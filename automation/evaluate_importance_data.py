import time
import numpy as np
from typing import Tuple, Dict, List
from tensorflow_core.python.keras.datasets import mnist
from data.data_handler import ImportanceDataHandler
from data.model_data import ModelData
from tensorflow import keras


def process_train_test_data() -> Tuple[np.array, np.array, np.array, np.array]:
    num_classes: int = 10
    img_size: int = 28 * 28

    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(x_train.shape[0], img_size, 1)
    x_test = x_test.reshape(x_test.shape[0], img_size, 1)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    return x_train, y_train, x_test, y_test


def get_added_bn_importance_class(edge_alpha: float, classes_importance: List[float],
                                  relevant_classes: List[int]) -> float:
    edge_sum_class_importance: float = 0.0
    class_count: int = 0
    for class_id, class_importance in enumerate(classes_importance):
        if class_id in relevant_classes:
            class_count += 1
            edge_sum_class_importance += class_importance * edge_alpha
    edge_sum_class_importance = edge_sum_class_importance / float(class_count)
    return edge_sum_class_importance


def get_added_bn_importance(edge_alpha: float, classes_importance: List[float],
                            relevant_classes: List[int] = None) -> float:
    if relevant_classes is not None:
        return get_added_bn_importance_class(edge_alpha, classes_importance, relevant_classes)
    edge_sum_class_importance: float = 0.0
    for class_importance in classes_importance:
        edge_sum_class_importance += class_importance * edge_alpha
    edge_sum_class_importance = edge_sum_class_importance / float(len(classes_importance))
    return edge_sum_class_importance


def get_only_bn_importance(classes_importance: List[float], relevant_classes: List[int] = None) -> float:
    edge_sum_class_importance: float = 0.0
    if relevant_classes is not None:
        for class_index, class_importance in enumerate(classes_importance):
            if class_index in relevant_classes:
                edge_sum_class_importance += class_importance
        edge_sum_class_importance = edge_sum_class_importance / float(len(relevant_classes))
    else:
        for class_index, class_importance in enumerate(classes_importance):
            edge_sum_class_importance += class_importance
        edge_sum_class_importance = edge_sum_class_importance / float(len(classes_importance))
    return edge_sum_class_importance


def prune_model(importance_type: str, importance_prune_percent: str, importance_mode: str, model_data: ModelData,
                importance_data: ImportanceDataHandler, importance_threshold: float,
                relevant_classes: List[int] = None):
    data: Dict[any, any] = dict()

    pruned_edges: int = 0
    remaining_edges: int = 0

    model_data.reload_model()
    for layer_id, layer in enumerate(importance_data.edge_importance_data):
        layer_weights = model_data.model.layers[layer_id + 1].get_weights()
        for input_node_id, input_node in enumerate(layer):
            for edge_id, edge_alpha in enumerate(input_node):
                edge_importance: float = 0.0
                if "bn_node_importance_added" in importance_mode:
                    edge_importance = get_added_bn_importance(edge_alpha,
                                                              importance_data.node_importance_data[layer_id][
                                                                  input_node_id], relevant_classes)
                elif "bn_node_importance_only" in importance_mode:
                    edge_importance = get_only_bn_importance(importance_data.node_importance_data[layer_id][
                                                                 input_node_id], relevant_classes)
                else:
                    edge_importance = edge_alpha
                if edge_importance <= importance_threshold:
                    pruned_edges += 1
                    layer_weights[0][input_node_id][edge_id] = 0.0
                else:
                    remaining_edges += 1
        model_data.model.layers[layer_id + 1].set_weights(layer_weights)

    data["pruned_edges"] = pruned_edges
    data["overall_edged"] = remaining_edges + pruned_edges
    data["actual_prune_percentage"] = str((100 * pruned_edges) / (remaining_edges + pruned_edges))
    data["importance_threshold"] = str(importance_threshold)

    print("Pruned edges: %i" % pruned_edges)

    model_data.model.compile(loss=keras.losses.categorical_crossentropy,
                             optimizer=keras.optimizers.Adam(0.001),
                             metrics=['accuracy'])

    model_data.store_data(importance_type, importance_prune_percent, importance_mode, data)


def test_model(importance_type: str, importance_prune_percent: str, importance_mode: str, model_data: ModelData,
               x_train, y_train, x_test, y_test):
    train_score = model_data.model.evaluate(x_train, y_train, verbose=0)
    test_score = model_data.model.evaluate(x_test, y_test, verbose=0)

    print('Train loss: %f, Train accuracy: %f' % (train_score[0], train_score[1]))
    print('Test loss: %f, Test accuracy: %f' % (test_score[0], test_score[1]))

    c_y_train = np.argmax(y_train, axis=1)  # Convert one-hot to index
    prediction_train = model_data.model.predict_classes(x_train)
    # train_c_report: dict = classification_report(c_y_train, prediction_train, output_dict=True)

    c_y_test = np.argmax(y_test, axis=1)  # Convert one-hot to index
    prediction_test = model_data.model.predict_classes(x_test)
    # test_c_report: dict = classification_report(c_y_test, prediction_test, output_dict=True)

    num_classes: int = 10

    train_class_accuracy_report: Dict[str, any] = dict()
    for i in range(num_classes):
        true_positives: int = 0
        true_negatives: int = 0
        false_positives: int = 0
        false_negatives: int = 0
        for truth, prediction in zip(c_y_train, prediction_train):
            if prediction == i and truth == i:
                true_positives += 1
            if prediction != i and truth != i:
                true_negatives += 1
            if prediction == i and truth != i:
                false_positives += 1
            if prediction != i and truth == i:
                false_negatives += 1

        true_positive_rate: float = float(true_positives) / (float(true_positives + false_negatives))
        true_negative_rate: float = float(true_negatives) / (float(true_negatives + false_positives))
        train_class_accuracy_report[str(i)] = (true_positive_rate + true_negative_rate) / 2.0

    test_class_accuracy_report: Dict[str, any] = dict()
    for i in range(num_classes):
        true_positives: int = 0
        true_negatives: int = 0
        false_positives: int = 0
        false_negatives: int = 0
        for truth, prediction in zip(c_y_test, prediction_test):
            if prediction == i and truth == i:
                true_positives += 1
            if prediction != i and truth != i:
                true_negatives += 1
            if prediction == i and truth != i:
                false_positives += 1
            if prediction != i and truth == i:
                false_negatives += 1

        true_positive_rate: float = float(true_positives) / (float(true_positives + false_negatives))
        true_negative_rate: float = float(true_negatives) / (float(true_negatives + false_positives))
        test_class_accuracy_report[str(i)] = (true_positive_rate + true_negative_rate) / 2.0

    importance_prune_data: Dict[str, any] = dict()
    importance_prune_data['train_loss'] = str(train_score[0])
    importance_prune_data['train_accuracy'] = str(train_score[1])
    importance_prune_data['test_loss'] = str(test_score[0])
    importance_prune_data['test_accuracy'] = str(test_score[1])
    # importance_prune_data['train_c_report'] = train_c_report
    # importance_prune_data['test_c_report'] = test_c_report
    importance_prune_data['train_class_accuracy'] = train_class_accuracy_report
    importance_prune_data['test_class_accuracy'] = test_class_accuracy_report

    model_data.store_data(importance_type, importance_prune_percent, importance_mode, importance_prune_data)


def create_evaluation_data(model_data: ModelData, importance_type: str,
                           importance_mode: str = "bn_node_importance_added", step_size: int = 1,
                           relevant_classes: List[int] = None, start_percentage: int = 0, end_percentage: int = 100):
    x_train, y_train, x_test, y_test = process_train_test_data()

    importance_data: ImportanceDataHandler = ImportanceDataHandler(
        model_data.get_path() + importance_type + "_importance_data.npz")

    edge_importance_data: List[float] = []
    for layer_id, layer in enumerate(importance_data.edge_importance_data):
        for input_node_id, input_node in enumerate(layer):
            for edge_alpha in input_node:
                edge_importance: float = 0.0
                if "bn_node_importance_added" in importance_mode:
                    edge_importance = get_added_bn_importance(edge_alpha,
                                                              importance_data.node_importance_data[layer_id][
                                                                  input_node_id], relevant_classes)
                elif "bn_node_importance_only" in importance_mode:
                    edge_importance = get_only_bn_importance(importance_data.node_importance_data[layer_id][
                                                                 input_node_id], relevant_classes)
                else:
                    edge_importance = edge_alpha
                edge_importance_data.append(edge_importance)

    sorted_edge_importance: np.array = np.sort(np.array(edge_importance_data))

    for prune_percentage in range(start_percentage, end_percentage + step_size, step_size):
        importance_threshold: float = -1.0
        if prune_percentage > 0:
            if int((prune_percentage * sorted_edge_importance.shape[0]) / 100) >= len(sorted_edge_importance):
                importance_threshold = sorted_edge_importance[len(sorted_edge_importance) - 1]
            else:
                importance_threshold = sorted_edge_importance[
                    int((prune_percentage * sorted_edge_importance.shape[0]) / 100)]
        time.sleep(1)

        imp_mode: str = importance_mode
        if relevant_classes is not None:
            imp_mode = importance_mode + "_" + str(relevant_classes)
        prune_model(importance_type, str(prune_percentage), imp_mode, model_data,
                    importance_data, importance_threshold, relevant_classes)
        test_model(importance_type, str(prune_percentage), imp_mode, model_data,
                   x_train, y_train, x_test, y_test)
