import os
from typing import Tuple, Dict, List

from sklearn.metrics import classification_report
from tensorflow_core.python.keras.datasets import mnist

from data.data_handler import ImportanceDataHandler
from data.model_data import ModelData
from tensorflow import keras
import numpy as np


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


def prune_model(model_data: ModelData, importance_data: ImportanceDataHandler, importance_threshold: float) -> \
        Dict[any, any]:
    data: Dict[any, any] = dict()

    pruned_edges: int = 0

    model_data.reload_model()
    for layer_id, layer in enumerate(importance_data.edge_importance_data):
        layer_weights = model_data.model.layers[layer_id + 1].get_weights()
        print(layer_weights)
        print(len(layer_weights[0]))
        print(len(layer_weights[1]))
        for input_node_id, input_node in enumerate(layer):
            for edge_id, edge in enumerate(input_node):
                edge_sum_class_importance: float = 0.0
                for class_importance in importance_data.node_importance_data[layer_id][input_node_id]:
                    edge_sum_class_importance += class_importance * edge
                edge_sum_class_importance = edge_sum_class_importance / float(
                    len(importance_data.node_importance_data[layer_id][input_node_id]))
                if edge_sum_class_importance <= importance_threshold:
                    pruned_edges += 1
                    layer_weights[0][input_node_id][edge_id] = 0.0
        print(layer_weights)
        model_data.model.layers[layer_id + 1].set_weights(layer_weights)

    print("Pruned edges: %i" % pruned_edges)

    return data


def test_model(importance_type: str, importance_prune_percent: str, model_data: ModelData, x_train, y_train, x_test,
               y_test):
    train_score = model_data.model.evaluate(x_test, y_test, verbose=0)
    test_score = model_data.model.evaluate(x_test, y_test, verbose=0)

    print('Train loss: %f, Train accuracy: %f' % (train_score[0], train_score[1]))
    print('Test loss: %f, Test accuracy: %f' % (test_score[0], test_score[1]))

    c_y_train = np.argmax(y_train, axis=1)  # Convert one-hot to index
    prediction_train = model_data.model.predict_classes(x_train)
    train_c_report: dict = classification_report(c_y_train, prediction_train, output_dict=True)
    print(train_c_report)

    c_y_test = np.argmax(y_test, axis=1)  # Convert one-hot to index
    prediction_test = model_data.model.predict_classes(x_test)
    test_c_report: dict = classification_report(c_y_test, prediction_test, output_dict=True)
    print(test_c_report)

    importance_prune_data: Dict[str, any] = dict()
    importance_prune_data['train_loss'] = str(train_score[0])
    importance_prune_data['train_accuracy'] = str(train_score[1])
    importance_prune_data['test_loss'] = str(test_score[0])
    importance_prune_data['test_accuracy'] = str(test_score[1])
    importance_prune_data['train_c_report'] = train_c_report
    importance_prune_data['test_c_report'] = test_c_report

    model_data.store_main_data(importance_type, importance_prune_percent, importance_prune_data)


def create_evaluation_data(model_data: ModelData, importance_type: str):
    x_train, y_train, x_test, y_test = process_train_test_data()

    importance_data: ImportanceDataHandler = ImportanceDataHandler(
        model_data.get_path() + importance_type + "_importance_data.npz")

    edge_importance_data: List[float] = []
    for layer_id, layer in enumerate(importance_data.edge_importance_data):
        for input_node_id, input_node in enumerate(layer):
            for edge_alpha in input_node:
                edge_sum_class_importance: float = 0.0
                for class_importance in importance_data.node_importance_data[layer_id][input_node_id]:
                    edge_sum_class_importance += class_importance * edge_alpha
                edge_importance_data.append(edge_sum_class_importance / float(
                    len(importance_data.node_importance_data[layer_id][input_node_id])))

    sorted_edge_importance: np.array = np.sort(np.array(edge_importance_data))

    data: Dict[any, any] = prune_model(model_data, importance_data, 0.9)
    '''for prune_percent in range(0, 10, 1):
        importance_threshold: float = 0.0
        if prune_percent != 0:
            importance_threshold = sorted_edge_importance[int((prune_percent * sorted_edge_importance.shape[0]) / 100)]
            data: Dict[any, any] = prune_model(model_data, importance_data.edge_importance_data, 0.9)

    for prune_percent in range(10, 100, 5):
        importance_threshold: float = sorted_edge_importance[
            int((prune_percent * sorted_edge_importance.shape[0]) / 100)]

    # modified_model: Model = prune_model(model_data)'''
