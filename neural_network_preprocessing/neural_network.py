import os

import numpy as np
from typing import List, Tuple, Dict

from sklearn.metrics import classification_report
from tensorflow import keras
from tensorflow_core.python.keras import Model, Input, Sequential
from tensorflow_core.python.keras.layers import BatchNormalization, Dense
from tensorflow_core.python.keras.regularizers import l1_l2, l1, l2
from tensorflow_core.python.layers.base import Layer

from automation.create_mnist_model import generate_model_description
from data.model_data import ModelData
from definitions import BASE_PATH, DATA_PATH

LOG_SOURCE: str = "NEURAL_NETWORK"


def modify_model(model: Model, class_index: int, batch_norm_centering: bool = False,
                 importance_start_at_one: bool = False, regularize_gamma: str = "None") -> Model:
    gamma_initializer: str = "zeros"
    if importance_start_at_one:
        gamma_initializer = "ones"

    gamma_regularizer = None
    if regularize_gamma == "l1":
        gamma_regularizer = l1()
    if regularize_gamma == "l2":
        gamma_regularizer = l2()
    if regularize_gamma == "l1l2":
        gamma_regularizer = l1_l2()

    max_layer: int = len(model.layers)
    last_output: Input = None
    network_input: Input = None
    for i, layer in enumerate(model.layers):
        if i == 0:
            last_output = layer.output
            network_input = layer.input
        if 0 < i < max_layer:
            new_layer: Layer = BatchNormalization(center=batch_norm_centering, gamma_initializer=gamma_initializer,
                                                  gamma_regularizer=gamma_regularizer)
            last_output = new_layer(last_output)
        if i == max_layer - 1:
            new_end_layer: Layer = Dense(2, activation='softmax', name="binary_output_layer")
            last_output = new_end_layer(last_output)

            old_weights = layer.get_weights()
            old_weights[0] = np.transpose(old_weights[0], (1, 0))
            new_weights: List[np.array] = [
                np.append(old_weights[0][class_index:class_index + 1],
                          np.subtract(np.sum(old_weights[0], axis=0, keepdims=True),
                                      old_weights[0][class_index:class_index + 1]), axis=0),
                np.append(old_weights[1][class_index:class_index + 1],
                          np.subtract(np.sum(old_weights[1], axis=0, keepdims=True),
                                      old_weights[1][class_index:class_index + 1]), axis=0)
            ]
            new_weights[0] = np.transpose(new_weights[0], (1, 0))
            new_end_layer.set_weights(new_weights)
        elif i > 0:
            last_output = layer(last_output)

    return Model(inputs=network_input, outputs=last_output)


class ProcessedNetwork:
    def __init__(self, model_data: ModelData, store_path: str = None):
        self.model_data: ModelData = model_data
        self.name: str = "Undefined"
        self.num_classes: int = -1
        self.path: str = store_path if model_data is None else model_data.get_path()
        self.original_name: str = None if model_data is None else model_data.name

        self.model_data.reload_model()
        self.architecture_data: List[int] = []
        self.node_importance_value: List[List[np.array]] = []
        self.edge_importance_value: List[np.array] = []
        self.edge_importance_set: bool = False
        for i, layer in enumerate(model_data.model.layers):
            self.architecture_data.append(layer.output_shape[1])
            if i is not 0:
                self.node_importance_value.append([])
                self.edge_importance_value.append(None)
            if i is len(model_data.model.layers) - 1:
                self.num_classes = layer.output_shape[1]

        self.centering: bool = False
        self.gamma_one: bool = True
        self.regularize_gamma: str = "None"

    def get_fine_tuned_model_data(self, class_index: int, train_data: Tuple[np.array, np.array],
                                  test_data: Tuple[np.array, np.array]) -> Model:
        batch_size: int = 128
        epochs: int = 20
        learning_rate: float = 0.001

        x_train, y_train = train_data
        x_test, y_test = test_data

        # convert class vectors to binary class matrices
        y_train = keras.utils.to_categorical(y_train, 2)
        y_test = keras.utils.to_categorical(y_test, 2)

        self.model_data.reload_model()
        modified_model: Model = modify_model(self.model_data.model, class_index, self.centering, self.gamma_one,
                                             self.regularize_gamma)

        for layer in modified_model.layers:
            layer.trainable = False
            if layer.__class__.__name__ == "BatchNormalization":
                layer.trainable = True

        modified_model.compile(loss=keras.losses.categorical_crossentropy,
                               optimizer=keras.optimizers.Adam(learning_rate),
                               metrics=['accuracy'])

        modified_model.fit(x_train, y_train,
                           batch_size=batch_size,
                           epochs=epochs,
                           verbose=0,
                           validation_data=(x_test, y_test))

        train_score = modified_model.evaluate(x_train, y_train, verbose=0)
        test_score = modified_model.evaluate(x_test, y_test, verbose=0)
        print('[%s] Class %i: Train loss: %f, Train accuracy: %f, Test loss: %f, Test accuracy: %f' % (
            LOG_SOURCE, class_index, train_score[0], train_score[1], test_score[0], test_score[1]))

        c_y_test = np.argmax(y_test, axis=1)  # Convert one-hot to index
        prediction_test = np.argmax(modified_model.predict(x_test), axis=1)
        c_report: any = classification_report(c_y_test, prediction_test, output_dict=True)

        fine_tuned_data: Dict[any, any] = dict()
        fine_tuned_data['batch_size'] = str(batch_size)
        fine_tuned_data['learning_rate'] = str(learning_rate)
        fine_tuned_data['train_loss'] = str(train_score[0])
        fine_tuned_data['train_accuracy'] = str(train_score[1])
        fine_tuned_data['test_loss'] = str(test_score[0])
        fine_tuned_data['test_accuracy'] = str(test_score[1])
        fine_tuned_data['classification_report'] = c_report

        self.model_data.store_data("modified_fine_tuned_performance", self.name, "class_%i" % class_index,
                                   fine_tuned_data)

        return modified_model

    def extract_importance_from_model(self, original_model: Model, fine_tuned_model: Model):
        count: int = 0
        for layer in fine_tuned_model.layers:
            if layer.__class__.__name__ == "BatchNormalization":
                self.node_importance_value[count].append(layer.get_weights()[0])
                count += 1

        if not self.edge_importance_set:
            count: int = 0
            for i, layer in enumerate(original_model.layers):
                if layer.__class__.__name__ == "Dense":
                    self.edge_importance_value[count] = original_model.layers[i].get_weights()[0]
                    count += 1
            self.edge_importance_set = True

    def generate_importance_for_data(self, train_data_path: str, test_data_path: str) -> Tuple[List[np.array],
                                                                                               List[np.array]]:
        raw_train_data: dict = np.load("%s/%s.npz" % (DATA_PATH, train_data_path), allow_pickle=True)
        train_data: List[np.array] = raw_train_data["arr_0"]

        raw_test_data: dict = np.load("%s/%s.npz" % (DATA_PATH, test_data_path), allow_pickle=True)
        test_data: List[np.array] = raw_test_data["arr_0"]

        if (len(train_data) is not self.num_classes and self.num_classes is not None) or (
                len(test_data) is not self.num_classes and self.num_classes is not None):
            raise Exception("[%s] Data does not match number of classes %i." % (LOG_SOURCE, self.num_classes))

        for i, (class_test_data, class_train_data) in enumerate(zip(test_data, train_data)):
            fine_tuned_model = self.get_fine_tuned_model_data(i, class_train_data, class_test_data)
            self.extract_importance_from_model(self.model_data.model, fine_tuned_model)

        result_node_importance: List[np.array] = []
        for importance_values in self.node_importance_value:
            result_node_importance.append(np.stack(importance_values, axis=1))
        result_edge_importance: List[np.array] = []
        for importance_values in self.edge_importance_value:
            result_edge_importance.append(importance_values)
        return result_node_importance, result_edge_importance

    def store_importance_data(self, train_data_path: str, test_data_path: str, centering: bool = False,
                              gamma_one: bool = True, regularize_gamma: str = "l1"):
        self.centering = centering
        self.gamma_one = gamma_one
        self.regularize_gamma = regularize_gamma

        self.name: str = "beta_" if self.centering else "nobeta_"
        self.name += "gammaone_" if self.gamma_one else "gammazero_"
        if self.regularize_gamma is not "None":
            self.name += "l1_" if self.regularize_gamma == "l1" else "l2_" if self.regularize_gamma == "l2" else "l1l2_"

        importance_data: Tuple[List[np.array], List[np.array]] = self.generate_importance_for_data(train_data_path,
                                                                                                   test_data_path)
        node_importance_data: List[np.array] = importance_data[0]
        edge_importance_data: List[np.array] = importance_data[1]

        normalized_node_importance_data: List[np.array] = []
        min_node_importance: float = 1000000.0
        max_node_importance: float = 0.0
        for layer_importance in node_importance_data:
            normalized_layer_importance: np.array = np.absolute(layer_importance)
            for node_importance in normalized_layer_importance:
                for node_class_importance in node_importance:
                    if min_node_importance > node_class_importance:
                        min_node_importance = node_class_importance
                    if max_node_importance < node_class_importance:
                        max_node_importance = node_class_importance
            normalized_node_importance_data.append(normalized_layer_importance)
        for i, normalized_layer_importance in enumerate(normalized_node_importance_data):
            normalized_node_importance_data[i] = normalized_layer_importance / max_node_importance
        node_importance_data = normalized_node_importance_data
        print("[%s] Node importance - Min: %f, Max: %f" % (LOG_SOURCE, min_node_importance, max_node_importance))

        min_edge_importance: float = 1000000.0
        max_edge_importance: float = 0.0
        for layer_data in edge_importance_data:
            for i in range(layer_data.shape[0]):
                absolute_layer_data: np.array = np.abs(layer_data[i])
                current_edge_max: float = float(np.max(absolute_layer_data))
                current_edge_min: float = float(np.min(absolute_layer_data))
                if max_edge_importance < current_edge_max:
                    max_edge_importance = current_edge_max
                if min_edge_importance > current_edge_min:
                    min_edge_importance = current_edge_min
        normalized_edge_importance_data: List[np.array] = []
        for layer_data in edge_importance_data:
            new_layer_data: List[np.array] = []
            for i in range(layer_data.shape[0]):
                absolute_layer_data: np.array = np.abs(layer_data[i])
                absolute_layer_data /= max_edge_importance
                new_layer_data.append(absolute_layer_data)
            normalized_edge_importance_data.append(np.stack(new_layer_data, axis=0))
        edge_importance_data = normalized_edge_importance_data
        print("[%s] Edge importance - Min: %f, Max: %f" % (LOG_SOURCE, min_edge_importance, max_edge_importance))

        last_layer_node_importance = []
        for i in range(self.num_classes):
            new_node_data = np.zeros(self.num_classes, dtype=np.float32)
            new_node_data[i] = 1.0
            last_layer_node_importance.append(new_node_data)
        node_importance_data.append(last_layer_node_importance)

        data_path: str = self.model_data.get_path() + self.name + "_importance_data"
        if not os.path.exists(os.path.dirname(data_path)):
            os.makedirs(os.path.dirname(data_path))
        np.savez(data_path, (node_importance_data, edge_importance_data))

        importance_value_range_data: Dict[str, str] = dict()
        importance_value_range_data['min_node_importance'] = str(min_node_importance)
        importance_value_range_data['max_node_importance'] = str(max_node_importance)
        importance_value_range_data['min_edge_importance'] = str(min_edge_importance)
        importance_value_range_data['max_edge_importance'] = str(max_edge_importance)
        self.model_data.store_data("modified_fine_tuned_performance", self.name, "importance_value_range",
                                   importance_value_range_data)

    def store_importance_data_layer_normalized(self, train_data_path: str, test_data_path: str, centering: bool = False,
                              gamma_one: bool = True, regularize_gamma: str = "l1"):
        self.centering = centering
        self.gamma_one = gamma_one
        self.regularize_gamma = regularize_gamma

        self.name: str = "beta_" if self.centering else "nobeta_"
        self.name += "gammaone_" if self.gamma_one else "gammazero_"
        if self.regularize_gamma is not "None":
            self.name += "l1_" if self.regularize_gamma == "l1" else "l2_" if self.regularize_gamma == "l2" else "l1l2_"

        importance_data: Tuple[List[np.array], List[np.array]] = self.generate_importance_for_data(train_data_path,
                                                                                                   test_data_path)
        node_importance_data: List[np.array] = importance_data[0]
        edge_importance_data: List[np.array] = importance_data[1]

        normalized_node_importance_data: List[np.array] = []
        for layer_importance in node_importance_data:
            min_node_importance: float = 1000000.0
            max_node_importance: float = 0.0
            normalized_layer_importance: np.array = np.absolute(layer_importance)
            for node_importance in normalized_layer_importance:
                for node_class_importance in node_importance:
                    if min_node_importance > node_class_importance:
                        min_node_importance = node_class_importance
                    if max_node_importance < node_class_importance:
                        max_node_importance = node_class_importance
            normalized_node_importance_data.append(normalized_layer_importance / max_node_importance)
            print("[%s] Node importance - Min: %f, Max: %f" % (LOG_SOURCE, min_node_importance, max_node_importance))
        node_importance_data = normalized_node_importance_data
        # print("[%s] Node importance - Min: %f, Max: %f" % (LOG_SOURCE, min_node_importance, max_node_importance))

        normalized_edge_importance_data: List[np.array] = []
        for layer_data in edge_importance_data:
            min_edge_importance: float = 1000000.0
            max_edge_importance: float = 0.0
            new_layer_data: List[np.array] = []
            for i in range(layer_data.shape[0]):
                absolute_layer_data: np.array = np.abs(layer_data[i])
                current_edge_max: float = float(np.max(absolute_layer_data))
                current_edge_min: float = float(np.min(absolute_layer_data))
                if max_edge_importance < current_edge_max:
                    max_edge_importance = current_edge_max
                if min_edge_importance > current_edge_min:
                    min_edge_importance = current_edge_min
                absolute_layer_data /= max_edge_importance
                new_layer_data.append(absolute_layer_data)
            print("[%s] Edge importance - Min: %f, Max: %f" % (LOG_SOURCE, min_edge_importance, max_edge_importance))
            normalized_edge_importance_data.append(np.stack(new_layer_data, axis=0))
        edge_importance_data = normalized_edge_importance_data

        last_layer_node_importance = []
        for i in range(self.num_classes):
            new_node_data = np.zeros(self.num_classes, dtype=np.float32)
            new_node_data[i] = 1.0
            last_layer_node_importance.append(new_node_data)
        node_importance_data.append(last_layer_node_importance)

        data_path: str = self.model_data.get_path() + self.name + "_importance_data"
        if not os.path.exists(os.path.dirname(data_path)):
            os.makedirs(os.path.dirname(data_path))
        np.savez(data_path, (node_importance_data, edge_importance_data))

        importance_value_range_data: Dict[str, str] = dict()
        '''importance_value_range_data['min_node_importance'] = str(min_node_importance)
        importance_value_range_data['max_node_importance'] = str(max_node_importance)
        importance_value_range_data['min_edge_importance'] = str(min_edge_importance)
        importance_value_range_data['max_edge_importance'] = str(max_edge_importance)'''
        self.model_data.store_data("modified_fine_tuned_performance", self.name, "importance_value_range",
                                   importance_value_range_data)
