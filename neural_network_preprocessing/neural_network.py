import os

import numpy as np
from typing import List, Tuple

from tensorflow import keras
from tensorflow_core.python.keras import Model, Input
from tensorflow_core.python.keras.layers import BatchNormalization, Dense
from tensorflow_core.python.layers.base import Layer

from definitions import BASE_PATH, DATA_PATH

LOG_SOURCE: str = "NEURAL_NETWORK"


def modify_model(model: Model, class_index: int, batch_norm_centering: bool = False,
                 importance_start_at_one: bool = False) -> Model:
    gamma_initializer: str = "zeros"
    if importance_start_at_one:
        gamma_initializer = "ones"
    max_layer: int = len(model.layers)
    last_output: Input = None
    network_input: Input = None
    for i, layer in enumerate(model.layers):
        if i == 0:
            last_output = layer.output
            network_input = layer.input
        if 0 < i < max_layer:
            new_layer: Layer = BatchNormalization(center=batch_norm_centering, gamma_initializer=gamma_initializer)
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
    def __init__(self, file: str):
        self.num_classes: int = -1
        self.model_path: str = BASE_PATH + '/storage/models/' + file
        model: Model = keras.models.load_model(self.model_path)
        print(model.summary())
        self.architecture_data: List[int] = []
        self.node_importance_value: List[List[np.array]] = []
        self.edge_importance_value: List[np.array] = []
        self.edge_importance_set: bool = False
        for i, layer in enumerate(model.layers):
            self.architecture_data.append(layer.output_shape[1])
            if i is not 0:
                self.node_importance_value.append([])
                self.edge_importance_value.append(None)
            if i is len(model.layers) - 1:
                self.num_classes = layer.output_shape[1]

        self.batch_norm_centering: bool = False
        self.importance_start_at_one: bool = False

    def get_fine_tuned_model_data(self, class_index: int, train_data: Tuple[np.array, np.array],
                                  test_data: Tuple[np.array, np.array]) -> Tuple[Model, Model]:
        batch_size: int = 128
        epochs: int = 20

        x_train, y_train = train_data
        x_test, y_test = test_data

        # convert class vectors to binary class matrices
        y_train = keras.utils.to_categorical(y_train, 2)
        y_test = keras.utils.to_categorical(y_test, 2)

        print('x_train shape:', x_train.shape)
        print(x_train.shape[0], 'train samples')
        print(x_test.shape[0], 'test samples')

        model: Model = keras.models.load_model(self.model_path)
        modified_model: Model = modify_model(model, class_index, self.batch_norm_centering,
                                             self.importance_start_at_one)

        for layer in modified_model.layers:
            layer.trainable = False
            if layer.__class__.__name__ == "BatchNormalization":
                layer.trainable = True

        print(modified_model.summary())

        modified_model.compile(loss=keras.losses.categorical_crossentropy,
                               optimizer=keras.optimizers.Adam(0.005),
                               metrics=['accuracy'])

        modified_model.fit(x_train, y_train,
                           batch_size=batch_size,
                           epochs=epochs,
                           verbose=1,
                           validation_data=(x_test, y_test))
        return model, modified_model

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
            original_model, fine_tuned_model = self.get_fine_tuned_model_data(i, class_train_data, class_test_data)
            self.extract_importance_from_model(original_model, fine_tuned_model)

        result_node_importance: List[np.array] = []
        for importance_values in self.node_importance_value:
            result_node_importance.append(np.stack(importance_values, axis=1))
        result_edge_importance: List[np.array] = []
        for importance_values in self.edge_importance_value:
            result_edge_importance.append(importance_values)
        return result_node_importance, result_edge_importance

    def store_importance_data(self, export_path: str, train_data_path: str, test_data_path: str,
                              normalize: bool = False, batch_norm_centering: bool = False,
                              importance_start_at_one: bool = False):
        self.batch_norm_centering: bool = batch_norm_centering
        self.importance_start_at_one: bool = importance_start_at_one

        importance_data: Tuple[List[np.array], List[np.array]] = self.generate_importance_for_data(train_data_path,
                                                                                                   test_data_path)
        node_importance_data: List[np.array] = importance_data[0]
        edge_importance_data: List[np.array] = importance_data[1]
        if normalize:
            normalized_node_importance_data: List[np.array] = []
            min_importance: float = 1000000.0
            max_importance: float = 0.0
            for layer_importance in node_importance_data:
                normalized_layer_importance: np.array = np.absolute(layer_importance)
                for node_importance in normalized_layer_importance:
                    for node_class_importance in node_importance:
                        if min_importance > node_class_importance:
                            min_importance = node_class_importance
                        if max_importance < node_class_importance:
                            max_importance = node_class_importance
                normalized_node_importance_data.append(normalized_layer_importance)
            for i, normalized_layer_importance in enumerate(normalized_node_importance_data):
                normalized_node_importance_data[i] = normalized_layer_importance / max_importance
            node_importance_data = normalized_node_importance_data
            print("[%s] Node importance - Min: %f, Max: %f" % (LOG_SOURCE, min_importance, max_importance))

            normalized_edge_importance_data: List[np.array] = []
            min_importance: float = 1000000.0
            max_importance: float = 0.0
            for layer_data in edge_importance_data:
                new_layer_data: List[np.array] = []
                for i in range(layer_data.shape[0]):
                    absolute_layer_data: np.array = np.abs(layer_data[i])
                    input_node_max: float = float(np.max(absolute_layer_data))
                    input_node_min: float = float(np.min(absolute_layer_data))
                    if max_importance < input_node_max:
                        max_importance = input_node_max
                    if min_importance > input_node_min:
                        min_importance = input_node_min
                    absolute_layer_data /= input_node_max
                    new_layer_data.append(absolute_layer_data)
                normalized_edge_importance_data.append(np.stack(new_layer_data, axis=0))
            edge_importance_data = normalized_edge_importance_data
            print("[%s] Edge importance - Min: %f, Max: %f" % (LOG_SOURCE, min_importance, max_importance))

        last_layer_node_importance = []
        for i in range(self.num_classes):
            new_node_data = np.zeros(self.num_classes, dtype=np.float32)
            new_node_data[i] = 1.0
            last_layer_node_importance.append(new_node_data)
        node_importance_data.append(last_layer_node_importance)

        data_path: str = DATA_PATH + export_path
        if not os.path.exists(os.path.dirname(data_path)):
            os.makedirs(os.path.dirname(data_path))
        np.savez(data_path, (node_importance_data, edge_importance_data))
