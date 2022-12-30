import logging
import os
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.metrics import classification_report
from tensorflow import keras
from tensorflow.keras import Model

from data.model_data import ModelData
from definitions import DATA_PATH
from neural_network_preprocessing.importance import (ImportanceType,
                                                     get_importance_type_name)
from neural_network_preprocessing.modify_model import modify_model


class ProcessedNetwork:
    def __init__(self, model_data: ModelData, store_path: str = None):
        self.model_data: ModelData = model_data
        self.name: str = 'Undefined'
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
            if i != 0:
                self.node_importance_value.append([])
                self.edge_importance_value.append(None)
            if i == len(model_data.model.layers) - 1:
                self.num_classes = layer.output_shape[1]

        self.importance_type: ImportanceType = ImportanceType(0)

    def get_fine_tuned_model_data(self, class_index: int, train_data: Tuple[np.array, np.array],
                                  test_data: Tuple[np.array, np.array]) -> Model:
        batch_size: int = 128
        epochs: int = 20
        learning_rate: float = 0.001

        x_train, y_train = train_data
        x_test, y_test = test_data

        y_train = keras.utils.to_categorical(y_train, 2)
        y_test = keras.utils.to_categorical(y_test, 2)

        self.model_data.reload_model()
        modified_model: Model = modify_model(
            self.model_data.model, class_index, self.importance_type)

        for layer in modified_model.layers:
            layer.trainable = False
            if layer.__class__.__name__ == 'BatchNormalization':
                layer.trainable = True

        modified_model.compile(loss=keras.losses.categorical_crossentropy,
                               optimizer=keras.optimizers.Adam(learning_rate),
                               metrics=['accuracy'])

        modified_model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=0,
                           validation_data=(x_test, y_test))

        train_score = modified_model.evaluate(x_train, y_train, verbose=0)
        test_score = modified_model.evaluate(x_test, y_test, verbose=0)
        logging.info('Class %i: Train loss: %f, Train accuracy: %f, Test loss: %f, Test accuracy: %f' % (
            class_index, train_score[0], train_score[1], test_score[0], test_score[1]))

        c_y_test = np.argmax(y_test, axis=1)
        prediction_test = np.argmax(modified_model.predict(x_test), axis=1)
        c_report: Any = classification_report(
            c_y_test, prediction_test, output_dict=True)

        fine_tuned_data: Dict[Any, Any] = dict()
        fine_tuned_data['batch_size'] = str(batch_size)
        fine_tuned_data['learning_rate'] = str(learning_rate)
        fine_tuned_data['train_loss'] = str(train_score[0])
        fine_tuned_data['train_accuracy'] = str(train_score[1])
        fine_tuned_data['test_loss'] = str(test_score[0])
        fine_tuned_data['test_accuracy'] = str(test_score[1])
        fine_tuned_data['classification_report'] = c_report

        self.model_data.store_data('modified_fine_tuned_performance', self.name, 'class_%i' % class_index,
                                   fine_tuned_data)

        return modified_model

    def extract_importance_from_model(self, original_model: Model, fine_tuned_model: Model):
        count: int = 0
        for layer in fine_tuned_model.layers:
            if layer.__class__.__name__ == 'BatchNormalization':
                self.node_importance_value[count].append(
                    layer.get_weights()[0])
                count += 1

        if not self.edge_importance_set:
            count: int = 0
            for i, layer in enumerate(original_model.layers):
                if layer.__class__.__name__ == 'Dense':
                    self.edge_importance_value[count] = original_model.layers[i].get_weights()[
                        0]
                    count += 1
            self.edge_importance_set = True

    def generate_importance_for_data(self, train_data_path: str, test_data_path: str) -> Tuple[List[np.array],
                                                                                               List[np.array]]:
        raw_train_data: dict = np.load(
            '%s/%s.npz' % (DATA_PATH, train_data_path), allow_pickle=True)
        train_data: List[np.array] = raw_train_data['arr_0']

        raw_test_data: dict = np.load(
            '%s/%s.npz' % (DATA_PATH, test_data_path), allow_pickle=True)
        test_data: List[np.array] = raw_test_data['arr_0']

        if (len(train_data) is not self.num_classes and self.num_classes is not None) or (
                len(test_data) is not self.num_classes and self.num_classes is not None):
            raise Exception(
                'Data does not match number of classes %i.' % self.num_classes)

        for i, (class_test_data, class_train_data) in enumerate(zip(test_data, train_data)):
            fine_tuned_model = self.get_fine_tuned_model_data(
                i, class_train_data, class_test_data)
            self.extract_importance_from_model(
                self.model_data.model, fine_tuned_model)

        result_node_importance: List[np.array] = []
        for importance_values in self.node_importance_value:
            result_node_importance.append(np.stack(importance_values, axis=1))
        result_edge_importance: List[np.array] = []
        for importance_values in self.edge_importance_value:
            result_edge_importance.append(importance_values)
        return result_node_importance, result_edge_importance

    def generate_importance_data(self, train_data_path: str, test_data_path: str,
                                 importance_type: ImportanceType = ImportanceType(
                                     ImportanceType.GAMMA | ImportanceType.L1)):
        self.importance_type = importance_type
        self.model_data.set_importance_type(importance_type)
        self.name = get_importance_type_name(self.importance_type)
        importance_data: Tuple[List[np.array], List[np.array]] = self.generate_importance_for_data(train_data_path,
                                                                                                   test_data_path)
        node_importance_data: List[np.array] = importance_data[0]
        edge_importance_data: List[np.array] = importance_data[1]

        normalized_node_importance_data: List[np.array] = []
        for layer_importance in node_importance_data:
            min_node_importance: float = 1000000.0
            max_node_importance: float = 0.0
            normalized_layer_importance: np.array = np.absolute(
                layer_importance)
            for node_importance in normalized_layer_importance:
                for node_class_importance in node_importance:
                    if min_node_importance > node_class_importance:
                        min_node_importance = node_class_importance
                    if max_node_importance < node_class_importance:
                        max_node_importance = node_class_importance
            normalized_node_importance_data.append(
                normalized_layer_importance / max_node_importance)
            logging.info('Node importance - Min: %f, Max: %f' %
                         (min_node_importance, max_node_importance))
        node_importance_data = normalized_node_importance_data

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
            logging.info('Edge importance - Min: %f, Max: %f' %
                         (min_edge_importance, max_edge_importance))
            normalized_edge_importance_data.append(
                np.stack(new_layer_data, axis=0))
        edge_importance_data = normalized_edge_importance_data

        last_layer_node_importance = []
        for i in range(self.num_classes):
            new_node_data = np.zeros(self.num_classes, dtype=np.float32)
            new_node_data[i] = 1.0
            last_layer_node_importance.append(new_node_data)
        node_importance_data.append(last_layer_node_importance)

        data_path: str = self.model_data.get_path() + self.name + '.imp.npz'
        if not os.path.exists(os.path.dirname(data_path)):
            os.makedirs(os.path.dirname(data_path))
        np.savez(data_path, (node_importance_data, edge_importance_data))

        importance_value_range_data: Dict[str, str] = dict()

        self.model_data.store_data('modified_fine_tuned_performance', self.name, 'importance_value_range',
                                   importance_value_range_data)
