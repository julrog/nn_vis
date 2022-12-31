import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np
from tensorflow import keras

from data.data_handler import ImportanceDataHandler
from data.model_data import ModelData
from neural_network_preprocessing.importance import (ImportanceCalculation,
                                                     ImportanceType,
                                                     get_importance_type_name)


class ImportanceEvaluator:
    def __init__(self, model_data: ModelData) -> None:
        self.model_data: ModelData = model_data
        self.importance_type: ImportanceType = ImportanceType(
            model_data.get_importance_type())
        self.importance_calculation: ImportanceCalculation = ImportanceCalculation.BNN_EDGE
        self.relevant_classes: Optional[List[int]] = None

        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None

    def set_train_and_test_data(self, x_train, y_train, x_test, y_test) -> None:
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def setup(self, importance_type: ImportanceType = ImportanceType.L1,
              importance_calculation: ImportanceCalculation = ImportanceCalculation.BNN_EDGE,
              relevant_classes: Optional[List[int]] = None) -> None:
        self.importance_type = importance_type
        self.importance_calculation = importance_calculation
        self.relevant_classes = relevant_classes

    def get_importance(self, edge_alpha: float, classes_importance: List[float]) -> float:
        if self.importance_calculation == ImportanceCalculation.EDGE_ONLY:
            return edge_alpha
        edge_sum_class_importance: float = 0.0
        if self.relevant_classes is not None:
            for r_class_id, class_importance in enumerate(classes_importance):
                if r_class_id in self.relevant_classes:
                    edge_sum_class_importance += class_importance
            if self.importance_calculation == ImportanceCalculation.BNN_EDGE:
                edge_sum_class_importance *= edge_alpha
            edge_sum_class_importance = edge_sum_class_importance / \
                float(len(self.relevant_classes))
        else:
            for class_index, class_importance in enumerate(classes_importance):
                edge_sum_class_importance += class_importance
            if self.importance_calculation == ImportanceCalculation.BNN_EDGE:
                edge_sum_class_importance *= edge_alpha
            edge_sum_class_importance = edge_sum_class_importance / \
                float(len(classes_importance))
        return edge_sum_class_importance

    def prune_model(self,
                    importance_prune_percent: str,
                    importance_data: ImportanceDataHandler,
                    importance_threshold: float) -> None:
        data: Dict[Any, Any] = dict()

        pruned_edges: int = 0
        remaining_edges: int = 0

        self.model_data.reload_model()
        for layer_id, layer in enumerate(importance_data.edge_importance_data):
            layer_weights = self.model_data.model.layers[layer_id + 1].get_weights(
            )
            for input_node_id, input_node in enumerate(layer):
                for edge_id, edge_alpha in enumerate(input_node):
                    classes_importance: List[float] = importance_data.node_importance_data[layer_id][input_node_id]
                    edge_importance: float = self.get_importance(
                        edge_alpha, classes_importance)
                    if edge_importance <= importance_threshold:
                        pruned_edges += 1
                        layer_weights[0][input_node_id][edge_id] = 0.0
                    else:
                        remaining_edges += 1
            self.model_data.model.layers[layer_id +
                                         1].set_weights(layer_weights)

        data['pruned_edges'] = pruned_edges
        data['overall_edged'] = remaining_edges + pruned_edges
        data['actual_prune_percentage'] = str(
            (100 * pruned_edges) / (remaining_edges + pruned_edges))
        data['importance_threshold'] = str(importance_threshold)

        data_key: str = get_importance_type_name(self.importance_type)
        if self.relevant_classes is not None:
            data_key = data_key + str(self.relevant_classes)
        self.model_data.store_data(
            data_key,
            importance_prune_percent,
            self.importance_calculation.name,
            data)

        logging.info('Pruned edges: %i' % pruned_edges)

    def accuracy_report(self, truths: np.array, predictions: np.array) -> Dict[str, Any]:
        accuracy_report: Dict[str, Any] = dict()
        for i in range(self.model_data.get_num_classes()):
            true_positives: int = 0
            true_negatives: int = 0
            false_positives: int = 0
            false_negatives: int = 0
            for truth, prediction in zip(truths, predictions):
                if prediction == i and truth == i:
                    true_positives += 1
                if prediction != i and truth != i:
                    true_negatives += 1
                if prediction == i and truth != i:
                    false_positives += 1
                if prediction != i and truth == i:
                    false_negatives += 1

            true_positive_rate: float = float(
                true_positives) / (float(true_positives + false_negatives))
            true_negative_rate: float = float(
                true_negatives) / (float(true_negatives + false_positives))
            accuracy_report[str(i)] = (
                true_positive_rate + true_negative_rate) / 2.0
        return accuracy_report

    def test_model(self, importance_prune_percent: str) -> None:
        self.model_data.model.compile(loss=keras.losses.categorical_crossentropy,
                                      optimizer=keras.optimizers.Adam(0.001),
                                      metrics=['accuracy'])
        train_score = self.model_data.model.evaluate(
            self.x_train, self.y_train, verbose=0)
        test_score = self.model_data.model.evaluate(
            self.x_test, self.y_test, verbose=0)

        logging.info('Train loss: %f, Train accuracy: %f' %
                     (train_score[0], train_score[1]))
        logging.info('Test loss: %f, Test accuracy: %f' %
                     (test_score[0], test_score[1]))

        truth_train: np.array = np.argmax(self.y_train, axis=1)
        prediction_train: np.array = self.model_data.model.predict_classes(
            self.x_train)
        truth_test: np.array = np.argmax(self.y_test, axis=1)
        prediction_test: np.array = self.model_data.model.predict_classes(
            self.x_test)

        train_class_accuracy_report: Dict[str, Any] = self.accuracy_report(
            truth_train, prediction_train)
        test_class_accuracy_report: Dict[str, Any] = self.accuracy_report(
            truth_test, prediction_test)

        importance_prune_data: Dict[str, Any] = dict()
        importance_prune_data['train_loss'] = str(train_score[0])
        importance_prune_data['train_accuracy'] = str(train_score[1])
        importance_prune_data['test_loss'] = str(test_score[0])
        importance_prune_data['test_accuracy'] = str(test_score[1])
        importance_prune_data['train_class_accuracy'] = train_class_accuracy_report
        importance_prune_data['test_class_accuracy'] = test_class_accuracy_report
        importance_type_name: str = get_importance_type_name(
            self.importance_type)
        if self.relevant_classes is not None:
            importance_type_name = importance_type_name + \
                str(self.relevant_classes)
        self.model_data.store_data(
            importance_type_name,
            importance_prune_percent,
            self.importance_calculation.name,
            importance_prune_data)

    def create_evaluation_data(self, step_size: int = 1, start_percentage: int = 0, end_percentage: int = 100) -> None:
        importance_data: ImportanceDataHandler = ImportanceDataHandler(
            self.model_data.get_path() + get_importance_type_name(self.importance_type) + '.imp.npz')

        edge_importance_data: List[float] = []
        self.model_data.reload_model()
        for layer_id, layer in enumerate(importance_data.edge_importance_data):
            for input_node_id, input_node in enumerate(layer):
                for edge_id, edge_alpha in enumerate(input_node):
                    classes_importance: List[float] = importance_data.node_importance_data[layer_id][input_node_id]
                    edge_importance: float = self.get_importance(
                        edge_alpha, classes_importance)
                    edge_importance_data.append(edge_importance)

        sorted_edge_importance: np.array = np.sort(
            np.array(edge_importance_data))

        for prune_percentage in range(start_percentage, end_percentage + step_size, step_size):
            importance_threshold: float = -1.0
            if prune_percentage > 0:
                if int((prune_percentage * sorted_edge_importance.shape[0]) / 100) >= len(sorted_edge_importance):
                    importance_threshold = sorted_edge_importance[len(
                        sorted_edge_importance) - 1]
                else:
                    importance_threshold = sorted_edge_importance[
                        int((prune_percentage * sorted_edge_importance.shape[0]) / 100)]
            time.sleep(1)

            self.prune_model(str(prune_percentage),
                             importance_data, importance_threshold)
            self.test_model(str(prune_percentage))
