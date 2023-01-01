import os
from enum import Enum
from typing import Any, Dict, List, Optional

from tensorflow import keras
from tensorflow.keras.models import Model

from definitions import DATA_PATH
from utility.file import EvaluationFile


class ModelTrainType(Enum):
    UNTRAINED = 0
    BALANCED = 1
    UNBALANCED = 2


SUPPORTED_LAYER: List[str] = ['Dense']
IGNORED_LAYER: List[str] = ['Flatten']


class ModelData:
    def __init__(self, name: str, description: Optional[str] = None, model: Optional[Model] = None) -> None:
        self.name: str = name
        self.model: Model = model if model is not None else keras.models.load_model(
            self.get_model_path())
        self.check_model_supported_layer()
        self.data: dict = dict()
        self.data['name'] = self.name
        if description is not None:
            self.data['description'] = description
        self.data_file: EvaluationFile = EvaluationFile(self.name)
        self.data_file.read_data()

    def set_parameter(self, batch_size: int, epochs: int, layer_data: List[int], learning_rate: float,
                      training_samples: int, test_samples: int) -> None:
        self.data['batch_size'] = batch_size
        self.data['epochs'] = epochs
        self.data['layer_data'] = layer_data
        self.data['num_classes'] = layer_data[len(layer_data) - 1]
        self.data['learning_rate'] = str(learning_rate)
        self.data['training_samples'] = training_samples
        self.data['test_samples'] = test_samples

    def set_initial_performance(self, test_loss: float, test_accuracy: float, train_loss: float, train_accuracy: float,
                                classification_report: Any) -> None:
        self.data['test_loss'] = str(test_loss)
        self.data['test_accuracy'] = str(test_accuracy)
        self.data['train_loss'] = str(train_loss)
        self.data['train_accuracy'] = str(train_accuracy)
        self.data['classification_report'] = classification_report

    def set_class_selection(self, class_selection: Optional[List[int]]) -> None:
        importance: dict = dict()
        importance['class_selection'] = class_selection
        self.data_file.append_main_data('processed', 'importance', importance)
        self.data_file.write_data()

    def set_importance_type(self, importance_type: int) -> None:
        importance: dict = dict()
        importance['importance_type'] = importance_type
        self.data_file.append_main_data('processed', 'importance', importance)
        self.data_file.write_data()

    def get_num_classes(self) -> int:
        return self.data_file.data_cache['overall']['basic_model']['num_classes']

    def get_class_selection(self) -> Optional[List[int]]:
        return self.data_file.data_cache['processed']['importance']['class_selection']

    def get_importance_type(self) -> int:
        return self.data_file.data_cache['processed']['importance']['importance_type']

    def store_model_data(self) -> None:
        self.data_file.append_main_data('overall', 'basic_model', self.data)
        self.data_file.write_data()

    def store_main_data(self, key: str, sub_key: str, data: Dict[Any, Any]) -> None:
        self.data_file.append_main_data(key, sub_key, data)
        self.data_file.write_data()

    def store_data(self, key: str, sub_key: str, sub_sub_key: str, data: Dict[Any, Any]) -> None:
        self.data_file.append_data(key, sub_key, sub_sub_key, data)
        self.data_file.write_data()

    def save_model(self) -> None:
        path: str = DATA_PATH + 'model/' + self.name + '/tf_model'
        if not os.path.exists(path):
            os.makedirs(path)
        self.model.save(path)

    def reload_model(self) -> None:
        self.model = keras.models.load_model(self.get_model_path())
        self.check_model_supported_layer()

    def get_model_path(self) -> str:
        return DATA_PATH + 'model/' + self.name + '/tf_model'

    def get_path(self) -> str:
        return DATA_PATH + 'model/' + self.name + '/'

    def save_data(self) -> None:
        self.data_file.write_data()

    def check_model_supported_layer(self) -> None:
        for index, layer in enumerate(self.model.layers):
            if layer.__class__.__name__ not in SUPPORTED_LAYER and layer.__class__.__name__ not in IGNORED_LAYER:
                raise Exception(
                    "'%s' layer type of model not supported!" % layer.__class__.__name__)
