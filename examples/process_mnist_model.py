import os.path
import sys
from typing import List

sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(sys.modules[__name__].__file__), '..')))

if True:
    from data.mnist_data_handler import split_mnist_data
    from data.model_data import ModelData
    from definitions import DATA_PATH
    from neural_network_preprocessing.create_mnist_model import create
    from neural_network_preprocessing.importance import (
        ImportanceType, get_importance_type_name)
    from neural_network_preprocessing.neural_network import ProcessedNetwork
    from processing.processing_handler import RecordingProcessingHandler
    from utility.log_handling import setup_logger
    from utility.recording_config import RecordingConfig


setup_logger('sample_processing')

# -------------------------------------------------change these settings-----------------------------------------------#
name: str = 'default'
class_selection: List[int] or None = None  # [0, 1, 2, 3, 4]
importance_type: ImportanceType = ImportanceType(
    ImportanceType.GAMMA | ImportanceType.L1)

basic_model_data: ModelData = create(name=name, batch_size=128, epochs=15, layer_data=[81, 49], regularized=False,
                                     class_selection=class_selection)
# ---------------------------------------------------------------------------------------------------------------------#

split_suffix: str = ''
if class_selection is not None:
    ('_' + ''.join(str(e) + '_' for e in class_selection))
if not os.path.exists('%smnist/mnist_train_split%s' % (DATA_PATH, split_suffix)) or not os.path.exists(
        '%smnist/mnist_test_split' % DATA_PATH):
    split_mnist_data(class_selection)

pn = ProcessedNetwork(model_data=basic_model_data)
pn.generate_importance_data('mnist/mnist_train_split%s' % split_suffix,
                            'mnist/mnist_test_split%s' % split_suffix,
                            importance_type)
basic_model_data.store_model_data()
basic_model_data.save_data()

recording_config: RecordingConfig = RecordingConfig()
processHandler: RecordingProcessingHandler = RecordingProcessingHandler(name, get_importance_type_name(importance_type),
                                                                        recording_config)
processHandler.process()
