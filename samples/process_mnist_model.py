import os.path
from typing import List

from data.mnist_data_handler import split_mnist_data
from data.model_data import ModelData
from definitions import DATA_PATH
from neural_network_preprocessing.neural_network import ProcessedNetwork
from automation.automation_config import AutomationConfig
from automation.create_processed_model import process_network
from automation.create_mnist_model import create

name: str = "8_class"
class_selection: List[int] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
basic_model_data: ModelData = create(name=name, batch_size=128, epochs=15, layer_data=[81, 49], regularized=True,
                                     class_selection=class_selection)

split_suffix: str = '_' + ''.join(str(e) + '_' for e in class_selection) if class_selection is not None else ''
if not os.path.exists("%smnist/mnist_train_split%s" % (DATA_PATH, split_suffix)) or not os.path.exists(
        "%smnist/mnist_test_split" % DATA_PATH):
    split_mnist_data(class_selection)

pn = ProcessedNetwork(model_data=basic_model_data)
pn.store_importance_data_layer_normalized("mnist/mnist_train_split%s" % split_suffix,
                                          "mnist/mnist_test_split%s" % split_suffix,
                                          centering=False,
                                          gamma_one=True,
                                          regularize_gamma="l1")
basic_model_data.store_model_data()
basic_model_data.save_data()

automation_config: AutomationConfig = AutomationConfig()
process_network(name, "nobeta_gammaone_l1_", automation_config)
