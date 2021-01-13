import os.path

from data.mnist_data_handler import split_mnist_data
from data.model_data import ModelData
from definitions import DATA_PATH
from neural_network_preprocessing.neural_network import ProcessedNetwork
from automation.automation_config import AutomationConfig
from automation.create_processed_model import process_network
from automation.create_mnist_model import create

name: str = "showcase_v3_regu"
basic_model_data: ModelData = create(name=name, batch_size=128, epochs=15, layer_data=[81, 49], regularized=True)

if not os.path.exists("%smnist/mnist_train_split" % DATA_PATH) or not os.path.exists(
        "%smnist/mnist_test_split" % DATA_PATH):
    split_mnist_data()

pn = ProcessedNetwork(model_data=basic_model_data)
pn.store_importance_data_layer_normalized("mnist/mnist_train_split", "mnist/mnist_test_split", centering=False,
                                          gamma_one=True,
                                          regularize_gamma="l1")
basic_model_data.store_model_data()
basic_model_data.save_data()

automation_config: AutomationConfig = AutomationConfig()
process_network(name, "nobeta_gammaone_l1_", automation_config)
