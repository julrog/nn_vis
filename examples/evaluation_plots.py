import os
from typing import List

from data.mnist_data_handler import split_mnist_data, get_prepared_data
from data.model_data import ModelData
from definitions import DATA_PATH
from evaluation.create_plot import setup_plot, create_importance_plot, create_importance_plot_compare_classes_vs_all, \
    create_importance_plot_compare_regularizer, create_importance_plot_compare_bn_parameter
from evaluation.evaluator import ImportanceEvaluator
from neural_network_preprocessing.create_mnist_model import create
from neural_network_preprocessing.importance import ImportanceType, get_importance_type_name, ImportanceCalculation
from neural_network_preprocessing.neural_network import ProcessedNetwork


def create_importance_data(model_data: ModelData, importance_type: ImportanceType):
    pn = ProcessedNetwork(model_data=model_data)
    pn.generate_importance_data("mnist/mnist_train_split%s" % split_suffix,
                                "mnist/mnist_test_split%s" % split_suffix,
                                importance_type)
    model_data.store_model_data()
    model_data.save_data()


def evaluate_importance(name: str, importance_calculation: ImportanceCalculation):
    model_data: ModelData = ModelData(name)
    model_data.reload_model()
    importance_handler: ImportanceEvaluator = ImportanceEvaluator(model_data)
    importance_handler.setup(importance_calculation)
    (x_train, y_train), (x_test, y_test), input_shape, num_classes = get_prepared_data(model_data.get_class_selection())
    importance_handler.set_train_and_test_data(x_train, y_train, x_test, y_test)
    importance_handler.create_evaluation_data(10)


setup_plot()

name: str = "default_all"
layer_data: List[int] = [81, 49]

model_data: ModelData = create(name=name, batch_size=128, epochs=15, layer_data=layer_data, regularized=False)
split_suffix: str = ""
if not os.path.exists("%smnist/mnist_train_split%s" % (DATA_PATH, split_suffix)) or not os.path.exists(
        "%smnist/mnist_test_split" % DATA_PATH):
    split_mnist_data()

importance_types: List[ImportanceType] = [
    ImportanceType(ImportanceType.GAMMA),
    ImportanceType(ImportanceType.GAMMA | ImportanceType.L1),
    ImportanceType(ImportanceType.GAMMA | ImportanceType.L2),
    ImportanceType(ImportanceType.GAMMA | ImportanceType.L1 | ImportanceType.L2),
    ImportanceType(ImportanceType.L1),
    ImportanceType(ImportanceType.CENTERING | ImportanceType.L1),
    ImportanceType(ImportanceType.CENTERING | ImportanceType.GAMMA | ImportanceType.L1),
]

importance_calculations: List[ImportanceCalculation] = [
    ImportanceCalculation.BNN_EDGE,
    ImportanceCalculation.BNN_ONLY,
    ImportanceCalculation.EDGE_ONLY
]

for importance_type in importance_types:
    create_importance_data(model_data, importance_type)
    for importance_calculation in importance_calculations:
        evaluate_importance(name, importance_calculation)

    importance_type_name: str = get_importance_type_name(importance_type)
    create_importance_plot("default_all", importance_type_name, True, False)
    for importance_calculation in importance_calculations:
        create_importance_plot_compare_classes_vs_all(name, importance_type_name, importance_calculation.name,
                                                      False, True, False)
create_importance_plot_compare_regularizer(name, ["nobeta_gammaone", "nobeta_gammaone_l1", "nobeta_gammaone_l2",
                                                  "nobeta_gammaone_l1l2"], "BNN_EDGE", True, False)
create_importance_plot_compare_bn_parameter(name, ["nobeta_gammaone_l1", "beta_gammaone_l1", "beta_gammazero_l1",
                                                   "nobeta_gammazero_l1"], "BNN_EDGE", True, False)
