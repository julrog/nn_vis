import logging
import os
import os.path
import sys

sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(sys.modules[__name__].__file__), '..')))

if True:
    from typing import List

    from progressbar import ProgressBar

    from data.mnist_data_handler import get_prepared_data, split_mnist_data
    from data.model_data import ModelData
    from definitions import DATA_PATH
    from evaluation.create_plot import (
        create_importance_plot, create_importance_plot_compare_bn_parameter,
        create_importance_plot_compare_classes_vs_all,
        create_importance_plot_compare_regularizer, setup_plot)
    from evaluation.evaluator import ImportanceEvaluator
    from neural_network_preprocessing.create_mnist_model import create
    from neural_network_preprocessing.importance import (
        ImportanceCalculation, ImportanceType, get_importance_type_name)
    from neural_network_preprocessing.neural_network import ProcessedNetwork
    from utility.log_handling import setup_logger


def create_importance_data(model_data: ModelData, importance_type: ImportanceType):
    pn = ProcessedNetwork(model_data=model_data)
    pn.generate_importance_data('mnist/mnist_train_split%s' % split_suffix,
                                'mnist/mnist_test_split%s' % split_suffix,
                                importance_type)
    model_data.store_model_data()
    model_data.save_data()


def evaluate_importance(model_data: ModelData, importance_type: ImportanceType, importance_calculation: ImportanceCalculation):
    model_data.reload_model()
    importance_handler: ImportanceEvaluator = ImportanceEvaluator(model_data)
    importance_handler.setup(importance_type, importance_calculation)
    (x_train, y_train), (x_test, y_test), input_shape, num_classes = get_prepared_data(
        model_data.get_class_selection())
    importance_handler.set_train_and_test_data(
        x_train, y_train, x_test, y_test)
    importance_handler.create_evaluation_data(10)


setup_logger('evaluation')
logging.info('Evaluation will take some time...')
setup_plot()

name: str = 'default_all'
layer_data: List[int] = [81, 49]

model_data: ModelData = create(
    name=name, batch_size=128, epochs=15, layer_data=layer_data, regularized=False)
split_suffix: str = ''
if not os.path.exists('%smnist/mnist_train_split%s' % (DATA_PATH, split_suffix)) or not os.path.exists(
        '%smnist/mnist_test_split' % DATA_PATH):
    split_mnist_data()

importance_types: List[ImportanceType] = [
    ImportanceType(ImportanceType.GAMMA),
    ImportanceType(ImportanceType.GAMMA | ImportanceType.L1),
    ImportanceType(ImportanceType.GAMMA | ImportanceType.L2),
    ImportanceType(ImportanceType.GAMMA |
                   ImportanceType.L1 | ImportanceType.L2),
    ImportanceType(ImportanceType.L1),
    ImportanceType(ImportanceType.CENTERING | ImportanceType.L1),
    ImportanceType(ImportanceType.CENTERING |
                   ImportanceType.GAMMA | ImportanceType.L1),
]

importance_calculations: List[ImportanceCalculation] = [
    ImportanceCalculation.BNN_EDGE,
    ImportanceCalculation.BNN_ONLY,
    ImportanceCalculation.EDGE_ONLY
]

bar: ProgressBar = ProgressBar(max_value=len(
    importance_calculations) * len(importance_types))
bar.start()
count: int = 0
for it in importance_types:
    create_importance_data(model_data, it)
    for ic in importance_calculations:
        evaluate_importance(model_data, it, ic)

    importance_type_name: str = get_importance_type_name(it)
    create_importance_plot(name, importance_type_name, True, False)
    for ic in importance_calculations:
        create_importance_plot_compare_classes_vs_all(name, importance_type_name, ic.name,
                                                      False, True, False)
        count += 1
        bar.update(count)
create_importance_plot_compare_regularizer(name, ['nobeta_gammaone', 'nobeta_gammaone_l1', 'nobeta_gammaone_l2',
                                                  'nobeta_gammaone_l1l2'], 'BNN_EDGE', True, False)
create_importance_plot_compare_bn_parameter(name, ['nobeta_gammaone_l1', 'beta_gammaone_l1', 'beta_gammazero_l1',
                                                   'nobeta_gammazero_l1'], 'BNN_EDGE', True, False)
