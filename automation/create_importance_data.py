from data.model_data import ModelData
from neural_network_preprocessing.neural_network import ProcessedNetwork


def create_importance_all_types(model_data: ModelData = None):
    pn = ProcessedNetwork(model_data)
    pn.store_importance_data("dense_784_128_10_v4/not_centering_ones", "mnist/mnist_train_split",
                             "mnist/mnist_test_split", True, False, True, False)

    pn = ProcessedNetwork("dense_784_128_10")
    pn.store_importance_data("dense_784_128_10_v4/not_centering_ones_r", "mnist/mnist_train_split",
                             "mnist/mnist_test_split", True, False, True, True)

    pn = ProcessedNetwork("dense_784_128_10")
    pn.store_importance_data("dense_784_128_10_v4/centering_ones", "mnist/mnist_train_split",
                             "mnist/mnist_test_split", True, True, True, False)

    pn = ProcessedNetwork("dense_784_128_10")
    pn.store_importance_data("dense_784_128_10_v4/centering_ones_r", "mnist/mnist_train_split",
                             "mnist/mnist_test_split", True, True, True, True)
    pass
