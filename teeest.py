from neural_network_preprocessing.neural_network import ProcessedNetwork

pn = ProcessedNetwork("dense_784_128_10")
pn.store_importance_data("dense_784_128_10_v3/not_centering_ones", "mnist/mnist_train_split",
                         "mnist/mnist_test_split", True, False, True)
