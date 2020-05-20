from neural_network_preprocessing.neural_network import ProcessedNetwork

pn = ProcessedNetwork("dense_784_128_10")
importance_data = pn.store_importance_data("dense_784_128_10/importance", "mnist/mnist_train_split",
                                           "mnist/mnist_test_split", True)
