import numpy as np
from tensorflow_core.python.keras.utils.data_utils import get_file

from neural_network_preprocessing.neural_network import ProcessedNetwork

pn = ProcessedNetwork("dense_784_128_10")
importance_data = pn.generate_importance_for_data("mnist/mnist_train_split", "mnist/mnist_test_split")
for layer in importance_data:
    for vec in layer:
        for val in vec:
            if val > 0.0:
                print(val)
    print(layer)
    print(layer.shape)
