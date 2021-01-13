import os
from typing import List, Tuple, Any
from tensorflow import keras
import numpy as np
from tensorflow_core.python.keras.datasets import mnist

from definitions import DATA_PATH

LOG_SOURCE: str = "MNIST_DATA_HANDLER"


def get_basic_data() -> Tuple[Tuple[Any, Any], Tuple[Any, Any], Any, Any]:
    num_classes: int = 10
    img_size: int = 28 * 28

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(x_train.shape[0], img_size, 1)
    x_test = x_test.reshape(x_test.shape[0], img_size, 1)
    input_shape = (img_size, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    return (x_train, y_train), (x_test, y_test), input_shape, num_classes


def get_prepared_data() -> Tuple[Tuple[Any, Any], Tuple[Any, Any], Any, Any]:
    (x_train, y_train), (x_test, y_test), input_shape, num_classes = get_basic_data()

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    return (x_train, y_train), (x_test, y_test), input_shape, num_classes


def get_unbalance_data(main_class: int, other_class_percentage: float) \
        -> Tuple[Tuple[Any, Any], Tuple[Any, Any], Any, Any]:
    (x_train, y_train), (x_test, y_test), input_shape, num_classes = get_basic_data()

    x_unbalanced_train = []
    y_unbalanced_train = []
    other_class_samples: int = int(other_class_percentage * x_train.shape[0])

    for result, image in zip(y_train, x_train):
        if result == main_class:
            x_unbalanced_train.append(image)
            y_unbalanced_train.append(result)

    for result, image in zip(y_train, x_train):
        if result != main_class and other_class_samples > 0:
            other_class_samples -= 1
            x_unbalanced_train.append(image)
            y_unbalanced_train.append(result)

    x_train, y_train = np.array(x_unbalanced_train).reshape([-1, input_shape[0], 1]), np.array(
        y_unbalanced_train).reshape([-1, 1])

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    return (x_train, y_train), (x_test, y_test), input_shape, num_classes


def split_mnist_data():
    (x_train, y_train), (x_test, y_test), input_shape, num_classes = get_basic_data()
    print("[%s] splitting %i train samples" % (LOG_SOURCE, x_train.shape[0]))
    print("[%s] splitting %i test samples" % (LOG_SOURCE, x_test.shape[0]))

    separated_train_data: List[Tuple[np.array or List[any], np.array or List[any]]] = [([], []) for _ in
                                                                                       range(num_classes)]
    separated_test_data: List[Tuple[np.array or List[any], np.array or List[any]]] = [([], []) for _ in
                                                                                      range(num_classes)]

    for result, image in zip(y_train, x_train):
        separated_train_data[result][0].append(image)
        separated_train_data[result][1].append(0)

    for result, image in zip(y_test, x_test):
        separated_test_data[result][0].append(image)
        separated_test_data[result][1].append(0)

    for i in range(num_classes):
        separated_train_data[i] = (np.array(separated_train_data[i][0]).reshape([-1, input_shape[0], 1]),
                                   np.array(separated_train_data[i][1]).reshape([-1, 1]))
        separated_test_data[i] = (np.array(separated_test_data[i][0]).reshape([-1, input_shape[0], 1]),
                                  np.array(separated_test_data[i][1]).reshape([-1, 1]))

    processed_separated_train_data: List[Tuple[np.array, np.array]] = [([], []) for _ in range(num_classes)]
    processed_separated_test_data: List[Tuple[np.array, np.array]] = [([], []) for _ in range(num_classes)]
    for i in range(num_classes):
        processed_separated_train_data[i] = (np.copy(separated_train_data[i][0]), np.copy(separated_train_data[i][1]))
        processed_separated_test_data[i] = (np.copy(separated_test_data[i][0]), np.copy(separated_test_data[i][1]))

    for i in range(num_classes):
        for j in range(num_classes):
            np.random.shuffle(separated_train_data[j][0])
            split_portion: int = int(len(separated_train_data[j][0]) / num_classes)
            processed_separated_train_data[i] = (
                np.append(processed_separated_train_data[i][0], separated_train_data[j][0][0:split_portion], axis=0),
                np.append(processed_separated_train_data[i][1], np.ones(split_portion).reshape(-1, 1), axis=0)
            )
            np.random.shuffle(separated_test_data[j][0])
            split_portion: int = int(len(separated_test_data[j][0]) / num_classes)
            processed_separated_test_data[i] = (
                np.append(processed_separated_test_data[i][0], separated_test_data[j][0][0:split_portion], axis=0),
                np.append(processed_separated_test_data[i][1], np.ones(split_portion).reshape(-1, 1), axis=0)
            )

    for i in range(num_classes):
        print("[%s] %i train samples for class #%i" % (LOG_SOURCE, processed_separated_train_data[i][0].shape[0], i))
        print("[%s] %i test samples for class #%i" % (LOG_SOURCE, processed_separated_test_data[i][0].shape[0], i))

    data_path: str = DATA_PATH + "mnist"
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    np.savez("%s/mnist_train_split" % data_path, processed_separated_train_data)
    np.savez("%s/mnist_test_split" % data_path, processed_separated_test_data)

    print("[%s] saved split data to \"%s\"" % (LOG_SOURCE, data_path))
