import logging
import os
from typing import Any, List, Optional, Tuple

import numpy as np
from tensorflow import keras
from tensorflow.keras.datasets import mnist

from definitions import DATA_PATH


def get_basic_data(categorical: bool = False) -> Tuple[Tuple[Any, Any], Tuple[Any, Any], Any, Any]:
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

    if categorical:
        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)

    return (x_train, y_train), (x_test, y_test), input_shape, num_classes


def get_prepared_data(class_selection: Optional[List[int]] = None) -> Tuple[Tuple[Any, Any], Tuple[Any, Any], Any, Any]:
    (x_train, y_train), (x_test, y_test), input_shape, num_classes = get_basic_data()

    if class_selection is not None:
        x_selected_train = []
        y_selected_train = []
        x_selected_test = []
        y_selected_test = []
        for i, class_id in enumerate(class_selection):
            for result, image in zip(y_train, x_train):
                if result == class_id:
                    x_selected_train.append(image)
                    y_selected_train.append(i)
            for result, image in zip(y_test, x_test):
                if result == class_id:
                    x_selected_test.append(image)
                    y_selected_test.append(i)
        x_train, y_train = np.array(x_selected_train).reshape([-1, input_shape[0], 1]), np.array(
            y_selected_train).reshape([-1, 1])
        x_test, y_test = np.array(x_selected_test).reshape([-1, input_shape[0], 1]), np.array(
            y_selected_test).reshape([-1, 1])
        y_train = keras.utils.to_categorical(y_train, len(class_selection))
        y_test = keras.utils.to_categorical(y_test, len(class_selection))
    else:
        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)

    return (x_train, y_train), (x_test, y_test), input_shape, num_classes


def get_unbalance_data(main_class: int, other_class_percentage: float, class_selection: Optional[List[int]] = None) \
        -> Tuple[Tuple[Any, Any], Tuple[Any, Any], Any, Any]:
    (x_train, y_train), (x_test, y_test), input_shape, num_classes = get_basic_data()

    x_unbalanced_train = []
    y_unbalanced_train = []

    considered_class_count: int = num_classes if class_selection is None else len(
        class_selection)
    other_class_samples: int = int(
        other_class_percentage * x_train.shape[0] * (considered_class_count / num_classes))

    for result, image in zip(y_train, x_train):
        if result == main_class:
            x_unbalanced_train.append(image)
            y_unbalanced_train.append(result)

    if class_selection is not None:
        x_selected_test = []
        y_selected_test = []
        for result, image in zip(y_train, x_train):
            if result != main_class and result in class_selection and other_class_samples > 0:
                other_class_samples -= 1
                x_unbalanced_train.append(image)
                y_unbalanced_train.append(result)
        for result, image in zip(y_test, x_test):
            if result in class_selection:
                x_selected_test.append(image)
                y_selected_test.append(result)
        x_test, y_test = np.array(x_selected_test).reshape([-1, input_shape[0], 1]), np.array(
            y_selected_test).reshape([-1, 1])
    else:
        for result, image in zip(y_train, x_train):
            if result != main_class and other_class_samples > 0:
                other_class_samples -= 1
                x_unbalanced_train.append(image)
                y_unbalanced_train.append(result)

    x_train, y_train = np.array(x_unbalanced_train).reshape([-1, input_shape[0], 1]), np.array(
        y_unbalanced_train).reshape([-1, 1])

    if class_selection is not None:
        y_train = keras.utils.to_categorical(y_train, len(class_selection))
        y_test = keras.utils.to_categorical(y_test, len(class_selection))
    else:
        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)
    return (x_train, y_train), (x_test, y_test), input_shape, num_classes


def split_mnist_data(class_selection: Optional[List[int]] = None) -> None:
    (x_train, y_train), (x_test, y_test), input_shape, num_classes = get_basic_data()
    logging.info(f'splitting {x_train.shape[0]} train examples')
    logging.info(f'splitting {x_test.shape[0]} test examples')

    separated_train_data: List[Tuple[np.array, np.array]] = [([], []) for _ in
                                                             range(num_classes)]
    separated_test_data: List[Tuple[np.array, np.array]] = [([], []) for _ in
                                                            range(num_classes)]
    ensured_class_selection: List[int] = list(range(
        num_classes)) if class_selection is None else class_selection

    for result, image in zip(y_train, x_train):
        if result in ensured_class_selection:
            separated_train_data[result][0].append(image)
            separated_train_data[result][1].append(0)

    for result, image in zip(y_test, x_test):
        if result in ensured_class_selection:
            separated_test_data[result][0].append(image)
            separated_test_data[result][1].append(0)

    for i, class_id in enumerate(ensured_class_selection):
        separated_train_data[i] = (np.array(separated_train_data[class_id][0]).reshape([-1, input_shape[0], 1]),
                                   np.array(separated_train_data[class_id][1]).reshape([-1, 1]))
        separated_test_data[i] = (np.array(separated_test_data[class_id][0]).reshape([-1, input_shape[0], 1]),
                                  np.array(separated_test_data[class_id][1]).reshape([-1, 1]))

    processed_separated_train_data: List[Tuple[np.array, np.array]] = [
        ([], []) for _ in range(len(ensured_class_selection))]
    processed_separated_test_data: List[Tuple[np.array, np.array]] = [
        ([], []) for _ in range(len(ensured_class_selection))]
    for i in range(len(ensured_class_selection)):
        processed_separated_train_data[i] = (
            np.copy(separated_train_data[i][0]), np.copy(separated_train_data[i][1]))
        processed_separated_test_data[i] = (
            np.copy(separated_test_data[i][0]), np.copy(separated_test_data[i][1]))

    for i in range(len(ensured_class_selection)):
        for j in range(len(ensured_class_selection)):
            np.random.shuffle(separated_train_data[j][0])
            split_portion = int(
                len(separated_train_data[j][0]) / len(ensured_class_selection))
            processed_separated_train_data[i] = (
                np.append(
                    processed_separated_train_data[i][0], separated_train_data[j][0][0:split_portion], axis=0),
                np.append(processed_separated_train_data[i][1], np.ones(
                    split_portion).reshape(-1, 1), axis=0)
            )
            np.random.shuffle(separated_test_data[j][0])
            split_portion = int(
                len(separated_test_data[j][0]) / len(ensured_class_selection))
            processed_separated_test_data[i] = (
                np.append(
                    processed_separated_test_data[i][0], separated_test_data[j][0][0:split_portion], axis=0),
                np.append(processed_separated_test_data[i][1], np.ones(
                    split_portion).reshape(-1, 1), axis=0)
            )

    for i, class_id in enumerate(ensured_class_selection):
        logging.info(
            f'{processed_separated_train_data[i][0].shape[0]} train examples for class #{class_id}')
        logging.info(
            f'{processed_separated_test_data[i][0].shape[0]} test examples for class #{class_id}')

    data_path: str = DATA_PATH + 'mnist'
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    if len(ensured_class_selection) == num_classes:
        np.savez(f'{data_path}/mnist_train_split',
                 processed_separated_train_data)
        np.savez(f'{data_path}/mnist_test_split',
                 processed_separated_test_data)
    else:
        np.savez(f"{data_path}/mnist_train_split_{''.join(str(e) + '_' for e in ensured_class_selection)}",
                 processed_separated_train_data)
        np.savez(f"{data_path}/mnist_test_split_{''.join(str(e) + '_' for e in ensured_class_selection)}",
                 processed_separated_test_data)

    logging.info(f"saved split data to \"{data_path}\"")
