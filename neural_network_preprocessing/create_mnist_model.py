import logging
from typing import Any, List, Optional

import numpy as np
from sklearn.metrics import classification_report
from tensorflow import keras
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Flatten, Layer

from data.mnist_data_handler import get_prepared_data, get_unbalance_data
from data.model_data import ModelData, ModelTrainType


def generate_model_description(batch_size: int, epochs: int, layers: List[Layer], learning_rate: float = 0.001) -> str:
    name: str = 'mnist_dense'
    for layer in layers:
        name += '_' + str(layer.output_shape[1])
    name += '_' + batch_size.__str__()
    name += '_' + epochs.__str__()
    name += '_' + learning_rate.__str__()
    return name


def build_mnist_model(layer_data: List[int], num_classes: int, input_shape: Any, learning_rate: float,
                      regularized: bool = False) -> Model:
    model: Model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    if regularized:
        for nodes in layer_data:
            model.add(Dense(nodes, activation='relu',
                      kernel_regularizer=keras.regularizers.l1(0.001)))
        model.add(Dense(num_classes, activation='softmax',
                  kernel_regularizer=keras.regularizers.l1(0.001)))
    else:
        for nodes in layer_data:
            model.add(Dense(nodes, activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(learning_rate),
                  metrics=['accuracy'])
    return model


def create(name: str, batch_size: int, epochs: int, layer_data: List[int], learning_rate: float = 0.001,
           regularized: bool = False, train_type: ModelTrainType = ModelTrainType.BALANCED, main_class: Optional[int] = None,
           other_class_percentage: Optional[float] = None, class_selection: Optional[List[int]] = None) -> ModelData:
    logging.info(
        "Create MNIST neural network model with training type \"%s\"." % train_type.name)

    if train_type is not ModelTrainType.UNBALANCED:
        (x_train, y_train), (x_test,
                             y_test), input_shape, num_classes = get_prepared_data(class_selection)
    else:
        if main_class is None:
            raise Exception(
                'No main class is given for creating an unbalanced dataset.')
        if other_class_percentage is None:
            raise Exception(
                'No percentage of other classes is given for creating an unbalanced dataset.')
        (x_train, y_train), (x_test, y_test), input_shape, num_classes = get_unbalance_data(main_class,
                                                                                            other_class_percentage,
                                                                                            class_selection)
    logging.info('Train examples: %i' % x_train.shape[0])
    logging.info('Test examples: %i' % x_test.shape[0])

    if class_selection is not None:
        num_classes = len(class_selection)

    model: Model = build_mnist_model(
        layer_data, num_classes, input_shape, learning_rate, regularized)
    if train_type is not ModelTrainType.UNTRAINED:
        model.fit(x_train, y_train, batch_size=batch_size,
                  epochs=epochs, verbose=1, validation_data=(x_test, y_test))

    model_description: str = generate_model_description(
        batch_size, epochs, model.layers, learning_rate)
    model_layer_nodes: List[int] = [input_shape[0]]
    model_layer_nodes.extend(layer_data)
    model_layer_nodes.append(num_classes)
    model_data: ModelData = ModelData(name, model_description, model)
    model_data.set_parameter(batch_size, epochs, model_layer_nodes,
                             learning_rate, x_train.shape[0], x_test.shape[0])
    model_data = evaluate_model(model_data, x_train, y_train, x_test, y_test)
    model_data.set_class_selection(class_selection)
    model_data.save_model()
    model_data.store_model_data()

    return model_data


def evaluate_model(model_data: ModelData, x_train: Any, y_train: Any, x_test: Any, y_test: Any) \
        -> ModelData:
    train_score = model_data.model.evaluate(x_train, y_train, verbose=0)
    test_score = model_data.model.evaluate(x_test, y_test, verbose=0)
    logging.info('Train loss: %f, Train accuracy: %f, Test loss: %f, Test accuracy: %f' % (
        train_score[0], train_score[1], test_score[0], test_score[1]))

    c_y_test = np.argmax(y_test, axis=1)

    predict_x = model_data.model.predict(x_test)
    prediction_test = np.argmax(predict_x, axis=1)
    c_report: Any = classification_report(
        c_y_test, prediction_test, output_dict=True)
    model_data.set_initial_performance(
        test_score[0], test_score[1], train_score[0], train_score[1], c_report)

    return model_data


def calculate_performance_of_model(model_data: ModelData) -> ModelData:
    (x_train, y_train), (x_test, y_test), input_shape, num_classes = get_prepared_data()

    logging.info('Train examples: %i' % x_train.shape[0])
    logging.info('Test examples: %i' % x_test.shape[0])

    model_data.reload_model()
    model_data.model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(0.001),
                             metrics=['accuracy'])

    model_data = evaluate_model(model_data, x_train, y_train, x_test, y_test)
    return model_data
