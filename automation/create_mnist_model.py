import os
import numpy as np
from typing import List

from tensorflow import keras
from tensorflow_core.python.keras import Sequential
from tensorflow_core.python.keras.datasets import mnist
from tensorflow_core.python.keras.layers import Flatten, Dense
from tensorflow_core.python.layers.base import Layer
from sklearn.metrics import classification_report

from data.model_data import ModelData


def generate_model_description(batch_size: int, epochs: int, layers: List[Layer], learning_rate: float = 0.001) -> str:
    name: str = "mnist_dense"
    for layer in layers:
        name += "_" + str(layer.output_shape[1])
    name += "_" + batch_size.__str__()
    name += "_" + epochs.__str__()
    name += "_" + learning_rate.__str__()
    return name


def create(name: str, batch_size: int, epochs: int, layer_data: List[int], learning_rate: float = 0.001) -> ModelData:
    num_classes: int = 10

    # input image dimensions
    img_size: int = 28 * 28

    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(x_train.shape[0], img_size, 1)
    x_test = x_test.reshape(x_test.shape[0], img_size, 1)
    input_shape = (img_size, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    training_samples: int = x_train.shape[0]
    test_samples: int = x_test.shape[0]
    print(training_samples, 'train samples')
    print(test_samples, 'test samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    for nodes in layer_data:
        model.add(Dense(nodes, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(learning_rate),
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    c_y_test = np.argmax(y_test, axis=1)  # Convert one-hot to index
    prediction_test = model.predict_classes(x_test)
    c_report: any = classification_report(c_y_test, prediction_test, output_dict=True)
    print(c_report)

    model_layer_nodes: List[int] = [input_shape[0]]
    model_layer_nodes.extend(layer_data)
    model_layer_nodes.append(num_classes)

    model_description: str = generate_model_description(batch_size, epochs, model.layers, learning_rate)

    model_data: ModelData = ModelData(name, model_description, model)
    model_data.set_parameter(batch_size, epochs, model_layer_nodes, learning_rate, training_samples, test_samples)
    model_data.set_initial_performance(score[0], score[1], c_report)
    model_data.save_model()
    model_data.store_model_data()

    return model_data
