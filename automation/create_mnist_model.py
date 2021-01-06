import numpy as np
from typing import List, Tuple, Any
from tensorflow import keras
from tensorflow_core.python.keras import Sequential
from tensorflow_core.python.keras.datasets import mnist
from tensorflow_core.python.keras.layers import Flatten, Dense
from tensorflow_core.python.keras.models import Model
from tensorflow_core.python.layers.base import Layer
from sklearn.metrics import classification_report
from data.model_data import ModelData, ModelTrainType

LOG_SOURCE = "MNIST_MODEL_CREATION"


def generate_model_description(batch_size: int, epochs: int, layers: List[Layer], learning_rate: float = 0.001) -> str:
    name: str = "mnist_dense"
    for layer in layers:
        name += "_" + str(layer.output_shape[1])
    name += "_" + batch_size.__str__()
    name += "_" + epochs.__str__()
    name += "_" + learning_rate.__str__()
    return name


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


def evaluate_model(model: Model, x_train: Any, y_train: Any, x_test: Any, y_test: Any) \
        -> Tuple[Tuple[float, float], Tuple[float, float], str]:
    train_score = model.evaluate(x_train, y_train, verbose=0)
    test_score = model.evaluate(x_test, y_test, verbose=0)
    print('[%s] Train loss: %f, Train accuracy: %f, Test loss: %f, Test accuracy: %f' % (
        LOG_SOURCE, train_score[0], train_score[1], test_score[0], test_score[1]))

    c_y_test = np.argmax(y_test, axis=1)  # Convert one-hot to index
    prediction_test = model.predict_classes(x_test)
    c_report: any = classification_report(c_y_test, prediction_test, output_dict=True)
    return (train_score[0], train_score[1]), (test_score[0], test_score[1]), c_report


def create(name: str, batch_size: int, epochs: int, layer_data: List[int], learning_rate: float = 0.001,
           regularized: bool = False, train_type: ModelTrainType = ModelTrainType.BALANCED, main_class: int = None,
           other_class_percentage: float = None) -> ModelData:
    print("[%s] Create MNIST neural network model with training type '%s'." % (LOG_SOURCE, train_type.name))

    (x_train, y_train), (x_test, y_test), input_shape, num_classes = get_prepared_data() \
        if train_type is not ModelTrainType.UNBALANCED else get_unbalance_data(main_class, other_class_percentage)

    print("[%s] Train samples: %i" % (LOG_SOURCE, x_train.shape[0]))
    print("[%s] Test samples: %i" % (LOG_SOURCE, x_test.shape[0]))

    model: Model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    if regularized:
        for nodes in layer_data:
            model.add(Dense(nodes, activation='relu', kernel_regularizer=keras.regularizers.l1(0.001)))
        model.add(Dense(num_classes, activation='softmax', kernel_regularizer=keras.regularizers.l1(0.001)))
    else:
        for nodes in layer_data:
            model.add(Dense(nodes, activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(learning_rate),
                  metrics=['accuracy'])

    if train_type is not ModelTrainType.UNTRAINED:
        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  validation_data=(x_test, y_test))

    (train_loss, train_acc), (test_loss, test_acc), c_report = evaluate_model(model, x_train, y_train, x_test, y_test)

    model_layer_nodes: List[int] = [input_shape[0]]
    model_layer_nodes.extend(layer_data)
    model_layer_nodes.append(num_classes)

    model_description: str = generate_model_description(batch_size, epochs, model.layers, learning_rate)

    model_data: ModelData = ModelData(name, model_description, model)
    model_data.set_parameter(batch_size, epochs, model_layer_nodes, learning_rate, x_train.shape[0], x_test.shape[0])
    model_data.set_initial_performance(test_loss, test_acc, train_loss, train_acc, c_report)
    model_data.save_model()
    model_data.store_model_data()

    return model_data


def calculate_performance_of_model(model_data: ModelData):
    (x_train, y_train), (x_test, y_test), input_shape, num_classes = get_prepared_data()

    print("[%s] Train samples: %i" % (LOG_SOURCE, x_train.shape[0]))
    print("[%s] Test samples: %i" % (LOG_SOURCE, x_test.shape[0]))

    model_data.reload_model()
    model_data.model.compile(loss=keras.losses.categorical_crossentropy,
                             optimizer=keras.optimizers.Adam(0.001),
                             metrics=['accuracy'])

    (train_loss, train_acc), (test_loss, test_acc), c_report = evaluate_model(model_data.model, x_train, y_train,
                                                                              x_test, y_test)

    model_data.set_initial_performance(test_loss, test_acc, train_loss, train_acc, c_report)
    return model_data
