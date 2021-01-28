import numpy as np
from typing import List, Tuple, Any
from tensorflow import keras
from tensorflow_core.python.keras import Sequential
from tensorflow_core.python.keras.layers import Flatten, Dense
from tensorflow_core.python.keras.models import Model
from tensorflow_core.python.layers.base import Layer
from sklearn.metrics import classification_report

from data.mnist_data_handler import get_prepared_data, get_unbalance_data
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
           other_class_percentage: float = None, class_selection: List[int] = None) -> ModelData:
    print("[%s] Create MNIST neural network model with training type '%s'." % (LOG_SOURCE, train_type.name))

    (x_train, y_train), (x_test, y_test), input_shape, num_classes = get_prepared_data(class_selection) \
        if train_type is not ModelTrainType.UNBALANCED else get_unbalance_data(main_class, other_class_percentage,
                                                                               class_selection)

    print("[%s] Train samples: %i" % (LOG_SOURCE, x_train.shape[0]))
    print("[%s] Test samples: %i" % (LOG_SOURCE, x_test.shape[0]))

    if class_selection is not None:
        num_classes = len(class_selection)

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
