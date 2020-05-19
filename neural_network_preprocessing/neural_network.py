import numpy as np
from typing import List, Tuple

from tensorflow import keras
from tensorflow_core.python.keras import Model, Input
from tensorflow_core.python.keras.layers import BatchNormalization
from tensorflow_core.python.layers.base import Layer

from definitions import BASE_PATH, DATA_PATH

LOG_SOURCE: str = "NEURAL_NETWORK"


def insert_batchnorm_layer(model: Model):
    max_layer: int = len(model.layers)
    last_output: Input = None
    network_input: Input = None
    for i, layer in enumerate(model.layers):
        if i == 0:
            last_output = layer.output
            network_input = layer.input
        if 0 < i < max_layer:
            new_layer: Layer = BatchNormalization(center=False, gamma_initializer="zeros")
            last_output = new_layer(last_output)
            last_output = layer(last_output)
    return Model(inputs=network_input, outputs=last_output)


class ProcessedNetwork:
    def __init__(self, file: str):
        self.num_classes = None
        self.model_path: str = BASE_PATH + '/storage/models/' + file
        model: Model = keras.models.load_model(self.model_path)
        print(model.summary())
        self.architecture_data: List[int] = []
        self.importance_value: List[List[np.array]] = []
        for i, layer in enumerate(model.layers):
            self.architecture_data.append(layer.output_shape[1])
            if i is not 0:
                self.importance_value.append([])
            if i is len(model.layers) - 1:
                self.num_classes = layer.output_shape[1]

    def get_fine_tuned_model_data(self, train_data: Tuple[np.array, np.array],
                                  test_data: Tuple[np.array, np.array]) -> Model:
        batch_size = 128
        epochs = 3

        x_train, y_train = train_data
        x_test, y_test = test_data

        # convert class vectors to binary class matrices
        y_train = keras.utils.to_categorical(y_train, self.num_classes)
        y_test = keras.utils.to_categorical(y_test, self.num_classes)

        print('x_train shape:', x_train.shape)
        print(x_train.shape[0], 'train samples')
        print(x_test.shape[0], 'test samples')

        model: Model = keras.models.load_model(self.model_path)
        modified_model: Model = insert_batchnorm_layer(model)

        for layer in modified_model.layers:
            layer.trainable = False
            if type(layer) is BatchNormalization:
                layer.trainable = True

        print(modified_model.summary())

        modified_model.compile(loss=keras.losses.categorical_crossentropy,
                               optimizer=keras.optimizers.Adam(0.001),
                               metrics=['accuracy'])

        modified_model.fit(x_train, y_train,
                           batch_size=batch_size,
                           epochs=epochs,
                           verbose=1,
                           validation_data=(x_test, y_test))
        return modified_model

    def extract_importance_from_model(self, fine_tuned_model: Model):
        count: int = 0
        for layer in fine_tuned_model.layers:
            if type(layer) == BatchNormalization:
                self.importance_value[count].append(layer.get_weights()[0])
                count += 1

    def generate_importance_for_data(self, train_data_path: str, test_data_path: str) -> List[np.array]:
        raw_train_data: dict = np.load("%s/%s.npz" % (DATA_PATH, train_data_path), allow_pickle=True)
        train_data: List[np.array] = raw_train_data["arr_0"]

        raw_test_data: dict = np.load("%s/%s.npz" % (DATA_PATH, test_data_path), allow_pickle=True)
        test_data: List[np.array] = raw_test_data["arr_0"]

        if (len(train_data) is not self.num_classes and self.num_classes is not None) or (
                len(test_data) is not self.num_classes and self.num_classes is not None):
            raise Exception("[%s] Data does not match number of classes %i." % (LOG_SOURCE, self.num_classes))

        for class_test_data, class_train_data in zip(test_data, train_data):
            fine_tuned_model = self.get_fine_tuned_model_data(class_train_data, class_test_data)
            self.extract_importance_from_model(fine_tuned_model)

        result = []
        print(len(self.importance_value))
        for importance_values in self.importance_value:
            print(len(importance_values))
            result.append(np.stack(importance_values, axis=1))
        return result
