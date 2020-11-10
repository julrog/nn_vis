import numpy as np
from tensorflow import keras
from tensorflow_core.python.keras import Model, Sequential
from tensorflow_core.python.keras.datasets import mnist
from tensorflow_core.python.keras.layers import BatchNormalization
from definitions import BASE_PATH
import tensorflow as tf


model_path: str = BASE_PATH + '/storage/models/' + "dense_784_128_10"
model: Model = keras.models.load_model(model_path)
print(model.summary())


def create_bn_layer_model(model):
    max_layer = len(model.layers)
    x = None
    input = None
    for i, layer in enumerate(model.layers):
        if i == 0:
            x = layer.output
            input = layer.input
        if 0 < i < max_layer:
            new_layer = BatchNormalization(center=False, gamma_initializer="zeros", gamma_constraint=tf.keras.constraints.NonNeg())
            x = new_layer(x)
            x = layer(x)
    return Model(inputs=input, outputs=x)


# create graph of your new model
bn_model = create_bn_layer_model(model)

for layer in bn_model.layers:
    print(type(layer))
    if type(layer) != BatchNormalization:
        print("not")
        layer.trainable = False
    else:
        layer.trainable = True
        print(layer.get_weights()[0])
        print("yes")

# compile the model
bn_model.compile(loss=keras.losses.categorical_crossentropy,
                 optimizer=keras.optimizers.Adam(0.001),
                 metrics=['accuracy'])

print(bn_model.summary())

batch_size = 128
num_classes = 10
epochs = 2

# input image dimensions
img_size = 28 * 28

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
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

separated_train_data = [([], []) for _ in range(num_classes)]
separated_test_data = [([], []) for _ in range(num_classes)]

for result, image in zip(y_train, x_train):
    separated_train_data[result][0].append(image)
    separated_train_data[result][1].append(result)

for result, image in zip(y_test, x_test):
    separated_test_data[result][0].append(image)
    separated_test_data[result][1].append(result)

for i in range(0, 1):
    separated_train_data[i] = (np.array(separated_train_data[i][0]).reshape([-1, img_size, 1]),
                               np.array(separated_train_data[i][1]).reshape([-1, 1]))
    separated_test_data[i] = (np.array(separated_test_data[i][0]).reshape([-1, img_size, 1]),
                              np.array(separated_test_data[i][1]).reshape([-1, 1]))

x_train, y_train = separated_train_data[0]
x_test, y_test = separated_test_data[0]
# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

bn_model.fit(x_train, y_train,
             batch_size=batch_size,
             epochs=epochs,
             verbose=1,
             validation_data=(x_test, y_test))
score = bn_model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

for layer in bn_model.layers:
    print(type(layer))
    if type(layer) == BatchNormalization:
        print(layer.get_weights()[0])
