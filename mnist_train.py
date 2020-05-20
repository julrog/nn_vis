import os

from tensorflow import keras
from tensorflow_core.python.keras import Sequential
from tensorflow_core.python.keras.datasets import mnist
from tensorflow_core.python.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense

from definitions import BASE_PATH

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

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Flatten(input_shape=input_shape))
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(0.001),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

model_name: str = "dense"
for layer in model.layers:
    model_name += "_" + str(layer.output_shape[1])

model_path: str = BASE_PATH + '/storage/models/' + model_name
if not os.path.exists(model_path):
    os.makedirs(model_path)

model.save(model_path)
