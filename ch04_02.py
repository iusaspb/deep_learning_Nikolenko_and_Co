from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from keras.datasets import mnist
image_width=28
image_size=image_width*image_width
num_classes=10
num_epochs=30
num_runnings=10
(x_train, y_train), (x_test, y_test) = mnist.load_data()
Y_train = np_utils.to_categorical(y_train, num_classes)
Y_test = np_utils.to_categorical(y_test, num_classes)
X_train = x_train.reshape([-1, image_size]) / 255.
X_test = x_test.reshape([-1, image_size]) / 255.
def create_model(init):
    model = Sequential()
    model.add(Dense(100, input_shape=(image_size,), kernel_initializer=init, activation='tanh'))
    model.add(Dense(100, kernel_initializer=init, activation='tanh'))
    model.add(Dense(100, kernel_initializer=init, activation='tanh'))
    model.add(Dense(100, kernel_initializer=init, activation='tanh'))
    model.add(Dense(num_classes, kernel_initializer=init, activation='softmax'))
    return model
uniform_hist = {'loss':[],'acc':[]}
glorot_hist={'loss':[],'acc':[]}
for running in range(num_runnings):
    uniform_model = create_model("uniform")
    uniform_model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    hist = uniform_model.fit(X_train, Y_train,batch_size=64, epochs=num_epochs, verbose=2, validation_data=(X_test, Y_test))
    uniform_hist['loss'].append(hist.history['loss'])
    uniform_hist['acc'].append(hist.history['acc'])
    glorot_model = create_model("glorot_normal")
    glorot_model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    hist =glorot_model.fit(X_train, Y_train,batch_size=64, epochs=num_epochs, verbose=2, validation_data=(X_test, Y_test))
    glorot_hist['loss'].append(hist.history['loss'])
    glorot_hist['acc'].append(hist.history['acc'])

import numpy as np
for hist in (uniform_hist,glorot_hist):
    for key in ('loss', 'acc') :
        hist[key] = [np.mean(hist[key], 0), np.std(hist[key], 0) * 2]

import matplotlib.pyplot as plt
t = np.arange(num_epochs)
fig, axs = plt.subplots(2, 1)
axs[0].errorbar(t, uniform_hist['loss'][0], yerr=uniform_hist['loss'][1], label='Случайная инициализация')
axs[0].errorbar(t, glorot_hist['loss'][0], yerr=glorot_hist['loss'][1], label='Инициализация Ксавье')
axs[0].set_xlabel('эпохи')
axs[0].set_ylabel('Функция потерь')
axs[0].set_title('Функция потерь')
axs[0].grid(True)
axs[0].legend()

axs[1].errorbar(t, uniform_hist['acc'][0], yerr=uniform_hist['acc'][1], label='Случайная инициализация')
axs[1].errorbar(t, glorot_hist['acc'][0], yerr=glorot_hist['acc'][1], label='Инициализация Ксавье')
axs[1].set_xlabel('эпохи')
axs[1].set_ylabel('Точность')
axs[1].set_title('Точность')

axs[1].grid(True)
axs[1].legend()

fig.tight_layout()
fig.canvas.set_window_title('Рис. 4.2. Сравнение инициализации Ксавье и случайной инициализации весов')
plt.show()
