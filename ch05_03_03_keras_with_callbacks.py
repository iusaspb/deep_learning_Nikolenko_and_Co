from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

(X_train, y_train), (X_test, y_test) = mnist.load_data()

num_classes=10
scale=2
channels_1=32
channels_2=64
layer_size=1024
batch_size, img_rows, img_cols = 64, 28, 28
X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

X_train = X_train.astype("float32")
X_test = X_test.astype("float32")
X_train /= 255
X_test /= 255

Y_train = np_utils.to_categorical(y_train, num_classes)
Y_test = np_utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Convolution2D(channels_1, 5, 5, border_mode="same", input_shape=input_shape))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(scale, scale), strides=(scale, scale), border_mode="same"))
model.add(Convolution2D(channels_2, 5, 5, border_mode="same", input_shape=input_shape))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(scale, scale), strides=(scale, scale), border_mode="same"))

model.add(Flatten())
model.add(Dense(layer_size))
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation("softmax"))

model.compile(loss="categorical_crossentropy",optimizer="adam", metrics=["accuracy"])
model.fit(X_train, Y_train,callbacks=[ModelCheckpoint("model.hdf5", monitor="val_acc",
save_best_only=True, save_weights_only=False, mode="auto")],
validation_split=0.1, epochs=10, verbose=2, batch_size=batch_size)


score = model.evaluate(X_test, Y_test, verbose=0)
print("Test score: %f" % score[0])
print("Test accuracy: %f" % score[1])