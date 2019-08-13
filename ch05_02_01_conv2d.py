import tensorflow as tf
import numpy as np

x_inp = tf.placeholder(tf.float32, [5, 5])
w_inp = tf.placeholder(tf.float32, [3, 3])
x = tf.reshape(x_inp, [1, 5, 5, 1])
w = tf.reshape(w_inp, [3, 3, 1, 1])

x_valid = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding="VALID")
x_same = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding="SAME")
x_valid_half = tf.nn.conv2d(x, w, strides=[1, 2, 2, 1], padding="VALID")
x_same_half = tf.nn.conv2d(x, w, strides=[1, 2, 2, 1], padding="SAME")

x = np.array([
    [0, 1, 2, 1, 0],
    [4, 1, 0, 1, 0],
    [2, 0, 1, 1, 1],
    [1, 2, 3, 1, 0],
    [0, 4, 3, 2, 0]])
w = np.array([
    [0, 1, 0],
    [1, 0, 1],
    [2, 1, 0]])
with tf.Session() as sess:
    y_valid, y_same, y_valid_half, y_same_half = sess.run([x_valid, x_same, x_valid_half, x_same_half],feed_dict={x_inp: x, w_inp: w})
print(r"padding=VALID:\n", y_valid[0, :, :, 0])
print(r"padding=SAME:\n", y_same[0, :, :, 0])
print(r"padding=VALID, stride 2:\n", y_valid_half[0, :, :, 0])
print(r"padding=SAME, stride 2:\n", y_same_half[0, :, :, 0])