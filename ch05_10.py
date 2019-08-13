import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
batch_size, learning_rate = 64, 0.01
image_width,image_height=28,28
num_batches=100000
image_shape=(image_height, image_width)
image_size=image_height*image_width
scale=2
strides=[1, scale, scale, 1]
channels_1=4
total_params = 5*5*channels_1+channels_1+5*5*channels_1+1


ae_weights = {
    # слой свёртки
    "conv": tf.Variable(tf.truncated_normal([5, 5, 1, channels_1], stddev=0.1)),
    "b_hidden": tf.Variable(tf.truncated_normal([channels_1], stddev=0.1)),
    # слой развёртки
    "deconv": tf.Variable(tf.truncated_normal([5, 5, 1, channels_1], stddev=0.1)),
    "b_visible": tf.Variable(tf.truncated_normal([1], stddev=0.1))}

ae_input = tf.placeholder(tf.float32, [batch_size, image_size])
images = tf.reshape(ae_input, [-1, image_height, image_width, 1])
hidden_logits = tf.nn.conv2d(images, ae_weights["conv"], strides=strides, padding="SAME") + ae_weights["b_hidden"]
hidden = tf.nn.sigmoid(hidden_logits)
visible_logits = tf.nn.conv2d_transpose(hidden,ae_weights["deconv"],[batch_size, image_height, image_width, 1], strides=strides,padding="SAME") + ae_weights["b_visible"]
visible = tf.nn.sigmoid(visible_logits)
optimizer = tf.train.AdagradOptimizer(learning_rate)
conv_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=visible_logits, labels=images))
conv_op = optimizer.minimize(conv_cost)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for batch in range(num_batches):
        x_batch, _ = mnist.train.next_batch(batch_size)
        _,conv_cost_val = sess.run([conv_op,conv_cost], feed_dict={ae_input: x_batch})
    print(conv_cost_val)
