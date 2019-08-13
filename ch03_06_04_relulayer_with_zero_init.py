# Инициализация нулями в данном случае была бы совсем бессмыслен-
# ной, потому что ReLU(0) = 0, а значит, при инициализации нулями градиенты
# совсем не распространялись бы по сети.
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
image_size=28*28
num_classes=10
batch_size=100
num_iterations=2000
layer_size=100
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
x = tf.placeholder(tf.float32, [None, image_size])
#
W_relu = tf.Variable(tf.zeros([image_size, layer_size]))
b_relu = tf.Variable(tf.zeros([layer_size]))
h = tf.nn.relu(tf.matmul(x, W_relu) + b_relu)
#
W = tf.Variable(tf.zeros([layer_size, num_classes]))
b = tf.Variable(tf.zeros([num_classes]))
y = tf.nn.softmax(tf.matmul(h, W) + b)

y_ = tf.placeholder(tf.float32, [None, num_classes])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
init = tf.global_variables_initializer()
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
with tf.Session() as sess:
    sess.run(init)
    for iteration in range(num_iterations):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
        W_relu_val,b_relu_val = sess.run([W_relu,b_relu])
        print(W_relu_val.min(),W_relu_val.max(),b_relu_val.min(),b_relu_val.max()) # каждый элемент остаётся равным нулю.

    print("Точность: %s" % sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
