# add relu layer
# add droup out layer
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
image_size=28*28
num_classes=10
batch_size=100
num_batches=4000 # instead of 2000
layer_size=100
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
x = tf.placeholder(tf.float32, [None, image_size])
#
W_relu = tf.Variable(tf.truncated_normal([image_size, layer_size], stddev=0.1))
b_relu = tf.Variable(tf.truncated_normal([layer_size], stddev=0.1))
h = tf.nn.relu(tf.matmul(x, W_relu) + b_relu)
keep_probability = tf.placeholder(tf.float32)
h_drop = tf.nn.dropout(h, keep_probability)
#
W = tf.Variable(tf.zeros([layer_size, num_classes]))
b = tf.Variable(tf.zeros([num_classes]))
y = tf.nn.softmax(tf.matmul(h_drop, W) + b)

y_ = tf.placeholder(tf.float32, [None, num_classes])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
init = tf.global_variables_initializer()
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
with tf.Session() as sess:
    sess.run(init)
    for i in range(num_batches):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys,keep_probability: 0.5})
    print("b=%s не числа" % sess.run(b))
    print("Точность: %s (=%s приближённо)!!!!!)" % (sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels,keep_probability: 1.}),1/num_classes))
