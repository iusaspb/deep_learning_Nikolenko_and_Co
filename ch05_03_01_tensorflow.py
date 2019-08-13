import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
image_width,image_height=28,28
image_size=image_width*image_height
num_classes=10
layer_size=1024
channels_1=32
channels_2=64
scale=2
small_image_size=int(image_width/scale/scale)*int(image_height/scale/scale)
batch_size=64
num_iterations=10000
# 3289546
total_model_params=(5*5+1)*channels_1+(5*5+1)*channels_2+(small_image_size+1) * channels_2*layer_size+(layer_size+1)* num_classes
#
#
#
x = tf.placeholder(tf.float32, [None, image_size])
y = tf.placeholder(tf.float32, [None, num_classes])
x_image = tf.reshape(x, [-1,image_width,image_height,1]) # 1d->2d
#
# первый сверточный слой
#
W_conv_1 = tf.Variable(tf.truncated_normal([5, 5, 1, channels_1], stddev=0.1))
b_conv_1 = tf.Variable(tf.constant(0.1, shape=[channels_1]))
conv_1 = tf.nn.conv2d(x_image, W_conv_1, strides=[1, 1, 1, 1], padding="SAME") + b_conv_1
h_conv_1 = tf.nn.relu(conv_1)
h_pool_1 = tf.nn.max_pool(h_conv_1, ksize=[1, scale, scale, 1], strides=[1, scale, scale, 1], padding="SAME")
#
# второй сверточный слой
#
W_conv_2 = tf.Variable(tf.truncated_normal([5, 5, channels_1, channels_2], stddev=0.1))
b_conv_2 = tf.Variable(tf.constant(0.1, shape=[channels_2]))
conv_2 = tf.nn.conv2d(h_pool_1, W_conv_2, strides=[1, 1, 1, 1], padding="SAME") + b_conv_2
h_conv_2 = tf.nn.relu(conv_2)
h_pool_2 = tf.nn.max_pool(h_conv_2, ksize=[1, scale, scale, 1], strides=[1, scale, scale, 1], padding="SAME")
h_pool_2_flat = tf.reshape(h_pool_2, [-1, small_image_size * channels_2])
#
# первый полносвязный слой
#
W_fc_1 = tf.Variable(tf.truncated_normal([small_image_size * channels_2, layer_size], stddev=0.1))
b_fc_1 = tf.Variable(tf.constant(0.1, shape=[layer_size]))
h_fc_1 = tf.nn.relu(tf.matmul(h_pool_2_flat, W_fc_1) + b_fc_1)
keep_probability = tf.placeholder(tf.float32)
h_fc_1_drop = tf.nn.dropout(h_fc_1, keep_probability)
#
# выходной полносвязный слой
#
W_fc_2 = tf.Variable(tf.truncated_normal([layer_size, num_classes], stddev=0.1))
b_fc_2 = tf.Variable(tf.constant(0.1, shape=[num_classes]))
logit_conv = tf.matmul(h_fc_1_drop, W_fc_2) + b_fc_2
y_conv = tf.nn.softmax(logit_conv)
#
#
#
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logit_conv))
train_step = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for iteration in range(num_iterations):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        sess.run(train_step,feed_dict={x: batch_xs, y: batch_ys, keep_probability: 0.5})
    print(sess.run(accuracy,feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_probability: 1.}))

