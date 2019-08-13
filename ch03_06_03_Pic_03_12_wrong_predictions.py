# Кроме того, давайте ради интереса посмотрим на некоторые из тех изображе-
# ний, метки которых угадать не получилось.
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
image_size=28*28
num_classes=10
batch_size=100
num_batches=1000
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

test_size= len(mnist.test.labels)
x = tf.placeholder(tf.float32, [None, image_size])
W = tf.Variable(tf.zeros([image_size, num_classes]))
b = tf.Variable(tf.zeros([num_classes]))
y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder(tf.float32, [None, num_classes])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
init = tf.global_variables_initializer()
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
test_labels = tuple(np.argmax(l) for l in mnist.test.labels)
predict_stats = [0]*test_size
with tf.Session() as sess:
    sess.run(init)
    for i in range(num_batches):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
        preds = sess.run(tf.cast(tf.argmax(y, 1), tf.int64), feed_dict={x: mnist.test.images})
        for i in range(test_size):
            if preds[i] !=  test_labels[i]:
                predict_stats[i] += 1

    print("Точность: %s" % sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
    inter_class_counts = np.zeros((num_classes, num_classes), np.int64)
    # popular errors
    # 0->6
    # 1->8
    # 2->0,1,3,4,6,7,8
    # 3->2,5,8,9
    # 4 ->6,8,9
    # 5->0,3,6,8
    # 6->0,
    # 7->2,3,9
    # 8->3,5,6
    # 9->0,3,4,7
    for pred,label in zip(preds,test_labels):
        inter_class_counts[label][pred]+=1
    print(inter_class_counts)
bad_images=[x for x in reversed(sorted(enumerate(predict_stats),key=lambda p:p[1]))][:25]
fig=plt.figure(figsize=(10,10))
fig.canvas.set_window_title('Рис. 3.12')
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(np.reshape(mnist.test.images[bad_images[i][0]] * 255, (28, 28)))
    plt.xlabel(test_labels[bad_images[i][0]])
plt.show()


