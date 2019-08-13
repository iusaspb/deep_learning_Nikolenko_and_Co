# Рис. 4.7. Сравнение разных методов для улучшения обучения нейронных сетей
# TODO: картинка получается не такая, как в книге
# 1. Начальная модель
# 2. Xavier
# 3. BN
# 4. Adam
# 5. Adam+BN+Xavier
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
image_size=28*28
num_classes=10
batch_size=100
num_batches=30
layer_size=100
num_runnings=10
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

def fullyconnected_layer(tensor, input_size, out_size):
    W = tf.Variable(tf.truncated_normal([input_size, out_size], stddev=0.1))
    b = tf.Variable(tf.truncated_normal([out_size], stddev=0.1))
    return tf.nn.relu(tf.matmul(tensor, W) + b)

def fullyconnected_xavier_layer(tensor, input_size, out_size):
    W = tf.Variable(tf.contrib.layers.xavier_initializer()(shape=[input_size, out_size]))
    b = tf.Variable(tf.contrib.layers.xavier_initializer()(shape=[out_size]))
    return tf.nn.relu(tf.matmul(tensor, W) + b)

def batchnorm_layer(tensor, size):
    batch_mean, batch_var = tf.nn.moments(tensor, [0])
    beta = tf.Variable(tf.zeros([size]))
    scale = tf.Variable(tf.ones([size]))
    return tf.nn.batch_normalization(tensor, batch_mean, batch_var, beta, scale, 0.001)

def build_normal_model():
    x = tf.placeholder(tf.float32, [None, image_size])
    #
    h1 = fullyconnected_layer(x, image_size, layer_size)
    h2 = fullyconnected_layer(h1, layer_size, layer_size)
    h3 = fullyconnected_layer(h2, layer_size, layer_size)
    h = fullyconnected_layer(h3, layer_size, layer_size)
    #
    W_out = tf.Variable(tf.zeros([layer_size, num_classes]))
    b_out = tf.Variable(tf.zeros([num_classes]))
    logit = tf.matmul(h, W_out) + b_out
    y = tf.nn.softmax(logit)
    y_ = tf.placeholder(tf.float32, [None, num_classes])
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=logit))
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return x,y,y_,cross_entropy,accuracy

def build_xavier_model():
    x = tf.placeholder(tf.float32, [None, image_size])
    #
    h1 = fullyconnected_xavier_layer(x, image_size, layer_size)
    h2 = fullyconnected_xavier_layer(h1, layer_size, layer_size)
    h3 = fullyconnected_xavier_layer(h2, layer_size, layer_size)
    h = fullyconnected_xavier_layer(h3, layer_size, layer_size)
    #
    W_out = tf.Variable(tf.zeros([layer_size, num_classes]))
    b_out = tf.Variable(tf.zeros([num_classes]))
    logit = tf.matmul(h, W_out) + b_out
    y = tf.nn.softmax(logit)
    y_ = tf.placeholder(tf.float32, [None, num_classes])
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=logit))
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return x,y,y_,cross_entropy,accuracy

def build_batchnorm_model():
    x = tf.placeholder(tf.float32, [None, image_size])
    #
    h1 = fullyconnected_layer(x, image_size, layer_size)
    h1_bn = batchnorm_layer(h1, layer_size)
    h2 = fullyconnected_layer(h1_bn, layer_size, layer_size)
    h2_bn = batchnorm_layer(h2, layer_size)
    h3 = fullyconnected_layer(h2_bn, layer_size, layer_size)
    h3_bn = batchnorm_layer(h3, layer_size)
    h = fullyconnected_layer(h3_bn, layer_size, layer_size)
    #
    W_out = tf.Variable(tf.zeros([layer_size, num_classes]))
    b_out = tf.Variable(tf.zeros([num_classes]))
    logit = tf.matmul(h, W_out) + b_out
    y = tf.nn.softmax(logit)
    y_ = tf.placeholder(tf.float32, [None, num_classes])
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=logit))
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return x,y,y_,cross_entropy,accuracy

def build_xavier_batchnorm_model():
    x = tf.placeholder(tf.float32, [None, image_size])
    #
    h1 = fullyconnected_layer(x, image_size, layer_size)
    h1_bn = batchnorm_layer(h1, layer_size)
    h2 = fullyconnected_layer(h1_bn, layer_size, layer_size)
    h2_bn = batchnorm_layer(h2, layer_size)
    h3 = fullyconnected_layer(h2_bn, layer_size, layer_size)
    h3_bn = batchnorm_layer(h3, layer_size)
    h = fullyconnected_layer(h3_bn, layer_size, layer_size)
    #
    W_out = tf.Variable(tf.zeros([layer_size, num_classes]))
    b_out = tf.Variable(tf.zeros([num_classes]))
    logit = tf.matmul(h, W_out) + b_out
    y = tf.nn.softmax(logit)
    y_ = tf.placeholder(tf.float32, [None, num_classes])
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=logit))
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return x,y,y_,cross_entropy,accuracy


def run_SGD_model(build_model,stats):
    x, y,y_,loss,accuracy = build_model()
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for batch in range(num_batches):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            _,loss_val,acc_val = sess.run([train_step,loss,accuracy], feed_dict={x: batch_xs, y_: batch_ys})
            stats['loss'][batch].append(loss_val)
            loss_val,acc_val = sess.run([loss,accuracy], feed_dict={x: mnist.test.images, y_: mnist.test.labels})
            stats['acc'][batch].append(acc_val)

def run_Adam_model(build_model,stats):
    x, y,y_,loss,accuracy = build_model()
    train_step = tf.train.AdamOptimizer().minimize(loss)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for batch in range(num_batches):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            _,loss_val,acc_val = sess.run([train_step,loss,accuracy], feed_dict={x: batch_xs, y_: batch_ys})
            stats['loss'][batch].append(loss_val)
            loss_val,acc_val = sess.run([loss,accuracy], feed_dict={x: mnist.test.images, y_: mnist.test.labels})
            stats['acc'][batch].append(acc_val)

def prep_stats(stats):
    for k in ('loss','acc'):
        l = stats[k]
        stats[k]={'mean':np.mean(l, 1)}

def xxx(name,model,runner):
    print(name)
    stats = {'name': name, 'loss': [[] for _ in range(num_batches)],'acc': [[] for _ in range(num_batches)]}
    for running in range(num_runnings):
        print(f'running={running}')
        runner(model, stats)
    prep_stats(stats)
    return stats

sgd_stats = xxx('Начальная модель',build_normal_model,run_SGD_model)
xavier_stats=xxx('Xavier',build_xavier_model,run_SGD_model)
bn_stats=xxx('BN',build_batchnorm_model,run_SGD_model)
adam_stats=xxx('Adam',build_normal_model,run_Adam_model)
adam_bn_xavier_stats=xxx('Adam+BN+Xavier',build_xavier_batchnorm_model,run_Adam_model)
#
#
#
t = np.arange(num_batches)
fig, axs = plt.subplots(2, 1)
fig.canvas.set_window_title('Рис. 4.7. Сравнение разных методов для улучшения обучения нейронных сетей')
def plot_stats(stats):
    axs[0].plot(t, stats['loss']['mean'], label=stats['name'])
    axs[1].plot(t, stats['acc']['mean'], label=stats['name'])

plot_stats(sgd_stats)
plot_stats(xavier_stats)
plot_stats(bn_stats)
plot_stats(adam_stats)
plot_stats(adam_bn_xavier_stats)

axs[0].set_xlabel('batches')
axs[0].set_ylabel('Функция потерь')
axs[0].set_title('Функция потерь')
axs[0].grid(True)
axs[0].legend()
axs[1].set_xlabel('batches')
axs[1].set_ylabel('Точность')
axs[1].set_title('Точность')
axs[1].grid(True)
axs[1].legend()
fig.tight_layout()
plt.show()




