# Рис. 4.6. Сравнение стохастического градиентного спуска и Adam
#  чем больше слоёв у модели, чем больший выигрыщ даёт adam
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


sgd_stats={'name': 'SGD', 'loss':[[] for _ in range(num_batches)], 'acc':[[] for _ in range(num_batches)]}
adam_stats={'name': 'Adam', 'loss':[[] for _ in range(num_batches)], 'acc':[[] for _ in range(num_batches)]}

print('SGD')
for running in range(num_runnings) :
    print(f'running={running}')
    run_SGD_model(build_normal_model,sgd_stats)

print('Adam')
for running in range(num_runnings) :
    print(f'running={running}')
    run_Adam_model(build_normal_model,adam_stats)


def prep_stats(stats):
    for k in ('loss','acc'):
        l = stats[k]
        stats[k]={'mean':np.mean(l, 1),'disp':np.std(l, 1)}

prep_stats(sgd_stats)
prep_stats(adam_stats)
#
#
#
t = np.arange(num_batches)
fig, axs = plt.subplots(2, 1)
fig.canvas.set_window_title('Рис. 4.6. Сравнение стохастического градиентного спуска и Adam')
# Функция потерь
axs[0].errorbar(t, sgd_stats['loss']['mean'], yerr=sgd_stats['loss']['disp'], label=sgd_stats['name'])
axs[0].errorbar(t, adam_stats['loss']['mean'], yerr=adam_stats['loss']['disp'], label=adam_stats['name'])
axs[0].set_xlabel('batches')
axs[0].set_ylabel('Функция потерь')
axs[0].set_title('Функция потерь')
axs[0].grid(True)
axs[0].legend()
# Точность
axs[1].errorbar(t, sgd_stats['acc']['mean'], yerr=sgd_stats['acc']['disp'], label=sgd_stats['name'])
axs[1].errorbar(t, adam_stats['acc']['mean'], yerr=adam_stats['acc']['disp'], label=adam_stats['name'])
axs[1].set_xlabel('batches')
axs[1].set_ylabel('Точность')
axs[1].set_title('Точность')
axs[1].grid(True)
axs[1].legend()
fig.tight_layout()
plt.show()
