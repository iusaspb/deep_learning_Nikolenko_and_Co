# Рис. 3.14. Графики изменения функции ошибки и точности на тренировочном и тестовом множествах по мере обучения модели
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
image_size=28*28
num_classes=10
batch_size=100
num_batches=10000
layer_size=100
num_runnings=100
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
train_history={'name':'Тренировочная выборка','loss':np.zeros((num_batches, num_runnings)), 'acc':np.zeros((num_batches, num_runnings))}
test_history={'name':'Тестовая выборка','loss':np.zeros((num_batches, num_runnings)), 'acc':np.zeros((num_batches, num_runnings))}


def run_model(running):
    print(f'running={running}')

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
    logit = tf.matmul(h_drop, W) + b
    y = tf.nn.softmax(logit)

    y_ = tf.placeholder(tf.float32, [None, num_classes])
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=logit))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
    init = tf.global_variables_initializer()
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    with tf.Session() as sess:
        sess.run(init)
        for batch in range(num_batches):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            _,loss_val,acc_value = sess.run([train_step,cross_entropy,accuracy], feed_dict={x: batch_xs, y_: batch_ys, keep_probability: 0.5})
            train_history['loss'][batch][running] = loss_val
            train_history['acc'][batch][running] = acc_value
            loss_val,acc_value = sess.run([cross_entropy,accuracy], feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_probability: 1.})
            test_history['loss'][batch][running] = loss_val
            test_history['acc'][batch][running] = acc_value

for running in range(num_runnings) :
    run_model(running)

train_mean_history={}
train_disp_history={}
for k in ('loss','acc'):
    l = train_history[k]
    train_mean_history[k]= np.mean(l, 1)
    train_disp_history[k] = np.std(l, 1)

test_mean_history={}
test_disp_history={}
for k in ('loss','acc'):
    l = test_history[k]
    test_mean_history[k]= np.mean(l, 1)
    test_disp_history[k] = np.std(l, 1)

import matplotlib.pyplot as plt

t = np.arange(num_batches)
fig, axs = plt.subplots(2, 1)
axs[0].errorbar(t, train_mean_history['loss'], yerr=train_disp_history['loss'], label=train_history['name'])
axs[0].errorbar(t, test_mean_history['loss'], yerr=test_disp_history['loss'], label=test_history['name'])
axs[0].set_xlabel('batches')
axs[0].set_ylabel('Функция потерь')
axs[0].set_title('Функция потерь')
axs[0].grid(True)
axs[0].legend()
axs[1].errorbar(t, train_mean_history['acc'], yerr=train_disp_history['acc'], label=train_history['name'])
axs[1].errorbar(t, test_mean_history['acc'], yerr=test_disp_history['acc'], label=test_history['name'])
axs[1].set_xlabel('batches')
axs[1].set_ylabel('Точность')
axs[1].set_title('Точность')
axs[1].grid(True)
axs[1].legend()
fig.tight_layout()
fig.canvas.set_window_title('Рис. 3.14. Графики изменения функции ошибки и точности на тренировочном и тестовом множествах по мере обучения модели')
plt.show()
