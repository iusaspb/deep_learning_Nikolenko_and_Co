# Рис. 3.11. Графики изменения функции ошибки и точности на тренировочном и тестовом множествах по мере обучения модели
#  у меня картинка получилась более регулярной - с меньшими выбросами.
# почему так - не пойму
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
image_size=784 #количество пикселов в картинке. Оно совпадает с размерностью входного слоя.
num_classes=10 #количество классов картинок. Оно совпадает с размерностью выходного слоя.
batch_size=100
num_batches=1000
num_runnings=100
train_history={'name':'Тренировочная выборка','loss':np.zeros((num_batches, num_runnings)), 'acc':np.zeros((num_batches, num_runnings))}
test_history={'name':'Тестовая выборка','loss':np.zeros((num_batches, num_runnings)), 'acc':np.zeros((num_batches, num_runnings))}
def run_model(running):
    print(f'running={running}')
    x = tf.placeholder(tf.float32, [None, image_size])
    W = tf.Variable(tf.zeros([image_size, num_classes]))
    b = tf.Variable(tf.zeros([num_classes]))
    y = tf.nn.softmax(tf.matmul(x, W) + b)
    y_ = tf.placeholder(tf.float32, [None, num_classes])
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for batch in range(num_batches):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            _,loss_val,acc_value = sess.run([train_step,cross_entropy,accuracy], feed_dict={x: batch_xs, y_: batch_ys})
            train_history['loss'][batch][running] = loss_val
            train_history['acc'][batch][running] = acc_value
            loss_val,acc_value = sess.run([cross_entropy,accuracy], feed_dict={x: mnist.test.images, y_: mnist.test.labels})
            test_history['loss'][batch][running] = loss_val
            test_history['acc'][batch][running] = acc_value

for running in range(num_runnings) :
    run_model(running)

train_m_history={}
train_v_history={}
for k in ('loss','acc'):
    l = train_history[k]
    train_m_history[k]= np.mean(l,1)
    train_v_history[k] = np.std(l, 1)

test_m_history={}
test_v_history={}
for k in ('loss','acc'):
    l = test_history[k]
    test_m_history[k]= np.mean(l,1)
    test_v_history[k] = np.std(l, 1)

import matplotlib.pyplot as plt

t = np.arange(num_batches)
fig, axs = plt.subplots(2, 1)
axs[0].errorbar(t, train_m_history['loss'], yerr=train_v_history['loss'], label=train_history['name'])
axs[0].errorbar(t, test_m_history['loss'], yerr=test_v_history['loss'], label=test_history['name'])
axs[0].set_xlabel('batches')
axs[0].set_ylabel('Функция потерь')
axs[0].set_title('Функция потерь')
axs[0].grid(True)
axs[0].legend()
axs[1].errorbar(t, train_m_history['acc'], yerr=train_v_history['acc'], label=train_history['name'])
axs[1].errorbar(t, test_m_history['acc'], yerr=test_v_history['acc'], label=test_history['name'])
axs[1].set_xlabel('batches')
axs[1].set_ylabel('Точность')
axs[1].set_title('Точность')
axs[1].grid(True)
axs[1].legend()
fig.tight_layout()
fig.canvas.set_window_title('Рис. 3.11. Графики изменения функции ошибки и точности на тренировочном и тестовом множествах по мере обучения модели')
plt.show()
