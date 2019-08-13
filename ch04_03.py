# Рис. 4.5. Сравнение сети с нормализацией по мини-батчам и без нее
# TODO : не получилось воспроизвести поведение сети так, как указано на картинке в книге
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

image_width,image_height=28,28
image_size=image_width*image_height
num_classes=10
batch_size=100
num_epochs=20
num_runnings=2
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
batchnorm_history={'name': 'С нормализацией', 'loss':np.zeros((num_epochs, num_runnings)), 'acc':np.zeros((num_epochs, num_runnings))}
history={'name': 'Без нормализацией', 'loss':np.zeros((num_epochs, num_runnings)), 'acc':np.zeros((num_epochs, num_runnings))}
def fullyconnected_layer(tensor, input_size, out_size):
    W = tf.Variable(tf.truncated_normal([input_size, out_size], stddev=0.1))
    b = tf.Variable(tf.truncated_normal([out_size], stddev=0.1))
    return tf.nn.tanh(tf.matmul(tensor, W) + b)
def batchnorm_layer(tensor, size):
    batch_mean, batch_var = tf.nn.moments(tensor, [0])
    beta = tf.Variable(tf.zeros([size]))
    scale = tf.Variable(tf.ones([size]))
    return tf.nn.batch_normalization(tensor, batch_mean, batch_var, beta, scale, 0.001)

def run_batchnorm_model():
    x = tf.placeholder(tf.float32, [None, image_size])
    h1 = fullyconnected_layer(x, image_size, 100)
    h1_bn = batchnorm_layer(h1, 100) # c нормализацией
    h2 = fullyconnected_layer(h1_bn, 100, 100)
    y_logit = fullyconnected_layer(h2, 100, num_classes)
    y = tf.nn.softmax(y_logit)
    y_ = tf.placeholder(tf.float32, [None, num_classes])
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_, logits=y_logit))
    train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
    init = tf.global_variables_initializer()
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    for running in range(num_runnings):
        with tf.Session() as sess:
            sess.run(init)
            for i in range(num_epochs):
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                # accuracy вычисляем не по тестовым данным, а по тренировачным
                _,loss_val,acc_val =sess.run([train_op,loss,accuracy], feed_dict={x: batch_xs, y_: batch_ys})
                acc_val = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
                batchnorm_history['loss'][i][running] = loss_val
                batchnorm_history['acc'][i][running] = acc_val


def run_model():
    x = tf.placeholder(tf.float32, [None, image_size])
    h1 = fullyconnected_layer(x, image_size, 100)
    # h1_bn = batchnorm_layer(h1, 100) # c нормализацией
    # h2 = fullyconnected_layer(h1, 100, 100)
    y_logit = fullyconnected_layer(h1, 100, num_classes)
    y = tf.nn.softmax(y_logit)
    y_ = tf.placeholder(tf.float32, [None, num_classes])
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_logit))
    train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
    init = tf.global_variables_initializer()
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    for running in range(num_runnings):
        with tf.Session() as sess:
            sess.run(init)
            for i in range(num_epochs):
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                # accuracy вычисляем не по тестовым данным, а по тренировачным
                _,loss_val,acc_val =sess.run([train_op,loss,accuracy], feed_dict={x: batch_xs, y_: batch_ys})
                acc_val = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
                history['loss'][i][running] = loss_val
                history['acc'][i][running] = acc_val


run_batchnorm_model()
run_model()

mean_history={}
disp_history={}
for k in ('loss','acc'):
    l = history[k]
    mean_history[k]= np.mean(l, 1)
    disp_history[k] = np.std(l, 1)

batchnorm_mean_history={}
batchnorm_disp_history={}
for k in ('loss','acc'):
    l = batchnorm_history[k]
    batchnorm_mean_history[k]= np.mean(l, 1)
    batchnorm_disp_history[k] = np.std(l, 1)


#Рис. 4.5. Сравнение сети с нормализацией по мини-батчам и без нее
import matplotlib.pyplot as plt
t = np.arange(num_epochs)
fig, axs = plt.subplots(2, 1)
axs[0].errorbar(t, batchnorm_mean_history['loss'], yerr=batchnorm_disp_history['loss'], label=batchnorm_history['name'])
axs[0].errorbar(t, mean_history['loss'], yerr=disp_history['loss'], label=history['name'])
axs[0].set_xlabel('epochs')
axs[0].set_ylabel('Функция потерь')
axs[0].set_title('Функция потерь')
axs[0].grid(True)
axs[0].legend()
axs[1].errorbar(t, batchnorm_mean_history['acc'], yerr=batchnorm_disp_history['acc'], label=batchnorm_history['name'])
axs[1].errorbar(t, mean_history['acc'], yerr=disp_history['acc'], label=history['name'])
axs[1].set_xlabel('epochs')
axs[1].set_ylabel('Точность')
axs[1].set_title('Точность')
axs[1].grid(True)
axs[1].legend()
fig.tight_layout()
fig.canvas.set_window_title('Рис. 4.5. Сравнение сети с нормализацией по мини-батчам и без нее')
plt.show()
