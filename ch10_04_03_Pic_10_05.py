import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
image_size=28*28

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
n_samples = mnist.train.num_examples

def xavier_init(fan_in, fan_out, constant=1):
    low = -constant*np.sqrt(6.0/(fan_in + fan_out))
    high = constant*np.sqrt(6.0/(fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out),minval=low, maxval=high,dtype=tf.float32)

w, n_input, n_z = {}, image_size, 2 #20
n_hidden_recog_1, n_hidden_recog_2 = 500, 500
n_hidden_gener_1, n_hidden_gener_2 = 500, 500
w['w_recog'] = {
    'h1': tf.Variable(xavier_init(n_input, n_hidden_recog_1)),
    'h2': tf.Variable(xavier_init(n_hidden_recog_1, n_hidden_recog_2)),
    'out_mean': tf.Variable(xavier_init(n_hidden_recog_2, n_z)),
    'out_log_sigma': tf.Variable(xavier_init(n_hidden_recog_2, n_z))}
w['b_recog'] = {
    'b1': tf.Variable(tf.zeros([n_hidden_recog_1], dtype=tf.float32)),
    'b2': tf.Variable(tf.zeros([n_hidden_recog_2], dtype=tf.float32)),
    'out_mean': tf.Variable(tf.zeros([n_z], dtype=tf.float32)),
    'out_log_sigma': tf.Variable(tf.zeros([n_z], dtype=tf.float32))}
w['w_gener'] = {
    'h1': tf.Variable(xavier_init(n_z, n_hidden_gener_1)),
    'h2': tf.Variable(xavier_init(n_hidden_gener_1, n_hidden_gener_2)),
    'out_mean': tf.Variable(xavier_init(n_hidden_gener_2, n_input)),
    'out_log_sigma': tf.Variable(xavier_init(n_hidden_gener_2, n_input))}
w['b_gener'] = {
    'b1': tf.Variable(tf.zeros([n_hidden_gener_1], dtype=tf.float32)),
    'b2': tf.Variable(tf.zeros([n_hidden_gener_2], dtype=tf.float32)),
    'out_mean': tf.Variable(tf.zeros([n_input], dtype=tf.float32)),
    'out_log_sigma': tf.Variable(tf.zeros([n_input], dtype=tf.float32))}

l_rate=0.001
batch_size=100

x = tf.placeholder(tf.float32, [None, n_input])
enc_layer_1 = tf.nn.softplus(tf.add(tf.matmul(x, w["w_recog"]['h1']), w["b_recog"]['b1']))
enc_layer_2 = tf.nn.softplus(tf.add(tf.matmul(enc_layer_1, w["w_recog"]['h2']), w["b_recog"]['b2']))
z_mean = tf.add( tf.matmul(enc_layer_2, w["w_recog"]['out_mean']),w["b_recog"]['out_mean'])
z_log_sigma_sq = tf.add( tf.matmul(enc_layer_2, w["w_recog"]['out_log_sigma']), w["b_recog"]['out_log_sigma'])

# eps = tf.random_normal((batch_size, n_z), 0, 1, dtype=tf.float32)
eps = tf.placeholder(tf.float32, [None, n_z])
z = tf.add(z_mean, tf.multiply(tf.sqrt(tf.exp(z_log_sigma_sq)), eps))

dec_layer_1 = tf.nn.softplus(tf.add(tf.matmul(z, w["w_gener"]['h1']), w["b_gener"]['b1']))
dec_layer_2 = tf.nn.softplus(tf.add(tf.matmul(dec_layer_1, w["w_gener"]['h2']), w["b_gener"]['b2']))
x_reconstr_mean = tf.nn.sigmoid(tf.add(tf.matmul(dec_layer_2, w["w_gener"]['out_mean']),w["b_gener"]['out_mean']))

reconstr_loss = -tf.reduce_sum(x * tf.log(1e-10 + x_reconstr_mean) +(1-x) * tf.log(1e-10 + 1 - x_reconstr_mean), 1)

latent_loss = -0.5 * tf.reduce_sum(1 + z_log_sigma_sq- tf.square(z_mean) - tf.exp(z_log_sigma_sq), 1)
cost = tf.reduce_mean(reconstr_loss + latent_loss)
reconstr_loss_mean = tf.reduce_mean(reconstr_loss)
latent_loss_mean = tf.reduce_mean(latent_loss)
optimizer = tf.train.AdamOptimizer(learning_rate=l_rate).minimize(cost)

show_step=[20]
xs_show, _ = mnist.test.next_batch(5000)
def show_scatter(sess,xs,epoch):
    z_mu, y_sample = sess.run((z_mean, x_reconstr_mean),
                              feed_dict={x: xs, eps: np.random.normal(loc=0.0, scale=0.0, size=(xs.shape[0], n_z))})
    fig = plt.figure(figsize=(8, 6))
    plt.gray()
    fig.canvas.set_window_title(f'Рис. 10.5. 2D-представление распределения рукописных цифр в датасете MNIST эпоха {epoch}')
    plt.scatter(z_mu[:, 0], z_mu[:, 1], c=np.argmax(y_sample, 1))
    plt.colorbar()
    plt.show()

def train(sess, batch_size=100, training_epochs=10, display_step=5):
    show_scatter(sess, xs_show, 0)
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(n_samples / batch_size)
        # Цикл по мини-батчам
        for i in range(total_batch):
            xs, _ = mnist.train.next_batch(batch_size)
            # Обучаем на текущем мини-батче
            _, c,_reconstr_loss,_latent_loss = sess.run((optimizer, cost, reconstr_loss_mean,latent_loss_mean),
                                                        feed_dict={x: xs, eps:np.random.normal(loc=0.0, scale=1.0, size=(batch_size, n_z))})
            # Compute average loss
            avg_cost += c / n_samples * batch_size

        # Каждые display_step шагов выводим текущую функцию потерь
        if epoch % display_step == 0:
            print("Epoch: %04d\tcost: %.9f\trec_loss: %.9f\tlatent_loss: %.9f" % (epoch+1, avg_cost,_reconstr_loss,_latent_loss))
        if epoch in show_step:
            show_scatter(sess, xs_show, epoch)
    show_scatter(sess, xs_show, training_epochs)

init = tf.global_variables_initializer()
sess = tf.InteractiveSession()
sess.run(init)
train(sess, training_epochs=200, batch_size=batch_size)

sess.close()



