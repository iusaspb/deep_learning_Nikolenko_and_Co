import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
image_size=28*28
image_shape=(28,28)

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
n_samples = mnist.train.num_examples

def xavier_init(fan_in, fan_out, constant=1):
    low = -constant*np.sqrt(6.0/(fan_in + fan_out))
    high = constant*np.sqrt(6.0/(fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out),minval=low, maxval=high,dtype=tf.float32)

w, n_input, n_z = {}, image_size, 20
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

saver = tf.train.Saver()
sess = tf.InteractiveSession()
saver.restore(sess, "checkpoints/ch10_04_01.ckpt")
nrows,ncols=4,6
batch_size=nrows*ncols
x_sample,_ = mnist.test.next_batch(batch_size)
x_logits = sess.run(x_reconstr_mean,feed_dict={x: x_sample, eps:np.random.normal(loc=0., scale=0., size=(batch_size, n_z))})
gen_logits = sess.run(x_reconstr_mean,feed_dict={z_mean:np.zeros(shape=(batch_size,n_z))
                                               , eps:np.random.normal(loc=0., scale=1., size=(batch_size, n_z))
                                               ,  z_log_sigma_sq:np.ones(shape=(batch_size,n_z))})


fig, axs = plt.subplots(nrows=nrows, ncols=3*ncols+2, figsize=(3*ncols+2,nrows),subplot_kw={'xticks': [], 'yticks': []})
plt.gray()
fig.canvas.set_window_title('Рис. 10.4. Пример работы вариационного автокодировщика а — исходные изображения б — реконструированные в — сэмплированные')
for i in range(nrows*ncols):
    i_row, i_col = divmod(i, ncols)
    axs[i_row][i_col].imshow(np.reshape(x_sample[i] * 255, image_shape))
    i_col+=ncols+1
    axs[i_row][i_col].imshow(np.reshape(x_logits[i] * 255, image_shape))
    i_col += ncols + 1
    axs[i_row][i_col].imshow(np.reshape(gen_logits[i] * 255, image_shape))

plt.tight_layout()
plt.show()
sess.close()



