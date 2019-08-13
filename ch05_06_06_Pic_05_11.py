# Рис. 5.11. Некоторые фильтры разреженного автокодировщика
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

batch_size = 64
latent_space = 128
learning_rate = 0.1
image_width,image_height=28,28
image_shape=(image_height, image_width)
image_size=image_height*image_width
num_batches=100000
rho = 0.05
beta = 1.0

ae_weights = {"encoder_w": tf.Variable(tf.truncated_normal([image_size, latent_space], stddev=0.1)),
                "encoder_b": tf.Variable(tf.truncated_normal([latent_space], stddev=0.1)),
                "decoder_w": tf.Variable(tf.truncated_normal([latent_space, image_size], stddev=0.1)),
                "decoder_b": tf.Variable(tf.truncated_normal([image_size], stddev=0.1))}

ae_input = tf.placeholder(tf.float32, [batch_size, image_size])
hidden = tf.nn.sigmoid(tf.matmul(ae_input, ae_weights["encoder_w"]) + ae_weights["encoder_b"])
noised_hidden = tf.nn.relu(hidden - 0.1) + 0.1
noised_visible = tf.nn.sigmoid(tf.matmul(noised_hidden, ae_weights["decoder_w"]) + ae_weights["decoder_b"])
data_rho = tf.reduce_mean(hidden, 0)
reg_cost = - tf.reduce_mean(tf.log(data_rho/rho) * rho + tf.log((1-data_rho)/(1-rho)) * (1-rho))
visible_logits = tf.matmul(hidden, ae_weights["decoder_w"]) + ae_weights["decoder_b"]
visible = tf.nn.sigmoid(visible_logits)
ae_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=visible_logits, labels=ae_input))
total_cost = ae_cost + beta * reg_cost
optimizer = tf.train.AdagradOptimizer(learning_rate)
ae_op = optimizer.minimize(total_cost)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for batch in range(num_batches):
        x_batch, _ = mnist.train.next_batch(batch_size)
        sess.run(ae_op, feed_dict={ae_input: x_batch})
    decoder_w_val = sess.run(ae_weights["decoder_w"])

nrows,ncols=5,12
fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(3*ncols,3*nrows),subplot_kw={'xticks': [], 'yticks': []})
plt.gray()
fig.canvas.set_window_title('Рис. 5.11. Некоторые фильтры разреженного автокодировщика')
for i in range(nrows*ncols):
    i_row, i_col = divmod(i, ncols)
    axs[i_row][i_col].imshow(np.reshape(decoder_w_val[i] * 255, image_shape))
plt.tight_layout()
plt.show()
