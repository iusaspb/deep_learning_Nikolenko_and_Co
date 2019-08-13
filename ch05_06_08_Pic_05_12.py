# Рис. 5.12. Пример работы шумоподавляющего автокодировщика: а — исходные изображения; б — реконструированные; в — зашумленные; г — восстановленные из зашумленных
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
noise_prob=0.3

ae_weights = {  "encoder_w": tf.Variable(tf.truncated_normal([image_size, latent_space], stddev=0.1)),
                "encoder_b": tf.Variable(tf.truncated_normal([latent_space], stddev=0.1)),
                "decoder_w": tf.Variable(tf.truncated_normal([latent_space, image_size], stddev=0.1)),
                "decoder_b": tf.Variable(tf.truncated_normal([image_size], stddev=0.1)),
                "noisy_encoder_w": tf.Variable(tf.truncated_normal([image_size, latent_space], stddev=0.1)),
                "noisy_encoder_b": tf.Variable(tf.truncated_normal([latent_space], stddev=0.1)),
                "noisy_decoder_w": tf.Variable(tf.truncated_normal([latent_space, image_size], stddev=0.1)),
                "noisy_decoder_b": tf.Variable(tf.truncated_normal([image_size], stddev=0.1))}

ae_input = tf.placeholder(tf.float32, [batch_size, image_size])
hidden = tf.nn.sigmoid(tf.matmul(ae_input, ae_weights["encoder_w"]) + ae_weights["encoder_b"])
visible_logits = tf.matmul(hidden, ae_weights["decoder_w"]) + ae_weights["decoder_b"]
visible = tf.nn.sigmoid(visible_logits)
ae_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=visible_logits, labels=ae_input))
optimizer = tf.train.AdagradOptimizer(learning_rate)
ae_op = optimizer.minimize(ae_cost)

noisy_ae_input = tf.placeholder(tf.float32, [batch_size, image_size])
noisy_hidden = tf.nn.sigmoid(tf.matmul(noisy_ae_input, ae_weights["noisy_encoder_w"]) + ae_weights["noisy_encoder_b"])
noisy_visible_logits = tf.matmul(hidden, ae_weights["noisy_decoder_w"]) + ae_weights["noisy_decoder_b"]
noisy_visible = tf.nn.sigmoid(noisy_visible_logits)
noisy_ae_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=noisy_visible_logits, labels=ae_input)) # !!! сравниваем с ae_input. не с noisy_ae_input
noisy_optimizer = tf.train.AdagradOptimizer(learning_rate)
noisy_ae_op = noisy_optimizer.minimize(noisy_ae_cost)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for batch in range(num_batches):
        x_batch, _ = mnist.train.next_batch(batch_size)
        noise_mask = np.random.uniform(0., 1., [batch_size, image_size]) < noise_prob
        noisy_batch = x_batch.copy()
        noisy_batch[noise_mask] = 0.0
        sess.run(ae_op, feed_dict={ae_input: x_batch})
        sess.run(noisy_ae_op, feed_dict={ae_input: x_batch,noisy_ae_input: noisy_batch})
    ae_cost_val, visible_val,noisy_ae_cost_val, noisy_visible_val \
        = sess.run([ae_cost, visible,noisy_ae_cost, noisy_visible],
                   feed_dict={ae_input: x_batch,noisy_ae_input: noisy_batch})
    print(f'ae_cost={ae_cost_val},noisy_ae_cost={noisy_ae_cost_val}')
#
#
#
nrows,ncols=5,5
fig, axs = plt.subplots(nrows=nrows, ncols=4*ncols+3, figsize=(4*ncols+3,nrows),subplot_kw={'xticks': [], 'yticks': []})
plt.gray()
fig.canvas.set_window_title('Рис. 5.12. Пример работы шумоподавляющего автокодировщика: а — исходные изображения; б — реконструированные; в — зашумленные; г — восстановленные из зашумленных')
for i in range(nrows*ncols):
    i_row, i_col = divmod(i, ncols)
    axs[i_row][i_col].imshow(np.reshape(x_batch[i] * 255, image_shape))
    i_col+=ncols+1
    axs[i_row][i_col].imshow(np.reshape(visible_val[i] * 255, image_shape))
    i_col += ncols + 1
    axs[i_row][i_col].imshow(np.reshape(noisy_batch[i] * 255, image_shape))
    i_col += ncols + 1
    axs[i_row][i_col].imshow(np.reshape(noisy_visible_val[i] * 255, image_shape))
plt.tight_layout()
plt.show()
