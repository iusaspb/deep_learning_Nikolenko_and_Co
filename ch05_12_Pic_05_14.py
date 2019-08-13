#  Рис. 5.14. Пример работы сверточного автокодировщика: а — исходные изображения; # б — реконструированные; в — зашумленные; г — восстановленные из зашумленных
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
batch_size, learning_rate = 64, 0.01
image_width,image_height=28,28
channels_1=4
channels_2=16
scale=2
strides=[1, scale, scale, 1]
num_batches=100000
noise_prob=0.3
image_shape=(image_height, image_width)
image_size=image_height*image_width
total_params = 5*5*4+4+5*5*4+1
ae_weights = {
                "conv1": tf.Variable(tf.truncated_normal([5, 5, 1, channels_1], stddev=0.1)),
                "b_conv1": tf.Variable(tf.truncated_normal([channels_1], stddev=0.1)),

                "conv2": tf.Variable(tf.truncated_normal([5, 5, channels_1, channels_2], stddev=0.1)),
                "b_hidden": tf.Variable(tf.truncated_normal([channels_2], stddev=0.1)),
                #
                "deconv1": tf.Variable(tf.truncated_normal([5, 5, channels_1, channels_2], stddev=0.1)),
                "b_deconv": tf.Variable(tf.truncated_normal([channels_1], stddev=0.1)),

                "deconv2": tf.Variable(tf.truncated_normal([5, 5, 1, channels_1], stddev=0.1)),
                "b_visible": tf.Variable(tf.truncated_normal([1], stddev=0.1)),
                #  noisy model
                "noisy_conv1": tf.Variable(tf.truncated_normal([5, 5, 1, channels_1], stddev=0.1)),
                "noisy_b_conv1": tf.Variable(tf.truncated_normal([channels_1], stddev=0.1)),

                "noisy_conv2": tf.Variable(tf.truncated_normal([5, 5, channels_1, channels_2], stddev=0.1)),
                "noisy_b_hidden": tf.Variable(tf.truncated_normal([channels_2], stddev=0.1)),
                #
                "noisy_deconv1": tf.Variable(tf.truncated_normal([5, 5, channels_1, channels_2], stddev=0.1)),
                "noisy_b_deconv": tf.Variable(tf.truncated_normal([channels_1], stddev=0.1)),

                "noisy_deconv2": tf.Variable(tf.truncated_normal([5, 5, 1, channels_1], stddev=0.1)),
                "noisy_b_visible": tf.Variable(tf.truncated_normal([1], stddev=0.1))
}
ae_input = tf.placeholder(tf.float32, [batch_size, image_size])
images = tf.reshape(ae_input, [-1, image_height, image_width, 1])
# первый свёрточный слой 1->4
conv_h1_logits = tf.nn.conv2d(images, ae_weights["conv1"],strides=strides, padding="SAME") + ae_weights["b_conv1"]
conv_h1 = tf.nn.relu(conv_h1_logits)
# # второй свёрточный слой 4->16
hidden_logits = tf.nn.conv2d(conv_h1, ae_weights["conv2"],strides=strides, padding="SAME") + ae_weights["b_hidden"]
hidden = tf.nn.relu(hidden_logits)
# # первый развёрточный  слой 16->4
deconv_h1_logits = tf.nn.conv2d_transpose(hidden,ae_weights["deconv1"],[batch_size, int(image_height/scale), int(image_width/scale), channels_1], strides=strides,padding="SAME") + ae_weights["b_deconv"]
deconv_h1 = tf.nn.relu(deconv_h1_logits)
# второй развёрточный  слой 4->1
visible_logits = tf.nn.conv2d_transpose(deconv_h1,ae_weights["deconv2"],[batch_size, image_height, image_width, 1], strides=strides,padding="SAME") + ae_weights["b_visible"]
visible = tf.nn.sigmoid(visible_logits)

optimizer = tf.train.AdagradOptimizer(learning_rate)
conv_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=visible_logits, labels=images))
conv_op = optimizer.minimize(conv_cost)
#
# noisy
#
noisy_ae_input = tf.placeholder(tf.float32, [batch_size, image_size])
noisy_images = tf.reshape(noisy_ae_input, [-1, image_height, image_width, 1])
# первый свёрточный слой 1->4
noisy_conv_h1_logits = tf.nn.conv2d(noisy_images, ae_weights["noisy_conv1"],strides=strides, padding="SAME") + ae_weights["noisy_b_conv1"]
noisy_conv_h1 = tf.nn.relu(noisy_conv_h1_logits)
# # второй свёрточный слой 4->16
noisy_hidden_logits = tf.nn.conv2d(noisy_conv_h1, ae_weights["noisy_conv2"],strides=strides, padding="SAME") + ae_weights["noisy_b_hidden"]
noisy_hidden = tf.nn.relu(noisy_hidden_logits)
# # первый развёрточный  слой 16->4
noisy_deconv_h1_logits = tf.nn.conv2d_transpose(noisy_hidden,ae_weights["noisy_deconv1"],[batch_size, int(image_height/scale), int(image_width/scale), channels_1], strides=strides,padding="SAME") + ae_weights["noisy_b_deconv"]
noisy_deconv_h1 = tf.nn.relu(noisy_deconv_h1_logits)
# второй развёрточный  слой 4->1
noisy_visible_logits = tf.nn.conv2d_transpose(noisy_deconv_h1,ae_weights["noisy_deconv2"],[batch_size, image_height, image_width, 1], strides=strides,padding="SAME") + ae_weights["noisy_b_visible"]
noisy_visible = tf.nn.sigmoid(noisy_visible_logits)

noisy_optimizer = tf.train.AdagradOptimizer(learning_rate)
noisy_conv_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=noisy_visible_logits, labels=images)) # !!! сравниваем с images. не с noisy_images
noisy_conv_op = optimizer.minimize(noisy_conv_cost)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for batch in range(num_batches):
        x_batch, _ = mnist.train.next_batch(batch_size)
        noise_mask = np.random.uniform(0., 1., [batch_size, image_size]) < noise_prob
        noisy_batch = x_batch.copy()
        noisy_batch[noise_mask] = 0.0
        sess.run(conv_op,feed_dict={ae_input: x_batch})
        sess.run(noisy_conv_op, feed_dict={ae_input: x_batch,noisy_ae_input: noisy_batch})
    conv_cost_val, visible_val,noisy_conv_cost_val, noisy_visible_val \
        = sess.run([conv_cost, visible,noisy_conv_cost, noisy_visible],
                   feed_dict={ae_input: x_batch,noisy_ae_input: noisy_batch})
    print(f'cost={conv_cost_val},noisy_cost={noisy_conv_cost_val}')
    print(conv_cost_val)
#
#
#
nrows,ncols=5,5
fig, axs = plt.subplots(nrows=nrows, ncols=4*ncols+3, figsize=(4*ncols+3,nrows),subplot_kw={'xticks': [], 'yticks': []})
plt.gray()
fig.canvas.set_window_title('Рис. 5.14. Пример работы сверточного автокодировщика а - исходные изображения; б - реконструированные; в - зашумленные; г - восстановленные из зашумленных')
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
