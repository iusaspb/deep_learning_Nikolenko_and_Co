import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

batch_size = 64
latent_space = 128
learning_rate = 0.1
image_size=28*28
num_batches=100000
rho = 0.05
beta = 0#1.0
noise_prob=0.3

ae_weights = {"encoder_w": tf.Variable(tf.truncated_normal([image_size, latent_space], stddev=0.1)),
                "encoder_b": tf.Variable(tf.truncated_normal([latent_space], stddev=0.1)),
                "decoder_w": tf.Variable(tf.truncated_normal([latent_space, image_size], stddev=0.1)),
                "decoder_b": tf.Variable(tf.truncated_normal([image_size], stddev=0.1))}

ae_input = tf.placeholder(tf.float32, [batch_size, image_size])
noisy_input = tf.placeholder(tf.float32, [batch_size, image_size])
hidden = tf.nn.sigmoid(tf.matmul(noisy_input, ae_weights["encoder_w"]) + ae_weights["encoder_b"])
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
        noise_mask = np.random.uniform(0., 1., [batch_size, image_size]) < noise_prob
        noisy_batch = x_batch.copy()
        noisy_batch[noise_mask] = 0.0
        _, ae_cost_val,reg_val = sess.run([ae_op,ae_cost, reg_cost], feed_dict={ae_input: x_batch,noisy_input: noisy_batch})
    print(f'ae_cost={ae_cost_val},reg_cost={reg_val},total={ae_cost_val+beta*reg_val}')

