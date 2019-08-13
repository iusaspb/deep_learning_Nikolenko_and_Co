import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

batch_size = 64
latent_space = 128
learning_rate = 0.1
image_size=28*28
num_batches=100000

ae_weights = {"encoder_w": tf.Variable(tf.truncated_normal([image_size, latent_space], stddev=0.1)),
                "encoder_b": tf.Variable(tf.truncated_normal([latent_space], stddev=0.1)),
                "decoder_w": tf.Variable(tf.truncated_normal([latent_space, image_size], stddev=0.1)),
                "decoder_b": tf.Variable(tf.truncated_normal([image_size], stddev=0.1))}

ae_input = tf.placeholder(tf.float32, [batch_size, image_size])
hidden = tf.nn.sigmoid(tf.matmul(ae_input, ae_weights["encoder_w"]) + ae_weights["encoder_b"])
visible_logits = tf.matmul(hidden, ae_weights["decoder_w"]) + ae_weights["decoder_b"]
visible = tf.nn.sigmoid(visible_logits)
ae_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=visible_logits, labels=ae_input))
optimizer = tf.train.AdagradOptimizer(learning_rate)
ae_op = optimizer.minimize(ae_cost)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for batch in range(num_batches):
        x_batch, _ = mnist.train.next_batch(batch_size)
        _, ae_cost_val = sess.run([ae_op,ae_cost], feed_dict={ae_input: x_batch})
    print(f'ae_cost={ae_cost_val}')

