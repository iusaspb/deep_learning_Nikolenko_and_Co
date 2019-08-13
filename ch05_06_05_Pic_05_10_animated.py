# Рис. 5.10. Пример работы разреженного автокодировщика: а — исходные изображения; б — реконструированные; в — восстановленные из «обрезанного» скрытого слоя
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

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

class Batcher(object):
  def __init__(self,
               images):
    np.random.seed(87654321)
    self._num_examples = images.shape[0]
    self._images = images
    self._indexes = np.arange(self._num_examples)
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

  @property
  def indexes(self):
    return self._indexes

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size):
    """Return the next `batch_size` examples from this data set."""
    start = self._index_in_epoch
    # Shuffle for the first epoch
    if self._epochs_completed == 0 and start == 0:
      perm0 = np.arange(self._num_examples)
      np.random.shuffle(perm0)
      self._indexes = self.indexes[perm0]
      self._images = self.images[perm0]
    # Go to the next epoch
    if start + batch_size > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Get the rest examples in this epoch
      rest_num_examples = self._num_examples - start
      images_rest_part = self._images[start:self._num_examples]
      indexes_rest_part = self._indexes[start:self._num_examples]
      # Shuffle the data
      perm = np.arange(self._num_examples)
      np.random.shuffle(perm)
      self._images = self.images[perm]
      self._indexes = self.indexes[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size - rest_num_examples
      end = self._index_in_epoch
      images_new_part = self._images[start:end]
      indexes_new_part = self._indexes[start:end]
      return np.concatenate((images_rest_part, images_new_part), axis=0),np.concatenate((indexes_rest_part, indexes_new_part), axis=0)
    else:
      self._index_in_epoch += batch_size
      end = self._index_in_epoch
      return self._images[start:end], self._indexes[start:end]

batcher = Batcher(mnist.train.images)
id_mon= any (index for index,label in  enumerate(mnist.train.labels) if label ==5)  # проследим на какой-нибудь 5
mon_hist=[]
with tf.Session() as sess:
    sess.run(init)
    for batch in range(num_batches):
        x_batch, indexes_val = batcher.next_batch(batch_size)
        _, visible_val = sess.run([ae_op,visible], feed_dict={ae_input: x_batch})
        if id_mon== -1:
            id_mon = indexes_val[np.random.randint(batch_size)]
        for index_batch  in range(batch_size) :
            if indexes_val[index_batch] == id_mon:
                mon_hist.append(np.reshape(visible_val[index_batch] * 255, image_shape))
                break
    ae_cost_val,reg_val,noised_visible_val,visible_val = sess.run([ae_cost, reg_cost,noised_visible,visible], feed_dict={ae_input: x_batch})
    print(f'ae_cost={ae_cost_val},reg_cost={reg_val},total={ae_cost_val+beta*reg_val}')

len_mon_hist=len(mon_hist)
print(f'mon_hist={len_mon_hist}')
if len_mon_hist > 0:
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10,5),subplot_kw={'xticks': [], 'yticks': []})
    plt.gray()
    ax=axs[0]
    axs[1].imshow(np.reshape(mnist.train.images[id_mon] * 255, image_shape))
    axs[1].set_title("Original")
    ax.set_xticks([])
    ax.set_yticks([])
    def update(frame_number):
        ax.cla()
        ax.imshow(mon_hist[frame_number % len_mon_hist])
        ax.set_title(f"frame {frame_number % len_mon_hist} ({len_mon_hist})")
    animation = FuncAnimation(fig, update, interval=100)
    plt.show()
