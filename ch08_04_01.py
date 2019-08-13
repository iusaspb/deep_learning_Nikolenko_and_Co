import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from operator import itemgetter

batch_size = 64
show_size= 1000
updates = 40002
updates_to_show=[0, 1001, updates - 1] # список итераций, перед которыми надо показать состояние
hist_data = []
learning_rate = 0.01
prior_mu = -2.5
prior_std = 0.5
noise_range = 5.
g_layer1_size=5
d_layer1_size=10
d_layer2_size=10


gen_weights = dict()
gen_weights['w1'] = tf.Variable(tf.random_normal([1, g_layer1_size]))
gen_weights['b1'] = tf.Variable(tf.random_normal([g_layer1_size]))
gen_weights['w2'] = tf.Variable(tf.random_normal([g_layer1_size, 1]))
gen_weights['b2'] = tf.Variable(tf.random_normal([1]))
disc_weights = dict()
disc_weights['w1'] = tf.Variable(tf.random_normal([1, d_layer1_size]))
disc_weights['b1'] = tf.Variable(tf.random_normal([d_layer1_size]))
disc_weights['w2'] = tf.Variable(tf.random_normal([d_layer1_size, d_layer2_size]))
disc_weights['b2'] = tf.Variable(tf.random_normal([d_layer2_size]))
disc_weights['w3'] = tf.Variable(tf.random_normal([d_layer2_size, 1]))
disc_weights['b3'] = tf.Variable(tf.random_normal([1]))
z_p = tf.placeholder('float', [None, 1])
x_d = tf.placeholder('float', [None, 1])
g_h = tf.nn.softplus(tf.add(tf.matmul(z_p, gen_weights['w1']), gen_weights['b1']))
x_g = tf.add(tf.matmul(g_h, gen_weights['w2']), gen_weights['b2'])

def discriminator(x):
    d_h1 = tf.nn.tanh(tf.add(tf.matmul(x, disc_weights['w1']), disc_weights['b1']))
    d_h2 = tf.nn.tanh(tf.add(tf.matmul(d_h1, disc_weights['w2']), disc_weights['b2']))
    score = tf.nn.sigmoid(tf.add(tf.matmul(d_h2, disc_weights['w3']), disc_weights['b3']))
    return score

x_data_score = discriminator(x_d)
x_gen_score = discriminator(x_g)

D_cost = -tf.reduce_mean(tf.log(x_data_score) + tf.log(1.0 - x_gen_score))
G_cost = tf.reduce_mean(tf.log(1.0 - x_gen_score))

optimizer = tf.train.GradientDescentOptimizer(learning_rate)
D_optimizer = optimizer.minimize(D_cost, var_list=[v for v in disc_weights.values()])
G_optimizer = optimizer.minimize(G_cost, var_list=[v for v in gen_weights.values()])

def sample_z(size=batch_size):
    return np.random.uniform(-noise_range, noise_range, size=[size, 1])
def sample_x(size=batch_size, mu=prior_mu, std=prior_std):
    return np.random.normal(mu, std, size=[size, 1])

init = tf.global_variables_initializer()
alpha = 1e-3
with tf.Session() as sess:
    sess.run(init)
    for i in range(updates):
        if i in updates_to_show: # собираем данные для рисунка
            norm = sample_x(show_size) # сэмплы из настоящего нормального распределения
            z_show = sample_z(show_size)
            # X_g - данные, которые в данный момент порождает генератор
            # Z_gen_score - вероятность, которую дискриминатор присваивает тому,
            #   что данные настоящие по точкам из генератора
            # Z_data_score - вероятность, которую дискриминатор присваивает тому,
            #   что данные настоящие по точкам из сэмплов
            X_g,Z_gen_score,Z_data_score  = sess.run([x_g,x_gen_score,x_data_score], feed_dict={z_p: z_show, x_d: norm})
            hist_data.append({'n':norm, 'x':X_g, 'd':Z_gen_score, 'dd':Z_data_score})
        z_batch = sample_z()
        x_batch = sample_x()
        sess.run(D_optimizer, feed_dict={z_p: z_batch, x_d: x_batch})
        z_batch = sample_z()
        # _, x_z = stats.normaltest(z_batch)
        _, X_g = sess.run([G_optimizer,x_g], feed_dict={z_p: z_batch})
        # _, x_p = stats.normaltest(X_g)
        # print(f'i={i}  init data is normal? {x_z>= alpha}. p={x_p} is generated data normal distributed? {x_p >= alpha}')



# Рис. 8.10. Результат работы GAN
fig, axs = plt.subplots(1, len(updates_to_show))
fig.canvas.set_window_title('Рис. 8.10. Результат работы GAN')
def find_bins_number(df):
    q75, q25 = np.percentile(df, [75, 25])
    h = 2 * (q75 - q25) * len(df) ** (-1. / 3.)
    bins = int((max(df) - min(df)) / h)
    return bins

for i, stat in enumerate(hist_data):
    axs[i].set_title(f'После {updates_to_show[i] - 1} итераций')
    norm = stat['n']
    bins = find_bins_number(norm)
    axs[i].hist(norm, bins, density=True, color='Gray', label='норм расп.')
    x = stat['x']
    bins = find_bins_number(x)
    axs[i].hist(x, bins, density=True, histtype='step', color='Red',label='генератор')
    x = np.concatenate((x, norm), axis=0)
    z = np.concatenate((stat['d'],stat['dd']), axis=0)
    xz = sorted(zip(x, z), key=itemgetter(0))
    axs[i].plot([x for x,_ in xz], [z for _,z in xz],label='дискриминатор')
    axs[i].legend()
    axs[i].grid(True)

axs[0].set_title('До начала обучения')
plt.show()