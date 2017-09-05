import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np
import pdb
import tensorflow as tf
layers = tf.layers
from scipy.stats import norm


# Set up true, standard normal data.
data_num = 500
data = np.random.randn(data_num)

def get_random_z(gen_num, z_dim):
    return np.random.uniform(size=[gen_num, z_dim],
            low=-1.0, high=1.0)


# Set up generator.
z_dim = 20
width = 5
depth = 3
out_dim = 1
activation = tf.nn.elu
def generator(z, width=3, depth=3, activation=tf.nn.elu, out_dim=1, reuse=False):
    with tf.variable_scope('generator', reuse=reuse) as vs:
        x = layers.dense(z, width, activation=activation)

        for idx in range(depth - 1):
            x = layers.dense(x, width, activation=activation)

        out = layers.dense(x, out_dim, activation=None)
    return out


# Build model.
x = tf.expand_dims(tf.constant(data), 1)
z = tf.placeholder(tf.float64, [data_num, z_dim], name='z')
g = generator(z)
v = tf.concat([x, g], 0)
VVT = tf.matmul(v, tf.transpose(v))
sqs = tf.reshape(tf.diag_part(VVT), [-1, 1])
sqs_tiled_horiz = tf.tile(sqs, tf.transpose(sqs).get_shape())
exp_object = sqs_tiled_horiz - 2 * VVT + tf.transpose(sqs_tiled_horiz)
sigma = 1
K = tf.exp(-0.5 * (1 / sigma) * exp_object)
K_xx = K[:data_num, :data_num]
K_xy = K[:data_num, data_num:]
K_yy = K[data_num:, data_num:]
K_xx_upper = tf.matrix_band_part(K_xx, 0, -1)
K_xy_upper = tf.matrix_band_part(K_xy, 0, -1)
K_yy_upper = tf.matrix_band_part(K_yy, 0, -1)
num_combos = data_num * (data_num - 1) / 2 
mmd = (tf.reduce_sum(K_xx_upper) / num_combos -
    2 * tf.reduce_sum(K_xy_upper) / num_combos +
    tf.reduce_sum(K_yy_upper) / num_combos)
g_vars = [var for var in tf.global_variables() if 'generator' in var.name]
g_optim = tf.train.AdagradOptimizer(1e-3).minimize(mmd, var_list=g_vars)


# Train.
init_op = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init_op)
for i in range(500000):
    sess.run(g_optim, feed_dict={z: get_random_z(data_num, z_dim)})
    if i % 1000 == 0:
        mmd_out, g_out = sess.run(
                [mmd, g], feed_dict={z: get_random_z(data_num, z_dim)})
        print 'iter:{} mmd = {}'.format(i, mmd_out)
        fig, ax = plt.subplots()
        ax.hist(g_out, 50, normed=True, color='blue', alpha=0.3)
        ax.hist(data, 50, normed=True, color='green', alpha=0.3)
        xs = np.arange(-3, 3, 0.01)
        ax.plot(xs, norm.pdf(xs), 'r-', alpha=0.3)
        ax.set_ylim([0, 1.5])
        ax.set_title('mmd = {}'.format(mmd_out))
        plt.savefig('hist_g_out_{}'.format(i))
        plt.close(fig)
