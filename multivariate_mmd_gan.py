import argparse
from time import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import pdb
import tensorflow as tf
layers = tf.layers
import pandas as pd
import seaborn as sb


# Config.
parser = argparse.ArgumentParser()
parser.add_argument('--data_num', type=int, default=1000)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--z_dim', type=int, default=2)
parser.add_argument('--width', type=int, default=5,
                    help='width of generator layers')
parser.add_argument('--depth', type=int, default=5,
                    help='num of generator layers')
parser.add_argument('--log_step', type=int, default=2500)
parser.add_argument('--max_step', type=int, default=200000)
parser.add_argument('--learning_rate', type=float, default=1e-3)
parser.add_argument('--optimizer', type=str, default='rmsprop',
                    choices=['adagrad', 'adam', 'gradientdescent', 'rmsprop'])
parser.add_argument('--data_file', type=str, default=None)

args = parser.parse_args()
data_num = args.data_num
batch_size = args.batch_size
z_dim = args.z_dim
width = args.width
depth = args.depth
log_step = args.log_step
max_step = args.max_step
learning_rate = args.learning_rate
optimizer = args.optimizer
data_file = args.data_file
activation = tf.nn.elu

# Load data.
if data_file:
    data = np.loadtxt(open(data_file, 'rb'), delimiter=' ')
    data_num = len(data)
    out_dim = data.shape[1]
else:
    n1 = data_num / 2
    n2 = data_num - n1
    cluster1 = np.random.multivariate_normal(
        [-2., 5.], [[1., .9], [.9, 1.]], n1)
    cluster2 = np.random.multivariate_normal(
        [6., 6.], [[1., 0.], [0., 1.]], n2)
    data = np.concatenate((cluster1, cluster2))
    out_dim = data.shape[1]

# Set save tag, as a function of config parameters.
save_tag = 'dn{}_zd{}_w{}_d{}_lr{}_op_{}'.format(
    data_num, z_dim, width, depth, learning_rate, optimizer)

# Set up log dir.
log_dir = 'logs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)


def get_random_z(gen_num, z_dim):
    """Generates 2d array of noise input data."""
    return np.random.uniform(size=[gen_num, z_dim],
                             low=-1.0, high=1.0)


def generator(z, width=3, depth=3, activation=tf.nn.elu, out_dim=1,
              reuse=False):
    """Generates output, given noise input."""
    with tf.variable_scope('generator', reuse=reuse):
        x = layers.dense(z, width, activation=activation)

        for idx in range(depth - 1):
            x = layers.dense(x, width, activation=activation)

        out = layers.dense(x, out_dim, activation=None)
    return out


# Build model.
x = tf.placeholder(tf.float64, [batch_size, out_dim], name='x')
z = tf.placeholder(tf.float64, [batch_size, z_dim], name='z')
g = generator(z, width=width, depth=depth, activation=activation,
              out_dim=out_dim)
v = tf.concat([x, g], 0)
VVT = tf.matmul(v, tf.transpose(v))
sqs = tf.reshape(tf.diag_part(VVT), [-1, 1])
sqs_tiled_horiz = tf.tile(sqs, tf.transpose(sqs).get_shape())
exp_object = sqs_tiled_horiz - 2 * VVT + tf.transpose(sqs_tiled_horiz)
K = 0.0
sigma_list = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0]
for sigma in sigma_list:
    gamma = 1.0 / (2 * sigma**2)
    K += tf.exp(-gamma * exp_object)
K_xx = K[:batch_size, :batch_size]
K_yy = K[batch_size:, batch_size:]
K_xy = K[:batch_size, batch_size:]
K_xx_upper = tf.matrix_band_part(K_xx, 0, -1)
K_yy_upper = tf.matrix_band_part(K_yy, 0, -1)
num_combos = batch_size * (batch_size - 1) / 2
mmd = (tf.reduce_sum(K_xx_upper) / num_combos +
       tf.reduce_sum(K_yy_upper) / num_combos -
       2 * tf.reduce_sum(K_xy) / (batch_size * batch_size))
g_vars = [var for var in tf.global_variables() if 'generator' in var.name]
if optimizer == 'adagrad':
    opt = tf.train.AdagradOptimizer
elif optimizer == 'adam':
    opt = tf.train.AdamOptimizer
elif optimizer == 'rmsprop':
    opt = tf.train.RMSPropOptimizer
else:
    opt = tf.train.GradientDescentOptimizer
g_optim = opt(learning_rate).minimize(mmd, var_list=g_vars)

# Train.
init_op = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init_op)
print save_tag
start_time = time()
for step in range(max_step):
    random_batch_data = np.array(
        [data[d] for d in np.random.choice(len(data), batch_size)])
    random_batch_z = get_random_z(batch_size, z_dim)
    sess.run(g_optim,
             feed_dict={
                 z: random_batch_z,
                 x: random_batch_data})

    # Occasionally log/plot results.
    if step % log_step == 0:
        mmd_out, g_out = sess.run(
            [mmd, g], feed_dict={
                z: random_batch_z,
                x: random_batch_data})
        print '\niter:{} mmd = {}'.format(step, mmd_out)
        np.save('g_out.npy', g_out)

        if out_dim > 2:
            indices_to_plot = [0, 1, 2]
        elif out_dim == 2:
            indices_to_plot = range(out_dim)
            fig, ax = plt.subplots()
            ax.scatter(*zip(*data), color='gray', alpha=0.05)
            ax.scatter(*zip(*g_out), color='green', alpha=0.3)
            plt.savefig(os.path.join(
                log_dir, 'scatter_{}_i{}.png'.format(save_tag, step)))
            plt.close(fig)
        else:
            indices_to_plot = range(out_dim)

        pairplot_data = sb.pairplot(
            pd.DataFrame(random_batch_data[:, indices_to_plot]))
        pairplot_data.savefig('pairplot_data.png')
        pairplot_simulation = sb.pairplot(
            pd.DataFrame(g_out[:, indices_to_plot]))
        pairplot_simulation.savefig('pairplot_simulation.png')
        plt.close('all')

        if step > 0:
            elapsed_time = time() - start_time
            time_per_iter = elapsed_time / step
            total_est = elapsed_time / step * max_step
            m, s = divmod(total_est, 60)
            h, m = divmod(m, 60)
            total_est_str = '{:.0f}:{:02.0f}:{:02.0f}'.format(h, m, s)
            print ('\nIter: {}, time (s): {:.2f}, time/iter: {:.4f},'
                   ' Total est.: {}').format(step, elapsed_time, time_per_iter,
                                             total_est_str)
