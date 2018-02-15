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


# Config.
parser = argparse.ArgumentParser()
parser.add_argument('--data_num', type=int, default=100)
parser.add_argument('--z_dim', type=int, default=10)
parser.add_argument('--width', type=int, default=5,
                    help='width of generator layers')
parser.add_argument('--depth', type=int, default=5,
                    help='num of generator layers')
parser.add_argument('--log_step', type=int, default=500)
parser.add_argument('--max_step', type=int, default=200000)
parser.add_argument('--learning_rate', type=float, default=1e-4)
parser.add_argument('--optimizer', type=str, default='adam',
                    choices=['adagrad', 'adam', 'gradientdescent', 'rmsprop'])
parser.add_argument('--data_file', type=str, default=None)

args = parser.parse_args()
data_num = args.data_num
z_dim = args.z_dim
width = args.width
depth = args.depth
log_step = args.log_step
max_step = args.max_step
learning_rate = args.learning_rate
optimizer = args.optimizer
data_file = args.data_file
activation = tf.nn.elu
save_tag = 'dn{}_zd{}_w{}_d{}_lr{}_op_{}'.format(
    data_num, z_dim, width, depth, learning_rate, optimizer)

# Load data.
if data_file:
    data = np.loadtxt(data_file,
                      converters={0: lambda s: float(s.strip() or 0)},
                      skiprows=1)
    out_dim = data.shape[1]
    pdb.set_trace()
else:
    n1 = data_num / 2
    n2 = data_num - n1
    cluster1 = np.random.multivariate_normal([0., 0.], [[1., 0.], [0., 1.]], n1)
    cluster2 = np.random.multivariate_normal([6., 6.], [[1., 0.], [0., 1.]], n2)
    data = np.concatenate((cluster1, cluster2))
    out_dim = data.shape[1]

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
x = tf.placeholder(tf.float64, [data_num, out_dim], name='x')
z = tf.placeholder(tf.float64, [data_num, z_dim], name='z')
g = generator(z, width=width, depth=depth, activation=activation,
              out_dim=out_dim)
v = tf.concat([x, g], 0)
VVT = tf.matmul(v, tf.transpose(v))
sqs = tf.reshape(tf.diag_part(VVT), [-1, 1])
sqs_tiled_horiz = tf.tile(sqs, tf.transpose(sqs).get_shape())
exp_object = sqs_tiled_horiz - 2 * VVT + tf.transpose(sqs_tiled_horiz)
sigma = 1
K = tf.exp(-0.5 * (1 / sigma) * exp_object)
# TODO: Add sum of kernels.
K_xx = K[:data_num, :data_num]
K_yy = K[data_num:, data_num:]
K_xy = K[:data_num, data_num:]
K_xx_upper = tf.matrix_band_part(K_xx, 0, -1)
K_yy_upper = tf.matrix_band_part(K_yy, 0, -1)
num_combos = data_num * (data_num - 1) / 2
mmd = (tf.reduce_sum(K_xx_upper) / num_combos +
       tf.reduce_sum(K_yy_upper) / num_combos -
       2 * tf.reduce_sum(K_xy) / (data_num * data_num))
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
print args
start_time = time()
for i in range(max_step):
    sess.run(g_optim,
             feed_dict={
                 z: get_random_z(data_num, z_dim),
                 # x: np.random.choice(data, (data_num, 1))})
                 x: data})

    # Occasionally log/plot results.
    if i % log_step == 0:
        mmd_out, g_out = sess.run(
            [mmd, g], feed_dict={
                z: get_random_z(data_num, z_dim),
                # x: np.random.choice(data, (data_num, 1))})
                x: data})
        print '\niter:{} mmd = {}'.format(i, mmd_out)
        fig, ax = plt.subplots()
        ax.scatter(*zip(*g_out), color='blue', alpha=0.3)
        ax.scatter(*zip(*data), color='green', alpha=0.3)
        plt.savefig(os.path.join(
            log_dir, 'scatter_{}_i{}.png'.format(save_tag, i)))
        plt.close(fig)

        if i > 0:
            elapsed_time = time() - start_time
            time_per_iter = elapsed_time / i
            total_est = elapsed_time / i * max_step
            m, s = divmod(total_est, 60)
            h, m = divmod(m, 60)
            total_est_str = '{:.0f}:{:02.0f}:{:02.0f}'.format(h, m, s)
            print ('\nTime (s). Elapsed: {:.2f}, Avg/iter: {:.4f},'
                   ' Total est.: {}').format(elapsed_time, time_per_iter,
                                             total_est_str)
