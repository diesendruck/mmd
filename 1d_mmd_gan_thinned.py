import argparse
from time import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pdb
import tensorflow as tf
layers = tf.layers
from scipy.stats import norm


# Config.
parser = argparse.ArgumentParser()
parser.add_argument('--starting_data_num', type=int, default=1000)
parser.add_argument('--z_dim', type=int, default=1)
parser.add_argument('--width', type=int, default=10,
                    help='width of generator layers')
parser.add_argument('--depth', type=int, default=20,
                    help='num of generator layers')
parser.add_argument('--learning_rate', type=float, default=1e-4)
parser.add_argument('--optimizer', type=str, default='adam',
                    choices=['adagrad', 'adam', 'gradientdescent',
                             'rmsprop'])
parser.add_argument('--total_num_runs', type=int, default=200101)
parser.add_argument('--save_iter', type=int, default=1000)

args = parser.parse_args()
starting_data_num = args.starting_data_num
z_dim = args.z_dim
width = args.width
depth = args.depth
learning_rate = args.learning_rate
optimizer = args.optimizer
total_num_runs = args.total_num_runs
save_iter = args.save_iter
out_dim = 1
activation = tf.nn.elu
save_tag = 'dn{}_zd{}_w{}_d{}_lr{}_op_{}'.format(starting_data_num, z_dim,
                                                 width, depth, learning_rate,
                                                 optimizer)

# Set up true, training data.
def generate_data(n):
    """Generates true data, and applies thinning function.
    
    Args:
      n: Number of data candidates to start with, to then be thinned.

    Returns:
      data_candidates: Thinned numpy array of points.
      data: Unthinned numpy array of points.
    """
    n_c1 = n/2
    n_c2 = n - n_c1
    data_candidates = np.concatenate((np.random.normal(0, 1, n_c1),
                                      np.random.normal(4, 1, n_c2)))

    data = np.array(thin(data_candidates))
    return data_candidates, data


def thin(data_candidates):
    """Thins data, accepting ~90% of Cluster 1, and ~10% Cluster 2.

    Args:
      data_candidates: List of scalar values.

    Returns:
      thinned: List of scalar values, thinned according to sigmoidal fn.
    """ 
    # Sigmoid mapping [-Inf, Inf] to [0.9, 0.1], centered at 2.
    def sigmoid(x):
        p = 0.8 / (1 + np.exp(10 * (x - 2))) + 0.1
        return p

    thinned = [d for d in data_candidates if 
               np.random.binomial(1, sigmoid(d))]
    return thinned 


def get_random_z(gen_num, z_dim):
    """Generates 2d array of noise input data."""
    #return np.random.uniform(size=[gen_num, z_dim],
    #                         low=-1.0, high=1.0)
    return np.random.normal(0, 1, size=[gen_num, z_dim])


# Set up generator.
def generator(z, width=3, depth=3, activation=tf.nn.elu, out_dim=1,
              reuse=False):
    """Generates output, given noise input."""
    with tf.variable_scope('generator', reuse=reuse):
        x = layers.dense(z, width, activation=activation)

        for idx in range(depth - 1):
            x = layers.dense(x, width, activation=activation)

        out = layers.dense(x, out_dim, activation=None)
    return out

# Set up data.
data_candidates, data = generate_data(starting_data_num)
data_num = len(data)

# Build model.
x = tf.placeholder(tf.float64, [data_num, 1], name='x')
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
    opt = tf.train.AdagradOptimizer(learning_rate)
elif optimizer == 'adam':
    opt = tf.train.AdamOptimizer(learning_rate)
elif optimizer == 'rmsprop':
    opt = tf.train.RMSPropOptimizer(learning_rate)
else:
    opt = tf.train.GradientDescentOptimizer(learning_rate)
# Set up objective function, and apply gradient clipping.
#g_optim = opt(learning_rate).minimize(mmd, var_list=g_vars)
gradients, variables = zip(*opt.compute_gradients(mmd))
gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
g_optim = opt.apply_gradients(zip(gradients, variables))

#gvs = opt(learning_rate).compute_gradients(mmd)
#capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
#g_optim = optimizer.apply_gradients(capped_gvs)

# Train.
init_op = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init_op)
print args
start_time = time()
for i in range(total_num_runs):
    sess.run(g_optim,
             feed_dict={
                 z: get_random_z(data_num, z_dim),
                 x: np.random.choice(data, (data_num, 1))})

    if i % save_iter == 100:
        z_sample = get_random_z(data_num, z_dim)
        x_sample = np.random.choice(data, (data_num, 1))
        mmd_out, g_out = sess.run(
            [mmd, g], feed_dict={
                z: z_sample,
                x: x_sample})
        np.save('z_sample', z_sample)
        np.save('x_sample', x_sample)
        np.save('g_sample', g_out)

        print '\niter:{} mmd = {}'.format(i, mmd_out)
        print 'min:{} max= {}'.format(min(g_out), max(g_out))
        fig, ax = plt.subplots()
        ax.hist(g_out, 30, normed=True, color='blue', alpha=0.3)
        ax.hist(x_sample, 30, normed=True, color='green', alpha=0.3)
        ax.set_ylim([0, 1.5])
        ax.set_title('mmd = {}'.format(mmd_out))
        plt.savefig('hist_{}_i{}.png'.format(save_tag, i))
        plt.close(fig)

        if i > 0:
            elapsed_time = time() - start_time
            time_per_iter = elapsed_time / i
            total_est = elapsed_time / i * total_num_runs
            m, s = divmod(total_est, 60)
            h, m = divmod(m, 60)
            total_est_str = '{:.0f}:{:02.0f}:{:02.0f}'.format(h, m, s)
            print ('\nTime (s). Elapsed: {:.2f}, Avg/iter: {:.4f},'
                   ' Total est.: {}').format(elapsed_time, time_per_iter,
                                             total_est_str)
