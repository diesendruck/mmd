import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pdb
import sys
from time import time
import tensorflow as tf
layers = tf.layers
from scipy.stats import norm
from energy_utils import energy, main


# Config.
parser = argparse.ArgumentParser()
parser.add_argument('--data_num', type=int, default=3)
parser.add_argument('--gen_num', type=int, default=2)
parser.add_argument('--z_dim', type=int, default=1)
parser.add_argument('--width', type=int, default=10,
                    help='width of generator layers')
parser.add_argument('--depth', type=int, default=20,
                    help='num of generator layers')
parser.add_argument('--learning_rate', type=float, default=1e-6)
parser.add_argument('--optimizer', type=str, default='rmsprop',
                    choices=['adagrad', 'adam', 'gradientdescent',
                             'rmsprop'])
parser.add_argument('--total_num_runs', type=int, default=20101)
parser.add_argument('--save_iter', type=int, default=1000)

args = parser.parse_args()
data_num = args.data_num
gen_num = args.gen_num
z_dim = args.z_dim
width = args.width
depth = args.depth
learning_rate = args.learning_rate
optimizer = args.optimizer
total_num_runs = args.total_num_runs
save_iter = args.save_iter
out_dim = 1
activation = tf.nn.elu
save_tag = 'dn{}_pn{}_zd{}_w{}_d{}_lr{}_op_{}'.format(data_num, gen_num, z_dim,
                                                      width, depth,
                                                      learning_rate, optimizer)

# GOAL: Draw 100 samples from standard normal. Call this the 'true' empirical
# distribution. Then propose 10 points that minimize the energy statistic
# between the 10 proposed points and the original 100 points.


def get_random_z(gen_num, z_dim):
    """Generates 2d array of noise input data."""
    #return np.random.uniform(size=[gen_num, z_dim],
    #                         low=-1.0, high=1.0)
    #return np.random.gamma(5, size=[gen_num, z_dim])
    #return np.random.standard_t(2, size=[gen_num, z_dim])
    return np.random.normal(size=[gen_num, z_dim])


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


# Set up true data, 100 points.
p = np.random.randn(data_num)  # Random normal.
p = np.array(range(data_num))  # Fixed integer range.
data = p
print 'TRUE DATA: ', p

# Test TensorFlow optimization.
x = tf.placeholder(tf.float64, [data_num, 1], name='x')
z = tf.placeholder(tf.float64, [gen_num, z_dim], name='z')
g = generator(z, width=width, depth=depth, activation=activation,
              out_dim=out_dim)
v = tf.concat([x, g], 0)

# Nodes for energy statistic.
v_tiled = tf.tile(v, (1, v.get_shape().as_list()[0]))
pairwise = v_tiled - tf.transpose(v_tiled)
energy_yx = abs(pairwise[data_num:, :data_num])
energy_yy = abs(pairwise[data_num:, data_num:])
between_group_term = tf.reduce_sum(energy_yx) / gen_num / data_num 
within_proposals_term = tf.reduce_sum(energy_yy) / gen_num / gen_num 
e = 2 * between_group_term - within_proposals_term

# Nodes for MMD statistic.
vvt = tf.matmul(v, tf.transpose(v))
sqs = tf.reshape(tf.diag_part(vvt), [-1, 1])
sqs_tiled_horiz = tf.tile(sqs, tf.transpose(sqs).get_shape())
exp_object = sqs_tiled_horiz - 2 * vvt + tf.transpose(sqs_tiled_horiz)
sigma = 0.5
K = tf.exp(-0.5 * (1 / sigma) * exp_object)
K_xx = K[:data_num, :data_num]
K_yy = K[data_num:, data_num:]
K_xy = K[:data_num, data_num:]
K_xx_upper = tf.matrix_band_part(K_xx, 0, -1)
K_yy_upper = tf.matrix_band_part(K_yy, 0, -1)
num_combos = data_num * (data_num - 1) / 2
mmd = (tf.reduce_sum(K_xx_upper) / num_combos +
       tf.reduce_sum(K_yy_upper) / num_combos -
       2 * tf.reduce_sum(K_xy) / (data_num * gen_num))

if optimizer == 'adagrad':
    opt = tf.train.AdagradOptimizer(learning_rate)
elif optimizer == 'adam':
    opt = tf.train.AdamOptimizer(learning_rate)
elif optimizer == 'rmsprop':
    opt = tf.train.RMSPropOptimizer(learning_rate)
else:
    opt = tf.train.GradientDescentOptimizer(learning_rate)

# Set up objective function, and apply gradient clipping.
g_vars = [var for var in tf.global_variables() if 'generator' in var.name]
OPTION = 1
objective = mmd
if OPTION == 1:
    g_optim = opt.minimize(objective, var_list=g_vars)
elif OPTION == 2:
    gradients, variables = zip(*opt.compute_gradients(objective))
    gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
    g_optim = opt.apply_gradients(zip(gradients, variables))
elif OPTION == 3:
    gvs = opt(learning_rate).compute_gradients(objective)
    capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
    g_optim = optimizer.apply_gradients(capped_gvs)

# Begin training.
init_op = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init_op)
print args
start_time = time()
mmd_e = []
for i in range(total_num_runs):
    sess.run(g_optim,
             feed_dict={
                 z: get_random_z(gen_num, z_dim),
                 x: data.reshape(-1, 1)})

    if i % save_iter == 100:
        z_sample = get_random_z(gen_num, z_dim)
        mmd_out, e_out, g_out = sess.run(
            [mmd, e, g], feed_dict={
                z: z_sample,
                x: data.reshape(-1, 1)})

        mmd_e.append([mmd_out, e_out])
        print 'Iter: {}, MMD: {:.2f}, e: {:.2f}'.format(i, mmd_out, e_out)
        if i % 1000 == 100:
            plt.plot([i for (i, j) in mmd_e], label='mmd')
            plt.plot([j for (i, j) in mmd_e], label='e')
            plt.legend()
            plt.savefig('mmd_e.png')
            plt.close()
            print 'saved plot'
        np.save('sample_z', z_sample)
        np.save('sample_x', data.reshape(-1, 1))
        np.save('sample_g', g_out)

        print g_out
        """
        fig, ax = plt.subplots()
        ax.hist(g_out, 20, normed=True, color='blue', alpha=0.3)
        ax.hist(np.random.randn(data_num, 1), 20, normed=True, color='green',
                alpha=0.3)
        xs = np.arange(-3, 3, 0.01)
        ax.plot(xs, norm.pdf(xs), 'r-', alpha=0.3)
        ax.set_ylim([0, 1.5])
        ax.set_title('mmd = {}'.format(mmd_out))
        plt.savefig('hist_{}_i{}.png'.format(save_tag, i))
        plt.close(fig)
        """

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

