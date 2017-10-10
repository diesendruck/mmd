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
from scipy.stats import norm


# Config.
parser = argparse.ArgumentParser()
parser.add_argument('--data_num', type=int, default=800)
parser.add_argument('--gen_num', type=int, default=300)
parser.add_argument('--z_dim', type=int, default=1)
parser.add_argument('--width', type=int, default=3,
                    help='width of generator layers')
parser.add_argument('--depth', type=int, default=6,
                    help='num of generator layers')
parser.add_argument('--sigma', type=float, default=0.5)
parser.add_argument('--thin_a', type=float, default=0.5)
parser.add_argument('--thin_b', type=float, default=0.5)
parser.add_argument('--learning_rate', type=float, default=1e-2)
parser.add_argument('--optimizer', type=str, default='adam',
                    choices=['adagrad', 'adam', 'gradientdescent',
                             'rmsprop'])
parser.add_argument('--total_num_runs', type=int, default=10101)
parser.add_argument('--save_iter', type=int, default=1500)
parser.add_argument('--expt', type=str, default='test')

args = parser.parse_args()
data_num = args.data_num
gen_num = args.gen_num
z_dim = args.z_dim
width = args.width
depth = args.depth
sigma = args.sigma
thin_a = args.thin_a
thin_b = args.thin_b
learning_rate = args.learning_rate
optimizer = args.optimizer
total_num_runs = args.total_num_runs
save_iter = args.save_iter
expt = args.expt
out_dim = 1
activation = tf.nn.elu
save_tag = 'expt{}_dn{}_gn{}_zd{}_w{}_d{}_lr{}_op_{}_sig{}_thin{}'.format(
    expt, data_num, gen_num, z_dim, width, depth, learning_rate, optimizer,
    sigma, str(thin_a)+str(thin_b))


def prepare_dirs_and_logging(expt):
    # Set up directories based on experiment name.
    logs_dir = './results/{}/logs'.format(expt)
    graphs_dir = './results/{}/graphs'.format(expt)
    checkpoints_dir = './results/{}/checkpoints'.format(expt)
    dirs_to_make = [logs_dir, graphs_dir, checkpoints_dir]
    for d in dirs_to_make:
        if not os.path.exists(d):
            os.makedirs(d)
            print('Made dir: {}'.format(d))

    saver = tf.train.Saver()
    summary_writer = tf.summary.FileWriter(logs_dir + '/train',
            sess.graph)
    step = tf.Variable(0, name='step', trainable=False)
    sv = tf.train.Supervisor(logdir=logs_dir,
                            is_chief=True,
                            saver=saver,
                            summary_op=None,
                            summary_writer=summary_writer,
                            save_model_secs=300,
                            global_step=step,
                            ready_for_local_init_op=None)
    return saver, checkpoints_dir, graphs_dir, logs_dir


def load_checkpoints(sess, saver, checkpoints_dir):
    print(' [*] Reading checkpoints') 
    ckpt = tf.train.get_checkpoint_state(checkpoints_dir)

    if ckpt and ckpt.model_checkpoint_path:

        # Check if user wants to continue.
        user_input = raw_input(
            'Found checkpoint {}. Proceed? (y/n) '.format(
                ckpt.model_checkpoint_path))
        if user_input != 'y':
            raise ValueError(
                ' [!] Cancelled. To start fresh, rm checkpoint files.')

        # Rewrite any necessary variables, based on loaded ckpt.
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        saver.restore(sess, os.path.join(checkpoints_dir,
            ckpt_name))
        print(' [*] Successfully loaded {}'.format(ckpt_name))
        could_load = True
        return could_load
    else:
        print(' [!] Failed to find a checkpoint')
        could_load = False 
        return could_load


def thin_data(data_unthinned):
    """Thins data, accepting ~90% of Cluster 1, and ~10% Cluster 2.
    Args:
      data_unthinned: List of scalar values.
    Returns:
      thinned: Numpy array of values, thinned according to logistic function.
    """ 

    thinned = [candidate for candidate in data_unthinned if 
               np.random.binomial(1, prob_of_keeping(candidate))]
    return thinned 


def prob_of_keeping(x):
    """Logistic mapping to preferentially thin samples.
    
    Maps [-Inf, Inf] to [0.9, 0.1], centered at 2.
    Args:
      x: Scalar, point from original distribution.
    Returns:
      p: Probability of being thinned.
    """
    p = 0.5 / (1 + np.exp(10 * (x - 1))) + 0.5
    return p


def generate_data(n):
    """Generates true data, and applies thinning function.
    
    Args:
      n: Number of data candidates to start with, to then be thinned.
    Returns:
      data_unthinned: Unthinned numpy array of points.
      data: Thinned numpy array of points.
    """
    n_c2 = n/2
    n_c1 = n - n_c2
    data_unthinned = np.concatenate((np.random.normal(0, 0.5, n_c1),
                                     np.random.normal(2, 0.5, n_c2)))

    data = thin_data(data_unthinned)
    return data_unthinned, data


def get_random_z(gen_num, z_dim):
    """Generates 2d array of noise input data."""
    #return np.random.uniform(size=[gen_num, z_dim],
    #                         low=-1.0, high=1.0)
    #return np.random.gamma(5, size=[gen_num, z_dim])
    #return np.random.standard_t(2, size=[gen_num, z_dim])
    return np.random.normal(0, 1, size=[gen_num, z_dim])


# Set up generator.
def generator(z, width=3, depth=3, activation=tf.nn.elu, out_dim=1,
              reuse=False):
    """Generates output, given noise input."""
    with tf.variable_scope('generator', reuse=reuse) as gen_vars:
        x = layers.dense(z, width, activation=activation)

        for idx in range(depth - 1):
            x = layers.dense(x, width, activation=activation)

        out = layers.dense(x, out_dim, activation=None)
    return out


# Define data.
data_unthinned, data = generate_data(data_num)
data_num = len(data)
gen_num = data_num

# Build model.
x = tf.placeholder(tf.float64, [data_num, 1], name='x')
z = tf.placeholder(tf.float64, [gen_num, z_dim], name='z')
x_len = tf.shape(x)[0]
z_len = tf.shape(z)[0]
g = generator(z, width=width, depth=depth, activation=activation,
              out_dim=out_dim)
v = tf.concat([x, g], 0)
VVT = tf.matmul(v, tf.transpose(v))
sqs = tf.reshape(tf.diag_part(VVT), [-1, 1])
sqs_tiled_horiz = tf.tile(sqs, tf.transpose(sqs).get_shape())
exp_object = sqs_tiled_horiz - 2 * VVT + tf.transpose(sqs_tiled_horiz)
K = tf.exp(-0.5 * (1 / sigma) * exp_object)
K_yy = K[data_num:, data_num:]
K_xy = K[:data_num, data_num:]
K_yy_upper = (tf.matrix_band_part(K_yy, 0, -1) - 
              tf.matrix_band_part(K_yy, 0, 0))
num_combos_yy = gen_num * (gen_num - 1) / 2

v_tiled_horiz = tf.tile(v, [1, x_len + z_len])
p1_inv = 1. / (thin_a / (1 + tf.exp(10 * (v_tiled_horiz - 1))) + thin_b)
p2_inv = tf.transpose(p1_inv)
p1_inv_xy = p1_inv[:data_num, data_num:]
p1_inv_xy_normed = p1_inv_xy / tf.reduce_sum(p1_inv_xy)
Kw_xy = K[:data_num, data_num:] * p1_inv_xy_normed
mmd = (tf.reduce_sum(K_yy_upper) / num_combos_yy -
       2 * tf.reduce_sum(Kw_xy))

#mmd = (tf.reduce_sum(K_yy_upper) / num_combos_yy -
#       2 * tf.reduce_sum(K_xy) / (data_num * gen_num))

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
OPTION = 2
if OPTION == 1:
    g_optim = opt.minimize(mmd, var_list=g_vars)
elif OPTION == 2:
    gradients, variables = zip(*opt.compute_gradients(mmd))
    gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
    g_optim = opt.apply_gradients(zip(gradients, variables))
else:
    gvs = opt(learning_rate).compute_gradients(mmd)
    capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
    g_optim = optimizer.apply_gradients(capped_gvs)



# BEGIN train.
init_op = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init_op)

# Prepare directories and checkpoints.
saver, checkpoints_dir, graphs_dir, logs_dir = prepare_dirs_and_logging(expt)
load_checkpoints(sess, saver, checkpoints_dir)
print args
start_time = time()
for i in range(total_num_runs):
    sess.run(g_optim,
             feed_dict={
                 z: get_random_z(gen_num, z_dim),
                 x: np.reshape(data, [-1, 1])})

    # Occasionally save and plot.
    if i % save_iter == 0:
        z_sample = get_random_z(gen_num, z_dim)
        x_sample = np.reshape(data, [-1, 1])
        mmd_xg_out, g_out = sess.run(
            [mmd, g], feed_dict={
                z: z_sample,
                x: x_sample})

        np.save(os.path.join(logs_dir, 'sample_z.npy'), z_sample)
        np.save(os.path.join(logs_dir, 'sample_x.npy'), x_sample)
        np.save(os.path.join(logs_dir, 'sample_g.npy'), g_out)
        with open(os.path.join(logs_dir, 'sample__log.txt'), 'w') as log_file:
            log_file.write(str(args))

        # Save checkpoint.
        saver.save(sess, os.path.join(checkpoints_dir, expt), global_step=i)

        # Print helpful summary.
        print '\niter:{} mmd_xg_out = {:.5f}'.format(i, mmd_xg_out)
        print ' min:{} max= {}'.format(min(g_out), max(g_out))
        print ' [*] Saved sample logs to {}'.format(logs_dir)
        print ' [*] Saved checkpoint {} to {}'.format(i, checkpoints_dir)

        if i > 0:
            elapsed_time = time() - start_time
            m, s = divmod(elapsed_time, 60)
            h, m = divmod(m, 60)
            elapsed_time_str = '{:.0f}:{:02.0f}:{:02.0f}'.format(h, m, s)
            total_est = elapsed_time / i * total_num_runs
            m, s = divmod(total_est, 60)
            h, m = divmod(m, 60)
            total_est_str = '{:.0f}:{:02.0f}:{:02.0f}'.format(h, m, s)
            time_per_iter = elapsed_time / i
            print ('\n  Time (s). Elapsed: {}, Avg/iter: {:.4f},'
                   ' Total est.: {}').format(elapsed_time_str, time_per_iter,
                                             total_est_str)

os.system('python eval_samples_thin.py --expt="{}"'.format(expt))
