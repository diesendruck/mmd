import argparse
from time import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pdb
import os
import tensorflow as tf
layers = tf.layers
from scipy.stats import norm


# Config.
def str2bool(v):
    return v.lower() in ('true', '1')
parser = argparse.ArgumentParser()
parser.add_argument('--starting_data_num', type=int, default=1000)
parser.add_argument('--z_dim', type=int, default=1)
parser.add_argument('--width', type=int, default=3,
                    help='width of generator layers')
parser.add_argument('--depth', type=int, default=6,
                    help='num of generator layers')
parser.add_argument('--learning_rate', type=float, default=1e-3)
parser.add_argument('--sigma', type=float, default=1.)
parser.add_argument('--optimizer', type=str, default='adam',
                    choices=['adagrad', 'adam', 'gradientdescent',
                             'rmsprop'])
parser.add_argument('--total_num_runs', type=int, default=200101)
parser.add_argument('--save_iter', type=int, default=200)
parser.add_argument('--expt', type=str, default='test')
parser.add_argument('--thinning_params', type=str, default='0.5,0.5',
                    help='parameters for prob_of_keeping function')
parser.add_argument('--weighting', type=str, default='naive',
                    choices=['naive', 'inverse', 'none'])
parser.add_argument('--gradient_clipping', type=str, default='none',
                    choices=['none', 'opt1', 'opt2'])

args = parser.parse_args()
starting_data_num = args.starting_data_num
z_dim = args.z_dim
width = args.width
depth = args.depth
learning_rate = args.learning_rate
sigma = args.sigma
optimizer = args.optimizer
total_num_runs = args.total_num_runs
save_iter = args.save_iter
expt = args.expt
thinning_params = args.thinning_params
thin_a, thin_b = [float(param) for param in thinning_params.split(',')]
weighting = args.weighting
gradient_clipping = args.gradient_clipping
out_dim = 1
activation = tf.nn.elu


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


# Set up true, training data.
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


def get_random_z(gen_num, z_dim):
    """Generates 2d array of noise input data.
    
    Args:
      gen_num: Number of values to generate.
      z_dim: Dimension of each value generated.

    Returns:
      z: Numpy array of dimension gen_num x z_dim. 
    
    """
    #return np.random.uniform(size=[gen_num, z_dim],
    #                         low=-1.0, high=1.0)
    #z =  np.random.standard_t(2, size=[gen_num, z_dim])
    z =  np.random.normal(0, 1, size=[gen_num, z_dim])
    return z


# Set up generator.
def generator(z, width=3, depth=3, activation=tf.nn.elu, out_dim=1,
              reuse=False):
    """Generates output, given noise input.
    
    Args:
      z: Numpy array as the noise input to the generator.
      width: Scalar, width of each neural net layer.
      depth: Scalar, number of neural net layers.
      activation: TensorFlow activation function for nodes.
      out_dim: Scalar, dimension of generator output.
      reuse: Flag of whether to reuse (without training).

    Returns:
      out: Numpy array of dimension z.shape[0] x out_dim. 
    """
    with tf.variable_scope('generator', reuse=reuse):
        x = layers.dense(z, width, activation=activation)

        for idx in range(depth - 1):
            x = layers.dense(x, width, activation=activation)

        out = layers.dense(x, out_dim, activation=None)
    return out


# Set up data.
data_unthinned, data = generate_data(starting_data_num)
data_num = len(data)

# Build model.
x = tf.placeholder(tf.float64, [None, 1], name='x')
z = tf.placeholder(tf.float64, [None, z_dim], name='z')
x_len = tf.shape(x)[0]
z_len = tf.shape(z)[0]
g = generator(z, width=width, depth=depth, activation=activation,
              out_dim=out_dim)
v = tf.concat([x, g], 0)
VVT = tf.matmul(v, tf.transpose(v))
sqs = tf.reshape(tf.diag_part(VVT), [-1, 1])

sqs_tiled_horiz = tf.tile(sqs, tf.shape(tf.transpose(sqs)))
exp_object = sqs_tiled_horiz - 2 * VVT + tf.transpose(sqs_tiled_horiz)
sigma = sigma 
K_orig = tf.exp(-0.5 * (1 / sigma) * exp_object)
v_tiled_horiz = tf.tile(v, [1, x_len + z_len])
# Define kernel with naive weights from thinning function, over entire kernel. 
p1 = thin_a / (1 + tf.exp(10 * (v_tiled_horiz - 1))) + thin_b 
p2 = tf.transpose(p1)
K_weighted_naive = p1 * p2 * K_orig
Kwn_xx = K_weighted_naive[:data_num, :data_num]
Kwn_yy = K_weighted_naive[data_num:, data_num:]
Kwn_xy = K_weighted_naive[:data_num, data_num:]
Kwn_xx_upper = tf.matrix_band_part(Kwn_xx, 0, -1) - tf.matrix_band_part(Kwn_xx, 0, 0)
Kwn_yy_upper = tf.matrix_band_part(Kwn_yy, 0, -1) - tf.matrix_band_part(Kwn_yy, 0, 0)
num_combos = data_num * (data_num - 1) / 2
mmd_wn = (tf.reduce_sum(Kwn_xx_upper) / num_combos +
          tf.reduce_sum(Kwn_yy_upper) / num_combos -
          2 * tf.reduce_sum(Kwn_xy) / (data_num * data_num))
# Define kernel with inverse weights only for data terms (i.e. with x).
p1_vert_unnormed = 1. / (thin_a / (1 + tf.exp(10 * (v -1))) + thin_b)
p1_vert_unnormed_sum = tf.reduce_sum(p1_vert_unnormed)
p1_vert = p1_vert_unnormed / p1_vert_unnormed_sum
p1_inverse = tf.tile(p1_vert, [1, x_len + z_len])
p2_inverse = tf.transpose(p1_inverse)
Kw_xx = (K_orig[:data_num, :data_num] *
         p1_inverse[:data_num, :data_num] *
         p2_inverse[:data_num, :data_num])
Kw_xy = (K_orig[:data_num, data_num:] *
         p1_inverse[:data_num, data_num:])
Kw_yy = K_orig[data_num:, data_num:]
Kw_xx_upper = (tf.matrix_band_part(Kw_xx, 0, -1) -
               tf.matrix_band_part(Kw_xx, 0, 0))
Kw_xy_upper = (tf.matrix_band_part(Kw_xy, 0, -1) -
               tf.matrix_band_part(Kw_xy, 0, 0))
Kw_yy_upper = (tf.matrix_band_part(Kw_yy, 0, -1) -
               tf.matrix_band_part(Kw_yy, 0, 0))
num_combos = data_num * (data_num - 1) / 2
mmd_w = (tf.reduce_sum(Kw_xx_upper) / num_combos + 
         tf.reduce_sum(Kw_yy_upper) / num_combos -
         2 * tf.reduce_sum(Kw_xy) / (data_num * data_num))
# Define kernel with no weights.
K_xx = K_orig[:data_num, :data_num]
K_yy = K_orig[data_num:, data_num:]
K_xy = K_orig[:data_num, data_num:]
K_xx_upper = tf.matrix_band_part(K_xx, 0, -1) - tf.matrix_band_part(K_xx, 0, 0)
K_yy_upper = tf.matrix_band_part(K_yy, 0, -1) - tf.matrix_band_part(K_yy, 0, 0)
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
if gradient_clipping == 'none':
    g_optim = opt.minimize(mmd, var_list=g_vars)
    g_optim_wn = opt.minimize(mmd_wn, var_list=g_vars)
    g_optim_w = opt.minimize(mmd_w, var_list=g_vars)
elif gradient_clipping == 'opt1':
    gradients, variables = zip(*opt.compute_gradients(mmd))
    gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
    g_optim = opt.apply_gradients(zip(gradients, variables))

    gradients_wn, variables_wn = zip(*opt.compute_gradients(mmd_wn))
    gradients_wn, _ = tf.clip_by_global_norm(gradients_wn, 1.0)
    g_optim_wn = opt.apply_gradients(zip(gradients_wn, variables_wn))

    gradients_w, variables_w = zip(*opt.compute_gradients(mmd_w))
    gradients_w, _ = tf.clip_by_global_norm(gradients_w, 1.0)
    g_optim_w = opt.apply_gradients(zip(gradients_w, variables_w))
elif gradient_clipping == 'opt2':
    gvs = opt.compute_gradients(mmd)
    capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
    g_optim = optimizer.apply_gradients(capped_gvs)

    gvs_wn = opt.compute_gradients(mmd_wn)
    capped_gvs_wn = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs_wn]
    g_optim_wn = optimizer.apply_gradients(capped_gvs_wn)

    gvs_w = opt.compute_gradients(mmd_w)
    capped_gvs_w = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs_w]
    g_optim_w = optimizer.apply_gradients(capped_gvs_w)

# Initialize session, graph, and checkpoint dir.
init_op = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init_op)

saver, checkpoints_dir, graphs_dir, logs_dir = prepare_dirs_and_logging(expt)
load_checkpoints(sess, saver, checkpoints_dir)
start_time = time()

# Train.
print '\nCONFIG'
print args
for it in range(total_num_runs):
    if it <= 10000:
        sess.run(g_optim,
                 feed_dict={
                     z: get_random_z(data_num, z_dim),
                     #x: np.random.choice(data, (data_num, 1))
                     x: np.reshape(data, [-1, 1])})
        if it == 10000:
            os.system('python eval_samples_thin.py --expt="{}"'.format(expt))
    else:
        sess.run(g_optim_w,
                 feed_dict={
                     z: get_random_z(data_num, z_dim),
                     #x: np.random.choice(data, (data_num, 1))
                     x: np.reshape(data, [-1, 1])})


    if 0:
        p1vu, p1vus, p1v, p1i = sess.run([p1_vert_unnormed, p1_vert_unnormed_sum, p1_vert, p1_inverse],
                 feed_dict={
                     z: get_random_z(data_num, z_dim),
                     #x: np.random.choice(data, (data_num, 1))
                     x: np.reshape(data, [-1, 1])})
        pdb.set_trace()

    # Occasionally save and plot.
    if it % save_iter == 0:
        z_sample = get_random_z(data_num, z_dim)
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
        saver.save(sess, os.path.join(checkpoints_dir, 'test'), global_step=it)

        # Print helpful summary.
        print '\niter:{} mmd_xg_out = {:.5f}'.format(it, mmd_xg_out)
        print ' min:{} max= {}'.format(min(g_out), max(g_out))
        print ' [*] Saved sample logs to {}'.format(logs_dir)
        print ' [*] Saved checkpoint {} to {}'.format(it, checkpoints_dir)

        # Print timing diagnostics.
        if it > 0:
            elapsed_time = time() - start_time
            time_per_iter = elapsed_time / it
            total_est = elapsed_time / it * total_num_runs
            m, s = divmod(total_est, 60)
            h, m = divmod(m, 60)
            total_est_str = '{:.0f}:{:02.0f}:{:02.0f}'.format(h, m, s)
            print ('\nTime (s). Elapsed: {:.2f}, Avg/iter: {:.4f},'
                   ' Total est.: {}').format(elapsed_time, time_per_iter,
                                             total_est_str)

os.system('python eval_samples_thin.py --expt="{}"'.format(expt))
