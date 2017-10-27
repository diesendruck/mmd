import argparse
from time import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np
import os
import pdb
import sys
import tensorflow as tf
layers = tf.layers
from scipy.stats import norm


# Config.
parser = argparse.ArgumentParser()
parser.add_argument('--data_num', type=int, default=1000)
parser.add_argument('--gen_num', type=int, default=300)
parser.add_argument('--z_dim', type=int, default=1)
parser.add_argument('--width', type=int, default=3,
                    help='width of generator layers')
parser.add_argument('--depth', type=int, default=6,
                    help='num of generator layers')
parser.add_argument('--sigma', type=float, default=0.5)
parser.add_argument('--thin_a', type=float, default=0.5)
parser.add_argument('--thin_b', type=float, default=0.5)
parser.add_argument('--learning_rate', type=float, default=1e-3)
parser.add_argument('--optimizer', type=str, default='adam',
                    choices=['adagrad', 'adam', 'gradientdescent',
                             'rmsprop'])
parser.add_argument('--max_iter', type=int, default=50101)
parser.add_argument('--save_iter', type=int, default=500)
parser.add_argument('--expt', type=str, default='test')
parser.add_argument('--weighting', type=str, default='logistic',
                    choices=['logistic', 'kernel'])
parser.add_argument('--graph_data', type=int, default=0, choices=[0, 1])
parser.add_argument('--testing', type=int, default=0, choices=[0, 1])

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
max_iter = args.max_iter
save_iter = args.save_iter
expt = args.expt
weighting = args.weighting
graph_data = args.graph_data
testing = args.testing
out_dim = 1
activation = tf.nn.elu
save_tag = 'expt{}_dn{}_gn{}_zd{}_w{}_d{}_lr{}_op_{}_sig{}_thin{}_wt{}'.format(
    expt, data_num, gen_num, z_dim, width, depth, learning_rate, optimizer,
    sigma, str(thin_a)+str(thin_b), weighting)


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
    '''Thins data, accepting ~90% of Cluster 1, and ~10% Cluster 2.
    Args:
      data_unthinned: List of scalar values.
    Returns:
      thinned: Numpy array of values, thinned according to logistic function.
    ''' 

    thinned = [candidate for candidate in data_unthinned if 
               np.random.binomial(1, prob_of_keeping(candidate, is_tf=False,
                                                     weighting=weighting))]
    return thinned 


def prob_of_keeping(x, is_tf=False, weighting='logistic'):
    '''Logistic mapping to preferentially thin samples.
    
    Maps [-Inf, Inf] to [0, 1], centered at 2.

    Args:
      x: Scalar, point from original distribution.
      is_tf: Boolean flag for whether computation is on Tensors or Numpy input.
      weighting: String, type of thinning function.
    Returns:
      p: Probability of being thinned.
    '''
    if weighting == 'logistic':
        if is_tf:
            p = thin_a / (1 + tf.exp(10 * (x - 1))) + thin_b
        else:
            p = thin_a / (1 + np.exp(10 * (x - 1))) + thin_b
    elif weighting == 'kernel':
        if is_tf:
            p = 1. - thin_a * tf.exp(-10. * tf.square(x - 2))
        else:
            p = 1. - thin_a * np.exp(-10. * np.square(x - 2))
    return p


def generate_data(n):
    '''Generates true data, and applies thinning function.
    
    Args:
      n: Number of data candidates to start with, to then be thinned.
    Returns:
      data_unthinned: Unthinned numpy array of points.
      data: Thinned numpy array of points.
    '''
    n_c2 = n/2
    n_c1 = n - n_c2
    data_unthinned = np.concatenate((np.random.normal(0, 0.5, n_c1),
                                     np.random.normal(2, 0.5, n_c2)))

    data = thin_data(data_unthinned)
    return data_unthinned, data


def get_random_z(gen_num, z_dim):
    '''Generates 2d array of noise input data.'''
    #return np.random.uniform(size=[gen_num, z_dim],
    #                         low=-1.0, high=1.0)
    #return np.random.gamma(5, size=[gen_num, z_dim])
    #return np.random.standard_t(2, size=[gen_num, z_dim])
    return np.random.normal(0, 1, size=[gen_num, z_dim])


# Set up generator.
def generator(z, width=3, depth=3, activation=tf.nn.elu, out_dim=1,
              reuse=False):
    '''Generates output, given noise input.'''
    with tf.variable_scope('generator', reuse=reuse) as gen_vars:
        x = layers.dense(z, width, activation=activation)

        for idx in range(depth - 1):
            x = layers.dense(x, width, activation=activation)

        out = layers.dense(x, out_dim, activation=None)
    return out


# Define data.
data_unthinned, data = generate_data(data_num)
data_thinned = data
data_num = len(data)
gen_num = data_num
if graph_data:
    xs = np.linspace(min(data_unthinned), max(data_unthinned), 100)
    ys1 = norm.pdf(xs, 0, 0.5)
    ys2 = norm.pdf(xs, 2, 0.5)
    y_thinned = 2. / 3. * ys1 + 1. / 3. * ys2
    y_unthinned = 0.5 * ys1 + 0.5 * ys2

    plt.plot(xs, y_unthinned, color='green', label='pdf', alpha=0.7)
    plt.hist(data_unthinned, 30, normed=True, color='green', label='x unthinned', alpha=0.3)
    plt.xlabel('x')
    plt.ylabel('Density')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('unthinned_data.png')
    plt.close()

    plt.plot(xs, y_thinned, color='blue', label='pdf', alpha=0.7)
    plt.hist(data, 30, normed=True, color='blue', label='x thinned', alpha=0.3)
    plt.xlabel('x')
    plt.ylabel('Density')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('thinned_data.png')
    plt.close()

    tx = 1 - prob_of_keeping(xs)
    plt.plot(xs, tx, color='black', alpha=0.7)
    plt.xlabel('x')
    plt.ylabel('T(x)')
    plt.ylim((-0.1, 1.1))
    plt.yticks(np.linspace(0, 1, 11))
    plt.axhline(0, color='black', alpha=0.2)
    plt.axvline(0, color='black', alpha=0.2)
    plt.tight_layout()
    plt.savefig('thinning_fn.png')
    plt.close()

    print('Plotted unthinned_data.png, thinned_data.png, thinning_fn.png')
    sys.exit('graph_data=1, graphed data')

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
num_combos_xx = data_num * (data_num - 1) / 2
num_combos_yy = gen_num * (gen_num - 1) / 2
v_tiled_horiz = tf.tile(v, [1, x_len + z_len])
thinning_fn = prob_of_keeping(v_tiled_horiz, is_tf=True, weighting=weighting)
p1_weights = 1. / thinning_fn 
p2_weights = tf.transpose(p1_weights) 
p1p2_weights_xx = p1_weights[:data_num, :data_num] * p2_weights[:data_num, :data_num]
p1p2_weights_xx_normed = p1p2_weights_xx / tf.reduce_sum(p1p2_weights_xx)
p1_weights_xy = p1_weights[:data_num, data_num:]
p1_weights_xy_normed = p1_weights_xy / tf.reduce_sum(p1_weights_xy)
Kw_xx = K[:data_num, :data_num] * p1p2_weights_xx_normed
Kw_xx_upper = (tf.matrix_band_part(Kw_xx, 0, -1) - 
               tf.matrix_band_part(Kw_xx, 0, 0))
Kw_xy = K[:data_num, data_num:] * p1_weights_xy_normed
mmd = (tf.reduce_sum(Kw_xx_upper) / num_combos_xx +
       tf.reduce_sum(K_yy_upper) / num_combos_yy -
       2 * tf.reduce_sum(Kw_xy))
#mmd = (tf.reduce_sum(K_yy_upper) / num_combos_yy -
#       2 * tf.reduce_sum(Kw_xy))

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
for i in range(max_iter):
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
            total_est = elapsed_time / i * max_iter
            m, s = divmod(total_est, 60)
            h, m = divmod(m, 60)
            total_est_str = '{:.0f}:{:02.0f}:{:02.0f}'.format(h, m, s)
            time_per_iter = elapsed_time / i
            print ('\n  Time (s). Elapsed: {}, Avg/iter: {:.4f},'
                   ' Total est.: {}').format(elapsed_time_str, time_per_iter,
                                             total_est_str)

    if testing == 1:
        if i == 6000 or i == 8000:
            make_samples = True
            pdb.set_trace()
            while make_samples == True:
                for j in range(10):
                    z_sample = get_random_z(gen_num, z_dim)
                    x_sample = np.random.choice(data_thinned, (data_num, 1))
                    mmd_out, g_out = sess.run(
                        [mmd, g], feed_dict={
                            z: z_sample,
                            x: x_sample})
                    print mmd_out
                print('got 10 for data_thinned, choose to set make_samples=False, or continue')
                pdb.set_trace()

                for k in range(10):
                    z_sample = get_random_z(gen_num, z_dim)
                    x_sample = np.random.choice(data_unthinned, (data_num, 1))
                    mmd_out, g_out = sess.run(
                        [mmd, g], feed_dict={
                            z: z_sample,
                            x: x_sample})
                    print mmd_out
                print('got 10 for data_unthinned, choose to set make_samples=False, or continue')
                pdb.set_trace()

os.system('python eval_samples_thin.py --expt="{}"'.format(expt))
