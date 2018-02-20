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
#import seaborn as sb


# Config.
parser = argparse.ArgumentParser()
parser.add_argument('--data_num', type=int, default=1000)
parser.add_argument('--batch_size', type=int, default=200)
parser.add_argument('--gen_num', type=int, default=200)
parser.add_argument('--z_dim', type=int, default=3)
parser.add_argument('--width', type=int, default=10,
                    help='width of generator layers')
parser.add_argument('--depth', type=int, default=20,
                    help='num of generator layers')
parser.add_argument('--log_step', type=int, default=1000)
parser.add_argument('--max_step', type=int, default=10000)
parser.add_argument('--learning_rate', type=float, default=1e-3)
parser.add_argument('--optimizer', type=str, default='rmsprop',
                    choices=['adagrad', 'adam', 'gradientdescent', 'rmsprop'])
parser.add_argument('--data_file', type=str, default=None)

args = parser.parse_args()
data_num = args.data_num
batch_size = args.batch_size
gen_num = args.gen_num
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
save_tag = 'dn{}_bs{}_gen{}_zd{}_w{}_d{}_lr{}_op_{}'.format(
    data_num, batch_size, gen_num, z_dim, width, depth, learning_rate,
    optimizer)

# Set up log dir.
log_dir = 'logs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)


def get_random_z(gen_num, z_dim):
    """Generates 2d array of noise input data."""
    return np.random.uniform(size=[gen_num, z_dim],
                             low=-1.0, high=1.0)


def autoencoder(x, width=3, depth=3, activation=tf.nn.elu, z_dim=3,
        reuse=False):
    """Decodes. Generates output, given noise input."""
    out_dim = x.shape[1]
    with tf.variable_scope('encoder', reuse=reuse) as vs_enc:
        x = layers.dense(x, width, activation=activation)

        for idx in range(depth - 1):
            x = layers.dense(x, width, activation=activation)

        h = layers.dense(x, z_dim, activation=None)

    with tf.variable_scope('decoder', reuse=reuse) as vs_dec:
        x = layers.dense(h, width, activation=activation)

        for idx in range(depth - 1):
            x = layers.dense(x, width, activation=activation)

        ae = layers.dense(x, out_dim, activation=None)

    vars_enc = tf.contrib.framework.get_variables(vs_enc)
    vars_dec = tf.contrib.framework.get_variables(vs_dec)
    return h, ae, vars_enc, vars_dec


def generator(z, width=3, depth=3, activation=tf.nn.elu, out_dim=2,
        reuse=False):
    """Decodes. Generates output, given noise input."""
    with tf.variable_scope('generator', reuse=reuse) as vs_g:
        x = layers.dense(z, width, activation=activation)

        for idx in range(depth - 1):
            x_ = layers.dense(x, width, activation=activation)
            x = layers.batch_normalization(x_) + x  # {batchnorm, shortcut}

        out = layers.dense(x, out_dim, activation=None)
    vars_g = tf.contrib.framework.get_variables(vs_g)
    return out, vars_g


def discriminator(x, width=3, depth=3, activation=tf.nn.elu, out_dim=1,
              reuse=False):
    """Generates output, given noise input."""
    with tf.variable_scope('discriminator', reuse=reuse) as d_vs:
        x = layers.dense(x, width, activation=activation)

        for idx in range(depth - 1):
            x = layers.dense(x, width, activation=activation)
            x = layers.batch_normalization(x)
            x = layers.dropout(x, rate=0.5)

        y_logit = layers.dense(x, out_dim, activation=None)
        y_prob = tf.nn.sigmoid(y_logit)
    d_vars = tf.contrib.framework.get_variables(d_vs)
    return y_logit, d_vars 


def compute_mmd(enc_x, enc_g):
    """Computes mmd between two inputs of size [batch_size, ...]."""
    v = tf.concat([enc_x, enc_g], 0)
    VVT = tf.matmul(v, tf.transpose(v))
    sqs = tf.reshape(tf.diag_part(VVT), [-1, 1])
    sqs_tiled_horiz = tf.tile(sqs, tf.transpose(sqs).get_shape())
    exp_object = sqs_tiled_horiz - 2 * VVT + tf.transpose(sqs_tiled_horiz)
    K = 0.0
    sigma_list = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 4.0]
    for sigma in sigma_list:
        gamma = 1.0 / (2 * sigma**2)
        K += tf.exp(-gamma * exp_object)
    K_xx = K[:batch_size, :batch_size]
    K_yy = K[batch_size:, batch_size:]
    K_xy = K[:batch_size, batch_size:]
    K_xx_upper = tf.matrix_band_part(K_xx, 0, -1)
    K_yy_upper = tf.matrix_band_part(K_yy, 0, -1)
    num_combos_x = batch_size * (batch_size - 1) / 2
    num_combos_y = gen_num * (gen_num - 1) / 2
    mmd = (tf.reduce_sum(K_xx_upper) / num_combos_x +
           tf.reduce_sum(K_yy_upper) / num_combos_y -
           2 * tf.reduce_sum(K_xy) / (batch_size * gen_num))
    return mmd


# Build model.
x = tf.placeholder(tf.float64, [batch_size, out_dim], name='x')
z = tf.placeholder(tf.float64, [gen_num, z_dim], name='z')

g, g_vars = generator(
    z, width=width, depth=depth, activation=activation, out_dim=out_dim)
h_out, ae_out, enc_vars, dec_vars = autoencoder(tf.concat([x, g], 0),
    width=width, depth=depth, activation=activation, z_dim=z_dim, reuse=False)
enc_x, enc_g = tf.split(h_out, [batch_size, gen_num])
ae_x, ae_g = tf.split(ae_out, [batch_size, gen_num])

ae_loss = tf.reduce_mean(tf.square(ae_x - x))

# DISCRIMINATOR LOSSES.
#d_x, d_vars = discriminator(x, width=width, depth=depth, activation=activation, reuse=False)
#d_g, _ = discriminator(g, width=width, depth=depth, activation=activation, reuse=True)
#def sigmoid_cross_entropy_with_logits(x_, y_):
#  try:
#    return tf.nn.sigmoid_cross_entropy_with_logits(logits=x_, labels=y_)
#  except:
#    return tf.nn.sigmoid_cross_entropy_with_logits(logits=x_, targets=y_)
#d_loss_real = tf.reduce_mean(sigmoid_cross_entropy_with_logits(d_x, tf.ones_like(d_x)))
#d_loss_fake = tf.reduce_mean(sigmoid_cross_entropy_with_logits(d_g, tf.zeros_like(d_g)))
#d_loss = d_loss_real + d_loss_fake
#g_loss = -1.0 * d_loss_fake
#g_loss = tf.reduce_mean(sigmoid_cross_entropy_with_logits(d_g, tf.ones_like(d_g)))

# SET UP MMD LOSS.
mmd = compute_mmd(enc_x, enc_g)

d_loss = ae_loss - 2.0 * mmd
g_loss = mmd

if optimizer == 'adagrad':
    d_opt = tf.train.AdagradOptimizer(learning_rate)
    g_opt = tf.train.AdagradOptimizer(learning_rate)
elif optimizer == 'adam':
    d_opt = tf.train.AdamOptimizer(learning_rate)
    g_opt = tf.train.AdamOptimizer(learning_rate)
elif optimizer == 'rmsprop':
    d_opt = tf.train.RMSPropOptimizer(learning_rate)
    g_opt = tf.train.RMSPropOptimizer(learning_rate)
else:
    d_opt = tf.train.GradientDescentOptimizer(learning_rate)
    g_opt = tf.train.GradientDescentOptimizer(learning_rate)

# Define optim nodes.
# Clip encoder gradients.
#d_optim = d_opt.minimize(d_loss, var_list=enc_vars + dec_vars)
enc_grads_, enc_vars_ = zip(*d_opt.compute_gradients(d_loss, var_list=enc_vars))
dec_grads_, dec_vars_ = zip(*d_opt.compute_gradients(d_loss, var_list=dec_vars))
enc_grads_clipped_ = tuple(
    [tf.clip_by_value(grad, -0.01, 0.01) for grad in enc_grads_])
d_grads_ = enc_grads_clipped_ + dec_grads_
d_vars_ = enc_vars_ + dec_vars_
d_optim = d_opt.apply_gradients(zip(d_grads_, d_vars_))
g_optim = g_opt.minimize(g_loss, var_list=g_vars)

# Train.
init_op = tf.global_variables_initializer()
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
sess_config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
sess = tf.Session(config=sess_config)
sess.run(init_op)
print save_tag
g_out_file = os.path.join(log_dir, 'g_out.txt')
if os.path.isfile(g_out_file):
    os.remove(g_out_file)
start_time = time()
for step in range(max_step):
    random_batch_data = np.array(
        [data[d] for d in np.random.choice(len(data), batch_size)])
    random_batch_z = get_random_z(gen_num, z_dim)
    sess.run([d_optim, g_optim],
             feed_dict={
                 z: random_batch_z,
                 x: random_batch_data})

    # Occasionally log/plot results.
    if step % log_step == 0:
        # Print some loss values.
        d_loss_, ae_loss_, mmd_, g_out = sess.run(
            [d_loss, ae_loss, mmd, g], feed_dict={
                z: random_batch_z,
                x: random_batch_data})
        print('Iter:{}, d_loss = {:.4f}, ae_loss = {:.4f}, mmd = {:.4f}'.format(
            step, d_loss_, ae_loss_, mmd_))

        # Make scatter plots.
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

        # Make pair plots.
        '''
        pairplot_data = sb.pairplot(
            pd.DataFrame(random_batch_data[:, indices_to_plot]))
        pairplot_data.savefig('pairplot_data.png')
        pairplot_simulation = sb.pairplot(
            pd.DataFrame(g_out[:, indices_to_plot]))
        pairplot_simulation.savefig('pairplot_simulation.png')
        plt.close('all')
        '''
        
        # Save generated data to file.
        np.save(os.path.join(log_dir, 'g_out.npy'), g_out)
        with open(g_out_file, 'a') as f:
            f.write(str(g_out) + '\n')

        # Print time performance.
        if step % 10 * log_step > 0:
            elapsed_time = time() - start_time
            time_per_iter = elapsed_time / step
            total_est = elapsed_time / step * max_step
            m, s = divmod(total_est, 60)
            h, m = divmod(m, 60)
            total_est_str = '{:.0f}:{:02.0f}:{:02.0f}'.format(h, m, s)
            print ('  time (s): {:.2f}, time/iter: {:.4f},'
                    ' Total est.: {:.4f}').format(step, elapsed_time, time_per_iter,
                                             total_est_str)
