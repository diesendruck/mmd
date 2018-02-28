import argparse
from time import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import pdb
import sys
import tensorflow as tf
layers = tf.layers
import pandas as pd
import seaborn as sb
from scipy.stats import scoreatpercentile


# Config.
parser = argparse.ArgumentParser()
parser.add_argument('--data_num', type=int, default=1000)
parser.add_argument('--batch_size', type=int, default=500)
parser.add_argument('--gen_num', type=int, default=200)
parser.add_argument('--z_dim', type=int, default=16)
parser.add_argument('--width', type=int, default=256,
    help='width of generator layers')
parser.add_argument('--depth', type=int, default=3, 
    help='num of generator layers')
parser.add_argument('--lambda_mmd', type=float, default=1e-4)
parser.add_argument('--log_step', type=int, default=2000)
parser.add_argument('--max_step', type=int, default=500000)
parser.add_argument('--learning_rate', type=float, default=1e-4)
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
lambda_mmd = args.lambda_mmd
log_step = args.log_step
max_step = args.max_step
learning_rate = args.learning_rate
optimizer = args.optimizer
data_file = args.data_file
activation = tf.nn.elu


def fivenum(v):
    """Returns Tukey's five number summary (minimum, lower-hinge, median, upper-hinge, maximum) for the input vector, a list or array of numbers based on 1.5 times the interquartile distance"""
    try:
        np.sum(v)
    except TypeError:
        print('Error: you must provide a list or array of only numbers')
    q1 = scoreatpercentile(v,25)
    q3 = scoreatpercentile(v,75)
    iqd = q3-q1
    md = np.median(v)
    whisker = 1.5*iqd
    #return np.min(v), md-whisker, md, md+whisker, np.max(v)
    return np.min(v), q1, md, q3, np.max(v)


def build_toy_data():
    num_clusters = 3
    d = 5
    n1 = data_num / num_clusters
    n2 = data_num / num_clusters
    n3 = data_num / num_clusters
    n1 = data_num / num_clusters
    cluster1 = np.random.multivariate_normal(
        [-2., 5., 10., 1., -15.], np.eye(d), n1)
    cluster2 = np.random.multivariate_normal(
        [6., 6., 6., 6., 6.], np.eye(d), n2)
    cluster3 = np.random.multivariate_normal(
        [0., -100., 2., 30, 1.5], np.eye(d), n3)
    cluster4 = np.random.multivariate_normal(
        [10., -10., 6., 0., 15], np.eye(d), n3)
    data = np.concatenate((cluster1, cluster2, cluster3, cluster4))
    return data
    #n1 = data_num / 2
    #n2 = data_num - n1
    #cluster1 = np.random.multivariate_normal(
    #    [-2., 5.], [[1., .9], [.9, 1.]], n1)
    #cluster2 = np.random.multivariate_normal(
    #    [6., 6.], [[1., 0.], [0., 1.]], n2)
    #data = np.concatenate((cluster1, cluster2))


def plot_summary_raw_data(raw_data):
    num_cols = raw_data.shape[1]
    sq_dim = int(np.ceil(np.sqrt(num_cols)))
    fig, axs = plt.subplots(sq_dim, sq_dim, figsize=(10,10))
    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    axs = axs.ravel()
    for i in range(num_cols):
        axs[i].hist(raw_data[:, i])
    plt.savefig('plot_raw_data_marginals.png')
    plt.close('all')


# load_data().
if data_file:
    raw_data = np.loadtxt(open(data_file, 'rb'), delimiter=',')
    num_cols = raw_data.shape[1]
    plot_summary_raw_data(raw_data)
    norm_data = 1

    print('\nRaw data:')
    for i in range(num_cols):
        print('{: >17},{: >17},{: >17},{: >17},{: >17}'.format(
            *fivenum(raw_data[:,i])))
    if norm_data:
        print('\nNormed data:')
        # Don't normalize binary vars.
        mean_mask = [1] * 17 + [0, 0]
        std_mask = [1] * 19
        mean_vec = raw_data.mean(0) * mean_mask
        std_vec = raw_data.std(0) * std_mask
        # Regularly normalize the rest.
        #data = (raw_data - raw_data.mean(0)) / raw_data.std(0)
        data = (raw_data - mean_vec) / std_vec 
        for i in range(data.shape[1]):
            print('{: >17},{: >17},{: >17},{: >17},{: >17}'.format(
                *fivenum(data[:,i])))

    # Optionally exclude some columns.
    excluded_cols = []
    cols_to_use = [c for c in range(num_cols) if c not in excluded_cols]
    do_subset_data = 1
    if do_subset_data:
        data = data[:, cols_to_use]

    data_num = len(data)
    out_dim = data.shape[1]
else:
    data = build_toy_data()
    data = (data - data.mean(0))/data.std(0)
    cols_to_use = np.arange(data.shape[1])
    out_dim = data.shape[1]

# Set save tag, as a function of config parameters.
save_tag = 'dn{}_bs{}_gen{}_zd{}_w{}_d{}_lr{}_op_{}'.format(
    data_num, batch_size, gen_num, z_dim, width, depth, learning_rate,
    optimizer)

# Set up log dir.
log_dir = 'logs'
checkpoint_dir = 'logs/checkpoints'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)


def get_random_z(gen_num, z_dim):
    """Generates 2d array of noise input data."""
    #return np.random.uniform(size=[gen_num, z_dim],
    #                         low=-1.0, high=1.0)
    return np.random.normal(size=[gen_num, z_dim])


def dense(x, width, activation, batch_residual=False):
    if not batch_residual:
        return layers.dense(x, width, activation=activation)
    else:
        x_ = layers.dense(x, width, activation=activation, use_bias=False)
        return layers.batch_normalization(x_) + x


def autoencoder(x, width=3, depth=3, activation=tf.nn.elu, z_dim=3,
        reuse=False):
    """Decodes. Generates output, given noise input."""
    out_dim = x.shape[1]
    with tf.variable_scope('encoder', reuse=reuse) as vs_enc:
        x = layers.dense(x, width, activation=activation)

        for idx in range(depth - 1):
            x = dense(x, width, activation=activation, batch_residual=True)

        h = layers.dense(x, z_dim, activation=None)

    with tf.variable_scope('decoder', reuse=reuse) as vs_dec:
        x = layers.dense(h, width, activation=activation)

        for idx in range(depth - 1):
            x = dense(x, width, activation=activation, batch_residual=True)

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
            x = dense(x, width, activation=activation, batch_residual=True)

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


def compute_mmd(enc_x, enc_g, use_tf=True):
    """Computes mmd between two inputs of size [batch_size, ...]."""
    #sigma_list = [1, 4, 16, 64, 256]
    sigma_list = [1.0, 4.0, 16.0, 64.0, 128.0]
    sigma_list = [0.1, 0.5, 1.0]

    if use_tf:
        v = tf.concat([enc_x, enc_g], 0)
        VVT = tf.matmul(v, tf.transpose(v))
        sqs = tf.reshape(tf.diag_part(VVT), [-1, 1])
        sqs_tiled_horiz = tf.tile(sqs, tf.transpose(sqs).get_shape())
        exp_object = sqs_tiled_horiz - 2 * VVT + tf.transpose(sqs_tiled_horiz)
        K = 0.0
        for sigma in sigma_list:
            gamma = 1.0 / (2.0 * sigma**2)
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
        return mmd, exp_object
    else:
        if len(enc_x.shape) == 1:
            enc_x = np.reshape(enc_x, [-1, 1])
            enc_g = np.reshape(enc_g, [-1, 1])
        v = np.concatenate((enc_x, enc_g), 0)
        VVT = np.matmul(v, np.transpose(v))
        sqs = np.reshape(np.diag(VVT), [-1, 1])
        sqs_tiled_horiz = np.tile(sqs, np.transpose(sqs).shape)
        exp_object = sqs_tiled_horiz - 2 * VVT + np.transpose(sqs_tiled_horiz)
        K = 0.0
        for sigma in sigma_list:
            gamma = 1.0 / (2.0 * sigma**2)
            K += np.exp(-gamma * exp_object)
        K_xx = K[:batch_size, :batch_size]
        K_yy = K[batch_size:, batch_size:]
        K_xy = K[:batch_size, batch_size:]
        K_xx_upper = np.triu(K_xx)
        K_yy_upper = np.triu(K_yy)
        num_combos_x = batch_size * (batch_size - 1) / 2
        num_combos_y = gen_num * (gen_num - 1) / 2
        mmd = (np.sum(K_xx_upper) / num_combos_x +
               np.sum(K_yy_upper) / num_combos_y -
               2 * np.sum(K_xy) / (batch_size * gen_num))
        return mmd, exp_object


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

# SET UP MMD LOSS.
mmd, exp_object = compute_mmd(enc_x, enc_g)

d_loss = ae_loss - lambda_mmd * mmd
g_loss = mmd

if optimizer == 'adagrad':
    ae_opt = tf.train.AdagradOptimizer(learning_rate)
    d_opt = tf.train.AdagradOptimizer(learning_rate)
    g_opt = tf.train.AdagradOptimizer(learning_rate)
elif optimizer == 'adam':
    ae_opt = tf.train.AdamOptimizer(learning_rate)
    d_opt = tf.train.AdamOptimizer(learning_rate)
    g_opt = tf.train.AdamOptimizer(learning_rate)
elif optimizer == 'rmsprop':
    ae_opt = tf.train.RMSPropOptimizer(learning_rate)
    d_opt = tf.train.RMSPropOptimizer(learning_rate)
    g_opt = tf.train.RMSPropOptimizer(learning_rate)
else:
    ae_opt = tf.train.GradientDescentOptimizer(learning_rate)
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
ae_optim = ae_opt.minimize(ae_loss, var_list=d_vars_)
g_optim = g_opt.minimize(g_loss, var_list=g_vars)


summary_op = tf.summary.merge([
    tf.summary.scalar("loss/d_loss", d_loss),
    tf.summary.scalar("loss/ae_loss", ae_loss),
    tf.summary.scalar("loss/mmd", mmd),
])

# train().
init_op = tf.global_variables_initializer()
saver = tf.train.Saver()
summary_writer = tf.summary.FileWriter(checkpoint_dir)
step = tf.Variable(0, name='step', trainable=False)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
sess_config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
sv = tf.train.Supervisor(logdir=checkpoint_dir,
                        is_chief=True,
                        saver=saver,
                        summary_op=None,
                        summary_writer=summary_writer,
                        save_model_secs=300,
                        global_step=step,
                        ready_for_local_init_op=None)
sess = tf.Session(config=sess_config)
#global_step = tf.train.get_or_create_global_step()
#sess = tf.train.MonitoredTrainingSession(
#    config=sess_config,
#    checkpoint_dir=checkpoint_dir,
#    hooks=[tf.train.StepCounterHook])

sess.run(init_op)
print save_tag
g_out_file = os.path.join(log_dir, 'g_out.txt')
if os.path.isfile(g_out_file):
    os.remove(g_out_file)
start_time = time()
pretrain_steps = 0
for step in range(max_step):
    random_batch_data = np.array(
        [data[d] for d in np.random.choice(len(data), batch_size)])
    random_batch_z = get_random_z(gen_num, z_dim)
    if step < pretrain_steps:
        sess.run([ae_optim],
                 feed_dict={
                     z: random_batch_z,
                     x: random_batch_data})
    else:
        sess.run([d_optim, g_optim],
                 feed_dict={
                     z: random_batch_z,
                     x: random_batch_data})

    # Occasionally log/plot results.
    if step % log_step == 0:
        saver.save(sess, os.path.join(checkpoint_dir, 'test'),
            global_step=step)
        summary_ = sess.run(summary_op,
            feed_dict={
                z: random_batch_z,
                x: random_batch_data})
        summary_writer.add_summary(summary_, step)
        summary_writer.flush()

        #enc_x_, enc_g_, exp_object_, mmd_ = sess.run(
        #    [enc_x, enc_g, exp_object, mmd],
        #        feed_dict={
        #            z: random_batch_z,
        #            x: random_batch_data})
        #print(np.exp(-0.5 * exp_object_))
        #pdb.set_trace()

        # Print some loss values.
        d_loss_, ae_loss_, mmd_, g_ = sess.run(
            [d_loss, ae_loss, mmd, g], feed_dict={
                z: random_batch_z,
                x: random_batch_data})
        print('Iter:{}, d_loss = {:.4f}, ae_loss = {:.4f}, '
            'L * mmd = {:.4f}, mmd = {:.4f}'.format(step, d_loss_, ae_loss_,
                lambda_mmd * mmd_, mmd_))

        # Print average of marginal MMD(data_i, gen_i).
        '''
        if data_file:
            #num_marginals = len(cols_to_use)
            marginal_mmds = np.zeros((num_cols, num_cols))
            for ind_i, i in enumerate(cols_to_use):
                for ind_j, j in enumerate(cols_to_use):
                    if j >= i:
                        mmd_ij_data = compute_mmd(
                            random_batch_data[:, ind_i], 
                            random_batch_data[:, ind_j], use_tf=False) 
                        mmd_ij_gen = compute_mmd(
                            g_[:, ind_i], g_[:, ind_j], use_tf=False) 
                        marginal_mmds[i][j] = mmd_ij_gen - mmd_ij_data

                        #marginal_mmds[i][j] = compute_mmd(
                        #    random_batch_data[:, i], g_[:, j], use_tf=False)
            print(np.round(marginal_mmds, 2))
            print
        '''

        # Save generated data to NumPy file and to output collection.
        g_out = (g_ * std_vec[cols_to_use] + mean_vec[cols_to_use])
        # Round values in binary cols.
        for i, row in enumerate(g_out):
            if row[17] < 0.5:
                row[17] = 0.0
            else:
                row[17] = 1.0
            if row[18] < 0.5:
                row[18] = 0.0
            else:
                row[18] = 1.0
            g_out[i] = row
        np.save(os.path.join(log_dir, 'g_out.npy'), g_out)
        np.savetxt(os.path.join(log_dir, 'g_out.csv'), g_out, delimiter=',')
        with open(g_out_file, 'a') as f:
            f.write(str(g_out) + '\n')

        # Print time performance.
        if step % log_step == 0:
            elapsed_time = time() - start_time
            time_per_iter = elapsed_time / step
            total_est = elapsed_time / step * max_step
            m, s = divmod(total_est, 60)
            h, m = divmod(m, 60)
            total_est_str = '{:.0f}:{:02.0f}:{:02.0f}'.format(h, m, s)
            print(('  time (s): {:.2f}, time/iter: {:.4f},'
                   ' Total est.: {}').format(
                       elapsed_time, time_per_iter, total_est_str))

        # PLOTTING RESULTS.
        plot = 1
        if plot:
            # Make scatter plots.
            '''
            if out_dim > 2:
                MAX_TO_PLOT = 5
                indices_to_plot = np.arange(min(MAX_TO_PLOT, out_dim))
                indices_to_plot = [11, 12, 13, 14, 15]
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
            pairplot_data = sb.pairplot(
                pd.DataFrame(random_batch_data[:, indices_to_plot]))
            plt.title('Marginals: {}'.format(indices_to_plot))
            pairplot_data.savefig('pairplot_data.png')
            pairplot_simulation = sb.pairplot(
                pd.DataFrame(g_out[:, indices_to_plot]))
            pairplot_simulation.savefig('pairplot_simulation.png')
            plt.close('all')
            '''
            num_cols = raw_data.shape[1]
            sq_dim = int(np.ceil(np.sqrt(num_cols)))
            fig, axs = plt.subplots(4, 5, figsize=(20, 16))
            fig.subplots_adjust(hspace=0.5, wspace=0.5)
            fig.suptitle('Marginals, it{}'.format(step))
            axs = axs.ravel()
            for ind_i, i in enumerate(cols_to_use):
                mmd_i_data_gen, _ = compute_mmd(
                    random_batch_data[:, ind_i], g_[:, ind_i], use_tf=False)
                axs[i].hist(raw_data[:, ind_i], normed=True, alpha=0.3, label='d')
                axs[i].hist(g_out[:, ind_i], normed=True, alpha=0.3, label='g')
                axs[i].set_xlabel('mmd = {:.3f}'.format(mmd_i_data_gen))
                axs[i].legend()
            #fig.tight_layout()
            plt.savefig('plot_marginals.png')
            plt.close('all')
