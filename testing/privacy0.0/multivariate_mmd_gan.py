import argparse
from time import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm
import os
import pdb
import tensorflow as tf
layers = tf.layers
from scipy.stats import multivariate_normal


# Config.
parser = argparse.ArgumentParser()
parser.add_argument('--model_type', type=str, default='mmd_ae',
                    choices=['mmd_ae', 'mmd_gan'])
parser.add_argument('--data_num', type=int, default=1000)
parser.add_argument('--percent_train', type=float, default=0.9)
parser.add_argument('--batch_size', type=int, default=200)
parser.add_argument('--gen_num', type=int, default=200)
parser.add_argument('--width', type=int, default=20,
                    help='width of generator layers')
parser.add_argument('--depth', type=int, default=10,
                    help='num of generator layers')
parser.add_argument('--z_dim', type=int, default=10)
parser.add_argument('--log_step', type=int, default=1000)
parser.add_argument('--max_step', type=int, default=200000)
parser.add_argument('--learning_rate', type=float, default=1e-3)
parser.add_argument('--optimizer', type=str, default='rmsprop',
                    choices=['adagrad', 'adam', 'gradientdescent', 'rmsprop'])
parser.add_argument('--data_file', type=str, default='gp_data.txt')
parser.add_argument('--tag', type=str, default='test')
parser.add_argument('--load_existing', default=False, action='store_true',
                    dest='load_existing')
args = parser.parse_args()
model_type = args.model_type
data_num = args.data_num
percent_train = args.percent_train
batch_size = args.batch_size
gen_num = args.gen_num
width = args.width
depth = args.depth
z_dim = args.z_dim
log_step = args.log_step
max_step = args.max_step
learning_rate = args.learning_rate
optimizer = args.optimizer
data_file = args.data_file
tag = args.tag
load_existing = args.load_existing

activation = tf.nn.elu
data_num = int(percent_train * data_num)  # Update based on % train.


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
        x = dense(z, width, activation=activation)

        for idx in range(depth - 1):
            x = dense(x, width, activation=activation, batch_residual=True)

        out = dense(x, out_dim, activation=None)
    vars_g = tf.contrib.framework.get_variables(vs_g)
    return out, vars_g


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

def evaluate_presence_risk(train, test, sim):
    """Assess privacy of simulations.
    
    Compute True Pos., True Neg., False Pos., and False Neg. rates of
    finding a neighbor in the simulations, for each of a subset of training
    data and a subset of test data.

    Args:
      train: Numpy array of all training data.
      test: Numpy array of all test data (smaller than train).
      sim: Numpy array of simulations.

    Return:
      sensitivity: Float of TP / (TP + FN).
      precision: Float of TP / (TP + FP).
    """
    assert len(test) < len(train), 'test should be smaller than train'
    num_samples = len(test)
    compromised_records = train[:num_samples]
    tp, tn, fp, fn = 0, 0, 0, 0
    ball_radius = 1e-2 

    # Count true positives and false negatives.
    for i in compromised_records:
        distances_from_i = norm(i - sim, axis=1)
        has_neighbor = np.any(distances_from_i < ball_radius)
        if has_neighbor:
            tp += 1
        else:
            fn += 1
    # Count false positives and true negatives.
    for i in test:
        distances_from_i = norm(i - sim, axis=1)
        has_neighbor = np.any(distances_from_i < ball_radius)
        if has_neighbor:
            fp += 1
        else:
            tn += 1

    sensitivity = float(tp) / (tp + fn)
    precision = float(tp + 1e-10) / (tp + fp + 1e-10)
    return sensitivity, precision, ball_radius


def load_checkpoint(saver, sess, checkpoint_dir):
    import re
    print(" [*] Reading checkpoints...")
    print("     {}".format(checkpoint_dir))
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
        #counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
        #counter = int(''.join([i for i in ckpt_name if i.isdigit()]))
        counter = int(ckpt_name.split('-')[-1])
        print(" [*] Success to read {}".format(ckpt_name))
        return True, counter
    else:
        print(" [*] Failed to find a checkpoint")
        return False, 0


def load_data(data_num, percent_train):
    # Load data.
    if data_file:
        data_raw = np.loadtxt(open(data_file, 'rb'), delimiter=' ')
        data_raw = np.random.permutation(data_raw)

        num_train = int(percent_train * data_raw.shape[0])
        data = data_raw[:num_train]
        data_test = data_raw[num_train:]

        data_num = data.shape[0]
        data_test_num = data_test.shape[0]
        out_dim = data.shape[1]
        return data, data_test, data_num, data_test_num, out_dim
    else:
        def sample_c1():
            sample = np.random.multivariate_normal(
                [-2., 5.], [[1., .6], [.6, 1.]], 1)
            return sample
        def sample_c2():
            sample = np.random.multivariate_normal(
                [6., 6.], [[1., 0.], [0., 1.]], 1)
            return sample
        data_raw = np.zeros((data_num, 2))
        for i in range(data_num):
            # Pick a Gaussian, then generate from that Gaussian.
            cluster_i = np.random.binomial(1, 0.0)  # NOTE: p=0.
            if cluster_i == 0:
                s = sample_c1()
                data_raw[i] = s
            else:
                s = sample_c2()
                data_raw[i] = s

        num_train = int(percent_train * data_raw.shape[0])
        data = data_raw[:num_train]
        data_test = data_raw[num_train:]

        data_num = data.shape[0]
        data_test_num = data_test.shape[0]
        out_dim = data.shape[1]
        return data, data_test, data_num, data_test_num, out_dim


def sample_loglik(mean, cov, data):
    d = cov.shape[1]
    residuals = data - mean

    def calc_loglik(residuals):
        ll = (-0.5 * (d * np.log(2 * np.pi) + np.log(np.linalg.det(cov))) +
              -0.5 * residuals.T.dot(np.linalg.inv(cov)).dot(residuals))
        return ll

    loglik = np.apply_along_axis(calc_loglik, 1, residuals)
    loglik_sum = float(loglik.sum())

    return loglik_sum


def prepare_dirs():
    log_dir = 'logs_{}'.format(tag)
    checkpoint_dir = os.path.join(log_dir, 'checkpoints')
    plot_dir = os.path.join(log_dir, 'plots')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    return log_dir, checkpoint_dir, plot_dir


def prepare_logging(log_dir, checkpoint_dir, sess):
    saver = tf.train.Saver()
    summary_writer = tf.summary.FileWriter(os.path.join(log_dir, 'summary'),
        sess.graph)
    step = tf.Variable(0, name='step', trainable=False)
    sv = tf.train.Supervisor(logdir=checkpoint_dir,
                            is_chief=True,
                            saver=saver,
                            summary_op=None,
                            summary_writer=summary_writer,
                            save_model_secs=300,
                            global_step=step,
                            ready_for_local_init_op=None)
    return saver, summary_writer


def add_nongraph_summary_items(summary_writer, step, dict_to_add):
    for k, v in dict_to_add.iteritems():
        summ = tf.Summary()
        summ.value.add(tag=k, simple_value=v)
        summary_writer.add_summary(summ, step)
    summary_writer.flush()


def avg_nearest_neighbor_distance(candidates, references):
    """Measures distance from candidate set to a reference set.

    For each element in candidate set, find distance to nearest neighbor in
    reference set. Return the average of these distances.

    Args:
      candidates: Numpy array of candidate points. (num_points x point_dim)
      references: Numpy array of reference points. (num_points x point_dim)

    Returns:
      avg_dist: Float, average over distances.
    """
    dists = []
    for i in xrange(candidates.shape[1]):
        c_i = tf.gather(candidates, [i]) 
        distances_from_i = tf.norm(c_i - references, axis=1)
        distances_negative = -1.0 * distances_from_i
        smallest_dist, _ = tf.nn.top_k(distances_negative)
        dists += [smallest_dist[0]]
    avg_dist = tf.reduce_mean(dists)
    return avg_dist


def build_model_mmd_ae(batch_size, data_num, data_test_num, gen_num, out_dim, z_dim):
    # Placeholders to precompute avg distance from data_test to data.
    x_precompute = tf.placeholder(tf.float64, [data_num, out_dim],
        name='x_precompute')
    x_test_precompute = tf.placeholder(tf.float64, [data_test_num, out_dim],
        name='x_test_precompute')
    avg_dist_x_test_to_x_precomputed = avg_nearest_neighbor_distance(
        x_test_precompute, x_precompute)

    # Regular training placeholders.
    x = tf.placeholder(tf.float64, [batch_size, out_dim], name='x')
    x_full = tf.placeholder(tf.float64, [data_num, out_dim], name='x_full')
    x_test = tf.placeholder(tf.float64, [batch_size, out_dim], name='x_test')
    avg_dist_x_test_to_x = tf.placeholder(tf.float64, shape=(),
        name='avg_dist_x_test_to_x')

    # Autoencoder.
    enc_x, ae_x, enc_vars, dec_vars = autoencoder(x,
        width=width, depth=depth, activation=activation, z_dim=z_dim,
        reuse=False)
    enc_x_full, ae_x_full, _, _ = autoencoder(x_full,
        width=width, depth=depth, activation=activation, z_dim=z_dim,
        reuse=True)

    ae_loss = tf.reduce_mean(tf.square(ae_x - x))
    g = ae_x
    g_full = ae_x_full

    # Build in differential privacy terms.
    # 1. Compute MMD between original input and random subset of input.
    # 2. Add this altered MMD to the original for all those MMDs.
    # 3. Include loss on closeness/min distance from ae(x) to x.
    mmd_subset = 0
    for i in xrange(batch_size):
        indices_to_keep = np.delete(np.arange(batch_size), i)
        x_i = tf.gather(x, indices_to_keep) 
        _, ae_x_i, _, _ = autoencoder(x_i,
            width=width, depth=depth, activation=activation, z_dim=z_dim,
            reuse=True)
        mmd_i = compute_mmd(x, ae_x_i)
        mmd_subset +=  mmd_i

    mmd_normal = compute_mmd(x, ae_x)
    mmd = mmd_subset

    # Simulations as close to data as heldouts are to data.
    # (Simulations aren't overfitting.)
    avg_dist_g_to_x = avg_nearest_neighbor_distance(g, x)
    loss1 = tf.abs(avg_dist_g_to_x - avg_dist_x_test_to_x)
    # Data as close to simulations as heldout to simulations.
    # (Simulations don't reveal "source", in being closer to data than to heldout.
    avg_dist_x_to_g = avg_nearest_neighbor_distance(x, g)
    avg_dist_x_test_to_g = avg_nearest_neighbor_distance(x_test, g)
    loss2 = tf.abs(avg_dist_x_to_g - avg_dist_x_test_to_g)

    d_loss = mmd + loss1 + loss2

    if optimizer == 'adagrad':
        d_opt = tf.train.AdagradOptimizer(learning_rate)
    elif optimizer == 'adam':
        d_opt = tf.train.AdamOptimizer(learning_rate)
    elif optimizer == 'rmsprop':
        d_opt = tf.train.RMSPropOptimizer(learning_rate)
    else:
        d_opt = tf.train.GradientDescentOptimizer(learning_rate)

    # Define optim nodes.
    # Clip encoder gradients.
    clip = 1
    if clip:
        enc_grads_, enc_vars_ = zip(*d_opt.compute_gradients(d_loss, var_list=enc_vars))
        dec_grads_, dec_vars_ = zip(*d_opt.compute_gradients(d_loss, var_list=dec_vars))
        enc_grads_clipped_ = tuple(
            [tf.clip_by_value(grad, -0.01, 0.01) for grad in enc_grads_])
        d_grads_ = enc_grads_clipped_ + dec_grads_
        d_vars_ = enc_vars_ + dec_vars_
        d_optim = d_opt.apply_gradients(zip(d_grads_, d_vars_))
    else:
        d_optim = d_opt.minimize(d_loss, var_list=enc_vars + dec_vars)

    # Define summary op for reporting.
    summary_op = tf.summary.merge([
	tf.summary.scalar("loss/ae_loss", ae_loss),
	tf.summary.scalar("loss/loss1", loss1),
	tf.summary.scalar("loss/loss2", loss2),
	tf.summary.scalar("loss/mmd", mmd),
	tf.summary.scalar("loss/d_loss", d_loss),
	tf.summary.scalar("misc/lr", learning_rate),
    ])

    return (x, x_full, x_test, avg_dist_x_test_to_x, x_precompute, x_test_precompute,
            avg_dist_x_test_to_x_precomputed, g, g_full, ae_loss, d_loss, mmd,
            d_optim, summary_op)


def build_model_mmd_gan(batch_size, gen_num, data_num, data_test_num, out_dim, z_dim):
    # Placeholders to precompute avg distance from data_test to data.
    x_precompute = tf.placeholder(tf.float64, [data_num, out_dim],
        name='x_precompute')
    x_test_precompute = tf.placeholder(tf.float64, [data_test_num, out_dim],
        name='x_test_precompute')
    avg_dist_x_test_to_x_precomputed = avg_nearest_neighbor_distance(
        x_test_precompute, x_precompute)

    # Regular training placeholders.
    x = tf.placeholder(tf.float64, [batch_size, out_dim], name='x')
    z = tf.placeholder(tf.float64, [gen_num, z_dim], name='z')
    z_full = tf.placeholder(tf.float64, [data_num, z_dim], name='z_full')
    x_test = tf.placeholder(tf.float64, [batch_size, out_dim], name='x_test')
    avg_dist_x_test_to_x = tf.placeholder(tf.float64, shape=(),
        name='avg_dist_x_test_to_x')


    g, g_vars = generator(
        z, width=width, depth=depth, activation=activation, out_dim=out_dim)
    g_full, _ = generator(
        z_full, width=width, depth=depth, activation=activation, out_dim=out_dim,
        reuse=True)
    h_out, ae_out, enc_vars, dec_vars = autoencoder(tf.concat([x, g], 0),
        width=width, depth=depth, activation=activation, z_dim=z_dim,
        reuse=False)
    enc_x, enc_g = tf.split(h_out, [batch_size, gen_num])
    ae_x, ae_g = tf.split(ae_out, [batch_size, gen_num])

    ae_loss = tf.reduce_mean(tf.square(ae_x - x))

    # SET UP MMD LOSS.
    mmd = compute_mmd(enc_x, enc_g)

    # Simulations as close to data as heldouts are to data.
    # (Simulations aren't overfitting.)
    avg_dist_g_to_x = avg_nearest_neighbor_distance(g, x)
    loss1 = tf.abs(avg_dist_g_to_x - avg_dist_x_test_to_x)
    # Data as close to simulations as heldout to simulations.
    # (Simulations don't reveal "source", in being closer to data than to heldout.
    avg_dist_x_to_g = avg_nearest_neighbor_distance(x, g)
    avg_dist_x_test_to_g = avg_nearest_neighbor_distance(x_test, g)
    loss2 = tf.abs(avg_dist_x_to_g - avg_dist_x_test_to_g)

    #d_loss = ae_loss - 2.0 * mmd
    d_loss = ae_loss - 2.0 * mmd - loss1 - loss2
    g_loss = mmd + loss1 + loss2

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

    # Define summary op for reporting.
    summary_op = tf.summary.merge([
	tf.summary.scalar("loss/ae_loss", ae_loss),
	tf.summary.scalar("loss/loss1", loss1),
	tf.summary.scalar("loss/loss2", loss2),
	tf.summary.scalar("loss/mmd", mmd),
	tf.summary.scalar("loss/d_loss", d_loss),
	tf.summary.scalar("misc/lr", learning_rate),
    ])

    return (x, z, z_full, x_test, avg_dist_x_test_to_x, x_precompute, x_test_precompute,
            avg_dist_x_test_to_x_precomputed, g, g_full, ae_loss, d_loss, mmd, d_optim,
            g_optim, summary_op)


def main():
    args = parser.parse_args()
    model_type = args.model_type
    data_num = args.data_num
    percent_train = args.percent_train
    batch_size = args.batch_size
    gen_num = args.gen_num
    width = args.width
    depth = args.depth
    z_dim = args.z_dim
    log_step = args.log_step
    max_step = args.max_step
    learning_rate = args.learning_rate
    optimizer = args.optimizer
    data_file = args.data_file
    tag = args.tag
    load_existing = args.load_existing
    activation = tf.nn.elu

    # Load data and prep dirs.
    data, data_test, data_num, data_test_num, out_dim = load_data(
        data_num, percent_train)
    log_dir, checkpoint_dir, plot_dir = prepare_dirs()
    save_tag = str(args)
    with open(os.path.join(log_dir, 'save_tag.txt'), 'w') as save_tag_file:
        save_tag_file.write(save_tag)
    print('Save tag: {}'.format(save_tag))

    g_out_file = os.path.join(log_dir, 'g_out.txt')
    if os.path.isfile(g_out_file):
        os.remove(g_out_file)

    # Compute sample mean and covariance, if using Gaussian data.
    if data_file == '':
        sample_mean = np.mean(data, axis=0)
        sample_cov = np.cov(data.T)

    # Build model.
    if model_type == 'mmd_ae':
        (x, x_full, x_test, avg_dist_x_test_to_x, x_precompute, x_test_precompute,
         avg_dist_x_test_to_x_precomputed, g, g_full, ae_loss, d_loss, mmd, d_optim,
         summary_op) = build_model_mmd_ae(
             batch_size, data_num, data_test_num, gen_num, out_dim, z_dim)
    elif model_type == 'mmd_gan':
        (x, z, z_full, x_test, avg_dist_x_test_to_x, x_precompute, x_test_precompute,
         avg_dist_x_test_to_x_precomputed, g, g_full, ae_loss, d_loss, mmd, d_optim,
         g_optim, summary_op) = build_model_mmd_gan(
             batch_size, gen_num, data_num, data_test_num, out_dim, z_dim)

    # Start session.
    init_op = tf.global_variables_initializer()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    sess_config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)

    with tf.Session(config=sess_config) as sess:
        saver, summary_writer = prepare_logging(log_dir, checkpoint_dir, sess)
        sess.run(init_op)

        # load_existing_model().
        if load_existing:
            could_load, checkpoint_counter = load_checkpoint(
                saver, sess, checkpoint_dir)
            if could_load:
                load_step = checkpoint_counter
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")
        else:
            load_step = 0

        # Once, compute average distance from heldout data to training data.
        avg_dist_x_test_to_x_precomputed_ = sess.run(
            avg_dist_x_test_to_x_precomputed,
            feed_dict=
                {x_precompute: data,
                 x_test_precompute: data_test})

        # train()
        start_time = time()
        for step in range(load_step, max_step):
            if model_type == 'mmd_ae':
                random_batch_data = np.array(
                    [data[d] for d in np.random.choice(len(data), batch_size)])
                random_batch_data_test = np.array(
                    [data_test[d] for d in np.random.choice(
                        len(data_test), batch_size)])
                sess.run(d_optim,
                     feed_dict={
                         x: random_batch_data,
                         x_test: random_batch_data_test,
                         avg_dist_x_test_to_x: avg_dist_x_test_to_x_precomputed_})
            elif model_type == 'mmd_gan':
                random_batch_data = np.array(
                    [data[d] for d in np.random.choice(len(data), batch_size)])
                random_batch_data_test = np.array(
                    [data_test[d] for d in np.random.choice(
                        len(data_test), batch_size)])
                random_batch_z = get_random_z(gen_num, z_dim)
                sess.run([d_optim, g_optim],
                     feed_dict={
                         z: random_batch_z,
                         x: random_batch_data,
                         x_test: random_batch_data_test,
                         avg_dist_x_test_to_x: avg_dist_x_test_to_x_precomputed_})

            ###################################################################
            # Occasionally log/plot results.
            if step % log_step == 0:
                # Read off from graph.
                if model_type == 'mmd_ae':
                    d_loss_, ae_loss_, mmd_, g_batch_, summary_result = sess.run(
                        [d_loss, ae_loss, mmd, g, summary_op],
                        feed_dict={
                             x: random_batch_data,
                             x_test: random_batch_data_test,
                             avg_dist_x_test_to_x: avg_dist_x_test_to_x_precomputed_})
                    g_full_ = sess.run(g_full, {x_full: data})
                    print(('Iter:{}, d_loss = {:.4f}, ae_loss = {:.4f}, '
                        'mmd = {:.4f}').format(step, d_loss_, ae_loss_, mmd_))
                elif model_type == 'mmd_gan':
                    d_loss_, ae_loss_, mmd_, summary_result = sess.run(
                        [d_loss, ae_loss, mmd, summary_op],
                        feed_dict={
                            z: random_batch_z,
                            x: random_batch_data,
                            x_test: random_batch_data_test,
                            avg_dist_x_test_to_x: avg_dist_x_test_to_x_precomputed_})
                    g_full_ = sess.run(g_full,
                        feed_dict={
                            z_full: get_random_z(data_num, z_dim)})
                    print(('Iter:{}, d_loss = {:.4f}, ae_loss = {:.4f}, '
                           'mmd = {:.4f}').format(
                               step, d_loss_, ae_loss_, mmd_))

                    c1_mean = np.array([-2., 5.])
                    c1_cov = np.array([[1., .6], [.6, 1.]])
                    both = np.concatenate((data, g_full_))
                    ll_gen = sample_loglik(c1_mean, c1_cov, g_full_)
                    ll_data = sample_loglik(c1_mean, c1_cov, data)
                    ll_both = 0.5 * sample_loglik(c1_mean, c1_cov, both)
                    ll_ratio_gen = ll_gen / ll_data
                    ll_ratio_both = ll_both / ll_data
                    both_mean = np.mean(both, axis=0)
                    both_cov = np.cov(both.T)
                    print(('  ll_gen = {:.4f}, ll_data = {:.4f}, ratio(g/d) = '
                        '{:.4f}, ratio(b/d) = {:.4f}').format(
                                ll_gen, ll_data, ll_ratio_gen, ll_ratio_both))
                    print('Union: ', both_mean)
                    print('Sample: ',sample_mean)
                    print('Union: ', both_cov)
                    print('Sample: ',sample_cov)

                # Compute disclosure risk.
                sensitivity, precision, ball_radius = evaluate_presence_risk(
                    data, data_test, g_full_)
                print('  Sensitivity={:.4f}, Precision={:.4f}'.format(
                    sensitivity, precision))

                # Add to summaries.
                summary_writer.add_summary(summary_result, step)
                add_nongraph_summary_items(summary_writer, step,
                    {'misc/sensitivity': sensitivity,
                     'misc/precision': precision,
                     'misc/ball_radius': ball_radius})
                if model_type == 'mmd_gan':
                    add_nongraph_summary_items(summary_writer, step,
                        {'misc/ll_gen': ll_gen,
                         'misc/ll_both': ll_both,
                         'misc/ll_ratio_gen': ll_ratio_gen,
                         'misc/ll_ratio_both': ll_ratio_both})

                # Save checkpoint.
                saver.save(
                    sess, 
                    os.path.join(log_dir, 'checkpoints', model_type),
                    global_step=step)

                # Save generated data to file.
                np.save(os.path.join(log_dir, 'g_out.npy'), g_full_)
                with open(g_out_file, 'a') as f:
                    f.write(str(g_full_) + '\n')

                # Print time performance.
                if step % (10 * log_step) == 0 and step > 0:
                    elapsed_time = time() - start_time
                    time_per_iter = elapsed_time / step
                    total_est = elapsed_time / step * max_step
                    m, s = divmod(total_est, 60)
                    h, m = divmod(m, 60)
                    total_est_str = '{:.0f}:{:02.0f}:{:02.0f}'.format(h, m, s)
                    print ('  Time (s): {:.2f}, time/iter: {:.4f},'
                            ' Total est.: {:.4f}').format(
                                step, elapsed_time, time_per_iter, total_est_str)

                    print('  Save tag: {}'.format(save_tag))

                # Make scatter plots.
                if out_dim > 2:
                    indices_to_plot = [0, 1, 2]
                elif out_dim == 2:
                    indices_to_plot = range(out_dim)
                    fig, ax = plt.subplots()
                    ax.scatter(*zip(*data), color='gray', alpha=0.2, label='data')
                    ax.scatter(*zip(*data_test), color='red', alpha=0.2, label='test')
                    ax.scatter(*zip(*g_full_), color='green', alpha=0.2, label='sim')
                    ax.legend()
                    plt.savefig(os.path.join(
                        plot_dir, 'scatter_i{}.png'.format(step)))
                    plt.close(fig)
                else:
                    indices_to_plot = range(out_dim)


if __name__ == "__main__":
    main()
