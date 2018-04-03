import argparse
from time import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm
import os
import pdb
import sys
import tensorflow as tf
layers = tf.layers
import pandas as pd
import seaborn as sb
from scipy.stats import scoreatpercentile
from scipy.stats import pearsonr 


# Config.
parser = argparse.ArgumentParser()
parser.add_argument('--data_num', type=int, default=1000)
parser.add_argument('--percent_train', type=float, default=0.9)
parser.add_argument('--batch_size', type=int, default=1000)
parser.add_argument('--gen_num', type=int, default=1000)
parser.add_argument('--z_dim', type=int, default=64)
parser.add_argument('--width', type=int, default=100,
                    help='width of generator layers')
parser.add_argument('--depth', type=int, default=3, 
                    help='num of generator layers')
parser.add_argument('--lambda_mmd', type=float, default=1e-1)
parser.add_argument('--log_step', type=int, default=500)
parser.add_argument('--max_step', type=int, default=500000)
parser.add_argument('--learning_rate', type=float, default=1e-3)
parser.add_argument('--lr_update_step', type=int, default=20000)
parser.add_argument('--optimizer', type=str, default='rmsprop',
                    choices=['adagrad', 'adam', 'gradientdescent', 'rmsprop'])
parser.add_argument('--data_file', type=str, default='data0.csv')
parser.add_argument('--tag', type=str, default='test')
parser.add_argument('--load_existing', action='store_true', default=False,
                    dest='load_existing')
parser.add_argument('--sample_n', type=int, default=0,
                    help='sample n from existing model then exit')
parser.add_argument('--plot_sparse', action='store_true', default=False)

args = parser.parse_args()
data_num = args.data_num
percent_train = args.percent_train
batch_size = args.batch_size
gen_num = args.gen_num
z_dim = args.z_dim
width = args.width
depth = args.depth
lambda_mmd = args.lambda_mmd
log_step = args.log_step
max_step = args.max_step
learning_rate = args.learning_rate
lr_update_step = args.lr_update_step
optimizer = args.optimizer
data_file = args.data_file
tag = args.tag
load_existing = args.load_existing
sample_n = args.sample_n
plot_sparse = args.plot_sparse
activation = tf.nn.elu


def fivenum(v):
    """Returns Tukey's five number summary (minimum, lower-hinge, median, upper-hinge, maximum) for the input vector, a list or array of numbers based on 1.5 times the interquartile distance"""
    try:
        np.sum(v)
    except TypeError:
        print('Error: you must provide a list or array of only numbers')
    q1 = scoreatpercentile(v,25)
    q3 = scoreatpercentile(v,75)
    iqd = q3 - q1
    md = np.median(v)
    whisker = 1.5 * iqd
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


def plot_marginals(raw_data_train, data, batch_size, step, g_, g_out, log_dir,
        filename_tag=None, plot_sparse=False):
    """Plots all marginals, and computes MMDs between marginals of real and sim.

    Args:
      raw_data_train: Numpy array, un-standardized data. 
      data: Numpy array, standardized data. 
      batch_size: Int, batch_size for MMD computation. 
      step: Int, used for logging. 
      g_: Numpy array, standardized simulation. 
      g_out: Numpy array, un-standardized simulation. 
      log_dir: String, path where plots are saved.
      filename_tag: String, used for naming plot.
      plot_spares: Bool, used to toggle labels on plots.
    """
    random_batch_data = np.array(
        [data[i] for i in np.random.choice(len(data), batch_size)])
    random_batch_gen = np.array(
        [g_[i] for i in np.random.choice(len(g_), batch_size)])
    num_cols = raw_data_train.shape[1]
    sq_dim = int(np.ceil(np.sqrt(num_cols)))
    fig, axs = plt.subplots(sq_dim, sq_dim, figsize=(20, 20))
    if not plot_sparse:
        fig.subplots_adjust(hspace=0.5, wspace=0.5)
        fig.suptitle('Marginals, it{}'.format(step))
    else:
        fig.subplots_adjust(hspace=0.05, wspace=0.05)
    axs = axs.ravel()
    bins = 30
    for i in range(num_cols):
        mmd_i_data_gen, _ = compute_mmd(
            random_batch_data[:, i], random_batch_gen[:, i], use_tf=False)
        plot_d = raw_data_train[:, i]
        plot_g = g_out[:, i]
        axs[i].hist(plot_d, normed=True, alpha=0.3, label='d', bins=bins)
        axs[i].hist(plot_g, normed=True, alpha=0.3, label='g', bins=bins)
        if not plot_sparse:
            axs[i].set_xlabel('mmd = {:.3f}'.format(mmd_i_data_gen))
            axs[i].legend()
        else:
            axs[i].tick_params(axis='both', which='both', bottom='off', top='off',
                labelbottom='off', right='off', left='off', labelleft='off')
    for i in range(num_cols, sq_dim ** 2):
        axs[i].axis('off')
    if filename_tag:
        filename = 'plot_marginals_{}.png'.format(filename_tag)
    else:
        filename = 'plot_marginals_{}.png'.format(step)
    plt.savefig(os.path.join(log_dir, filename))
    plt.close('all')


def plot_correlations(raw_data_train, step, g_out, log_dir):
    num_cols = raw_data_train.shape[1]
    corr_coefs_data = np.zeros((num_cols, num_cols))
    corr_coefs_gens = np.zeros((num_cols, num_cols))
    for i in range(num_cols):
        for j in range(num_cols):
            if j > i:
                corr_coefs_data[i][j], _ = pearsonr(
                        raw_data_train[:, i], raw_data_train[:, j])
                corr_coefs_gens[i][j], _ = pearsonr(
                        g_out[:, i], g_out[:, j])
    coefs_d = corr_coefs_data.flatten()
    coefs_g = corr_coefs_gens.flatten()
    coefs_d = coefs_d[coefs_d != 0]
    coefs_g = coefs_g[coefs_g != 0]
    fig, ax = plt.subplots()
    ax.scatter(coefs_d, coefs_g)
    ax.plot(ax.get_xlim(), ax.get_ylim(), ls='-')
    #ax.set_xlabel('Correlation data')
    #ax.set_ylabel('Correlation gens')
    plt.savefig(os.path.join(log_dir, 'plot_correlations_{}.png'.format(
        step)))
    plt.close('all')


def evaluate_presence_risk(all_train, test, sim):
    """Assess presence disclosure risk of simulations.
    
    For each candidate record in a subset of training data, compute counts of
    True Pos, True Neg, False Pos, and False Neg, where positive classification
    indicates that the candidate has a neighbor in the simulation set.
    
    Args:
      all_train: Numpy array of all training data.
      test: Numpy array of test data (smaller than train).
      sim: Numpy array of simulations.
      presence_margin: Float, size of ball around candidate point, used to
        compute whether a neighbor is found.

    Return:
      sensitivity: Float of TP / (TP + FN).
      precision: Float of TP / (TP + FP).
    """
    # Constants.
    presence_margin = 15. 

    assert len(test) < len(all_train), 'test should be smaller than train'
    num_samples = len(test)
    compromised_records = all_train[:num_samples]
    tp, tn, fp, fn = 0, 0, 0, 0

    # Count true positives and false negatives.
    for r in compromised_records:
        distances_from_r = norm(r - sim, axis=1)
        has_neighbor = np.any(distances_from_r < presence_margin)
        if has_neighbor:
            tp += 1
        else:
            fn += 1
    # Count false positives and true negatives.
    for t in test:
        distances_from_t = norm(t - sim, axis=1)
        has_neighbor = np.any(distances_from_t < presence_margin)
        if has_neighbor:
            fp += 1
        else:
            tn += 1

    sensitivity = float(tp) / (tp + fn)
    precision = float(tp + 1e-10) / (tp + fp + 1e-10)
    return sensitivity, precision, presence_margin, tp, fn, fp


def evaluate_attribute_risk(all_train, test, sim, binary_cols):
    """Assess attribute disclosure risk of simulations.
    
    For each candidate record in a subset of training data, consider that some
    subset of its features are known. Using only these known features, find
    neighbors in simulation set, and compute mean of remaining features. Count
    True Pos, True Neg, False Pos, and False Neg, where positive classification
    indicates that a feature's mean over simulations is within margin of error
    of the true feature values.

    Args:
      all_train: Numpy array of all training data.
      test: Numpy array of test data (smaller than train).
      sim: Numpy array of simulations.
      binary_cols: Numpy array of binary column indices.
      k: Int, number of neighbors to include.
      attribute_margin: Float, margin of error, for feature mean over
        simulations to be considered a match.

    Return:
      sensitivity: Float of TP / (TP + FN).
      precision: Float of TP / (TP + FP).
    """
    # Constants.
    k = 5 
    attribute_margin = 0.05
    pct_attributes_known = 0.5

    assert len(test) < len(all_train), 'test should be smaller than train'
    num_compromised = test.shape[0]
    num_attributes = test.shape[1]
    compromised_records = all_train[:num_compromised] 
    attr_idxs = np.random.choice(num_attributes,
        int(pct_attributes_known * num_attributes), replace=False)
    attr_bool = np.zeros(num_attributes, dtype=np.bool)
    attr_bool[attr_idxs] = True 
    attr_complement_idxs = np.arange(num_attributes)[~attr_bool]

    tp, tn, fp, fn = 0, 0, 0, 0

    # Count true positives and false negatives.
    for r in compromised_records:
        r = np.array(r)
        distances_from_r = norm(r[attr_bool] - sim[:, attr_bool], axis=1)
        k_nearest_idxs = distances_from_r.argsort()[:k]
        k_nearest_neighbors = sim[k_nearest_idxs]
        neighbors_complement_features = k_nearest_neighbors[:, ~attr_bool]
        neighbors_complement_features_mean = np.mean(neighbors_complement_features, axis=0)
        r_complement_features = r[~attr_bool]
        # Add TP and FN of each feature independently.
        for i, complement_idx in enumerate(attr_complement_idxs):
            neighbor_mean = neighbors_complement_features_mean[i]
            r_feature = r_complement_features[i]
            if complement_idx in binary_cols:
                # In binary case
                if ((neighbor_mean < 0.5 and r_feature == 0) or
                    (neighbor_mean >= 0.5 and r_feature == 1)):
                    tp += 1
                else:
                    fn += 1
            else:
                # In continuous case, measure relative error.
                relative_error = abs(float(neighbor_mean - r_feature) / r_feature)
                if relative_error < attribute_margin:
                    tp += 1
                else:
                    fn += 1

    # Count false positives and true negatives.
    for t in test:
        t = np.array(t)
        distances_from_t = norm(t[attr_bool] - sim[:, attr_bool], axis=1)
        k_nearest_idxs = distances_from_t.argsort()[:k]
        k_nearest_neighbors = sim[k_nearest_idxs]
        neighbors_complement_features = k_nearest_neighbors[:, ~attr_bool]
        neighbors_complement_features_mean = np.mean(neighbors_complement_features, axis=0)
        t_complement_features = t[~attr_bool]
        # Add TP and FN of each feature independently.
        for i, complement_idx in enumerate(attr_complement_idxs):
            neighbor_mean = neighbors_complement_features_mean[i]
            t_feature = t_complement_features[i]
            if complement_idx in binary_cols:
                # In binary case
                if ((neighbor_mean < 0.5 and t_feature == 0) or
                    (neighbor_mean >= 0.5 and t_feature == 1)):
                    fp += 1
                else:
                    tn += 1
            else:
                # In continuous case, measure relative error.
                relative_error = float(neighbor_mean - t_feature) / t_feature
                if relative_error < attribute_margin:
                    fp += 1
                else:
                    tn += 1

    sensitivity = float(tp) / (tp + fn)
    precision = float(tp + 1e-10) / (tp + fp + 1e-10)
    return sensitivity, precision, attribute_margin, tp, fn, fp


def load_checkpoint(saver, sess, checkpoint_dir):
    import re
    print(" [*] Reading checkpoints...")
    print("     {}".format(checkpoint_dir))
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
        counter = int(ckpt_name.split('-')[-1])
        print(" [*] Success to read {}".format(ckpt_name))
        return True, counter
    else:
        print(" [*] Failed to find a checkpoint")
        return False, 0


def prepare_dirs():
    """Set up log dir."""
    log_dir = 'logs_{}'.format(tag)
    checkpoint_dir = os.path.join(log_dir, 'checkpoints')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    return log_dir, checkpoint_dir


def evaluate_disclosure_risks(g_read_only, num_test, z_dim, std_vec, mean_vec,
        raw_data_train, raw_data_test, raw_data_eval, binary_cols,
        presence_risk_file, attribute_risk_file, temp=False):
    if temp:
        presence_risk_file = os.path.join(log_dir, 'presence_risk_temp.txt')
        attribute_risk_file = os.path.join(log_dir, 'attribute_risk_temp.txt')
        num_samples = 10
        print('Taking {} samples of presence and attribute risks...'.format(
            num_samples))
        for i in xrange(num_samples):
            sim_ = sess.run(g_read_only,
                feed_dict={
                    z: get_random_z(num_test, z_dim)})
            sim_full = (sim_ * std_vec + mean_vec)
            random_sample_eval = np.array(
                [raw_data_eval[d] for d in np.random.choice(
                    len(raw_data_eval), len(raw_data_eval))])
            sim_full = random_sample_eval 

            p_sens, p_prec, p_margin, p_tp, p_fn, p_fp = evaluate_presence_risk(
                raw_data_train, raw_data_test, sim_full)
            a_sens, a_prec, a_margin, a_tp, a_fn, a_fp = evaluate_attribute_risk(
                raw_data_train, raw_data_test, sim_full, binary_cols)
            with open(presence_risk_file, 'a') as pf:
                pf.write(','.join(map(str, [p_sens, p_prec])) + '\n')
            with open(attribute_risk_file, 'a') as af:
                af.write(','.join(map(str, [a_sens, a_prec])) + '\n')
            print('Got sample {}.'.format(i))

        os.system('python evaluate_disclosure_risks.py --tag="{}" --temp'.format(tag))

    else:
        sim_ = sess.run(g_read_only,
            feed_dict={
                z: get_random_z(num_test, z_dim)})
        sim_full = (sim_ * std_vec + mean_vec)

        p_sens, p_prec, p_margin, p_tp, p_fn, p_fp = evaluate_presence_risk(
            raw_data_train, raw_data_test, sim_full)
        a_sens, a_prec, a_margin, a_tp, a_fn, a_fp = evaluate_attribute_risk(
            raw_data_train, raw_data_test, sim_full, binary_cols)
        with open(presence_risk_file, 'a') as pf:
            pf.write(','.join(map(str, [p_sens, p_prec])) + '\n')
        with open(attribute_risk_file, 'a') as af:
            af.write(','.join(map(str, [a_sens, a_prec])) + '\n')

        print('  Disclosure risk on {} from train'.format(len(raw_data_test)))
        print('    Pres: sens={:.4f}, prec={:.4f}, pm={}, tp={}, fn={}, fp={} '.format(
            p_sens, p_prec, p_margin, p_tp, p_fn, p_fp))
        print('    Attr: sens={:.4f}, prec={:.4f}, am={}, tp={}, fn={}, fp={} '.format(
            a_sens, a_prec, a_margin, a_tp, a_fn, a_fp))


###############################################################################

# BEGIN: Data Preparation.

# Prepare directories for logging and checkpoints.
log_dir, checkpoint_dir = prepare_dirs()

# load_data().
if data_file:
    orig_raw_data = np.loadtxt(open(data_file, 'rb'), delimiter=',')
    num_rows = orig_raw_data.shape[0]
    num_cols = orig_raw_data.shape[1]

    # Separate train and test data.
    num_train = int(percent_train * num_rows)
    num_test_and_eval = num_rows - num_train
    num_test = num_test_and_eval / 2
    num_eval = num_test_and_eval - num_test
    raw_data_train = orig_raw_data[:num_train]
    raw_data_test = orig_raw_data[num_train: num_train + num_test]
    raw_data_eval = orig_raw_data[-num_eval:]

    print('\nRaw data:')
    for i in range(num_cols):
        print('{: >17},{: >17},{: >17},{: >17},{: >17}'.format(
            *fivenum(raw_data_train[:,i])))

    binary_cols = []
    for col in range(num_cols):
        col_data = orig_raw_data[:, col]
        if np.array_equal(col_data, col_data.astype(bool)):
            binary_cols.append(col)
    print('binary_cols={}'.format(binary_cols))

    # Don't standardize binary vars.
    mean_mask = np.array([1] * num_cols)
    mean_mask[binary_cols] = 0  
    std_mask = np.array([1] * num_cols) 
    mean_vec = raw_data_train.mean(0) * mean_mask
    std_vec = raw_data_train.std(0) * std_mask
    data = (raw_data_train - mean_vec) / std_vec 

    print('\nStandardized data:')
    for i in range(data.shape[1]):
        print('{: >17},{: >17},{: >17},{: >17},{: >17}'.format(
            *fivenum(data[:,i])))

    data_num = len(data)
    out_dim = data.shape[1]
else:
    data = build_toy_data()
    data = (data - data.mean(0))/data.std(0)
    out_dim = data.shape[1]

# END: Data Preparation.


# BEGIN: Build Model.

x = tf.placeholder(tf.float64, [None, out_dim], name='x')
z = tf.placeholder(tf.float64, [None, z_dim], name='z')

g, g_vars = generator(
    z, width=width, depth=depth, activation=activation, out_dim=out_dim)
g_read_only, _ = generator(
    z, width=width, depth=depth, activation=activation, out_dim=out_dim,
        reuse=True)
h_out, ae_out, enc_vars, dec_vars = autoencoder(tf.concat([x, g], 0),
    width=width, depth=depth, activation=activation, z_dim=z_dim, reuse=False)
enc_x, enc_g = tf.split(h_out, [batch_size, gen_num])
ae_x, ae_g = tf.split(ae_out, [batch_size, gen_num])

ae_loss = tf.reduce_mean(tf.square(ae_x - x))

# Set up MMD loss.
mmd, exp_object = compute_mmd(enc_x, enc_g)
d_loss = ae_loss - lambda_mmd * mmd
g_loss = mmd

lr = tf.Variable(learning_rate, name='lr', trainable=False)
lr_update = tf.assign(lr, tf.maximum(lr * 0.5, 1e-8), name='lr_update')
if optimizer == 'adagrad':
    ae_opt = tf.train.AdagradOptimizer(lr)
    d_opt = tf.train.AdagradOptimizer(lr)
    g_opt = tf.train.AdagradOptimizer(lr)
elif optimizer == 'adam':
    ae_opt = tf.train.AdamOptimizer(lr)
    d_opt = tf.train.AdamOptimizer(lr)
    g_opt = tf.train.AdamOptimizer(lr)
elif optimizer == 'rmsprop':
    ae_opt = tf.train.RMSPropOptimizer(lr)
    d_opt = tf.train.RMSPropOptimizer(lr)
    g_opt = tf.train.RMSPropOptimizer(lr)
else:
    ae_opt = tf.train.GradientDescentOptimizer(lr)
    d_opt = tf.train.GradientDescentOptimizer(lr)
    g_opt = tf.train.GradientDescentOptimizer(lr)

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
    tf.summary.scalar("lr/lr", lr),
])

# END : Build Model.

# BEGIN: Initialize Model.

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
sess.run(init_op)

# Set save tag, as a function of config parameters.
save_tag = str(args) + '_binary_cols={}'.format(binary_cols)
with open(os.path.join(log_dir, 'save_tag.txt'), 'w') as save_tag_file:
    save_tag_file.write(save_tag)
print save_tag

# Set up cumulative output files.
g_out_file = os.path.join(log_dir, 'g_out.txt')
presence_risk_file = os.path.join(log_dir, 'presence_risk.txt')
attribute_risk_file = os.path.join(log_dir, 'attribute_risk.txt')
for fi in [g_out_file, presence_risk_file, attribute_risk_file]:
    if os.path.isfile(fi) and not load_existing:
        os.remove(fi)

# Start time.
start_time = time()

# Load existing model.
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

# END: Initialize Model.


# BEGIN: Sample Existing Model. 

if load_existing and sample_n:
    random_batch_data = np.array(
        [data[d] for d in np.random.choice(len(data), batch_size)])
    random_batch_z = get_random_z(sample_n, z_dim)

    g_ = sess.run(g_read_only,
        feed_dict={
            z: random_batch_z})

    g_out = (g_ * std_vec + mean_vec)
    for i, row in enumerate(g_out):
        for col in binary_cols:
            if row[col] < 0.5:
                row[col] = 0.0
            else:
                row[col] = 1.0
        g_out[i] = row
    out_name = os.path.join(log_dir, 'g_out_sample_{}'.format(sample_n))
    np.save(out_name + '.npy', g_out)
    np.savetxt(out_name + '.csv', g_out, delimiter=',')
    print('Saved npy and csv of:\n{}'.format(out_name))

    plot_marginals(raw_data_train, data, batch_size, step, g_, g_out, log_dir,
        filename_tag='sample_{}'.format(sample_n), plot_sparse=plot_sparse)

    # Evaluate disclosure risks for existing model.
    evaluate_disclosure_risks(g_read_only, num_test, z_dim, std_vec, mean_vec,
        raw_data_train, raw_data_test, raw_data_eval, binary_cols,
        presence_risk_file, attribute_risk_file, temp=True)

    # If only sampling from existing model, exit before training.
    sys.exit('Finished sampling n.')

elif load_existing:
    print('Contining training on checkpoint_dir:\n  {}.'.format(
        checkpoint_dir))

else:
    print('Training new model, and storing at checkpoint_dir:\n  {}.'.format(
        checkpoint_dir))

# END: Sample Existing Model. 


# BEGIN: Train Model. 

if int(raw_input('\nContinue? [0/1]: ')) != 1:
    sys.exit()
for step in range(load_step, max_step):
    random_batch_data = np.array(
        [data[d] for d in np.random.choice(len(data), batch_size)])
    random_batch_z = get_random_z(gen_num, z_dim)

    sess.run([d_optim, g_optim],
             feed_dict={
                 z: random_batch_z,
                 x: random_batch_data})

    if step % lr_update_step == lr_update_step - 1:
        sess.run(lr_update)

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

        """
        # Inspect size of exponential object, for evaluating kernel bandwidth.
        enc_x_, enc_g_, exp_object_, mmd_ = sess.run(
            [enc_x, enc_g, exp_object, mmd],
                feed_dict={
                    z: random_batch_z,
                    x: random_batch_data})
        print(np.exp(-0.5 * exp_object_))
        pdb.set_trace()
        """

        # Print model loss values.
        d_loss_, ae_loss_, mmd_, g_ = sess.run(
            [d_loss, ae_loss, mmd, g], feed_dict={
                z: random_batch_z,
                x: random_batch_data})
        print('Iter:{}, d_loss = {:.4f}, ae_loss = {:.4f}, '
            'L * mmd = {:.4f}, mmd = {:.4f}'.format(step, d_loss_, ae_loss_,
                lambda_mmd * mmd_, mmd_))

        # Save generated data to NumPy file and to output collection.
        # NOTE: First un-norm, then round values in binary cols.
        g_out = (g_ * std_vec + mean_vec)

        for i, row in enumerate(g_out):
            for col in binary_cols:
                if row[col] < 0.5:
                    row[col] = 0.0
                else:
                    row[col] = 1.0
            g_out[i] = row
        np.save(os.path.join(log_dir, 'g_out.npy'), g_out)
        np.savetxt(os.path.join(log_dir, 'g_out.csv'), g_out, delimiter=',')
        with open(g_out_file, 'a') as gf:
            gf.write(str(g_out) + '\n')

        # Print presence and attribute risk values.
        evaluate_disclosure_risks(g_read_only, num_test, z_dim, std_vec,
            mean_vec, raw_data_train, raw_data_test, raw_data_eval, binary_cols,
            presence_risk_file, attribute_risk_file)

        # PLOTTING RESULTS.
        plot = 1
        if plot:
            plot_marginals(raw_data_train, data, batch_size, step, g_, g_out,
                log_dir, plot_sparse=plot_sparse)
            plot_correlations(raw_data_train, step, g_out, log_dir)

        # Print time performance.
        if step % log_step == 0 and step > 0:
            elapsed_time = time() - start_time
            time_per_iter = elapsed_time / step
            total_est = elapsed_time / step * max_step
            m, s = divmod(total_est, 60)
            h, m = divmod(m, 60)
            total_est_str = '{:.0f}:{:02.0f}:{:02.0f}'.format(h, m, s)
            print(('  time (s): {:.2f}, time/iter: {:.4f},'
                   ' Total est.: {}').format(
                       elapsed_time, time_per_iter, total_est_str))

# END: Train Model. 
