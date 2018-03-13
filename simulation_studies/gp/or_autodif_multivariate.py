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
from weighting import get_estimation_points


def compute_mmd(arr1, arr2, batch_size, gen_num):
    """Computes mmd between two inputs of size [batch_size, ...]."""
    v = tf.concat([arr1, arr2], 0)
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


def print_time_info(start_time, step, max_step):
    elapsed_time = time() - start_time
    time_per_iter = elapsed_time / step
    total_est = elapsed_time / step * max_step
    m, s = divmod(total_est, 60)
    h, m = divmod(m, 60)
    total_est_str = '{:.0f}:{:02.0f}:{:02.0f}'.format(h, m, s)
    print('  time (s): {:.2f}, time/iter: {:.4f},'
          ' Total est.: {}').format(
              elapsed_time, time_per_iter, total_est_str)


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


def load_data(data_file):
    data = np.loadtxt(open(data_file, 'rb'), delimiter=' ')
    data_num = data.shape[0]
    data_dim = data.shape[1]
    return data, data_num, data_dim


def sample_uniform_random_gen(data, gen_num, data_dim):
    d_xmin, d_ymin = np.min(data, axis=0)
    d_xmax, d_ymax = np.max(data, axis=0)
    gen_uniform = np.zeros((gen_num, data_dim))
    for i in range(gen_num):
        gen_uniform[i] = [np.random.uniform(d_xmin, d_xmax),
                          np.random.uniform(d_ymin, d_ymax)]
    return gen_uniform


def prepare_dirs(tag):
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


def build_model(batch_size, gen_num, data_dim, optimizer, learning_rate):
    x = tf.Variable(tf.zeros([batch_size, data_dim]), name='x', trainable=False)
    g = tf.Variable(tf.zeros([gen_num, data_dim]), name='g', trainable=True)
    x_update_val = tf.Variable(tf.zeros([batch_size, data_dim]),
                               name='x_update_val', trainable=False)
    g_update_val = tf.Variable(tf.zeros([gen_num, data_dim]),
                               name='g_update_val', trainable=False)
    x_update = tf.assign(x, x_update_val, name='x_update')
    g_update = tf.assign(g, g_update_val, name='g_update')

    # SET UP MMD LOSS.
    mmd = compute_mmd(x, g, batch_size, gen_num)

    if optimizer == 'adagrad':
        g_opt = tf.train.AdagradOptimizer(learning_rate)
    elif optimizer == 'adam':
        g_opt = tf.train.AdamOptimizer(learning_rate)
    elif optimizer == 'rmsprop':
        g_opt = tf.train.RMSPropOptimizer(learning_rate)
    else:
        g_opt = tf.train.GradientDescentOptimizer(learning_rate)

    g_optim = g_opt.minimize(mmd)
    #return x, g, g_update, g_update_val, mmd, g_optim
    return x, x_update, x_update_val, g, g_update, g_update_val, mmd, g_optim


def main(data_file, batch_size, gen_num, optimizer, learning_rate):
    start_time = time()
    log_step = 500
    max_step = 20000
    tag = 'test'
    load_existing = False
    activation = tf.nn.elu

    # Load data.
    data, data_num, data_dim = load_data(data_file)

    # Sample starting point for generations.
    gen_uniform = sample_uniform_random_gen(data, gen_num, data_dim)

    # Prepare directories.
    log_dir, checkpoint_dir, plot_dir = prepare_dirs(tag)

    # Build model.
    x, x_update, x_update_val, g, g_update, g_update_val, mmd, g_optim = build_model(
        batch_size, gen_num, data_dim, optimizer, learning_rate)

    # Initialize and start session.
    init_op = tf.global_variables_initializer()
    saver = tf.train.Saver()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    sess_config = tf.ConfigProto(allow_soft_placement=True,
                                 gpu_options=gpu_options)
    with tf.Session(config=sess_config) as sess:
        sess.run(init_op)

        # Set save tag, as a function of config parameters.
        save_tag = 'dn{}_bs{}_gen{}_lr{}_op_{}'.format(
            data_num, batch_size, gen_num, learning_rate, optimizer)
        with open(os.path.join(log_dir, 'save_tag.txt'), 'w') as save_tag_file:
            save_tag_file.write(save_tag)

        # Remove existing output file.
        g_out_file = os.path.join(log_dir, 'g_out.txt')
        if os.path.isfile(g_out_file):
            os.remove(g_out_file)

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

        # Set initial value for x, g, and mmd.
        random_batch_data = np.array(
            [data[d] for d in np.random.choice(len(data), batch_size)])
        initial_x = sess.run(x_update, {x_update_val: random_batch_data})
        initial_g = sess.run(g_update, {g_update_val: gen_uniform})
        mmd_ = 1e10

        # Do training iterations: train().
        for step in range(load_step, max_step):

            # Update generations based on MMD between gens and minibatch data.
            random_batch_data = np.array(
                [data[d] for d in np.random.choice(len(data), batch_size)])
            sess.run(g_optim, {x: random_batch_data})

            # Occasionally log/plot results.
            if step % log_step == 0:
                # Save checkpoint.
                saver.save(sess, os.path.join(log_dir, 'checkpoints', tag),
                    global_step=step)
                # Print some loss values.
                mmd_last = mmd_
                mmd_, g_out = sess.run([mmd, g])
                stopping_eps = 3e-4
                if np.abs(mmd_ - mmd_last) < stopping_eps:
                    print_time_info(start_time, step, max_step)
                    return g_out, log_dir

                print(save_tag)
                print('Iter:{}, mmd = {:.4f}'.format(step, mmd_))

                # Make scatter plots.
                if data_dim > 2:
                    indices_to_plot = [0, 1, 2]
                elif data_dim == 2:
                    indices_to_plot = range(data_dim)
                    fig, ax = plt.subplots()
                    ax.scatter(*zip(*data), color='gray', alpha=0.05)
                    ax.scatter(*zip(*g_out), color='green', alpha=0.3)
                    plt.savefig(os.path.join(
                        plot_dir, 'scatter_i{}.png'.format(step)))
                    plt.close(fig)
                else:
                    indices_to_plot = range(data_dim)

                # Save generated data to file.
                np.save(os.path.join(log_dir, 'g_out.npy'), g_out)
                with open(g_out_file, 'a') as f:
                    f.write(str(g_out) + '\n')

                # Print time performance.
                if step % 10 * log_step > 0:
                    print_time_info(start_time, step, max_step)


def get_points():
    data_file = 'gp_data.txt'
    batch_size = 100
    gen_num = 100
    optimizer = 'rmsprop'
    learning_rate = 1e-3
    mode = 'coreset'
    g_out, log_dir = main(data_file, batch_size, gen_num, optimizer, learning_rate)
    results = get_estimation_points(log_dir=log_dir, mode=mode, support_points=g_out)
    support_points, coreset, weights_estimation_pts, weights_data = results
    for r in results:
        print(r.shape)
    return g_out, support_points, coreset, weights_estimation_pts, weights_data


if __name__ == "__main__":
    get_points()
