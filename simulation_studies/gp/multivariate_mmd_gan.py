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
parser.add_argument('--batch_size', type=int, default=800)
parser.add_argument('--gen_num', type=int, default=800)
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
data_num = args.data_num
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


def build_model_disc():
    d_x, d_vars = discriminator(x, width=width, depth=depth, activation=activation, reuse=False)
    d_g, _ = discriminator(g, width=width, depth=depth, activation=activation, reuse=True)
    def sigmoid_cross_entropy_with_logits(x_, y_):
      try:
        return tf.nn.sigmoid_cross_entropy_with_logits(logits=x_, labels=y_)
      except:
        return tf.nn.sigmoid_cross_entropy_with_logits(logits=x_, targets=y_)
    d_loss_real = tf.reduce_mean(sigmoid_cross_entropy_with_logits(d_x, tf.ones_like(d_x)))
    d_loss_fake = tf.reduce_mean(sigmoid_cross_entropy_with_logits(d_g, tf.zeros_like(d_g)))
    d_loss = d_loss_real + d_loss_fake
    g_loss = -1.0 * d_loss_fake
    g_loss = tf.reduce_mean(sigmoid_cross_entropy_with_logits(d_g, tf.ones_like(d_g)))


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


def load_data(data_num):
    # Load data.
    if data_file:
        data = np.loadtxt(open(data_file, 'rb'), delimiter=' ')
        data_num = data.shape[0]
        out_dim = data.shape[1]
        return data, data_num, out_dim
    else:
        n1 = data_num / 2
        n2 = data_num - n1
        cluster1 = np.random.multivariate_normal(
            [-2., 5.], [[1., .9], [.9, 1.]], n1)
        cluster2 = np.random.multivariate_normal(
            [6., 6.], [[1., 0.], [0., 1.]], n2)
        data = np.concatenate((cluster1, cluster2))
        out_dim = data.shape[1]
        return data, data_num, out_dim


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


def build_model(batch_size, gen_num, out_dim, z_dim):
    x = tf.placeholder(tf.float64, [batch_size, out_dim], name='x')
    z = tf.placeholder(tf.float64, [gen_num, z_dim], name='z')

    g, g_vars = generator(
        z, width=width, depth=depth, activation=activation, out_dim=out_dim)
    g_read_only, _ = generator(
        z, width=width, depth=depth, activation=activation, out_dim=out_dim,
        reuse=True)
    h_out, ae_out, enc_vars, dec_vars = autoencoder(tf.concat([x, g], 0),
        width=width, depth=depth, activation=activation, z_dim=z_dim,
        reuse=False)
    enc_x, enc_g = tf.split(h_out, [batch_size, gen_num])
    ae_x, ae_g = tf.split(ae_out, [batch_size, gen_num])

    ae_loss = tf.reduce_mean(tf.square(ae_x - x))


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
    return x, z, g, g_read_only, ae_loss, d_loss, mmd, d_optim, g_optim


#def get_sample(gen_num=200, tag='test', checkpoint_dir=None):
#    'Separate callable fn to sample, given gen_num and checkpoint_dir.' 
#    # Set up config.
#    args = parser.parse_args()
#    data_num = args.data_num
#    batch_size = args.batch_size
#    z_dim = args.z_dim
#    width = args.width
#    depth = args.depth
#    log_step = args.log_step
#    max_step = args.max_step
#    learning_rate = args.learning_rate
#    optimizer = args.optimizer
#    data_file = args.data_file
#    activation = tf.nn.elu
#    assert gen_num <= data_num, 'gen_num must be < data_num'
#
#    # Set up data, dirs, and model.
#    data, data_num, out_dim = load_data(data_num)
#    if checkpoint_dir is None:
#        _, checkpoint_dir = prepare_dirs()
#    x, z, g, g_read_only, ae_loss, d_loss, mmd, d_optim, g_optim = build_model(
#        batch_size, gen_num, out_dim, z_dim)
#    init_op = tf.global_variables_initializer()
#    saver = tf.train.Saver()
#    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
#    sess_config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
#    sess = tf.Session(config=sess_config)
#    sess.run(init_op)
#
#    # Sample from a saved model.
#    could_load, checkpoint_counter = load_checkpoint(
#        saver, sess, checkpoint_dir)
#    if could_load:
#        load_step = checkpoint_counter
#        print(' [*] Load SUCCESS, checkpoint {}'.format(load_step))
#    else:
#        print(' [!] Load failed...')
#    random_batch_data = np.array(
#        [data[d] for d in np.random.choice(len(data), batch_size)])
#    random_batch_z = get_random_z(gen_num, z_dim)
#    g_out = sess.run(g_read_only,
#        feed_dict={
#            z: random_batch_z,
#            x: random_batch_data})
#    print(g_out)
#
#    sess.close()
#    return g_out


def main():
    # TODO: Adjust "width" and "depth" so they don't collide with model names.
    print('TODO: Sort why flags cause generation script to fail.')
    args = parser.parse_args()
    data_num = args.data_num
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

    data, data_num, out_dim = load_data(data_num)
    log_dir, checkpoint_dir, plot_dir = prepare_dirs()
    x, z, g, g_read_only, ae_loss, d_loss, mmd, d_optim, g_optim = build_model(
        batch_size, gen_num, out_dim, z_dim)
    init_op = tf.global_variables_initializer()
    saver = tf.train.Saver()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    sess_config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)

    with tf.Session(config=sess_config) as sess:
        sess.run(init_op)

        # Set save tag, as a function of config parameters.
        save_tag = 'dn{}_bs{}_gen{}_w{}_d{}_zd{}_lr{}_op_{}'.format(
            data_num, batch_size, gen_num, width, depth, z_dim, learning_rate,
            optimizer)
        with open(os.path.join(log_dir, 'save_tag.txt'), 'w') as save_tag_file:
            save_tag_file.write(save_tag)
        print(save_tag)

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

        # MAIN RUNNING FUNCTIONS.
        # train()
        start_time = time()
        for step in range(load_step, max_step):
            random_batch_data = np.array(
                [data[d] for d in np.random.choice(len(data), batch_size)])
            random_batch_z = get_random_z(gen_num, z_dim)
            sess.run([d_optim, g_optim],
                     feed_dict={
                         z: random_batch_z,
                         x: random_batch_data})

            # Occasionally log/plot results.
            if step % log_step == 0:
                # Save checkpoint.
                saver.save(sess, os.path.join(log_dir, 'checkpoints', tag),
                    global_step=step)
                # Print some loss values.
                d_loss_, ae_loss_, mmd_, g_out = sess.run(
                    [d_loss, ae_loss, mmd, g], feed_dict={
                        z: random_batch_z,
                        x: random_batch_data})
                print(save_tag)
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
                        plot_dir, 'scatter_i{}.png'.format(step)))
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
                            ' Total est.: {:.4f}').format(
                                step, elapsed_time, time_per_iter, total_est_str)


if __name__ == "__main__":
    main()
