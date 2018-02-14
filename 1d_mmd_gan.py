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
parser.add_argument('--data_num', type=int, default=1000)
parser.add_argument('--gen_num', type=int, default=300)
parser.add_argument('--z_dim', type=int, default=1)
parser.add_argument('--width', type=int, default=3,
                    help='width of generator layers')
parser.add_argument('--depth', type=int, default=6,
                    help='num of generator layers')
parser.add_argument('--sigma', type=float, default=0.5)
parser.add_argument('--learning_rate', type=float, default=1e-3)
parser.add_argument('--optimizer', type=str, default='adam',
                    choices=['adagrad', 'adam', 'gradientdescent',
                             'rmsprop'])
parser.add_argument('--total_num_runs', type=int, default=200101)
parser.add_argument('--save_iter', type=int, default=500)
parser.add_argument('--testing', type=int, default=0, choices=[0, 1])

args = parser.parse_args()
data_num = args.data_num
gen_num = args.gen_num
z_dim = args.z_dim
width = args.width
depth = args.depth
sigma = args.sigma
learning_rate = args.learning_rate
optimizer = args.optimizer
total_num_runs = args.total_num_runs
save_iter = args.save_iter
testing = args.testing
out_dim = 1
activation = tf.nn.elu
save_tag = 'dn{}_zd{}_w{}_d{}_lr{}_op_{}'.format(data_num, z_dim, width, depth,
                                                 learning_rate, optimizer)

# Set up true, training data.
#data_thinned = np.concatenate(
#    (np.random.normal(0, 0.5, 750), np.random.normal(2, 0.5, 250)))
def prob_of_thinning(x):
    return 0.5 / (1 + np.exp(10 * (x - 1))) + 0.5
data_unthinned = np.concatenate(
    (np.random.normal(0, 0.5, 500), np.random.normal(2, 0.5, 500)))
data_thinned = np.array([candidate for candidate in data_unthinned if
    np.random.binomial(1, prob_of_thinning(candidate))])

data = data_thinned
data_num = len(data)
gen_num = int(0.75 * data_num)
#data = np.concatenate(([0]*100, [3]*100, [4]*100)); data_num = len(data)  # TEST


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


# Build model.
x = tf.placeholder(tf.float64, [data_num, 1], name='x')
z = tf.placeholder(tf.float64, [gen_num, z_dim], name='z')

g = generator(z, width=width, depth=depth, activation=activation,
              out_dim=out_dim)
v = tf.concat([x, g], 0)
VVT = tf.matmul(v, tf.transpose(v))
sqs = tf.reshape(tf.diag_part(VVT), [-1, 1])
sqs_tiled_horiz = tf.tile(sqs, tf.transpose(sqs).get_shape())
exp_object = sqs_tiled_horiz - 2 * VVT + tf.transpose(sqs_tiled_horiz)
K = tf.exp(-0.5 * (1 / sigma) * exp_object)
K_xx = K[:data_num, :data_num]
K_yy = K[data_num:, data_num:]
K_xy = K[:data_num, data_num:]
K_xx_upper = (tf.matrix_band_part(K_xx, 0, -1) - 
              tf.matrix_band_part(K_xx, 0, 0))
K_yy_upper = (tf.matrix_band_part(K_yy, 0, -1) - 
              tf.matrix_band_part(K_yy, 0, 0))
num_combos_xx = data_num * (data_num - 1) / 2
num_combos_yy = gen_num * (gen_num - 1) / 2
mmd = (tf.reduce_sum(K_xx_upper) / num_combos_xx +
       tf.reduce_sum(K_yy_upper) / num_combos_yy -
       2 * tf.reduce_sum(K_xy) / (data_num * gen_num))
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
OPTION = 1
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


# Train.
init_op = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init_op)
print args
start_time = time()
for i in range(total_num_runs):
    sess.run(g_optim,
             feed_dict={
                 z: get_random_z(gen_num, z_dim),
                 #x: np.random.choice(data, (data_num, 1))})
                 x: np.reshape(data, [-1, 1])})  # TEST

    if i % save_iter == 0:
        """
        # TEST
        v_, k_, kyy_, kyyu_, mmd_ = sess.run([v, K, K_yy, K_yy_upper, mmd], feed_dict={
            z: get_random_z(gen_num, z_dim),  # TEST
            x: np.reshape(data, [-1, 1])})  # TEST
        """

        z_sample = get_random_z(gen_num, z_dim)
        #x_sample = np.random.choice(data, (data_num, 1))
        x_sample = np.reshape(data, [-1, 1])  # TEST
        mmd_out, g_out = sess.run(
            [mmd, g], feed_dict={
                z: z_sample,
                x: x_sample})
        np.save('sample_z', z_sample)
        np.save('sample_x', x_sample)
        np.save('sample_g', g_out)

        print '\niter:{}, mmd:{}'.format(i, mmd_out)
        print '  min:{}, max:{}'.format(min(g_out), max(g_out))

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

        if i > 0:
            elapsed_time = time() - start_time
            time_per_iter = elapsed_time / i
            total_est = elapsed_time / i * total_num_runs
            m, s = divmod(total_est, 60)
            h, m = divmod(m, 60)
            total_est_str = '{:.0f}:{:02.0f}:{:02.0f}'.format(h, m, s)
            print ('\n  Time (s). Elapsed: {:.2f}, Avg/iter: {:.4f},'
                   ' Total est.: {}').format(elapsed_time, time_per_iter,
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

os.system('python eval_samples.py; echo $PWD | mutt momod@utexas.edu -s "1d_mmd_gan_result_plot.png" -a "result_plot.png"')
print 'Emailed result_plot.png'
