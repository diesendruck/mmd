import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pdb
import sys
from time import time
import tensorflow as tf
layers = tf.layers
from scipy.stats import norm


# Config.
parser = argparse.ArgumentParser()
parser.add_argument('--data_num', type=int, default=700)
parser.add_argument('--z_dim', type=int, default=1)
parser.add_argument('--width', type=int, default=10,
                    help='width of generator layers')
parser.add_argument('--depth', type=int, default=20,
                    help='num of generator layers')
parser.add_argument('--learning_rate', type=float, default=1e-4)
parser.add_argument('--optimizer', type=str, default='adam',
                    choices=['adagrad', 'adam', 'gradientdescent',
                             'rmsprop'])
parser.add_argument('--total_num_runs', type=int, default=200101)
parser.add_argument('--save_iter', type=int, default=10000)

args = parser.parse_args()
data_num = args.data_num
z_dim = args.z_dim
width = args.width
depth = args.depth
learning_rate = args.learning_rate
optimizer = args.optimizer
total_num_runs = args.total_num_runs
save_iter = args.save_iter
out_dim = 1
activation = tf.nn.elu
save_tag = 'dn{}_zd{}_w{}_d{}_lr{}_op_{}'.format(data_num, z_dim, width, depth,
                                                 learning_rate, optimizer)

def get_random_z(gen_num, z_dim):
    """Generates 2d array of noise input data."""
    return np.random.uniform(size=[gen_num, z_dim],
                             low=-1.0, high=1.0)
    #return np.random.gamma(5, size=[gen_num, z_dim])
    #return np.random.standard_t(2, size=[gen_num, z_dim])


# Set up generator.
def generator(z, width=3, depth=3, activation=tf.nn.elu, out_dim=1,
              reuse=False):
    """Generates output, given noise input."""
    with tf.variable_scope('generator', reuse=reuse):
        x = layers.dense(z, width, activation=activation)

        for idx in range(depth - 1):
            x = layers.dense(x, width, activation=activation)

        out = layers.dense(x, out_dim, activation=None)
    return out


# GOAL: Draw 100 samples from standard normal. Call this the 'true' empirical
# distribution. Then propose 10 points that minimize the energy statistic
# between the 10 proposed points and the original 100 points.

# Set up true data, 100 points.
p = np.random.randn(10)

# Initialize 10 proposal points as a uniform spread over true points.
q = np.linspace(min(p), max(p), 3)

# Define (abbreviated) energy statistic.
def energy(data, prop):
    """Computes abbreviated energy statistic between two point sets.
    
    The smaller the value, the closer the sets.

    Args:
      data: 1D numpy array of any length, e.g. 100.
      prop: 1D numpy array of any length, e.g. 10.

    Returns:
      e: Scalar, the energy between the sets.
      gradients: Numpy array of gradients for each proposal point. 
    """
    x = data
    y = prop
    v = np.concatenate((x, y), 0)
    v_vert = v.reshape(-1, 1)
    v_tiled = np.tile(v_vert, (1, len(v)))
    pairwise = v_tiled - np.transpose(v_tiled)
    energy_xx = abs(pairwise[:len(x), :len(x)])
    energy_yx = abs(pairwise[len(x):, :len(x)])
    energy_yy = abs(pairwise[len(x):, len(x):])
    between_group_term = float(np.sum(energy_yx)) / len(y) / len(x)
    within_proposals_term = float(np.sum(energy_yy)) / len(y) / len(y) 
    e = 2 * between_group_term - within_proposals_term
    
    # Compute gradients.
    signed = np.sign(pairwise)
    signed_yx = signed[len(x):, :len(x)]
    signed_yy = signed[len(x):, len(x):]
    gradients = []
    for i in range(len(y)):
        grad_yi = 2 * sum(signed_yx[i]) - sum(signed_yy[i])
        gradients.append(grad_yi)
    
    return e, np.array(gradients)

def optimize(data, prop, n, learning_rate, joint=False):
    """Runs alternating optimizations, n times through proposal points.

    Args:
      data: 1D numpy array of any length, e.g. 100.
      prop: 1D numpy array of any length, e.g. 10.
      n: Scalar, number of times to loop through updates for all vars.
      learning_rate: Scalar, amount to move point with each gradient update.
      joint: Boolean, whether to move proposals jointly or alternating-ly.

    Returns:
      prop: 1D numpy array of updated proposal points.
    """
    e, _ = energy(data, prop)
    props = np.array(prop)
    energies = np.array([e])
    proposal_indices = range(len(prop))
    for run in range(n):
        if not joint:
            for index in proposal_indices:
                e, grads = energy(data, prop) 
                prop[index] -= learning_rate * grads[index] 
        else:
            e, grads = energy(data, prop) 
            prop -= learning_rate * grads 
        #print '{}, energy: {}'.format(prop, e)
        props = np.vstack((props, prop))
        energies = np.vstack((energies, e))

    plt.subplot(211)
    plt.plot(props)
    plt.subplot(212)
    plt.plot(energies)
    plt.savefig('props_joint_{}.png'.format(joint))
    plt.close()
optimize(p, q, 500, 0.001, joint=1)
optimize(p, q, 500, 0.001, joint=0)
sys.exit()



x = tf.placeholder(tf.float64, [data_num, 1], name='x')
z = tf.placeholder(tf.float64, [data_num, z_dim], name='z')
g = generator(z, width=width, depth=depth, activation=activation,
              out_dim=out_dim)
v = tf.concat([x, g], 0)
VVT = tf.matmul(v, tf.transpose(v))
sqs = tf.reshape(tf.diag_part(VVT), [-1, 1])
sqs_tiled_horiz = tf.tile(sqs, tf.transpose(sqs).get_shape())
exp_object = sqs_tiled_horiz - 2 * VVT + tf.transpose(sqs_tiled_horiz)
sigma = 1
K = tf.exp(-0.5 * (1 / sigma) * exp_object)
K_xx = K[:data_num, :data_num]
K_yy = K[data_num:, data_num:]
K_xy = K[:data_num, data_num:]
K_xx_upper = tf.matrix_band_part(K_xx, 0, -1)
K_yy_upper = tf.matrix_band_part(K_yy, 0, -1)
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
g_optim = opt.minimize(mmd, var_list=g_vars)
#gradients, variables = zip(*opt.compute_gradients(mmd))
#gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
#g_optim = opt.apply_gradients(zip(gradients, variables))

#gvs = opt(learning_rate).compute_gradients(mmd)
#capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
#g_optim = optimizer.apply_gradients(capped_gvs)

# Train.
init_op = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init_op)
print args
start_time = time()
for i in range(total_num_runs):
    sess.run(g_optim,
             feed_dict={
                 z: get_random_z(data_num, z_dim),
                 x: np.random.choice(data, (data_num, 1))})

    if i % save_iter == 100:
        z_sample = get_random_z(data_num, z_dim)
        x_sample = np.random.choice(data, (data_num, 1))
        mmd_out, g_out = sess.run(
            [mmd, g], feed_dict={
                z: z_sample,
                x: x_sample})
        np.save('sample_z', z_sample)
        np.save('sample_x', x_sample)
        np.save('sample_g', g_out)

        print '\niter:{} mmd = {}'.format(i, mmd_out)
        print 'min:{} max= {}'.format(min(g_out), max(g_out))
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
            print ('\nTime (s). Elapsed: {:.2f}, Avg/iter: {:.4f},'
                   ' Total est.: {}').format(elapsed_time, time_per_iter,
                                             total_est_str)
