import pdb
import numpy as np
import tensorflow as tf
layers = tf.layers

# Set up true, standard normal data.
data_num = 100
data = np.random.randn(data_num)

def get_random_z(gen_num, z_dim):
    return np.random.uniform(size=[gen_num, z_dim],
            low=-1.0, high=1.0)


# Set up generator.
width = 3
depth = 3
activation = tf.nn.elu
out_dim = 1
def generator(z, width=3, depth=3, activation=tf.nn.elu, out_dim=1, reuse=False):
    with tf.variable_scope('generator', reuse=reuse) as vs:
        x = layers.dense(z, width, activation=activation)

        for idx in range(depth - 1):
            x = layers.dense(x, width, activation=activation)

        out = layers.dense(x, out_dim, activation=None)
    return out


# Build model.
z_dim = 5
x = data
z = tf.placeholder(tf.float64, [None, z_dim], name='z')
g = generator(z)
#c = tf.concat([x, g], 0)
#C = tf.matmul(c, tf.transpose(c))
#sqs_tiled_horiz = tf.reshape(tf.diag_part(C), [-1, 1]) 
delta = x - g
norm_sq = tf.reduce_mean(tf.reduce_sum(delta[:-1] * delta[1:], 1))
g_vars = [var for var in tf.global_variables() if 'generator' in var.name]
g_optim = tf.train.AdagradOptimizer(1e-5).minimize(norm_sq, var_list=g_vars)


# Train.
init_op = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init_op)
for i in range(100000):
    sess.run(g_optim, feed_dict={z: get_random_z(data_num, z_dim)})
    if i % 1000 == 0:
        ns, z_out = sess.run([norm_sq, z], feed_dict={z: get_random_z(data_num, z_dim)})
        #ns, z, g, c, x = sess.run([norm_sq, z, g, c, x], feed_dict={z: get_random_z(data_num, z_dim)})
        print 'iter:{} ns = {}'.format(i, ns)



def mmd(x, y, estimation='full'):
    """Compute Maximum Mean Discrepancy (MMD) between two samples.
    Computes mmd between two nxd Numpy arrays, representing n samples of
    dimension d. The Gaussian Radial Basis Function is used as the kernel
    function.

    Inputs:
      x: Numpy array of n samples.
      y: Numpy array of n samples.

    Outputs:
      mmd: Scalar representing MMD.
    """
    total_mmd1 = 0 
    total_mmd2 = 0 
    total_mmd3 = 0 
    if estimation == 'sampling':
        sampling_n = (x.shape[0]**2) / 10  # Number of samples to estimate E[k(x,y)]. 
        for i in range(sampling_n):
            ind_x = np.random.randint(x.shape[0], size=2)  # Get two sample indices.
            ind_y = np.random.randint(y.shape[0], size=2)
            x1 = x[ind_x[0]]
            x2 = x[ind_x[1]]
            y1 = y[ind_y[0]]
            y2 = y[ind_y[1]]
            total_mmd1 += kernel(x1, x2) 
            total_mmd2 += kernel(y1, y2) 
            total_mmd3 += kernel(x1, y1)
        mmd = total_mmd1/sampling_n + total_mmd2/sampling_n - 2 * total_mmd3/sampling_n
    else:
        n = x.shape[0]
        m = y.shape[0]
        assert n==m 
        # Exact x-x term.
        for i in range(n):
            for j in range(i+1, n):
                x1 = x[i]
                x2 = x[j]
                total_mmd1 += kernel(x1, x2) 
        # Exact y-y term.
        for i in range(m):
            for j in range(i+1, m):
                y1 = y[i]
                y2 = y[j]
                total_mmd2 += kernel(y1, y2) 
        # Exact x-y term.
        for i in range(n):
            for j in range(i+1, m):
                x3 = x[i]
                y3 = y[j]
                total_mmd3 += kernel(x3, y3) 
        n_combos = n * (n - 1) / 2
        mmd = total_mmd1/n_combos + total_mmd2/n_combos - 2 * total_mmd3/n_combos
    return mmd


def kernel(a, b):
    """Gaussian Radial Basis Function.
    Output is between 0 and 1.

    Inputs:
      a: A single Numpy array of dimension d.
      b: A single Numpy array of dimension d.

    Outputs:
      k: Kernel value.
    """
    sigma = 1  # Bandwidth of kernel.
    a = np.array(a)
    b = np.array(b)
    k = np.exp(-1 * sum(np.square(a - b)) / (2 * sigma**2)) 

    return k
