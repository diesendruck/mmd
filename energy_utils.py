import matplotlib
matplotlib.use('Agg')
#matplotlib.rcParams.update({'font.size': 16})
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import matplotlib
matplotlib.rc('xtick', labelsize=12) 
matplotlib.rc('ytick', labelsize=12)
import numpy as np
import os
import pdb


# Define (abbreviated) energy statistic.
def energy(data, gen):
    """Computes abbreviated energy statistic between two point sets.
    
    The smaller the value, the closer the sets.

    Args:
      data: 1D numpy array of any length, e.g. 100.
      gen: 1D numpy array of any length, e.g. 10.

    Returns:
      e: Scalar, the energy between the sets.
      mmd: Scalar, the mmd between the sets.
      gradients_e: Numpy array of energy gradients for each proposal point.
      gradients_mmd: Numpy array of mmdgradients for each proposal point.
    """
    x = data
    y = gen
    data_num = len(x)
    gen_num = len(y)
    num_combos_yy = gen_num * (gen_num - 1) / 2.

    # Compute energy.
    v = np.concatenate((x, y), 0)
    v_vert = v.reshape(-1, 1)
    v_tiled = np.tile(v_vert, (1, len(v)))
    pairwise_difs = v_tiled - np.transpose(v_tiled)
    energy = abs(pairwise_difs)
    energy_yx = energy[data_num:, :data_num]
    energy_yy = energy[data_num:, data_num:]
    e = (2. / gen_num / data_num * np.sum(energy_yx) -
         1. / gen_num / gen_num * np.sum(energy_yy))

    # Compute MMD.
    pairwise_prods = np.matmul(v_vert, np.transpose(v_vert))
    sqs_vert = np.reshape(np.diag(pairwise_prods), [-1, 1])
    sqs_vert_tiled_horiz = np.tile(sqs_vert, (1, data_num + gen_num))
    exp_object = (sqs_vert_tiled_horiz - 2 * pairwise_prods +
                  np.transpose(sqs_vert_tiled_horiz))
    sigma = 1.
    K = np.exp(-0.5 / sigma * exp_object)
    K_yy = K[data_num:, data_num:]
    K_xy = K[:data_num, data_num:]
    K_yy_upper = np.triu(K_yy, 1)
    mmd = (1. / num_combos_yy * np.sum(K_yy_upper) -
           2. / data_num / gen_num * np.sum(K_xy))

    # Compute energy gradients.
    signed = np.sign(pairwise_difs)
    signed_yx = signed[data_num:, :data_num]
    signed_yy = signed[data_num:, data_num:]
    gradients_e = []
    for i in range(gen_num):
        grad_yi = (2. / gen_num / data_num * sum(signed_yx[i]) - 
                   1. / gen_num / gen_num * sum(signed_yy[i]))
        gradients_e.append(grad_yi)
    
    # Compute MMD gradients.
    mmd_grad = K * (-1. / sigma * pairwise_difs)
    mmd_grad_yx = mmd_grad[data_num:, :data_num] 
    mmd_grad_yy = mmd_grad[data_num:, data_num:] 
    mmd_grad_yy_upper = np.triu(mmd_grad_yy, 1)
    gradients_mmd = []
    for i in range(gen_num):
        grad_yi = (1. / num_combos_yy * 
                       (sum(mmd_grad_yy_upper[i]) -
                        sum(mmd_grad_yy_upper[:, i])) -
                   2. / gen_num / data_num * sum(mmd_grad_yx[i]))
                   
        gradients_mmd.append(grad_yi)

    return e, mmd, np.array(gradients_e), np.array(gradients_mmd)


def optimize(data, gen, n=5000, learning_rate=1e-4, dist='mmd', thin=False):
    """Runs alternating optimizations, n times through proposal points.

    Args:
      data: 1D numpy array of any length, e.g. 100.
      gen: 1D numpy array of any length, e.g. 10.
      n: Scalar, number of times to loop through updates for all vars.
      learning_rate: Scalar, amount to move point with each gradient update.
      dist: String, distance measure ['e', 'mmd'].
      thin: Boolean for whether to thin proposals.

    Returns:
      gen: 1D numpy array of updated proposal points.
    """

    '''
    if thin:
        # Sample from generator (accounting for thinning) until size matches data.
        g_thinned = np.array([])
        while len(g_thinned) < data_num:
            # Fetch a batch of candidates.
            g_candidates = sess.run(g, feed_dict={z: get_random_z(data_num, z_dim)})
            # Go through the batch, adding, until there are data_num candidates.
            for candidate in g_candidates:
                is_included = np.random.binomial(1, prob_of_keeping(candidate[0]))
                if is_included:
                    g_thinned = np.concatenate((g_thinned, candidate))
                    # Every time one is included, check if size matches.
                    if len(g_thinned) == data_num:
                        break
        gen = g_thinned.reshape(-1, 1)
    '''

    e, mmd, _, _ = energy(data, gen)
    gens = np.zeros((n, len(gen)))
    gens[0, :] = gen
    for it in range(1, n):
        e, mmd, grads_e, grads_mmd = energy(data, gen) 
        gen -= learning_rate * grads_mmd 

        if it % 100 == 0:
            print 'it{}: gen:{}, e: {:.8f}, mmd: {:.8f}'.format(it, gen, e, mmd)
        gens[it, :] = gen

    return gens
    

def test_0_2():
    p = np.array([0, 1, 2])
    # Show energy statistic around optimal point {0, 2}.
    plt.figure(figsize=(20,15)) 
    q1 = np.linspace(-1.5, 1.5, 1000)
    e_test = []
    mmd_test = []
    grad_e_test = []
    grad_mmd_test = []
    for i in q1:
        e, mmd, grad_e, grad_mmd = energy(p, [i, 2])
        e_test.append(e)
        mmd_test.append(mmd)
        grad_e_test.append(grad_e[0])
        grad_mmd_test.append(grad_mmd[0])
    plt.subplot(221)
    plt.axis('equal')
    plt.plot(q1, e_test, label='e1')
    plt.plot(q1, mmd_test, label='mmd1')
    plt.legend()
    plt.title('E({p, 2}, {0, 1, 2})', fontsize=24)
    plt.xlabel('p', fontsize=24)
    plt.ylabel('Energy, MMD', fontsize=24)
    plt.subplot(223)
    plt.axis('equal')
    plt.plot(q1, grad_e_test, label='grad_e1')
    plt.plot(q1, grad_mmd_test, label='grad_mmd1')
    plt.legend()
    plt.title('Grad(E({p, 2}, {0, 1, 2}))', fontsize=24)
    plt.xlabel('p', fontsize=24)
    plt.ylabel('Grad(Energy, MMD)', fontsize=24)
    

    q2 = np.linspace(0.5, 3.5, 1000)
    e_test = []
    mmd_test = []
    grad_e_test = []
    grad_mmd_test = []
    for i in q2:
        e, mmd, grad_e, grad_mmd = energy(p, [0, i])
        e_test.append(e)
        mmd_test.append(mmd)
        grad_e_test.append(grad_e[1])
        grad_mmd_test.append(grad_mmd[1])
    plt.subplot(222)
    plt.axis('equal')
    plt.plot(q2, e_test, label='e2')
    plt.plot(q2, mmd_test, label='mmd2')
    plt.legend()
    plt.xlabel('p', fontsize=24)
    plt.ylabel('Energy, MMD', fontsize=24)
    plt.title('E({0, p}, {0, 1, 2})', fontsize=24)
    plt.subplot(224)
    plt.axis('equal')
    plt.plot(q2, grad_e_test, label='grad_e2')
    plt.plot(q2, grad_mmd_test, label='grad_mmd2')
    plt.legend()
    plt.xlabel('p', fontsize=24)
    plt.ylabel('Grad(Energy, MMD)', fontsize=24)
    plt.title('Grad(E({0, p}, {0, 1, 2}))', fontsize=24)

    plt.subplots_adjust(hspace=0.5)
    plt.savefig('energy_utils_test_e_near_0_2.png')


def test_2d_grid():
    p = np.array([0, 1, 2])
    # Plot e and mmd over grid of {q1, q2} values.
    plt.figure(figsize=(20,8)) 
    grid_gran = 101
    q1 = np.linspace(-2, 4, grid_gran)
    q2 = np.linspace(4, -2, grid_gran)
    energies = np.zeros([grid_gran, grid_gran])
    mmds = np.zeros([grid_gran, grid_gran])
    for i, q1_i in enumerate(q1):
        for j, q2_j in enumerate(q2):
            e, mmd, _, _ = energy(p, [q1_i, q2_j])
            energies[i, j] = e
            mmds[i, j] = mmd

    plt.subplot(121)
    plt.imshow(energies, interpolation='nearest', aspect='equal',
               extent=[q1.min(), q1.max(), q2.min(), q2.max()])
    plt.title('Energies', fontsize=24)
    plt.colorbar()

    plt.subplot(122)
    plt.imshow(mmds, interpolation='nearest', aspect='equal',
               extent=[q1.min(), q1.max(), q2.min(), q2.max()])
    plt.title('MMDs', fontsize=24)
    plt.colorbar()

    plt.subplots_adjust(hspace=0.5)
    plt.savefig('energy_utils_test_grid_0_2.png')


def generate_data(n):
    """Generates true data, and applies thinning function.

    Args:
      n: Number of data candidates to start with, to then be thinned.

    Returns:
      data_unthinned: Unthinned numpy array of points.
      data: Thinned numpy array of points.
    """
    n_c2 = n/2
    n_c1 = n - n_c2
    data_unthinned = np.concatenate((np.random.normal(0, 0, n_c1),
        np.random.normal(2, 0, n_c2)))
    data = thin_data(data_unthinned)
    return data_unthinned, data


def thin_data(data_unthinned):
    """Thins data, accepting ~90% of Cluster 1, and ~10% Cluster 2.

    Args:
      data_unthinned: List of scalar values.

    Returns:
      thinned: Numpy array of values, thinned according to logistic function.
    """

    thinned = [candidate for candidate in data_unthinned if
               np.random.binomial(1, prob_of_keeping(candidate))]
    return thinned


def prob_of_keeping(x):
    """Logistic mapping to preferentially thin samples.
    
    Maps [-Inf, Inf] to [0.9, 0.1], centered at 2.

    Args:
      x: Scalar, point from original distribution.

    Returns:
      p: Probability of being thinned.
    """
    p = 0.6 / (1 + np.exp(10 * (x - 1))) + 0.4
    return p


def main():
    num_data = 40
    num_proposals = 80 
    n_iter = 50000
    lr = 1e-2
    thin = True

    p_thinned, p = generate_data(num_data)

    half_num_data = num_data / 2
    half_num_data_ = num_data - half_num_data
    p = np.concatenate((np.random.randn(half_num_data),
                        np.random.randn(half_num_data_) + 10))
    q_orig = np.linspace(min(p), max(p), num_proposals)
    q = list(q_orig)
    (low, high) = (np.floor(min(p)) - 1, np.ceil(max(p)) + 1)

    # Do some basic tests over the grid, and near {0, 2}.
    TEST = 0
    if TEST:
        test_0_2()
        test_2d_grid()

    if len(q) == 2:
        # Make heatmap.
        plt.figure(figsize=(12, 12)) 
        grid_gran = 201
        q1 = np.linspace(high, low, grid_gran)
        q2 = np.linspace(low, high, grid_gran)
        mmds = np.zeros([grid_gran, grid_gran], dtype=np.float64)
        for i, q1_i in enumerate(q1):
            for j, q2_j in enumerate(q2):
                _, mmd, _, _ = energy(p, [q1_i, q2_j])
                mmds[i, j] = mmd
        plt.imshow(mmds, interpolation='nearest', aspect='equal',
                   extent=[q1.min(), q1.max(), q2.min(), q2.max()])
        plt.colorbar()

        # Add optimization paths to heatmap.
        num_paths = 3
        for _ in range(num_paths):
            rand1 = np.random.uniform(low, high)
            rand2 = np.random.uniform(low, high)
            optimize(p, [rand1, rand2], n=n_iter, learning_rate=lr)
        plt.xlabel('q1')
        plt.ylabel('q2')
        plt.title('P={}, Q=[q1,q2], Dist=MMD\', it={}'.format(p, n_iter),
                  fontsize=16)
        plt.savefig('energy_utils_optimize_results.png')
        plt.close()

        # Email resulting plot.
        os.system(('echo $PWD | mutt momod@utexas.edu -s "energy_utils_test"'
                   ' -a "energy_utils_optimize_results.png"'))
        print 'Emailed energy_utils_optimize_results.png'

    else:
        gens_out = optimize(p, q, n=n_iter, learning_rate=lr, thin=thin)
        fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
        plt.suptitle('P_m ~ N(0,1), m={}, n={}'.format(len(p), len(q)))
        ax1.plot(gens_out, 'k-', alpha=0.3)
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Value of proposals Q_n')
        ax1.set_title('Proposals, Q_m')
        ax2.hist(p, bins=20, orientation='horizontal', color='blue', alpha=0.3)
        ax2.set_title('Data, P_n')
        plt.savefig('energy_utils_large.png')
        plt.close()

        # Compare to original data.
        offsets_before = []
        for prop in q_orig:
            dist_to_closest_datapoint = min(abs(prop - p))
            offsets_before.append(dist_to_closest_datapoint)
        last_gen = gens_out[-1, :]
        offsets_after = []
        for prop in last_gen:
            dist_to_closest_datapoint = min(abs(prop - p))
            offsets_after.append(dist_to_closest_datapoint)
        histogram_data = np.vstack([offsets_before, offsets_after]).T
        plt.hist(histogram_data, bins=30, alpha=0.3, label=['before', 'after'])
        plt.legend()
        plt.title('Offsets After Optimization. min:{:.6f}, max:{:.6f}'.format(
            min(offsets_after),
            max(offsets_after)))
        plt.savefig('offsets.png')

        # Email resulting plots.
        os.system(('echo $PWD | mutt momod@utexas.edu -s '
                   '"energy_utils_large" -a "energy_utils_large.png" '
                   '-a "offsets.png"'))
        print 'Emailed energy_utils_large.png, offsets.png'

        pdb.set_trace()

