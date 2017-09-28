import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 25})
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import matplotlib
matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20)
import numpy as np
import pdb

# Set up true data, 100 points.
#p = np.random.randn(10)

# Initialize 10 proposal points as a uniform spread over true points.
#q = np.linspace(min(p), max(p), 3)

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
    #between_group_term = float(np.sum(energy_yx)) / gen_num / data_num
    #within_proposals_term = float(np.sum(energy_yy)) / gen_num / gen_num
    #e = 2. * between_group_term - within_proposals_term
    #print e
    e = (2. / gen_num / data_num * np.sum(energy_yx) -
         1. / gen_num / gen_num * np.sum(energy_yy))

    # Also compute MMD.
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


def optimize(data, gen, n=5000, learning_rate=1e-4, dist='mmd'):
    """Runs alternating optimizations, n times through proposal points.

    Args:
      data: 1D numpy array of any length, e.g. 100.
      gen: 1D numpy array of any length, e.g. 10.
      n: Scalar, number of times to loop through updates for all vars.
      learning_rate: Scalar, amount to move point with each gradient update.
      dist: String, distance measure ['e', 'mmd'].

    Returns:
      gen: 1D numpy array of updated proposal points.
    """
    e, mmd, _, _ = energy(data, gen)
    gens = np.array(gen)
    energies = np.array([e])
    mmds = np.array([mmd])
    proposal_indices = range(len(gen))
    for it in range(n):
        for index in proposal_indices:
        #for index in [0]:
            #print 'INDEX-{}'.format(index)
            e_, mmd_, grads_e_, grads_mmd_ = energy(data, gen) 
            #print 'BEFORE'
            #print 'gen: {}, mmd: {}'.format(gen, mmd_)
            #print 'grads_mmd: {}'.format(grads_mmd_)
            if dist == 'e':
                gen[index] -= learning_rate * grads_e_[index] 
            else:
                gen[index] -= learning_rate * grads_mmd_[index] 
            e, mmd, grads_e, grads_mmd = energy(data, gen) 
            #print 'AFTER'
            #print 'gen: {}, mmd: {}'.format(gen, mmd)
            #print 'Change in mmd: {}'.format(mmd - mmd_)

        if it % 5000 == 0:
            print 'it{}: gen:{}, e: {:.8f}, mmd: {:.8f}'.format(it, gen, e, mmd)
        gens = np.vstack((gens, gen))
        energies = np.vstack((energies, e))
        mmds = np.vstack((mmds, mmd))

    plt.plot(gens[:, 0], gens[:, 1], color='red', label='gen')
    plt.scatter(gens[0, 0], gens[0, 1], color='red', s=50)
    plt.scatter(gens[-1, 0], gens[-1, 1], color='red', marker='x', s=50)
    

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


def main():
    p = np.array([0, 3, 4])
    #p = np.random.randn(10)
    #q = np.linspace(min(p), max(p), 2)
    q = [0, 0.1]
    e, mmd, grads_e, grads_mmd = energy(p, q)
    print 'e: {:.4f}, mmd: {:.4f}, grads_e: {}, grads_mmd: {}'.format(e, mmd,
                                                                      grads_e,
                                                                      grads_mmd)
    # Do some basic tests over the grid, and near {0, 2}.
    #test_0_2()
    #test_2d_grid()

    # Make heatmap.
    plt.figure(figsize=(10, 10)) 
    grid_gran = 101
    q1 = np.linspace(-3, 5, grid_gran)
    q2 = np.linspace(5, -3, grid_gran)
    energies = np.zeros([grid_gran, grid_gran])
    mmds = np.zeros([grid_gran, grid_gran])
    for i, q1_i in enumerate(q1):
        for j, q2_j in enumerate(q2):
            _, mmd, _, _ = energy(p, [q1_i, q2_j])
            mmds[i, j] = mmd

    plt.imshow(mmds, interpolation='nearest', aspect='equal',
               extent=[q1.min(), q1.max(), q2.min(), q2.max()])
    plt.colorbar()
    optimize(p, [0.0, 0.1], n=50000, learning_rate=1e-3)
    optimize(p, [0.5, 3.5], n=50000, learning_rate=1e-3)
    optimize(p, [3.8, 3.0], n=50000, learning_rate=1e-3)
    plt.xlabel('p1')
    plt.ylabel('p2')
    plt.title('Q = {0, 1, 2}, P = {p1, p2}, Distance = MMD\'', fontsize=16)
    plt.savefig('energy_utils_optimize_results.png')
    plt.close()

