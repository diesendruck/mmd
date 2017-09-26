import matplotlib
matplotlib.use('Agg')
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

    # Also compute MMD.
    vvt = np.matmul(v_vert, np.transpose(v_vert))
    sqs = np.reshape(np.diag(vvt), [-1, 1])
    sqs_tiled_horiz = np.tile(sqs, (1, len(vvt)))
    exp_object = sqs_tiled_horiz - 2 * vvt + np.transpose(sqs_tiled_horiz)
    sigma = 1.
    K = np.exp(-0.5 / sigma * exp_object)
    K_xx = K[:data_num, :data_num]
    K_yy = K[data_num:, data_num:]
    K_xy = K[:data_num, data_num:]
    K_xx_upper = np.triu(K_xx, 1)
    K_yy_upper = np.triu(K_yy, 1)
    num_combos_xx = data_num * (data_num - 1) / 2
    num_combos_yy = gen_num * (gen_num - 1) / 2
    mmd = (np.sum(K_xx_upper) / num_combos_xx +
           np.sum(K_yy_upper) / num_combos_yy -
           2 * np.sum(K_xy) / (data_num * gen_num))
    
    # Compute energy gradients.
    signed = np.sign(pairwise)
    signed_yx = signed[data_num:, :data_num]
    signed_yy = signed[data_num:, data_num:]
    gradients_e = []
    for i in range(gen_num):
        grad_yi = (2. / gen_num / data_num * sum(signed_yx[i]) - 
                   1. / gen_num / gen_num * sum(signed_yy[i]))
        gradients_e.append(grad_yi)
    
    # Compute MMD gradients.
    mmd_grad = K * (-1 / sigma * pairwise)
    mmd_grad_yx = mmd_grad[data_num:, :data_num] 
    mmd_grad_yy = mmd_grad[data_num:, data_num:] 
    gradients_mmd = []
    for i in range(gen_num):
        grad_yi = (1. / gen_num / gen_num * sum(mmd_grad_yy[i]) -
                   2. / gen_num / data_num * sum(mmd_grad_yx[i]))
                   
        gradients_mmd.append(grad_yi)

    return e, mmd, np.array(gradients_e), np.array(gradients_mmd)


def optimize(data, gen, n, learning_rate, joint=False, dist='mmd'):
    """Runs alternating optimizations, n times through proposal points.

    Args:
      data: 1D numpy array of any length, e.g. 100.
      gen: 1D numpy array of any length, e.g. 10.
      n: Scalar, number of times to loop through updates for all vars.
      learning_rate: Scalar, amount to move point with each gradient update.
      joint: Boolean, whether to move proposals jointly or alternating-ly.
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
        if not joint:
            for index in proposal_indices:
                print 'INDEX-{}'.format(index)
                e, mmd, grads_e, grads_mmd = energy(data, gen) 
                print 'BEFORE'
                print 'gen: {}, mmd: {}'.format(gen, mmd)
                print 'grads_mmd: {}'.format(grads_mmd)
                if dist == 'e':
                    gen[index] -= learning_rate * grads_e[index] 
                else:
                    gen[index] -= learning_rate * grads_mmd[index] 
                e, mmd, grads_e, grads_mmd = energy(data, gen) 
                print 'AFTER'
                print 'gen: {}, mmd: {}'.format(gen, mmd)

        else:
            e, mmd, grads_e, grads_mmd = energy(data, gen) 
            if dist == 'e':
                gen -= learning_rate * grads_e 
            else:
                gen -= learning_rate * grads_mmd
        pdb.set_trace()
        if it % 10000 == 0:
            print 'it{}: gen:{}, e: {:.4f}, mmd: {:.4f}'.format(it, gen, e, mmd)
        gens = np.vstack((gens, gen))
        energies = np.vstack((energies, e))
        mmds = np.vstack((mmds, mmd))

    plt.subplot(211)
    plt.plot(gens[:, 0], label='gen0')
    plt.plot(gens[:, 1], label='gen1')
    plt.legend()
    plt.title('gens')
    plt.subplot(212)
    plt.plot(energies, label='e')
    plt.plot(mmds, label='mmd')
    plt.legend()
    plt.title('Energy, MMD')
    plt.savefig('energy_utils_optimize_results_{}.png'.format(joint))
    plt.close()
    

def test_0_2(p):
    # Show energy statistic around optimal point {0, 2}.
    plt.figure(figsize=(20,8)) 
    q1 = np.linspace(-1, 1, 1000)
    e_test = []
    mmd_test = []
    for i in q1:
        e, mmd, _, _ = energy(p, [i, 2])
        e_test.append(e)
        mmd_test.append(mmd)
    plt.subplot(121)
    plt.plot(q1, e_test, label='e')
    plt.plot(q1, mmd_test, label='mmd')
    plt.legend()
    plt.title('E({p, 2}, {0, 1, 2})', fontsize=24)
    plt.xlabel('p', fontsize=24)
    plt.ylabel('Energy, MMD', fontsize=24)

    q2 = np.linspace(1, 3, 1000)
    e_test = []
    mmd_test = []
    for i in q2:
        e, mmd, _, _ = energy(p, [0, i])
        e_test.append(e)
        mmd_test.append(mmd)
    plt.subplot(122)
    plt.plot(q2, e_test, label='e')
    plt.plot(q2, mmd_test, label='mmd')
    plt.legend()
    plt.xlabel('p', fontsize=24)
    plt.ylabel('Energy, MMD', fontsize=24)
    plt.title('E({0, p}, {0, 1, 2})', fontsize=24)

    plt.subplots_adjust(hspace=0.5)
    plt.savefig('energy_utils_test_e_near_0_2.png')


def test_2d_grid(p):
    # Plot e and mmd over grid of {q1, q2} values.
    plt.figure(figsize=(20,8)) 
    grid_gran = 11
    q1 = np.linspace(-3, 6, grid_gran)
    q2 = np.linspace(6, -3, grid_gran)
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
    p = np.array([0, 1, 2])
    p = np.random.randn(10)
    #q = np.linspace(min(p), max(p), 2)
    q = [0.5, 1.5]
    e, mmd, grads_e, grads_mmd = energy(p, q)
    print 'e: {:.4f}, mmd: {:.4f}, grads_e: {}, grads_mmd: {}'.format(e, mmd,
                                                                      grads_e,
                                                                      grads_mmd)

    #test_0_2(p)
    test_2d_grid(p)

    #optimize(p, q, 300000, 1e-3, dist='mmd')


