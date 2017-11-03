import argparse
import math
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 12})
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import matplotlib
matplotlib.rc('xtick', labelsize=12) 
matplotlib.rc('ytick', labelsize=12)
import numpy as np
import os
import pdb
import sys


def str2bool(v):
        return v.lower() in ('true', '1')
parser = argparse.ArgumentParser()
parser.add_argument('--num_data', type=int, default=10000)
parser.add_argument('--chunk_size', type=int, default=200)
parser.add_argument('--proposals_per_chunk', type=int, default=10)
parser.add_argument('--lr', type=float, default=1.)
parser.add_argument('--sigma', type=float, default=1.)
parser.add_argument('--max_iter', type=int, default=501)
parser.add_argument('--save_iter', type=int, default=500)
parser.add_argument('--email', type=str2bool, default=False)
parser.add_argument('--data_source', type=str, default='correlated',
    choices=['gaussian', 'file', 'correlated'])
parser.add_argument('--corr', type=float, default=0.6, help='On [-1, 1]')
parser.add_argument('--plot', type=str2bool, default=False)
parser.add_argument('--test', type=str2bool, default=False)
args = parser.parse_args()
num_data = args.num_data
chunk_size = args.chunk_size
proposals_per_chunk = args.proposals_per_chunk
lr = args.lr
sigma = args.sigma
max_iter = args.max_iter
save_iter = args.save_iter
email = args.email
data_source = args.data_source
corr = args.corr
plot = args.plot
test = args.test


def generate_data(n, data_source):
    '''Generates data.

    Args:
      n: Number of data candidates.
      data_source: String flag for which data_source to use.

    Returns:
      data: Numpy array of points.
    '''
    if data_source == 'gaussian':
        two_modes = 1
        if two_modes:
            n_c2 = n/2
            n_c1 = n - n_c2
            data = np.concatenate((np.random.normal(0, 1, n_c1),
                                   np.random.normal(6, 1, n_c2)))
        else:
            data = np.random.normal(0, 0.5, n)
        return data
    elif data_source == 'file':
        with open('samp.dat') as f:
            data = [line.split() for line in f]
            data = np.array([float(d[1]) for d in data if len(d) == 2])
            return data[:2000]
    else:  # Generate correlated data.
        data = np.zeros(n)
        data[0] = np.random.normal()
        for i in xrange(1, n):
            data[i] = (data[i - 1] * corr +
                np.sqrt(1 - corr**2) * np.random.normal())

        plt.subplot(211);
        plt.plot(data)
        plt.subplot(212);
        plt.hist(data, 30)
        plt.savefig('orig_data.png'); plt.close()

        if test:
            num_proposals = num_data / chunk_size * proposals_per_chunk
            x_bars = []
            for i in range(1000):
                data = np.zeros(n)
                data[0] = np.random.normal()
                for i in xrange(1, n):
                    data[i] = (data[i - 1] * corr +
                        np.sqrt(1 - corr**2) * np.random.normal())
                x_bars.append(np.mean(
                    np.random.choice(data, num_proposals, replace=False)))
            print 'num proposals = {}'.format(num_proposals) 
            print '1/{} = {}'.format(num_proposals, 1./num_proposals)
            print 'var(x_bars) = {}'.format(np.var(x_bars))
            sys.exit('exited on test within generate_data')
        return data


def energy(data, gen, sigma=1.):
    '''Computes abbreviated energy statistic between two point sets.
    
    The smaller the value, the closer the sets.

    Args:
      data: 1D numpy array of any length, e.g. 100.
      gen: 1D numpy array of any length, e.g. 10.
      sigma: Float, kernel lengthscale.

    Returns:
      e: Scalar, the energy between the sets.
      mmd: Scalar, the mmd between the sets.
      gradients_e: Numpy array of energy gradients for each proposal point.
      gradients_mmd: Numpy array of mmdgradients for each proposal point.
    '''
    x = list(data)
    y = list(gen)
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
    sigma = sigma 
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

    #return e, mmd, np.array(gradients_e), np.array(gradients_mmd)
    return e, mmd, np.array(gradients_e), np.array(gradients_mmd)


def optimize(data, gen, n=5000, learning_rate=1e-2, dist='mmd',
             sigma=1.):
    '''Runs alternating optimizations, n times through proposal points.

    Args:
      data: 1D numpy array of any length, e.g. 100.
      gen: 1D numpy array of any length, e.g. 10.
      n: Scalar, number of times to loop through updates for all vars.
      learning_rate: Scalar, amount to move point with each gradient update.
      dist: String, distance measure ['e', 'mmd'].
      sigma: Float, kernel lengthscale.

    Returns:
      gens: 2D numpy array of trace of generated proposal points.
    '''
    e, mmd, _, _ = energy(data, gen, sigma=sigma)
    gens = np.zeros((n, len(gen)))
    #print '\nit0: gen:{}, e: {:.8f}, mmd: {:.8f}'.format(np.sort(gen), e, mmd)
    print '\n  it0: e: {:.8f}, mmd: {:.8f}'.format(e, mmd)
    gens[0, :] = gen
    for it in range(1, n):
        _, mmd_d_g, _, grads_mmd_d_g = energy(data, gen, sigma=sigma) 
        gen -= learning_rate * grads_mmd_d_g
        gens[it, :] = gen

        if it % save_iter == 0 or it == (n - 1):
            print '\n  it{}: mmd_d_g: {:.6f}'.format(it, mmd_d_g)
            #print np.sort(gen)

            print '\n  P: min,med,mean,max: {:4f},{:3f},{:3f},{:3f}'.format(
                np.min(partition), np.median(partition), np.mean(partition),
                np.max(partition))
            g = gens[:it]
            print '  Gens: min,med,mean,max: {:3f},{:3f},{:3f},{:3f}'.format(
                np.min(g), np.median(g), np.mean(g), np.max(g))
            if plot:
                plot_results(partition, q, q_orig, g, save_tag)

    return gens


def plot_results(p, q, q_orig, gens_out, save_tag):
    # Plot results.
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(10,6))
    plt.suptitle('P_m, Q_n, m={}, n={}'.format(len(p), len(q)))
    ax1.plot(gens_out, 'k-', alpha=0.3)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Value of proposals Q_n')
    ax1.set_title('Proposals, Q_n')
    ax1.set_ylim([-4,4])
    ax2.hist(p, bins=30, color='green', alpha=0.3, normed=True,
             orientation='horizontal', label='data')
    ax2.hist(gens_out[-1, :], bins=30, color='black', alpha=0.3, normed=True,
             orientation='horizontal', label='proposals')
    ax2.set_title('Data, P_m')
    ax2.legend()
    ax2.set_ylim([-4,4])
    plt.savefig('particles_{}.png'.format(save_tag))
    plt.close()

    # Email resulting plots.
    if email:
        os.system(
            ('echo $PWD -- {} | mutt momod@utexas.edu -s '
             '"optimal_representation particles" -a "particles_{}.png"').format(
                save_tag, save_tag))
        print '  Emailed particles.png'


# BEGIN MAIN RUN.
m = 1
x_bars = []
for i in range(m):
    print '\n\nEXPERIMENT {}\n'.format(i)
    p_orig = generate_data(num_data, data_source)
    num_data = len(p_orig)
    p = np.random.permutation(p_orig)
    partitions = [p[i: i + chunk_size] for i in xrange(0, len(p), chunk_size)]

    save_tag = 'data{}_nd{}_chunk{}_ppc{}_it{}_lr{}_sig{}'.format(
        data_source, num_data, chunk_size, proposals_per_chunk, max_iter, lr,
        sigma)
    print '  Config: {}'.format(save_tag)

    all_proposals = np.array([])

    for j, partition in enumerate(partitions):
        print '\nWorking on partition {}'.format(j)
        q_orig = np.linspace(min(partition), max(partition),
            proposals_per_chunk)
        #q_orig = np.random.uniform(min(partition), max(partition),
        #    proposals_per_chunk)
        q = list(q_orig)

        gens_out = optimize(partition, q, n=max_iter, learning_rate=lr,
            sigma=sigma)
        if plot:
            plot_results(partition, q, q_orig, gens_out, save_tag)

        all_proposals = np.concatenate(
            (all_proposals, gens_out[-1, :]))

    # Plot all generated proposals together.
    plt.hist(all_proposals, 30);
    plt.title('All proposals, n={}'.format(len(all_proposals)))
    plt.savefig('all_proposals.png'); plt.close()
    plt.close()

    x_bar = np.mean(all_proposals)
    x_bars.append(x_bar)
    with open('x_bars.txt', 'a') as f:
        f.write('\n{}'.format(x_bar))

# Compute variance of x_bars.
np.save('all_proposals.npy', all_proposals)
np.save('p_orig.npy', p_orig)
num_proposals_generated = num_data / chunk_size * proposals_per_chunk
print 'n = {}'.format(num_proposals_generated)
print '1/n = {}'.format(1. / num_proposals_generated)
print 'var(x_bars) = {}'.format(np.var(x_bars))
print 'x_bar = {}'.format(x_bar)
