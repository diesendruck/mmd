import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import pdb
from numpy.linalg import norm
from scipy.spatial.distance import pdist


def nearest(target, arr):
    nearest_dist = 1e10
    nearest_neighbor = [0, 0]
    nearest_index = 0
    for index, candidate in enumerate(arr):
        dist = norm(target - candidate)
        if dist < nearest_dist:
            nearest_dist = dist
            nearest_neighbor = candidate 
            nearest_index = index
    return nearest_dist, np.array(nearest_neighbor), nearest_index


def get_estimation_points(data_file='gp_data.txt', log_dir='logs_test',
        mode='coreset', support_points=None, email=False):
    # Load data and support points.
    data = np.loadtxt(open(data_file, 'rb'), delimiter=' ')
    if support_points is None:
        support_points = np.load(os.path.join(log_dir, 'g_out.npy'))
    assert data.shape[1] == support_points.shape[1], \
        'dimension of data and support points.g must match'

    # Get datapoints (the subset) closest to support points.
    if mode == 'coreset':
        subset = np.zeros(support_points.shape)
        for index, sp in enumerate(support_points):
            _, nearest_neighbor, _ = nearest(sp, data)
            subset[index] = nearest_neighbor
        np.save(os.path.join(log_dir, 'subset.npy'), subset)
    else:
        subset = support_points

    # Compute subset weights. First compute pairwise distances. Then find minimum
    # distance to nearest neighbor.
    n_subset = len(subset)
    pairwise_distances = np.zeros((n_subset, n_subset))
    for i in range(n_subset):
        for j in range(n_subset):
            distance = norm(subset[i] - subset[j]) 
            pairwise_distances[i][j] = distance
    weights_estimation_pts = np.zeros(n_subset) 
    for i in range(n_subset):
        min_non_diag_distance = sorted(pairwise_distances[i])[1]
        weights_estimation_pts[i] = min_non_diag_distance
    weights_estimation_pts = weights_estimation_pts / max(weights_estimation_pts)
    np.save(os.path.join(log_dir, 'weights_estimation_pts.npy'), weights_estimation_pts)

    # Compute data weights.
    weights_data = np.zeros(data.shape[0])
    for index, dp in enumerate(data):
        nearest_dist, _, nearest_index = nearest(dp, subset) 
        weights_data[index] = weights_estimation_pts[nearest_index] / (nearest_dist + 1e-6)
    weights_data = weights_data / max(weights_data)
    np.save(os.path.join(log_dir, 'weights_data.npy'), weights_data)


    # PLOTTING: Plot data, support points, and subset.
    fig, ax = plt.subplots()
    ax.scatter(*zip(*data), color='gray', alpha=0.05, label='data')
    if mode == 'coreset':
        ax.scatter(*zip(*support_points), color='green', alpha=0.3, label='support')
        ax.scatter(*zip(*subset), color='red', alpha=0.3, label='subset',
            s=75*weights_estimation_pts)
        ax.legend()
        plt.savefig('plots/plot_data_support_subset.png')
    else:
        ax.scatter(*zip(*support_points), color='green', alpha=0.3, label='support',
            s=75*weights_estimation_pts)
        ax.legend()
        plt.savefig('plots/plot_data_support.png')

    fig, ax = plt.subplots()
    ax.plot(weights_estimation_pts, color='gray', label='weights_estimation_pts')
    ax.legend()
    plt.savefig('plots/weights_estimation_pts.png')

    fig, ax = plt.subplots()
    ax.plot(weights_data, color='gray', label='weights_data')
    ax.legend()
    plt.savefig('plots/weights_data.png')
    plt.close('all')

    if email:
        if mode == 'coreset':
            os.system('echo $PWD{} | mutt momod@utexas.edu -s "gp_data" -a "gp_data.txt" '
                      '-a "plots/plot_data_support_subset.png" -a "plots/weights_data.png"'
                      ' -a "plots/weights_estimation_pts.png"'.format('  mode: '+str(mode)))
        else:
            os.system('echo $PWD{} | mutt momod@utexas.edu -s "gp_data" -a "gp_data.txt" '
                      '-a "plots/plot_data_support.png" '
                      '-a "plots/weights_data.png"'
                      ' -a "plots/weights_estimation_pts.png"'.format('  mode: '+str(mode)))
        print('Emailed results to momod@utexas.edu.')

    return support_points, subset, weights_estimation_pts, weights_data 


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='coreset',
        choices=['coreset', 'support'])
    parser.add_argument('--data_file', type=str, default='gp_data.txt')
    parser.add_argument('--tag', type=str, default='test')
    args = parser.parse_args()
    mode = args.mode
    data_file = args.data_file
    tag = args.tag
    log_dir = 'logs_{}'.format(tag)

    get_estimation_points(data_file, log_dir, mode)


