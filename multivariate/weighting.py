import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import pdb
from numpy.linalg import norm
from scipy.spatial.distance import pdist


parser = argparse.ArgumentParser()
parser.add_argument('--snap', type=int, default=0, choices=[0, 1])
parser.add_argument('--data_file', type=str, default='gp_data.txt')
args = parser.parse_args()
snap = args.snap
data_file = args.data_file


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


# Load data and support points.
data = np.loadtxt(open(data_file, 'rb'), delimiter=' ')
support_points = np.load('g_out.npy')

assert data.shape[1] == support_points.shape[1], \
    'dimension of data and support points.g must match'

# Get datapoints (the subset) closest to support points.
if snap:
    subset = np.zeros(support_points.shape)
    for index, sp in enumerate(support_points):
        _, nearest_neighbor, _ = nearest(sp, data)
        subset[index] = nearest_neighbor
    np.save('subset.npy', subset)
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
subset_weights = np.zeros(n_subset) 
for i in range(n_subset):
    min_non_diag_distance = sorted(pairwise_distances[i])[1]
    subset_weights[i] = min_non_diag_distance
subset_weights = subset_weights / max(subset_weights)
np.save('weights_subset.npy', subset_weights)

# Compute data weights.
data_weights = np.zeros(data.shape[0])
for index, dp in enumerate(data):
    nearest_dist, _, nearest_index = nearest(dp, subset) 
    data_weights[index] = subset_weights[nearest_index] / (nearest_dist + 1e-6)
data_weights = data_weights / max(data_weights)
np.save('weights_data.npy', data_weights)


# PLOTTING: Plot data, support points, and subset.
fig, ax = plt.subplots()
ax.scatter(*zip(*data), color='gray', alpha=0.05, label='data')
if snap:
    ax.scatter(*zip(*support_points), color='green', alpha=0.3, label='support')
    ax.scatter(*zip(*subset), color='red', alpha=0.3, label='subset', s=75*subset_weights)
    ax.legend()
    plt.savefig('plots/plot_data_support_subset.png')
else:
    ax.scatter(*zip(*support_points), color='green', alpha=0.3, label='support', s=75*subset_weights)
    ax.legend()
    plt.savefig('plots/plot_data_subset.png')

fig, ax = plt.subplots()
ax.plot(subset_weights, color='gray', label='subset_weights')
ax.legend()
plt.savefig('plots/weights_subset.png')

fig, ax = plt.subplots()
ax.plot(data_weights, color='gray', label='data_weights')
ax.legend()
plt.savefig('plots/weights_data.png')

os.system('echo $PWD | mutt momod@utexas.edu -s "gp_data" -a "gp_data.txt" '
          '-a "g_out.npy" -a "plots/plot_data_subset.png" -a '
          '"plots/plot_data_support_subset.png" -a "plots/weights_data.png" -a '
          '"plots/weights_subset.png"')
print('Emailed results to momod@utexas.edu.')
