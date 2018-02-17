import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import pdb
from numpy.linalg import norm


def nearest(target, arr):
    nearest_dist = 1e10
    nearest_neighbor = [0, 0]
    for candidate in arr:
        dist = norm(target - candidate)
        if dist < nearest_dist:
            nearest_dist = dist
            nearest_neighbor = candidate 
    return nearest_dist, np.array(nearest_neighbor)


# Load data and support points.
data = np.loadtxt(open('gp_data.txt', 'rb'), delimiter=' ')
support_points = np.load('g_out.npy')

assert data.shape[1] == support_points.shape[1], \
    'dimension of data and support points.g must match'

# Get datapoints (the subset) closest to support points.
subset = []
for sp in support_points:
    _, nearest_neighbor = nearest(sp, data)
    subset.append(nearest_neighbor)
np.save('subset.npy', np.array(subset))

# Plot data, support points, and subset.
fig, ax = plt.subplots()
ax.scatter(*zip(*data), color='gray', alpha=0.05, label='data')
ax.scatter(*zip(*support_points), color='green', alpha=0.3, label='support')
ax.scatter(*zip(*subset), color='red', alpha=0.3, label='subset')
ax.legend()
plt.savefig('plot_data_support_subset.png')

# Compute data weights.
c = 1
weights = np.zeros(data.shape[0])
for index, dp in enumerate(data):
    nearest_dist, _ = nearest(dp, subset) 
    weights[index] = c * 1.0 / (nearest_dist + 1e-6)
np.save('weights.npy', weights)
