import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import pdb
from numpy.linalg import norm


data = np.loadtxt(open('gp_data.txt', 'rb'), delimiter=' ')
support_points = np.load('g_out.npy')

assert data.shape[1] == support_points.shape[1], \
    'dimension of data and support points.g must match'

proper_subset = []
for sp in support_points:
    nearest_dist = 1e10
    nearest_neighbor = [0, 0]
    for dp in data:
        dist = norm(sp - dp)
        if dist < nearest_dist:
            nearest_dist = dist
            nearest_neighbor = dp
    proper_subset.append(nearest_neighbor)

fig, ax = plt.subplots()
ax.scatter(*zip(*data), color='gray', alpha=0.05, label='data')
ax.scatter(*zip(*support_points), color='green', alpha=0.3, label='support')
ax.scatter(*zip(*proper_subset), color='red', alpha=0.3, label='subset')
ax.legend()
plt.savefig('plot_proper_subset.png')
pdb.set_trace()

