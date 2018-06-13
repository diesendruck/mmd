import pdb
import numpy as np
from mmd_utils import compute_mmd, compute_kmmd


def get_mmds(sample_size):
    num_runs = 100
    mmds_norm_gamma = np.zeros(num_runs)
    kmmds_norm_gamma = np.zeros(num_runs)
    kmmds_norm_norm = np.zeros(num_runs)
    for i in range(num_runs):
        x = np.random.normal(1, 1, size=sample_size)
        y = np.random.gamma(1, 1, size=sample_size)
        z = np.random.normal(1, 1, size=sample_size)
        mmd_ng, _ = compute_mmd(x, y)  # Normal-gamma.
        kmmd_ng, _ = compute_kmmd(x, y)  # Normal-gamma.
        kmmd_nn, _ = compute_kmmd(x, z)  # Normal-normal.
        mmds_norm_gamma[i] = mmd_ng
        kmmds_norm_gamma[i] = kmmd_ng
        kmmds_norm_norm[i] = kmmd_nn
    return mmds_norm_gamma, kmmds_norm_gamma, kmmds_norm_norm


for sample_size in [50, 100, 500]:
    mmds_norm_gamma, kmmds_norm_gamma, kmmds_norm_norm = \
        get_mmds(sample_size)
    print
    print(('mmd_norm_gamma:  {:.2f}, {:.2f}\n'
           'kmmd_norm_gamma: {:.2f}, {:.2f}\n'
           'kmmd_norm_norm:  {:.2f}, {:.2f}').format(
        np.median(mmds_norm_gamma), np.var(mmds_norm_gamma),
        np.median(kmmds_norm_gamma), np.var(kmmds_norm_gamma),
        np.median(kmmds_norm_norm), np.var(kmmds_norm_norm)))
pdb.set_trace()
