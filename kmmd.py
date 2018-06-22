import sys
sys.path.append('/home/maurice/mmd')
import pdb
import numpy as np
from mmd_utils import compute_mmd, compute_kmmd


def compute_moments(x):
    m = [np.mean(x), np.mean(x**2), np.mean(x**3), np.mean(x**4)]
    return m


def get_mmds(sample_size):
    num_runs = 100

    # Set up containers for resuls.
    mmds_norm_gamma = np.zeros(num_runs)
    kmmds_norm_gamma = np.zeros(num_runs)
    kmmds_norm_norm = np.zeros(num_runs)
    x_moments = np.zeros((num_runs, 4))
    y_moments = np.zeros((num_runs, 4))
    z_moments = np.zeros((num_runs, 4))

    for i in range(num_runs):
        # Define distributions to sample.
        """
        x = np.random.normal(1, 1, size=sample_size)
        y = np.random.gamma(1, 1, size=sample_size)
        z = np.random.normal(1, 1, size=sample_size)
        """
        x = np.random.normal(10, 5, size=sample_size)
        y = np.random.gamma(20, 0.5, size=sample_size)
        z = np.random.normal(10, 5, size=sample_size)

        # Center and norm the data.
        x = (x - np.mean(x)) / np.std(x)
        y = (y - np.mean(y)) / np.std(y)
        z = (z - np.mean(z)) / np.std(z)
        
        # Compute metrics: reference, desired metric, comparison.
        mmds_norm_gamma[i] = compute_mmd(x, y, slim_output=True)
        kmmds_norm_gamma[i] = compute_kmmd(x, y, slim_output=True)
        kmmds_norm_norm[i] = compute_kmmd(x, z, slim_output=True)

        # Store empirical moments.
        x_moments[i] = compute_moments(x)
        y_moments[i] = compute_moments(y)
        z_moments[i] = compute_moments(z)

    return (mmds_norm_gamma, kmmds_norm_gamma, kmmds_norm_norm,
            x_moments, y_moments, z_moments)


def run_expt():
    for sample_size in [400]:
        # Get many samples of MMD and moments.
        (mmds_norm_gamma,
         kmmds_norm_gamma,
         kmmds_norm_norm,
         x_moments,
         y_moments,
         z_moments) = get_mmds(sample_size)

        # Print diagnostics.
        print(('mmd_norm_gamma:  {:.4f}, {:.4f}\n'
               'kmmd_norm_gamma: {:.4f}, {:.4f}\n'
               'kmmd_norm_norm:  {:.4f}, {:.4f}\n'
               'norm_moments: {}\n'
               'gamm_moments: {}\n'
               'norm_moments: {}\n\n').format(
            np.mean(mmds_norm_gamma), np.std(mmds_norm_gamma),
            np.mean(kmmds_norm_gamma), np.std(kmmds_norm_gamma),
            np.mean(kmmds_norm_norm), np.std(kmmds_norm_norm),
            np.round(np.mean(x_moments, axis=0), 4),
            np.round(np.mean(y_moments, axis=0), 4),
            np.round(np.mean(z_moments, axis=0), 4)))

run_expt()
