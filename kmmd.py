import pdb
import numpy as np
from mmd_utils import compute_mmd, compute_kmmd


# Define slim versions with only one ouput.
def compute_mmd_(x, y):
    out, _ = compute_mmd(x, y)
    return out
def compute_kmmd_(x, y):
    out, _ = compute_kmmd(x, y)
    return out


def compute_moments(x):
    m = [np.mean(x), np.mean(x**2), np.mean(x**3), np.mean(x**4)]
    return m


def get_mmds(sample_size):
    num_runs = 100

    mmds_norm_gamma = np.zeros(num_runs)
    kmmds_norm_gamma = np.zeros(num_runs)
    kmmds_norm_norm = np.zeros(num_runs)
    x_moments = np.zeros((num_runs, 4))
    y_moments = np.zeros((num_runs, 4))
    z_moments = np.zeros((num_runs, 4))

    for i in range(num_runs):
        # Define distributions to sample.
        x = np.random.normal(1, 1, size=sample_size)
        y = np.random.gamma(1, 1, size=sample_size)
        z = np.random.normal(1, 1, size=sample_size)
        x = (x - np.mean(x)) / np.std(x)
        y = (y - np.mean(y)) / np.std(y)
        z = (z - np.mean(z)) / np.std(z)
        mmds_norm_gamma[i] = compute_mmd_(x, y)
        kmmds_norm_gamma[i] = compute_kmmd_(x, y)
        kmmds_norm_norm[i] = compute_kmmd_(x, z)
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
