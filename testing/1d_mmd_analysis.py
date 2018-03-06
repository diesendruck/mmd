import numpy as np
import pdb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def generate_1d_data(n, p=0.8):
    """Generate 1D data with two clusters.
    Produces histogram of data.
    Inputs:
      n: Number of data points to generate.
      p: Percent of points in first (of two) clusters.
    Outputs:
      data: Numpy array with n points, coming from two clusters.
    """
    var1 = 0.2
    var2 = 0.2
    center1 = -1
    center2 = 1
    n_c1 = int(np.floor(n * p))
    n_c2 = n - n_c1
    # Define points as univariate normal.
    c1_points = np.random.normal(center1, var1, n_c1)
    c2_points = np.random.normal(center2, var2, n_c2)
    data = c1_points
    if len(c2_points) > 0:
        data = np.concatenate((c1_points, c2_points))
    assert len(data) == n
    return data


def in_target_region(x):
    """Checks if 1-dimensional Numpy array is in target region.
    Inputs:
      x: One-dimensional Numpy array.
    """
    if x > 0:
        return True
    else:
        return False


def run_trials(data, gen, conformal=False, c=2, sigma=1):
    """Performs and returns summary of MMD trials over various data mixtures.
    Inputs:
      data: One-dimensional Numpy array of true data.
      gen: One-dimensional Numpy array of generated data.
      conformal: Boolean passed to kernel() to trigger conformal map.
      c: Scalar of conformal map.
      sigma: Scalar, the lengthscale of the kernel.
    Outputs:
      mmd: MMD for one data mixture.
    """
    # TODO: Determine if this function is needed, if trials are needed.
    #       E.g. maybe the sample should be bootstrapped?
    mmd_out = mmd(data, gen, conformal=conformal, c=c, sigma=sigma)
    return mmd_out


def mmd(x, y, conformal=False, c=2, sigma=1):
    """Compute Maximum Mean Discrepancy (MMD) between two samples.
    Computes mmd between two nxd Numpy arrays, representing n samples of
    dimension d. The Gaussian Radial Basis Function is used as the kernel
    function.
    Inputs:
      x: Numpy array of n samples.
      y: Numpy array of n samples.
      conformal: Boolean passed to kernel() to trigger conformal map.
      c: Scalar of conformal map.
      sigma: Scalar, the lengthscale of the kernel.
    Outputs:
      mmd: Scalar representing MMD.
    """
    total_mmd1 = 0 
    total_mmd2 = 0 
    total_mmd3 = 0 
    n = x.shape[0]
    m = y.shape[0]
    assert n==m 
    # Exact x-x term.
    for i in range(n):
        for j in range(i+1, n):
            total_mmd1 += kernel(x[i], x[j], conformal=conformal, c=c,
                                 sigma=sigma) 
    # Exact y-y term.
    for i in range(m):
        for j in range(i+1, m):
            total_mmd2 += kernel(y[i], y[j], conformal=conformal, c=c,
                                 sigma=sigma) 
    # Exact x-y term.
    for i in range(n):
        for j in range(m):
            total_mmd3 += kernel(x[i], y[j], conformal=conformal, c=c,
                                 sigma=sigma) 
    mmd = (total_mmd1 / (n * (n - 1) / 2) +
           total_mmd2 / (n * (n - 1) / 2) -
           2 * total_mmd3 / (n * m))
    return mmd


def kernel(a, b, conformal=False, c=2, sigma=1):
    """Gaussian Radial Basis Function.
    Output is between 0 and 1.
    Inputs:
      a: A single Numpy array of dimension d.
      b: A single Numpy array of dimension d.
      conformal: Boolean passed to kernel() to trigger conformal map.
      c: Scalar of conformal map.
      sigma: Scalar, the lengthscale of the kernel.
    Outputs:
      k: Kernel value.
    """
    sigma = sigma  # Bandwidth of kernel.
    k = np.exp(-1 * np.square(a - b) / (2 * sigma**2)) 

    # Apply conformal map, where D(x) is indicator of being in target region.
    # i.e. D(x) = {1, if point is in cluster 1;
    #              c, if point is in cluster 2 [the target region]} 
    if conformal:
        # Multiply the kernel by D(x) for each kernel element a and b.
        if in_target_region(a):
            k *= c
        if in_target_region(b):
            k *= c
    return k


def plot_sample():
    """Plots one set of data and generated points.

    Saves to local directory.
    """
    p1 = 0.9
    p2 = 0.99
    data = generate_1d_data(100, p=p1)
    gen = generate_1d_data(100, p=p2)
    fig, ax = plt.subplots()
    ax.hist(data, bins=75, color='g', alpha=0.3, label='data')
    ax.hist(gen, bins=75, color='b', alpha=0.3, label='gen')
    ax.legend()
    ax.set_title('True and Generated Data: p1_{} p2_{}'.format(p1, p2))
    plt.savefig('plot_example.png')


def main():
    # Plot sample data for exposition.
    plot_sample()

    # Define settings for experiment.
    n = 1000
    gen_mixtures = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]
    gen_mixtures = np.arange(0.5, 1, 0.025)
    data_mixtures = [0.7, 0.8, 0.9]
    c_choices = [0.8, 1.2, 2, 3]
    sigma_choices = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0]

    for c in c_choices:
        for sigma in sigma_choices:
            for data_mix in data_mixtures:
                print('C={}, PERCENT_CLUSTER_1={}'.format(c, data_mix))
                print('conformal stats')
                to_plot_conf = []
                for gen_mix in gen_mixtures:
                    data = generate_1d_data(n, p=data_mix)
                    gen = generate_1d_data(n, p=gen_mix)
                    mmd = run_trials(data, gen, conformal=True, c=c,
                                     sigma=sigma)
                    to_plot_conf.append(mmd)
                    print 'gen_mix: {}'.format(gen_mix)

                print('plain stats')
                to_plot_plain = []
                for gen_mix in gen_mixtures:
                    data = generate_1d_data(n, p=data_mix)
                    gen = generate_1d_data(n, p=gen_mix)
                    mmd = run_trials(data, gen, conformal=False, c=c,
                                     sigma=sigma)
                    to_plot_plain.append(mmd)
                    print 'gen_mix: {}'.format(gen_mix)

                fig, ax = plt.subplots()
                ax.plot(gen_mixtures, to_plot_conf, label='conf')
                ax.plot(gen_mixtures, to_plot_plain, label='plain')
                ax.legend()
                ax.set_xlabel('% generated in cluster 1')
                ax.set_ylabel('MMD')
                save_tag = 'mix{}_n{}_c{}_sig{}'.format(data_mix, n, c, sigma)
                ax.set_title('Means of MMD: {}'.format(save_tag))
                plt.savefig('plots_{}.png'.format(save_tag))
    

main()
