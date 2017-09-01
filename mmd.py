import numpy as np
import pdb
import matplotlib.pyplot as plt


def generate_2d_data(n, p=0.9):
    """Generate 2D data with two clusters.

    Produces graph of data.

    Inputs:
      n: Number of data points to generate.
      p: Percent of points in first (of two) clusters.

    Outputs:
      data: Numpy array with n points, coming from two clusters.
    """
    var1 = 0.1
    var2 = 0.1
    center1 = [-1, -1]
    center2 = [1, 1]
    n_c1 = int(np.floor(n * p))
    n_c2 = n - n_c1
    c1_points = np.random.multivariate_normal(
            center1, np.identity(2) * var1, n_c1)
    c1_points = [p for p in c1_points if not in_target_region(p)]
    c2_points = np.random.multivariate_normal(
            center2, np.identity(2) * var2, n_c2)
    c2_points = [p for p in c2_points if in_target_region(p)]
    if len(c1_points) > 0 and len(c2_points) > 0:
        data = np.concatenate((c1_points, c2_points))
    elif len(c1_points) == 0 and len(c2_points) > 0:
        data = np.array(c2_points)
    elif len(c1_points) > 0 and len(c2_points) == 0:
        data = np.array(c1_points)
    else:
        raise ValueError('Issue with number of c1 and c2 points.')
    return data


def mmd(x, y, conformal=False):
    """Compute Maximum Mean Discrepancy (MMD) between two samples.

    Computes mmd between two nxd Numpy arrays, representing n samples of
    dimension d. The Gaussian Radial Basis Function is used as the kernel
    function.

    Inputs:
      x: Numpy array of n samples.
      y: Numpy array of n samples.
      conformal: Boolean passed to kernel() to trigger conformal map.

    Outputs:
      mmd: Scalar representing MMD.
    """
    n = 250
    total_mmd1 = 0 
    total_mmd2 = 0 
    total_mmd3 = 0 
    for i in range(n):
        ind_x = np.random.randint(x.shape[0], size=2)  # Get two sample indices.
        ind_y = np.random.randint(y.shape[0], size=2)
        x1 = x[ind_x[0]]
        x2 = x[ind_x[1]]
        y1 = y[ind_y[0]]
        y2 = y[ind_y[1]]
        total_mmd1 += kernel(x1, x2, conformal=conformal) 
        total_mmd2 += kernel(y1, y2, conformal=conformal) 
        total_mmd3 += kernel(x1, y1, conformal=conformal)

    mmd = total_mmd1/n + total_mmd2/n - 2 * total_mmd3/n
    return mmd


def in_target_region(x):
    if x[0] > 0.5 and x[1] > 0.5:
        return True
    else:
        return False


def kernel(a, b, conformal=False):
    """Gaussian Radial Basis Function.

    Output is between 0 and 1.

    Inputs:
      a: A single Numpy array of dimension d.
      b: A single Numpy array of dimension d.
      conformal: Boolean to trigger conformal map.

    Outputs:
      k: Kernel value.
    """
    sigma = 1  # Bandwidth of kernel.
    a = np.array(a)
    b = np.array(b)
    k = np.exp(-1 * sum(np.square(a - b)) / (2 * sigma**2)) 

    # Apply conformal map, where D(x) is indicator of being in target region.
    # i.e. D(x) = {1, if point is in cluster 1;
    #              c, if point is in cluster 2 [the target region]} 
    c = 1.2
    if conformal:
        # Multiply the kernel by D(x) for each kernel element a and b.
        if in_target_region(a):
            k *= c
        if in_target_region(b):
            k *= c

    return k


def run_trials(data, gen, conformal=False):
    num_runs = 100
    trials = [mmd(data, gen, conformal=conformal) for i in range(num_runs)]
    stats = {'min': round(min(trials), 2),
             'med': round(np.median(trials), 2),
             'max': round(max(trials), 2),
             'var': round(np.var(trials), 2)}
    return stats, np.median(trials) 


def plot_sample(data, gen):
    plt.scatter([i for i,j in data], [j for i,j in data], c='b', alpha=0.3)
    plt.scatter([i for i,j in gen], [j for i,j in gen], c='r', alpha=0.3)
    plt.show()


def main():
    n = 50
    data = generate_2d_data(n, p=0.8)
    gen = generate_2d_data(n, p=1)
    plot_sample(data, gen)
    various_mixtures = np.arange(0, 1.01, 0.1)  # [0.0, 0.1, ..., 1]

    print('Conformal stats')
    conf_medians = []
    for mix in various_mixtures:
        data = generate_2d_data(n, p=0.5)
        gen = generate_2d_data(n, p=mix)
        stats, conf_median = run_trials(data, gen, conformal=True)
        conf_medians.append(conf_median)
        print('Mix: {}, MMDs: min={}, med={}, max={}, var={}'.format(
            mix, stats['min'], stats['med'], stats['max'], stats['var']))

    print('Plain stats')
    plain_medians = []
    for mix in various_mixtures:
        data = generate_2d_data(n, p=0.5)
        gen = generate_2d_data(n, p=mix)
        stats, plain_median = run_trials(data, gen, conformal=False)
        plain_medians.append(plain_median)
        print('Mix: {}, MMDs: min={}, med={}, max={}, var={}'.format(
            mix, stats['min'], stats['med'], stats['max'], stats['var']))

    fig, ax = plt.subplots()
    ax.plot(conf_medians, label='conf')
    ax.plot(plain_medians, label='plain')
    ax.legend()
    plt.show()
    

main()
