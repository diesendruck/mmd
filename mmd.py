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


def in_target_region(x):
    """Checks if 2-dimensional Numpy array is in target region.

    Inputs:
      x: Two-dimensional Numpy array.
    """
    if x[0] > 0.5 and x[1] > 0.5:
        return True
    else:
        return False


def run_trials(data, gen, conformal=False, c=2):
    """Performs and returns summary of MMD trials over various data mixtures.

    Inputs:
      data: Two-dimensional Numpy array of true data.
      gen: Two-dimensional Numpy array of generated data.
      conformal: Boolean passed to kernel() to trigger conformal map.
      c: Scalar of conformal map.

    Outputs:
      stats: Dictionary of MMD statistics for one data mixture.
      med: Median of MMD statistics for one data mixture.
    """
    num_runs = 100
    trials = [mmd(data, gen, conformal=conformal, c=c) for i in range(num_runs)]
    stats = {'min': round(min(trials), 2),
             'med': round(np.median(trials), 2),
             'max': round(max(trials), 2),
             'var': round(np.var(trials), 2)}
    med = np.median(trials)
    return stats, med


def mmd(x, y, conformal=False, c=2):
    """Compute Maximum Mean Discrepancy (MMD) between two samples.

    Computes mmd between two nxd Numpy arrays, representing n samples of
    dimension d. The Gaussian Radial Basis Function is used as the kernel
    function.

    Inputs:
      x: Numpy array of n samples.
      y: Numpy array of n samples.
      conformal: Boolean passed to kernel() to trigger conformal map.
      c: Scalar of conformal map.

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
        total_mmd1 += kernel(x1, x2, conformal=conformal, c=c) 
        total_mmd2 += kernel(y1, y2, conformal=conformal, c=c) 
        total_mmd3 += kernel(x1, y1, conformal=conformal, c=c)

    mmd = total_mmd1/n + total_mmd2/n - 2 * total_mmd3/n
    return mmd


def kernel(a, b, conformal=False, c=2):
    """Gaussian Radial Basis Function.

    Output is between 0 and 1.

    Inputs:
      a: A single Numpy array of dimension d.
      b: A single Numpy array of dimension d.
      conformal: Boolean passed to kernel() to trigger conformal map.
      c: Scalar of conformal map.

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
    if conformal:
        # Multiply the kernel by D(x) for each kernel element a and b.
        if in_target_region(a):
            k *= c
        if in_target_region(b):
            k *= c

    return k


def plot_sample(data, gen, tag):
    """Plots one set of data and generated points.

    Saves to local directory.

    Inputs:
      data: Two-dimensional Numpy array of true data.
      gen: Two-dimensional Numpy array of generated data.
      tag: String included in title.
    """
    plt.scatter([i for i,j in data], [j for i,j in data], c='b', alpha=0.3)
    plt.scatter([i for i,j in gen], [j for i,j in gen], c='r', alpha=0.3)
    plt.savefig('plot_{}.png'.format(tag))


def main():
    n = 50
    gen_mixtures = np.arange(0, 1.01, 0.1)  # [0.0, 0.1, ..., 1]
    data_mixtures = np.arange(0, 1.01, 0.1)  # [0.0, 0.1, ..., 1]
    c_choices = [1.01, 1.02, 1.04, 1.08, 1.16, 1.32]

    for c in c_choices:
        print('C={}'.format(c))

        for data_mix in data_mixtures:
            print('Conformal stats')
            conf_medians = []
            for gen_mix in gen_mixtures:
                data = generate_2d_data(n, p=data_mix)
                gen = generate_2d_data(n, p=gen_mix)
                stats, conf_median = run_trials(data, gen, conformal=True, c=c)
                conf_medians.append(conf_median)
                print('Mix: {}, MMDs: min={}, med={}, max={}, var={}'.format(
                    gen_mix, stats['min'], stats['med'], stats['max'], stats['var']))

            print('Plain stats')
            plain_medians = []
            for gen_mix in gen_mixtures:
                data = generate_2d_data(n, p=data_mix)
                gen = generate_2d_data(n, p=gen_mix)
                stats, plain_median = run_trials(data, gen, conformal=False, c=c)
                plain_medians.append(plain_median)
                print('Mix: {}, MMDs: min={}, med={}, max={}, var={}'.format(
                    gen_mix, stats['min'], stats['med'], stats['max'], stats['var']))

            fig, ax = plt.subplots()
            ax.plot(gen_mixtures, conf_medians, label='conf')
            ax.plot(gen_mixtures, plain_medians, label='plain')
            ax.legend()
            ax.set_xlabel('Percent true data in Cluster 1')
            ax.set_ylabel('MMD')
            ax.set_title('c_{}_datamix_{}_genmix_{}'.format(c, data_mix, gen_mix))
            plt.savefig('plots_c_{}_datamix_{}_genmix_{}.png'.format(c, data_mix, gen_mix))
    

main()
