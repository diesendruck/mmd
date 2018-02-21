import pdb
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from multivariate_mmd_gan import get_sample
from weighting import get_estimation_points 


def generate_points(M, mode='coreset'):
    """Generates M estimation points for pre-trained model."""
    support_points_ = get_sample(M, tag='test')
    support_points, coreset, weights_estimation_pts, weights_data = \
        get_estimation_points(mode=mode, support_points=support_points_)

    #plt.scatter(*zip(*support_points), c='blue')
    #plt.scatter(*zip(*np.load('logs_test/g_out.npy')), c='red')
    #plt.savefig('test.png')

    pdb.set_trace()
    return support_points, coreset, weights_estimation_pts, weights_data


results = generate_points(200) 
support_points, coreset, weights_estimation_pts, weights_data = results
for r in results:
    print r.shape

pdb.set_trace()


