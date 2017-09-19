import matplotlib.pyplot as plt
import numpy as np
import pdb

# Set up true data, 100 points.
#p = np.random.randn(10)

# Initialize 10 proposal points as a uniform spread over true points.
#q = np.linspace(min(p), max(p), 3)

# Define (abbreviated) energy statistic.
def energy(data, prop):
    """Computes abbreviated energy statistic between two point sets.
    
    The smaller the value, the closer the sets.
    Args:
      data: 1D numpy array of any length, e.g. 100.
      prop: 1D numpy array of any length, e.g. 10.
    Returns:
      e: Scalar, the energy between the sets.
      gradients: Numpy array of gradients for each proposal point. 
    """
    x = data
    y = prop
    v = np.concatenate((x, y), 0)
    v_vert = v.reshape(-1, 1)
    v_tiled = np.tile(v_vert, (1, len(v)))
    pairwise = v_tiled - np.transpose(v_tiled)
    energy_xx = abs(pairwise[:len(x), :len(x)])
    energy_yx = abs(pairwise[len(x):, :len(x)])
    energy_yy = abs(pairwise[len(x):, len(x):])
    between_group_term = float(np.sum(energy_yx)) / len(y) / len(x)
    within_proposals_term = float(np.sum(energy_yy)) / len(y) / len(y) 
    e = 2 * between_group_term - within_proposals_term
    
    # Compute gradients.
    pdb.set_trace()
    signed = np.sign(pairwise)
    signed_yx = signed[len(x):, :len(x)]
    signed_yy = signed[len(x):, len(x):]
    gradients = []
    for i in range(len(y)):
        grad_yi = 2 * sum(signed_yx[i]) - sum(signed_yy[i])
        gradients.append(grad_yi)
    
    return e, np.array(gradients)

def optimize(data, prop, n, learning_rate, joint=False):
    """Runs alternating optimizations, n times through proposal points.
    Args:
      data: 1D numpy array of any length, e.g. 100.
      prop: 1D numpy array of any length, e.g. 10.
      n: Scalar, number of times to loop through updates for all vars.
      learning_rate: Scalar, amount to move point with each gradient update.
      joint: Boolean, whether to move proposals jointly or alternating-ly.
    Returns:
      prop: 1D numpy array of updated proposal points.
    """
    e, _ = energy(data, prop)
    props = np.array(prop)
    energies = np.array([e])
    proposal_indices = range(len(prop))
    for run in range(n):
        if not joint:
            for index in proposal_indices:
                e, grads = energy(data, prop) 
                prop[index] -= learning_rate * grads[index] 
        else:
            e, grads = energy(data, prop) 
            prop -= learning_rate * grads 
        if run % 10000 == 0:
            print 'it{}: prop:{}, energy: {}'.format(run, prop, e)
        props = np.vstack((props, prop))
        energies = np.vstack((energies, e))

    plt.subplot(211)
    plt.plot(props[:, 0])
    plt.plot(props[:, 1])
    plt.title('props')
    plt.subplot(212)
    plt.plot(energies)
    plt.title('energies')
    plt.savefig('props_joint_{}.png'.format(joint))
    plt.close()
    

def main():
    p = np.array([0, 1, 2])
    q = np.linspace(min(p), max(p), 2)
    q = [0.4, 2]
    energy(p, q)
    #optimize(p, q, 300000, 1e-5)
