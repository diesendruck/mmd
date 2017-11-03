import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 12})
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import pylab
import pdb
import numpy as np
import os
from scipy.stats import shapiro, probplot, norm


m = 10000
chunk_size = 200
proposals_per_chunk = 10
corr = 0.9


# MAKE QQ PLOTS
plt.subplot(131)
q = np.load('all_proposals.npy')
n = len(q)
probplot(q, dist='norm', plot=pylab)
plt.title('QQ Plot: G')

p = np.zeros(n)
p[0] = np.random.normal()
for i in xrange(1, n):
    p[i] = (p[i - 1] * corr +
        np.sqrt(1 - corr**2) * np.random.normal())
plt.subplot(132)
probplot(p, dist='norm', plot=pylab)
plt.title('QQ Plot: P')

plt.subplot(133)
probplot(np.random.normal(0, 1, n), dist='norm', plot=pylab)
plt.title('QQ Plot: N(0,1)')

plt.suptitle('n={}, corr={}'.format(n, corr))
plt.subplots_adjust(top=0.85)
plt.savefig('qq_plots.png')
plt.close()

os.system('echo $PWD | mutt momod@utexas.edu -s "optimal_representation" -a "orig_data.png"  -a "all_proposals.png" -a "qq_plots.png" -a "particles_datacorrelated_corr{}_nd{}_chunk{}_ppc{}_it1001_lr1.0_sig1.0.png"'.format(corr, m, chunk_size, proposals_per_chunk))


# COMPUTE COMPARISONS FOR VAR(X_BARS) 
x_bars_corr = []
for i in range(1000):
    data = np.zeros(m)
    data[0] = np.random.normal()
    for j in xrange(1, m):
        data[j] = (data[j - 1] * corr +
            np.sqrt(1 - corr**2) * np.random.normal())

    p = np.random.permutation(data)
    partitions = [p[k: k + chunk_size] for k in xrange(0, len(p), chunk_size)]
    props = []
    for part in partitions:
        props = np.concatenate(
            (props, np.random.choice(part, proposals_per_chunk, replace=False)))
    x_bars_corr.append(np.mean(props))

x_bars_stdnorm = []
for i in range(1000):
    data = np.random.normal(0, 1, m)

    p = np.random.permutation(data)
    partitions = [p[k: k + chunk_size] for k in xrange(0, len(p), chunk_size)]
    props = []
    for part in partitions:
        props = np.concatenate(
            (props, np.random.choice(part, proposals_per_chunk, replace=False)))
    x_bars_stdnorm.append(np.mean(props))

print '\ncorr = {}'.format(corr)
print 'num proposals = {}'.format(n)
print '1/{} = {}'.format(n, 1./n)

print 'CORR SAMPLING - 1000 expts'
print '  var(x_bars_corr) = {}'.format(np.var(x_bars_corr))
print 'STDNORM SAMPLING - 1000 expts'
print '  var(x_bars_stdnorm) = {}'.format(np.var(x_bars_stdnorm))

x_bars_mmd = np.genfromtxt('x_bars.txt')
print 'MMD SAMPLING - {} expts'.format(len(x_bars_mmd))
print '  var(x_bars_mmd) = {}'.format(np.var(x_bars_mmd))

pdb.set_trace()
