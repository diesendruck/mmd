import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np
import pdb
import pylab
from scipy.stats import shapiro, probplot, norm


x_orig = np.load('sample_x.npy')
g_orig = np.load('sample_g.npy')
z_orig = np.load('sample_z.npy')

x = x_orig.reshape(x_orig.shape[0],)
g = g_orig.reshape(g_orig.shape[0],)
if z_orig.shape[1] == 1:
    z = z_orig.reshape(z_orig.shape[0],)
else:
    z = z_orig[:, 1]

# Evaluate mapping between z and g, by reordering both arrays according to one
# of their sort orders. If they map smoothly, increase of one leads to increase
# of the other, but this is not necessary.
z_sort_order = np.argsort(z)
z_sorted = z[z_sort_order]
g_sorted_by_z = g[z_sort_order]

# Shapiro-Wilk test for normality of generated samples.
#(w_statistic, p_value) = shapiro(g)
(w_statistic, p_value) = 0, 0

# Plot results.
plt.figure(figsize=(25, 10))
plt.suptitle('Comparison: X, G, Z. ShapiroWilk(w,p)=({:.04f},{:.04f})'.format(
    w_statistic, p_value), fontsize=20)

plt.subplot(211)
plt.hist(x, 30, normed=True, color='green', label='x', alpha=0.3)
plt.hist(g, 30, normed=True, color='blue', label='g', alpha=0.3)
xs = np.linspace(min(x), max(x), 100)
ys = norm.pdf(xs, 0, 1)
plt.plot(xs, ys, color='red', label='pdf', alpha=0.3)
plt.title('Distribution comparison: X vs G')
plt.xlabel('Values')
plt.legend()

plt.subplot(245)
plt.plot(np.sort(g), color='blue', label='g_sorted', alpha=0.3)
plt.plot(np.sort(x), color='green', label='x_sorted', alpha=0.3)
plt.title('Ordered comparison: X vs G')
plt.xlabel('Samples')
plt.ylabel('Values g, x')
plt.legend()

plt.subplot(246)
probplot(g, dist='norm', plot=pylab)
plt.title('QQ Plot: G')

plt.subplot(247)
probplot(x, dist='norm', plot=pylab)
plt.title('QQ Plot: X')

plt.subplot(248)
plt.scatter(z, g)
plt.title('Mapping from noise z to g')
plt.xlabel('Noise z')
plt.ylabel('Generated value')

plt.savefig('result_plot.png')
plt.close()

# Additional graphic/image for paper.
xs = np.linspace(min(x), max(x), 100)
ys1 = norm.pdf(xs, 0, 0.5)
ys2 = norm.pdf(xs, 2, 0.5)
ys_unthinned = 2./3 * ys1 + 1./3 * ys2

plt.hist(g, 30, normed=True, color='blue', label='g', alpha=0.7)
plt.plot(xs, ys_unthinned, color='blue', label='pdf_unthinned', alpha=0.7)
plt.xlabel('Simulations')
plt.ylabel('Density')
plt.tight_layout()
plt.savefig('mmd_simulations.png')
