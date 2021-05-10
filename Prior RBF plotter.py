print(__doc__)

# Authors: Jan Hendrik Metzen <jhm@informatik.uni-bremen.de>
#
# License: BSD 3 clause

import numpy as np

from matplotlib import pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic,
                                              ExpSineSquared, DotProduct,
                                              ConstantKernel, WhiteKernel)
import GPy

kernel = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(0.5289, 0.5291))


alpha = 0.2
alpha2 = 0.5
fontsize = 12
labelsize = 11
summing = False

# Specify Gaussian Process
gp = GaussianProcessRegressor(kernel=kernel)

# # Plot prior
plt.figure(figsize=(6.25, 3.5)) #(6.25, 5.72)
plt.subplot(2, 1, 1)
X_ = np.linspace(0, 7., 100)
y_mean, y_std = gp.predict(X_[:, np.newaxis], return_std=True)
plt.plot(X_, y_mean, 'k', lw=2, zorder=9)
plt.fill_between(X_, y_mean - y_std, y_mean + y_std,
                 alpha=alpha, color='k')
y_samples = gp.sample_y(X_[:, np.newaxis], 7)
plt.plot(X_, y_samples, lw=1)
plt.xlim(0, 7)
# plt.ylim(-3, 3)
plt.tick_params(axis="both", which='both', direction="in", bottom=True, top=True, left=True, right=True, labelsize=labelsize)
plt.ylabel("Output", fontsize=fontsize)
plt.xlabel("Input", fontsize=fontsize)

# plt.title("Prior (kernel:  %s)" % kernel, fontsize=12)

# Generate data and fit GP
upper_bound = 7.
rng = np.random.RandomState(5)
X = rng.uniform(0, 5, 10)[:, np.newaxis]
y = 2*np.sin((X[:, 0] - 2.5) ** 2)
kernel = WhiteKernel(noise_level=1.0, noise_level_bounds=(0.0004239, 0.0004241)) \
          + 1.0 * RBF(length_scale=1.0, length_scale_bounds=(0.5289, 0.5291))
gp = GaussianProcessRegressor(kernel=kernel)
gp.fit(X, y)

# Plot posterior
plt.subplot(2, 1, 2)
X_ = np.linspace(0, upper_bound, 100)
y_mean, y_std = gp.predict(X_[:, np.newaxis], return_std=True)
plt.plot(X_, y_mean, 'k', lw=2, zorder=9)
plt.fill_between(X_, y_mean - y_std, y_mean + y_std,
                 alpha=alpha, color='k')

y_samples = gp.sample_y(X_[:, np.newaxis], 10)
gpy = GPy.models.GPRegression(X, y[:,None], GPy.kern.RBF(input_dim=1, variance=np.exp(gp.kernel_.theta)[1], lengthscale=np.exp(gp.kernel_.theta)[2]))
# gpy.optimize_restarts(num_restarts=3, verbose=True)
gpy.Gaussian_noise.variance = np.exp(gp.kernel_.theta)[0]
print(gpy)
y_samples = gpy.posterior_samples_f(X_[:, np.newaxis], size=7)
plt.plot(X_, y_samples[:,0,:], lw=1)
# gpy.plot(plot_limits=(0, 5), fixed_inputs=None, resolution=None, plot_raw=False, apply_link=False, which_data_ycols='all',
#      which_data_rows='all', visible_dims=None, levels=20, samples=7, samples_likelihood=0, lower=2.5, upper=97.5,
#      plot_data=False, plot_inducing=False, plot_density=False, predict_kw=None, projection='2d', legend=False)
plt.scatter(X[:, 0], y, c='r', s=30, zorder=10, edgecolors=(0, 0, 0))
plt.xlim(0, upper_bound)
# plt.ylim(-3, 3)
print("Posterior (kernel: %s)\n Log-Likelihood: %.3f"
          % (gp.kernel_, gp.log_marginal_likelihood(gp.kernel_.theta)))
plt.tick_params(axis="both", which='both', direction="in", bottom=True, top=True, left=True, right=True, labelsize=labelsize)
plt.ylabel("Output", fontsize=fontsize)
plt.xlabel("Input", fontsize=fontsize)
plt.tight_layout()
print(np.exp(gp.kernel_.theta))
plt.show()