# -*- coding: utf-8 -*-
"""
Look at the maximum scale length we can cover given data extent

@author: Tom Williams
"""

import os

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
import matplotlib.pyplot as plt
import matplotlib

from vars import top_dir, plot_dir

os.chdir(top_dir)

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['font.size'] = 14

plot_name = os.path.join(plot_dir, 'recovered_scale_length_illustration')

scale_lengths = np.arange(0.5, 10, 0.5)

n_points = 100
x = np.random.uniform(0, 1, n_points)
x = x.reshape(-1, 1)

recovered_scale_lengths = np.zeros_like(scale_lengths)

for i, scale_length in enumerate(scale_lengths):
    print(scale_length)
    kernel_perfect = Matern(length_scale=scale_length, length_scale_bounds=(0.01, 100), nu=1.5) + ConstantKernel(0.0)
    gp_perfect = GaussianProcessRegressor(kernel=kernel_perfect, n_restarts_optimizer=0)

    kernel_to_fit = Matern(length_scale=1, length_scale_bounds=(0.01, 100), nu=1.5) \
        + ConstantKernel(0.0) + WhiteKernel(0.1, noise_level_bounds=(1e-9, 10))
    gp_to_fit = GaussianProcessRegressor(kernel=kernel_to_fit, n_restarts_optimizer=25)
    y = gp_perfect.sample_y(x) + np.random.normal(loc=0, scale=1e-3, size=n_points).reshape(-1, 1)

    gp_to_fit.fit(x, y)
    fit_scale_length = gp_to_fit.kernel_.get_params()['k1__k1__length_scale']

    recovered_scale_lengths[i] = fit_scale_length

plt.figure(figsize=(5, 4))

ax = plt.subplot(111)

plt.scatter(scale_lengths, recovered_scale_lengths, c='k')
plt.plot([0, 10], [0, 10], c='k', ls='--')

plt.xlim([0, 10])
plt.ylim([0, 10])

plt.grid()

ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())

plt.xlabel(r'$\sigma_{l, \mathrm{input}}$ ($\times$ array range)')
plt.ylabel(r'$\sigma_{l, \mathrm{output}}$ ($\times$ array range)')

plt.tight_layout()

plt.savefig(plot_name + '.png', bbox_inches='tight')
plt.savefig(plot_name + '.pdf', bbox_inches='tight')

plt.show()
plt.close()

print('Complete!')
