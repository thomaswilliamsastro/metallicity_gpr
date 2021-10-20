# -*- coding: utf-8 -*-
"""
Illustrate the scale length using some simple 2D examples

@author: Tom Williams
"""

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

from vars import top_dir, metallicity_dir, plot_dir

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['font.size'] = 14

os.chdir(top_dir)

overwrite = False

array_dir = os.path.join(metallicity_dir, 'scale_length_illustrations')
if not os.path.exists(array_dir):
    os.makedirs(array_dir)

sig_ls = [1, 2.5, 5, 10]

lims = [-10, 10]
n_samps = 100

x, y = np.meshgrid(np.linspace(lims[0], lims[1], n_samps), np.linspace(lims[0], lims[1], n_samps))
x_flat, y_flat = x.flatten(), y.flatten()
xy = np.array([[x_flat[i], y_flat[i]] for i in range(len(x_flat))])

fig_name = os.path.join(plot_dir, 'scale_length_illustration')

plt.figure(figsize=(6, 6))

for i, sig_l in enumerate(sig_ls):

    print(sig_l)

    file_name = os.path.join(array_dir, '%s.npy' % sig_l)

    if not os.path.exists(file_name) or overwrite:
        kernel = Matern(length_scale=sig_l, nu=1.5)
        gp = GaussianProcessRegressor(kernel=kernel)

        z = gp.sample_y(xy).reshape(x.shape)
        np.save(file_name, z)
    else:
        z = np.load(file_name)

    ax = plt.subplot(2, 2, i + 1)
    plt.imshow(z, origin='lower', extent=[lims[0], lims[1], lims[0], lims[1]])

    plt.text(0.05, 0.95, r'$\sigma_l = %s$' % sig_l,
             ha='left', va='top',
             bbox=dict(facecolor='white', edgecolor='none', alpha=0.75),
             transform=ax.transAxes)

plt.tight_layout()
# plt.show()
plt.savefig(fig_name + '.png', bbox_inches='tight')
plt.savefig(fig_name + '.pdf', bbox_inches='tight')

plt.close()

print('Complete!')
