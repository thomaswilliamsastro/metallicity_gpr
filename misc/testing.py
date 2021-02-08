# -*- coding: utf-8 -*-
"""
Just some tests

@author: Tom Williams
"""

import os

from astropy.table import Table
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
import dill
from astropy.wcs import WCS

from vars import top_dir, muse_dir, muse_version, hii_mask_dir, metallicity_dir, phangs_master_table

os.chdir(top_dir)

x = np.linspace(-10, 10, 250)
x = x.reshape(-1, 1)

kernel = Matern(length_scale=3, length_scale_bounds=(0.1, 10), nu=1.5) #+ ConstantKernel(0) + WhiteKernel(0.1, noise_level_bounds=(1e-9, 10))
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=25)

y = gp.sample_y(x) #+ np.random.normal(scale=0.1, size=x.shape)

# Fit this

gp.fit(x, y)
y_pred = gp.sample_y(x).flatten()
print(gp.kernel_)

# https://arxiv.org/pdf/1408.5810.pdf, p4
det_cov = np.linalg.det(gp.kernel_(gp.X_train_))

log_likelihood = gp.log_marginal_likelihood()
print(log_likelihood)
no

plt.figure()
plt.plot(x, y, 'rx')
plt.plot(x, y_pred, 'bo')
plt.show()
