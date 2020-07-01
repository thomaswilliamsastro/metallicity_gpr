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
from sklearn.gaussian_process.kernels import Matern
import dill
from astropy.wcs import WCS

from vars import top_dir, muse_dir, muse_version, hii_mask_dir, metallicity_dir, phangs_master_table

os.chdir(top_dir)

# x = np.linspace(-10, 10, 100)
#
# nus = [0.5, 1.5, 2.5, 20]
#
# plt.figure()
#
# for nu in nus:
#
#     kernel = Matern(length_scale=1, nu=nu)
#     gp = GaussianProcessRegressor(kernel=kernel)
#
#     y_pred = gp.sample_y(x.reshape(-1, 1))
#
#     plt.plot(x,y_pred, label='nu=' + str(nu))
#
# plt.legend(loc='upper right')
# plt.show()
# no

hii_table = Table.read(hii_mask_dir + 'Nebulae_Catalogue.fits')

galaxy = 'NGC628'

hii_rows = hii_table[hii_table['gal_name'] == galaxy]
hii_rows = hii_rows[~np.isnan(hii_rows['met_scal'])]
hii_ids = list(hii_rows['region_ID'])

hii_hdu = fits.open(hii_mask_dir + 'spatial_masks/' + galaxy + '_HIIreg_mask.fits')[0]

data = hii_hdu.data.flatten()

total_n_pix = len(np.where(~np.isnan(data))[0])

hii_region_pix = 0
for hii_id in hii_ids:
    hii_region_pix += len(np.where(data == hii_id)[0])

print(hii_region_pix / total_n_pix)
