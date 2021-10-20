# -*- coding: utf-8 -*-
"""
Look at scale length as a function of various sizes

@author: Tom Williams
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.table import Table
from scipy.stats import kendalltau

from vars import top_dir, muse_dir, muse_version, metallicity_dir, extinction_curve, hii_only, phangs_master_table
from funcs import get_info_from_master_table


def get_tau_err(x, x_err, y, y_err, n_draws=1000):
    tau_array = np.zeros(n_draws)

    for i in range(n_draws):
        x_perturb = x + np.random.normal(loc=0, scale=x_err)
        y_perturb = y + np.random.normal(loc=0, scale=y_err)

        tau_array[i] = kendalltau(x_perturb, y_perturb)[0]

    tau = np.nanmedian(tau_array)
    tau_err = np.nanpercentile(tau_array, 84) - tau

    return tau, tau_err


os.chdir(top_dir)

metallicity_calib = 'pg16_scal'

scale_length_file_name = os.path.join(metallicity_dir, muse_version, '%s_ext_curve' % extinction_curve,
                                      'scale_lengths_%s' % metallicity_calib)
if hii_only:
    scale_length_file_name += '_hii_only'
scale_length_file_name += '.fits'
scale_length_table = Table.read(scale_length_file_name)
tab = scale_length_table[scale_length_table['SIGNIFICANT'] == 1]
galaxies = tab['GALAXY']
sigma_l = tab['SIGMA_L']
sigma_l_err = tab['SIGMA_L_ERR_UP']

neb_tab = Table.read('nebulae_catalogue/v2/Nebulae_catalogue_v2.fits')

total_kpc = []
masses = []
r25s = []
incs = []
incs_unc = []

for i, galaxy in enumerate(galaxies):
    pc_per_pix = neb_tab[neb_tab['gal_name'] == galaxy]['pc_per_pixel'][0]

    muse_hdu = fits.open(os.path.join(muse_dir, muse_version, '%s_MAPS.fits' % galaxy))[1]
    n_good = len(muse_hdu.data[~np.isnan(muse_hdu.data)])
    total_kpc_fov = np.sqrt(n_good * (pc_per_pix / 1000) ** 2)

    total_kpc.append(total_kpc_fov)

    mass, r25, dist, inc, inc_unc = get_info_from_master_table(phangs_master_table, galaxy.lower(),
                                                      ['props_mstar', 'size_r25', 'dist', 'orient_incl',
                                                       'orient_incl_unc'])
    masses.append(mass)
    r25_phys = dist * 1e3 * r25 / 3600 * np.pi / 180
    r25s.append(r25_phys)

    incs.append(inc)
    incs_unc.append(inc_unc)

total_kpc = np.array(total_kpc)
r25s = np.array(r25s)
incs = np.array(incs)

print(np.min(sigma_l / r25s), np.max(sigma_l / r25s), np.median(sigma_l / r25s))

print('scale length/FOV: ' + str(get_tau_err(sigma_l, sigma_l_err, total_kpc, 0.1 * total_kpc)))
print('scale length/r25: ' + str(get_tau_err(sigma_l, sigma_l_err, r25s, 0.1 * r25s)))
print('scale length/inc: ' + str(get_tau_err(sigma_l, sigma_l_err, incs, incs_unc)))

# plt.figure()
# plt.scatter(sigma_l, total_kpc)
#
# plt.figure()
# plt.scatter(total_kpc, masses)
# plt.show()


print('Complete!')
