# -*- coding: utf-8 -*-
"""
Estimate the number of 'DIG-like' pixels that are used in the fitting

@author: Tom Williams
"""

import os
import glob

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from astropy.io import fits
from reproject import reproject_interp
from astropy.convolution import convolve, Gaussian2DKernel

from vars import top_dir, metallicity_dir, muse_dir, muse_version, galaxies, phangs_master_table, plot_dir
from funcs import get_info_from_master_table

os.chdir(top_dir)

metallicity_calib = 'pg16_scal'

dig_fractions = []

target_phys_res = 120

copt_files = [file_name.split('/')[-1].split('asec')[0] for file_name in
              glob.glob(os.path.join(muse_dir, muse_version, 'copt/*'))]

for galaxy in galaxies:

    print(galaxy)

    copt_file_found = False

    for copt_file in copt_files:
        if galaxy.upper() in copt_file:
            copt_file_found = True
            break

    if not copt_file_found:
        raise Warning('COPT resolution not found!')

    dist = get_info_from_master_table(phangs_master_table, galaxy.lower(), 'dist')

    metallicity_map_file_name = os.path.join(metallicity_dir, 'pix_maps', muse_version, metallicity_calib,
                                             '%s_metallicity_%s.fits' % (galaxy, metallicity_calib))
    metallicity_map = fits.open(metallicity_map_file_name)[0]

    nebulae_map_file_name = os.path.join('nebulae_catalogue', muse_version, '%s_HIIreg_mask.fits' % galaxy)
    nebulae_map = fits.open(nebulae_map_file_name)[0]

    pix_scale = np.abs(nebulae_map.header['CD1_1']) * 3600

    # We only care about in and out of regions

    nebulae_map.data[nebulae_map.data > 0] = 1

    # Convolve up to the fixed spatial resolution.

    target_ang_res = target_phys_res / (dist * 1e6) * 180 / np.pi * 3600
    copt_ang_res = float(copt_file.split('-')[-1])

    conv_fwhm = np.sqrt(target_ang_res ** 2 - copt_ang_res ** 2)
    conv_std = conv_fwhm / 2.355 / pix_scale

    kernel = Gaussian2DKernel(conv_std)
    nebulae_map.data = convolve(nebulae_map.data, kernel, preserve_nan=True)

    nebulae_map.data[nebulae_map.data <= 0.05] = 0
    nebulae_map.data[nebulae_map.data > 0.05] = 1

    # Reproject to the conv map pix scale

    nebulae_map_reproj, _ = reproject_interp(nebulae_map, metallicity_map.header)
    nebulae_map_reproj[nebulae_map_reproj > 0] = 1

    dig_idx = np.where((nebulae_map_reproj != 1) & (~np.isnan(metallicity_map.data)))
    total_idx = np.where(~np.isnan(metallicity_map.data))
    dig_fraction = len(dig_idx[0])/len(total_idx[0])

    print(dig_fraction)

    dig_fractions.append(dig_fraction * 100)

plot_name = os.path.join(plot_dir, muse_version, 'dig_fraction')

plt.figure(figsize=(4, 3))
sns.kdeplot(np.log10(dig_fractions), bw='silverman', color='k', shade=True)

plt.xlabel(r'$\log_{10}$[DIG fraction (%)]')
plt.ylabel(r'Probability Density')

plt.savefig(plot_name + '.png', bbox_inches='tight')
plt.savefig(plot_name + '.pdf', bbox_inches='tight')

plt.close()

print('Complete!')
