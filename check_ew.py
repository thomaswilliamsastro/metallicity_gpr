# -*- coding: utf-8 -*-
"""
Check how EW cuts would affect the pixel metallicity maps

@author: Tom Williams
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from reproject import reproject_interp

from vars import top_dir, muse_dir, muse_version, galaxies, metallicity_dir

os.chdir(top_dir)

metallicity_calib = 'pg16_scal'
ew_cutoff = 6

ew_fracs = []

for galaxy in galaxies:
    pix_metal_file_name = os.path.join(metallicity_dir, 'pix_maps', muse_version, metallicity_calib,
                                       '%s_metallicity_%s.fits' % (galaxy, metallicity_calib))
    pix_metal_hdu = fits.open(pix_metal_file_name)[0]

    pix_metal_flat = pix_metal_hdu.data.flatten()
    n_pix = len(~np.isnan(pix_metal_flat))

    ew_file_name = os.path.join(muse_dir, 'ew', muse_version, '%s_ew.fits' % galaxy)
    ew_hdu = fits.open(ew_file_name)[0]

    # Create a mask of EW less than the cutoff

    ew_hdu.data = (ew_hdu.data < ew_cutoff).astype(int)

    ew_reproj, _ = reproject_interp(ew_hdu, pix_metal_hdu.header, order='nearest-neighbor')

    ew_flat = ew_reproj.flatten()
    ew_idx = np.where((ew_flat == 1) & ~np.isnan(pix_metal_flat))

    ew_frac = len(ew_idx[0]) / n_pix * 100
    ew_fracs.append(ew_frac)

print(galaxies)
print(ew_fracs)
no

plt.figure()
plt.hist(ew_fracs)

plt.xlabel('Fraction of pixels removed (%)')
plt.ylabel('N')
plt.show()

print('Complete!')
