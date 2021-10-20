# -*- coding: utf-8 -*-
"""
See the number of pixels that fall outside HII regions

@author: Tom Williams
"""

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.table import Table
from reproject import reproject_interp
from uncertainties import unumpy as unp

from vars import top_dir, muse_version, galaxies, metallicity_dir, hii_mask_dir, plot_dir, muse_dir, \
    metallicity_calibs, extinction_curve

os.chdir(top_dir)

nebulae_cat = Table.read(os.path.join(hii_mask_dir, 'Nebulae_catalogue_v2.fits'))

for metallicity_calib in metallicity_calibs:

    only_hii_dir = os.path.join(metallicity_dir, 'pix_maps', muse_version, '%s_ext_curve' % extinction_curve,
                                metallicity_calib + '_hii_only')

    if not os.path.exists(only_hii_dir):
        os.makedirs(only_hii_dir)

    for galaxy in galaxies:

        region_ids = nebulae_cat[(nebulae_cat['gal_name'] == galaxy) & (nebulae_cat['HII_class'] == 1)]['region_ID']
        nebulae_mask_file_name = os.path.join(hii_mask_dir, 'spatial_masks', '%s_nebulae_mask_V2.fits' % galaxy)
        nebulae_mask_hdu = fits.open(nebulae_mask_file_name)[0]

        hii_regions = np.zeros_like(nebulae_mask_hdu.data)

        for region_id in region_ids:
            idx = np.where(nebulae_mask_hdu.data == region_id)
            hii_regions[idx] = 1

        nebulae_mask_hdu.data = hii_regions

        pix_metal_file_name = os.path.join(metallicity_dir, 'pix_maps', muse_version, metallicity_calib,
                                           '%s_ext_curve' % extinction_curve,
                                           '%s_metallicity_%s.fits' % (galaxy, metallicity_calib))
        pix_metal_err_hdu = fits.open(pix_metal_file_name.replace('.fits', '_err.fits'))[0]
        pix_metal_hdu = fits.open(pix_metal_file_name)[0]

        hii_regions_reproj, _ = reproject_interp(nebulae_mask_hdu, pix_metal_hdu.header, order='nearest-neighbor')

        n_pix = len(np.where(~np.isnan(pix_metal_hdu.data))[0])
        n_outside_hii_regions = len(np.where((~np.isnan(pix_metal_hdu.data)) & (hii_regions_reproj == 0))[0])

        print(n_outside_hii_regions / n_pix * 100)

        # Write these out to disc

        mask = ~np.isnan(pix_metal_hdu.data) & (hii_regions_reproj == 0)

        pix_metal_hdu.data[mask] = np.nan
        pix_metal_err_hdu.data[mask] = np.nan

        pix_metal_hdu.writeto(
            os.path.join(only_hii_dir, '%s_metallicity_%s.fits' % (galaxy, metallicity_calib)),
            overwrite=True)
        pix_metal_err_hdu.writeto(
            os.path.join(only_hii_dir, '%s_metallicity_%s_err.fits' % (galaxy, metallicity_calib)),
            overwrite=True)

print('Complete!')
