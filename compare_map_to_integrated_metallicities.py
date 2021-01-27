# -*- coding: utf-8 -*-
"""
Compare the map to what we'd get from the nebula catalogues

@author: Tom Williams
"""

import os

from astropy.io import fits
from astropy.table import Table
import numpy as np
import matplotlib.pyplot as plt
import cmocean

from vars import top_dir, muse_version, muse_dir, plot_dir, galaxies, metallicity_dir, hii_mask_dir

os.chdir(top_dir)

galaxies = ['IC5332']

metallicity_tab = Table.read(os.path.join(hii_mask_dir, 'Nebulae_Catalogue.fits'))

for galaxy in galaxies:

    metallicity_rows = metallicity_tab[metallicity_tab['gal_name'] == galaxy]

    # Read in metallicity map

    hdu_name = os.path.join(metallicity_dir, muse_version, galaxy + '_gpr_pix', galaxy + '_positions.fits')
    metallicity_hdu = fits.open(hdu_name)[0]

    # Read in metallicities and region mask
    mask_name = os.path.join(hii_mask_dir, 'spatial_masks', galaxy + '_HIIreg_mask.fits')
    mask = fits.open(mask_name)[0]

    metallicity_regions = np.zeros_like(mask.data)
    metallicity_regions[metallicity_regions == 0] = np.nan

    idxs = np.where(~np.isnan(metallicity_rows['met_scal']))

    for idx in idxs[0]:

        region_id = metallicity_rows[idx]['region_ID']
        metallicity_regions[mask.data == region_id] = metallicity_rows[idx]['met_scal']

    # Calculate residual
    metallicity_residual = metallicity_regions - metallicity_hdu.data

    vmin = np.nanpercentile(metallicity_residual, 16)

    plt.figure()
    plt.imshow(metallicity_residual, origin='lower', vmin=vmin, vmax=-vmin, cmap=cmocean.cm.balance)
    plt.colorbar(label=r'$\Delta(\mathrm{O/H})$')
    plt.show()





