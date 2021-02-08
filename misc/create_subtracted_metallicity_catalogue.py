# -*- coding: utf-8 -*-
"""
Create a subtracted metallicity catalogue for Kathryn

@author: Tom Williams
"""

import os

import numpy as np
from astropy.table import Table
from astropy.io import fits
import shutil

from vars import top_dir, plot_dir, hii_mask_dir, muse_dir, muse_version, metallicity_dir, galaxies, phangs_master_table
from astronomical_gpr import AstronomicalGPR

os.chdir(top_dir)

table_name = os.path.join('metallicity', 'radial_gpr_subtract_catalogue.fits')
hii_table = Table.read(os.path.join(hii_mask_dir, 'Nebulae_Catalogue.fits'))

if not os.path.exists(os.path.join(plot_dir, 'radial_gprs')):
    os.makedirs(os.path.join(plot_dir, 'radial_gprs'))

metallicity_subtract_table = Table(names=('galaxy', 'region_id', 'metallicity', 'metallicity_err'),
                                   dtype=(str, int, float, float))

for galaxy in galaxies:

    galaxy_metallicity_dir = os.path.join(metallicity_dir, muse_version, galaxy + '_gpr')
    galaxy_plot_dir = os.path.join(plot_dir, muse_version, galaxy + '_gpr')

    radial_gpr_file_name = os.path.join(galaxy_metallicity_dir, galaxy + '_radial_gpr.pkl')

    hdu_file_name = os.path.join(muse_dir, muse_version, galaxy + '_MAPS.fits')
    data_hdu = fits.open(hdu_file_name)['HA6562_FLUX']

    galaxy_edit = galaxy.lower()
    if galaxy == 'NGC628':
        galaxy_edit = 'ngc0628'

    row = phangs_master_table[phangs_master_table['name'] == galaxy_edit]
    dist, ra, dec, pa, inc, r25 = row['dist'][0], row['orient_ra'][0], row['orient_dec'][0], row['orient_posang'][0], \
                                  row['orient_incl'][0], row['size_r25'][0]

    gpr_fitter = AstronomicalGPR(galaxy, data_hdu, dist, ra, dec, pa, inc, r25)
    gpr_fitter.calculate_pixel_positions()

    gpr_fitter.calculate_region_parameters(hii_table, 'met_scal')
    gpr_fitter.fit_radial_distribution_gpr(gpr_file=radial_gpr_file_name, verbose=True)
    gpr_fitter.subtract_radial_fit_gpr(radial_gpr_file_name, n_batch=100)

    for i in range(len(gpr_fitter.region_numbers)):

        region_number = int(gpr_fitter.region_numbers[i])
        metallicity = gpr_fitter.parameter_radial_subtract[i]
        metallicity_err = gpr_fitter.parameter_radial_subtract_err[i]

        metallicity_subtract_table.add_row([galaxy.lower(), region_number, metallicity, metallicity_err])

    in_plot_name = os.path.join(galaxy_plot_dir, galaxy + '_fit_gpr.png')
    out_plot_name = os.path.join(plot_dir, 'radial_gprs', galaxy + '.png')

    shutil.copy(in_plot_name, out_plot_name)

metallicity_subtract_table.write(table_name, overwrite=True)

print('Complete!')
