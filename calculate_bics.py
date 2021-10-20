# -*- coding: utf-8 -*-
"""
Let's have a look at calculating BICs for these radial/higher-order mapps

@author: Tom Williams
"""

import os
import socket
import time

import cmocean
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from uncertainties import ufloat
from uncertainties import unumpy as unp

from astronomical_gpr import AstronomicalGPR
from funcs import get_info_from_master_table
from vars import top_dir, plot_dir, hii_mask_dir, muse_dir, muse_version, metallicity_dir, phangs_master_table, \
    use_pix_maps, include_radial_subtract, gpr_step_val, use_regions, simulation, metallicity_calibs, use_conv, \
    galaxies, hii_only, extinction_curve


def ln_like_radial(m, c, scatter, x, y, y_err):
    """Likelihood function for emcee straight line fitting, including intrinsic scatter."""
    model = m * x + c
    total_err2 = y_err ** 2 + scatter ** 2

    return -0.5 * np.sum((y - model) ** 2 / total_err2 + np.log(2 * np.pi * total_err2))


def ln_like_gpr(y, y_err, model, model_err):
    total_err2 = y_err ** 2 + model_err ** 2

    return -0.5 * np.sum((y - model) ** 2 / total_err2 + np.log(2 * np.pi * total_err2))


matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['font.size'] = 14

os.chdir(top_dir)

emission_to_use = 'HA6562_FLUX'
galaxy_params = ['dist', 'orient_ra', 'orient_dec', 'orient_posang', 'orient_incl', 'size_r25']
metallicity_calib = 'pg16_scal'

scale_length_file_name = os.path.join(metallicity_dir, muse_version, '%s_ext_curve' % extinction_curve,
                                      'scale_lengths_%s' % metallicity_calib)
if hii_only:
    scale_length_file_name += '_hii_only'
scale_length_file_name += '.fits'
scale_length_table = Table.read(scale_length_file_name)

chisqs = []
delta_bics = []
fc = []

for galaxy in galaxies:

    print(galaxy)

    if scale_length_table[scale_length_table['GALAXY'] == galaxy]['SIGNIFICANT'] == 1:
        fc.append('k')
    else:
        fc.append('none')

    galaxy_metallicity_dir = os.path.join(metallicity_dir, muse_version, '%s_ext_curve' % extinction_curve,
                                          metallicity_calib, galaxy + '_' + metallicity_calib + '_gpr')
    if use_pix_maps:
        galaxy_metallicity_dir += '_pix'
    if hii_only:
        galaxy_metallicity_dir += '_hii_only'

    galaxy_plot_dir = os.path.join(plot_dir, muse_version, '%s_ext_curve' % extinction_curve,
                                   metallicity_calib, galaxy + '_' + metallicity_calib + '_gpr')
    if use_pix_maps:
        galaxy_plot_dir += '_pix'
    if hii_only:
        galaxy_plot_dir += '_hii_only'

    # Native resolution MUSE maps
    hdu_file_name = os.path.join(muse_dir, muse_version, galaxy + '_MAPS.fits')
    data_hdu = fits.open(hdu_file_name)[emission_to_use]
    wcs = WCS(data_hdu.header)

    # Get a simple (x, y, z) to contour to show data coverage
    coverage_hdu = fits.open(hdu_file_name)[1]
    x_grid, y_grid = np.meshgrid(np.arange(coverage_hdu.data.shape[1]),
                                 np.arange(coverage_hdu.data.shape[0]))
    z_grid = np.zeros_like(coverage_hdu.data)
    z_grid[~np.isnan(coverage_hdu.data)] = 1

    dist, ra, dec, pa, inc, r25 = get_info_from_master_table(phangs_master_table, galaxy.lower(), galaxy_params)

    mcmc_file_name = os.path.join(galaxy_metallicity_dir, galaxy)
    mcmc_file_name += '_mcmc.pkl'

    gpr_fitter = AstronomicalGPR(galaxy, data_hdu, dist, ra, dec, pa, inc, r25)
    gpr_fitter.calculate_pixel_positions()
    pix_map_file_name = os.path.join(metallicity_dir, 'pix_maps', muse_version,
                                     '%s_ext_curve' % extinction_curve, metallicity_calib)
    if hii_only:
        pix_map_file_name += '_hii_only'
    pix_map_file_name = os.path.join(pix_map_file_name,
                                     galaxy + '_metallicity_' + metallicity_calib + '.fits')
    pix_map = fits.open(pix_map_file_name)[0]
    pix_map_err = fits.open(pix_map_file_name.replace('.fits', '_err.fits'))[0]
    gpr_fitter.calculate_pixel_parameters(pix_map, pix_map_err, step_val=gpr_step_val)

    # Fit the radial distribution
    gpr_fitter.fit_radial_distribution(mcmc_file_name=mcmc_file_name)
    gpr_fitter.subtract_radial_fit()

    # Set up pixels to fit

    gpr_fitter.calc_xy_to_fit_positions(use_radial=include_radial_subtract)

    gpr_file = os.path.join(galaxy_metallicity_dir, galaxy)
    gpr_file += '_positions'
    gpr_file += '_linear_radial'
    gpr_file += '_gpr.pkl'

    # Throw into GPR

    pred_file = gpr_file.replace('_gpr.pkl', '_predict.npy')
    pred_file_central = gpr_file.replace('_gpr.pkl', '_predict_central.npy')

    # Skip reading in/generating GPR files if running on laptop

    if 'astro-node' in socket.gethostname():
        gpr_fitter.fit_gpr_regressor(gpr_file)

    gpr_fitter.make_predictions(pred_file_central, gpr_fitter.xy_regions, name='central')

    # Now it's likelihood time

    m, r_0, scatter = gpr_fitter.m, gpr_fitter.r_0, gpr_fitter.intrinsic_scatter

    r, metallicity, metallicity_err = gpr_fitter.r_to_fit, gpr_fitter.parameter, gpr_fitter.parameter_err
    metallicity_radial_subtract = gpr_fitter.parameter_radial_subtract
    metallicity_gpr_subtract = gpr_fitter.predictions['central'][0, :]
    metallicity_gpr_subtract_err = gpr_fitter.predictions['central'][1, :]

    radial_chisq = np.nansum((metallicity_radial_subtract / metallicity_err) ** 2)
    gpr_chisq = np.nansum(((metallicity_gpr_subtract - metallicity_radial_subtract) / metallicity_err) ** 2)
    chisqs.append(gpr_chisq / radial_chisq)

    n_pix = len(r)

    radial_like = ln_like_radial(m, r_0, scatter, r, metallicity, metallicity_err)
    gpr_like = ln_like_gpr(metallicity_radial_subtract, metallicity_err,
                           metallicity_gpr_subtract, metallicity_gpr_subtract_err)

    delta_bic = -2 * radial_like + 2 * gpr_like - 3 * np.log(n_pix)
    delta_bics.append(delta_bic)

# Look at the difference in R^2 values

plot_name = os.path.join(plot_dir, muse_version, '%s_ext_curve' % extinction_curve, metallicity_calib,
                         'model_comparisons_%s_bic' % metallicity_calib)
if use_pix_maps:
    plot_name += '_pix_maps'
if hii_only:
    plot_name += '_hii_only'

ylims_chisq = [10 ** -1.3, 10 ** 1.3]
ylims_bic = [-10 ** 5.9, 10 ** 5.9]

fig, axes = plt.subplots(2, 1, sharex='all', figsize=(12, 6))

ax1 = axes[0]

ax1.axhline(1, c='k', ls='--')

ax1.text(0, 1e1, 'Radial model better describes data', ha='left', va='bottom')
ax1.text(0, 1e-1, 'Radial and higher order model better describes data', ha='left', va='top')

ax1.scatter(np.arange(len(galaxies)), chisqs, edgecolors='k', facecolors=fc)

ax1.set_xticks(np.arange(len(galaxies)))
ax1.set_xticklabels(galaxies, rotation=45, ha='left')
ax1.xaxis.tick_top()
ax1.tick_params(labeltop=True)
ax1.set_yscale('log')

ax1.set_ylabel(r'$\frac{\chi^2_\mathrm{Radial+Higher Order}}{\chi^2_\mathrm{Radial}}$', fontsize=22)
ax1.set_ylim(ylims_chisq)

ax1.grid(axis='both')

plt.grid()

# plt.show()

ax2 = axes[1]

ax2.axhline(0, c='k', ls='--')

ax2.text(0, 10**4.5, 'Radial and higher order model preferred', ha='left', va='bottom')
ax2.text(0, -10**4.5, 'Radial model preferred', ha='left', va='top')

ax2.scatter(np.arange(len(galaxies)), delta_bics, edgecolors='k', facecolors=fc)

# ax2.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.1))
# ax2.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.5))

ax2.set_yscale('symlog')

ax2.set_xticks(np.arange(len(galaxies)))
ax2.set_xticklabels(galaxies, rotation=45, ha='right')

ax2.set_ylabel(r'$\Delta\mathrm{BIC}$')
ax2.set_ylim(ylims_bic)
ax2.set_yticks([-10 ** 5, -10 ** 3, -10 ** 1, 0, 10 ** 1, 10 ** 3, 10 ** 5])

ax2.grid(b=True, axis='both')

plt.subplots_adjust(wspace=0, hspace=0)

# plt.show()

plt.savefig(plot_name + '.pdf', bbox_inches='tight')
plt.savefig(plot_name + '.png', bbox_inches='tight')

plt.close()

print('Complete!')
