# -*- coding: utf-8 -*-
"""
Create two-point correlation scale tables.

@author: Tom Williams
"""

import os

import astropy.units as u
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.table import Table
from tqdm import tqdm

from astronomical_gpr import AstronomicalGPR
from funcs import get_info_from_master_table
from vars import top_dir, metallicity_dir, muse_version, phangs_master_table, extinction_curve, \
    hii_only, plot_dir


def calculate_two_point_correlation(x, y, z, tp_scales):
    avg_z = np.nanmean(z)
    sigma_z = np.nanmean((z - avg_z) ** 2)

    tp_corr = np.zeros_like(tp_scales)

    for scale_idx, scale in enumerate(tp_scales):

        averages = np.zeros_like(z)
        averages[averages == 0] = np.nan

        for i in range(len(z)):
            pos_idx = np.where(np.sqrt((x - x[i]) ** 2 + (y - y[i]) ** 2) <= scale)
            averages[i] = np.nanmean(z[i] * z[pos_idx])

        tp_corr[scale_idx] = np.nanmean((averages - avg_z ** 2) / sigma_z)

    return tp_corr


matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['font.size'] = 14

os.chdir(top_dir)

metallicity_calib = 'pg16_scal'
hii_only = False
two_point_step_val = 250
two_point_n_draws = 100
two_point_lims = [0, 5]
two_point_points = int((two_point_lims[1] - two_point_lims[0]) * 10 + 1)
two_point_scales = np.linspace(two_point_lims[0], two_point_lims[1], two_point_points)
target_two_point_corr = 0.5
overwrite_two_point_files = False

two_point = []
two_point_err_up = []
two_point_err_down = []

sl_table_file_name = os.path.join(metallicity_dir, muse_version, '%s_ext_curve' % extinction_curve,
                                  'scale_lengths_%s' % metallicity_calib)
if hii_only:
    sl_table_file_name += '_hii_only'
sl_table_file_name += '.fits'
sl_table = Table.read(sl_table_file_name)

two_point_corr_table_file_name = os.path.join(metallicity_dir, muse_version, '%s_ext_curve' % extinction_curve,
                                              '%spercent_two_point_corr_%s' %
                                              (int(100 * target_two_point_corr), metallicity_calib))
if hii_only:
    two_point_corr_table_file_name += '_hii_only'
two_point_corr_table_file_name += '.fits'

galaxies = list(sl_table[sl_table['SIGNIFICANT'] == 1]['GALAXY'])

galaxy_params = ['dist', 'orient_ra', 'orient_dec', 'orient_posang', 'orient_incl', 'size_r25']

colours = iter(matplotlib.cm.rainbow(np.linspace(0, 1, len(galaxies))))

plot_name = os.path.join(plot_dir, muse_version, '%s_ext_curve' % extinction_curve,
                         '%s_two_point_curve' % metallicity_calib)
if hii_only:
    plot_name += '_hii_only'

fig = plt.figure(figsize=(8, 4))
ax = plt.subplot(111)

for galaxy in galaxies:

    # Calculate the two-point correlation from the GPR

    hdu_file_name = os.path.join(metallicity_dir, muse_version, '%s_ext_curve' % extinction_curve, metallicity_calib,
                                 '%s_%s_gpr_pix' % (galaxy, metallicity_calib))
    if hii_only:
        hdu_file_name += '_hii_only'
    hdu_file_name = os.path.join(hdu_file_name, '%s_positions_linear_radial.fits' % galaxy)
    hdu = fits.open(hdu_file_name)[0]
    hdu_err = fits.open(hdu_file_name.replace('.fits', '_err.fits'))[0]

    dist, ra, dec, pa, inc, r25 = get_info_from_master_table(phangs_master_table, galaxy.lower(), galaxy_params)

    gpr_fitter = AstronomicalGPR(galaxy, hdu, dist, ra, dec, pa, inc, r25)
    gpr_fitter.calculate_pixel_positions()

    # Pull in the predictions file

    pred_file = os.path.join(metallicity_dir, muse_version, '%s_ext_curve' % extinction_curve, metallicity_calib,
                             '%s_%s_gpr_pix' % (galaxy, metallicity_calib))
    if hii_only:
        pred_file += '_hii_only'
    pred_file = os.path.join(pred_file, '%s_positions_linear_radial_predict.npy' % galaxy)

    metallicity_preds = np.load(pred_file)
    non_nan_mask = ~np.isnan(hdu.data)
    metallicity_radial_subtract = np.zeros_like(hdu.data)
    metallicity_radial_subtract[metallicity_radial_subtract == 0] = np.nan
    metallicity_radial_subtract[non_nan_mask] = metallicity_preds[0, :]

    nan_idx = np.where(~np.isnan(metallicity_radial_subtract))
    r25_phys = r25 / 3600 * np.pi / 180 * dist * 1e3

    x = gpr_fitter.x_full[nan_idx].flatten() * r25_phys
    y = gpr_fitter.y_full[nan_idx].flatten() * r25_phys
    z = metallicity_radial_subtract[nan_idx].flatten()
    z_err = hdu_err.data[nan_idx].flatten()

    two_point_corr_galaxy = np.zeros([len(two_point_scales),
                                      two_point_n_draws])

    two_point_dir = os.path.join(metallicity_dir, muse_version, '%s_ext_curve' % extinction_curve, 'two_point',
                                 metallicity_calib, galaxy)
    if hii_only:
        two_point_dir += '_hii_only'
    if not os.path.exists(two_point_dir):
        os.makedirs(two_point_dir)

    for i in tqdm(range(two_point_n_draws), desc=galaxy):

        two_point_file_name = os.path.join(two_point_dir, 'two_point_%d.txt' % i)
        if not os.path.exists(two_point_file_name) or overwrite_two_point_files:

            # Because fitting every point takes an age, also account for random stepping

            idx = np.random.choice(len(z), size=int(len(z) / two_point_step_val), replace=False)

            x_perturb = x[idx]
            y_perturb = y[idx]
            z_perturb = z[idx] + np.random.normal(loc=0, scale=z_err[idx])

            two_point_corr_perturb = calculate_two_point_correlation(x_perturb, y_perturb, z_perturb, two_point_scales)
            np.savetxt(two_point_file_name, two_point_corr_perturb)

        else:
            two_point_corr_perturb = np.loadtxt(two_point_file_name)

        two_point_corr_galaxy[:, i] = two_point_corr_perturb

    two_point_corr_percentiles = np.nanpercentile(two_point_corr_galaxy, [16, 50, 84], axis=-1)

    # OK, so now we figure out the two-point correlation scale

    idx = np.where(np.abs(two_point_corr_percentiles[1, :] - target_two_point_corr) ==
                   np.nanmin(np.abs(two_point_corr_percentiles[1, :] - target_two_point_corr)))[0][0]

    # We'll interpolate to a finer grid around this minimum

    try:

        two_point_scales_interp = np.linspace(two_point_scales[idx - 1], two_point_scales[idx + 1], 201)
        two_point_corr_low_interp = np.interp(two_point_scales_interp, two_point_scales,
                                              two_point_corr_percentiles[0, :])
        two_point_corr_med_interp = np.interp(two_point_scales_interp, two_point_scales,
                                              two_point_corr_percentiles[1, :])
        two_point_corr_high_interp = np.interp(two_point_scales_interp, two_point_scales,
                                               two_point_corr_percentiles[2, :])

        two_point_corr_median = two_point_scales_interp[
            np.abs(two_point_corr_med_interp - target_two_point_corr) ==
            np.nanmin(np.abs(two_point_corr_med_interp - target_two_point_corr))][0]
        two_point_corr_err_up = two_point_scales_interp[
                                    np.abs(two_point_corr_high_interp - target_two_point_corr) ==
                                    np.nanmin(np.abs(two_point_corr_high_interp - target_two_point_corr))][
                                    0] - two_point_corr_median
        two_point_corr_err_down = two_point_corr_median - two_point_scales_interp[
            np.abs(two_point_corr_low_interp - target_two_point_corr) ==
            np.nanmin(np.abs(two_point_corr_low_interp - target_two_point_corr))][0]

    except IndexError:
        two_point_corr_median = two_point_lims[-1]
        two_point_corr_err_up = 0
        two_point_corr_err_down = 0

    two_point.append(two_point_corr_median)
    two_point_err_up.append(two_point_corr_err_up)
    two_point_err_down.append(two_point_corr_err_down)

    # Put on a plot

    c = next(colours)

    plt.plot(two_point_scales, two_point_corr_percentiles[1, :], c=c, label=galaxy)
    plt.fill_between(two_point_scales, two_point_corr_percentiles[2, :], two_point_corr_percentiles[0, :],
                     color=c, alpha=0.5)

plt.xlim(two_point_lims)
plt.ylim(0, 1)

plt.grid()

ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())

plt.xlabel(r'$R$ (kpc)')
plt.ylabel('Two-point metal correlation')

plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1.0), frameon=False)

plt.savefig(plot_name + '.png', bbox_inches='tight')
plt.savefig(plot_name + '.pdf', bbox_inches='tight')
plt.close()

# Write out the table

two_point = np.array(two_point) * u.kpc
two_point_err_up = np.array(two_point_err_up) * u.kpc
two_point_err_down = np.array(two_point_err_down) * u.kpc

two_point_tab = Table([galaxies, two_point, two_point_err_up, two_point_err_down],
                      names=['GALAXY', 'TWO_POINT_CORR', 'TWO_POINT_CORR_ERR_UP', 'TWO_POINT_CORR_ERR_DOWN'])

two_point_tab.write(two_point_corr_table_file_name, overwrite=True)

print('Complete!')
