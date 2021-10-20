# -*- coding: utf-8 -*-
"""
Looks at the azimuthal variation with respect to Querejeta (21) environmental masks

@author: Tom Williams
"""

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from astropy.io import fits
from astropy.table import Table
from reproject import reproject_interp
from scipy.stats import gaussian_kde
import itertools

from vars import top_dir, muse_dir, metallicity_dir, muse_version, gpr_version, plot_dir, extinction_curve, hii_only


def get_mask_idx(mask, flags):
    pixel_mask = np.zeros_like(mask, dtype=bool)
    for flag in flags:
        pixel_mask = (pixel_mask) | (mask == flag)

    return pixel_mask


def plot_normalised_kde(x_kde, val_dict, regions, weight_vals=None, breakout_barred_sample=None):
    y_vals = {}

    for region in regions:

        if not breakout_barred_sample or region not in breakout_barred_sample:

            if weight_vals is not None:
                weights = weight_vals[region]
            else:
                weights = [1] * len(val_dict[region])

            if len(val_dict[region]) > 0:
                y_val = gaussian_kde(val_dict[region], weights=weights, bw_method='silverman').evaluate(x_kde)
                y_val *= len(val_dict[region]) / np.trapz(y_val, x_kde)
                y_vals[region] = y_val

        else:

            if weight_vals is not None:
                weights = weight_vals[region + '_bar']
            else:
                weights = [1] * len(val_dict[region + '_bar'])

            if len(val_dict[region + '_bar']) > 0:
                y_val = gaussian_kde(val_dict[region + '_bar'], weights=weights, bw_method='silverman').evaluate(x_kde)
                y_val *= len(val_dict[region + '_bar']) / np.trapz(y_val, x_kde)
                y_vals[region + ' (Barred)'] = y_val

            if weight_vals is not None:
                weights = weight_vals[region + '_non_bar']
            else:
                weights = [1] * len(val_dict[region + '_non_bar'])

            if len(val_dict[region + '_non_bar']) > 0:
                y_val = gaussian_kde(val_dict[region + '_non_bar'], weights=weights,
                                     bw_method='silverman').evaluate(x_kde)
                y_val *= len(val_dict[region + '_non_bar']) / np.trapz(y_val, x_kde)
                y_vals[region + ' (Non-Barred)'] = y_val

    all_vals = np.zeros(len(x_kde))
    for y_val in y_vals:
        if len(y_vals[y_val]) > 0:
            all_vals += y_vals[y_val]

    norm_factor = np.trapz(all_vals, x_kde)

    y_vals['All'] = all_vals

    for y_val in y_vals:
        y_vals[y_val] /= norm_factor

    return y_vals


matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['font.size'] = 14

os.chdir(top_dir)

n_draws = 1

weight_by_h_alpha = True
breakout_by_bar = ['Centre']

metallicity_calib = 'pg16_scal'

sl_table_file_name = os.path.join(metallicity_dir, muse_version, '%s_ext_curve' % extinction_curve,
                                  'scale_lengths_%s' % metallicity_calib)
if hii_only:
    sl_table_file_name += '_hii_only'
sl_table_file_name += '.fits'
sl_table = Table.read(sl_table_file_name)

mask_dir = os.path.join('environment', 'masks_v5p3_simple')

environment_plot_dir = os.path.join(plot_dir, muse_version, '%s_ext_curve' % extinction_curve, 'environment',
                                    metallicity_calib)
if hii_only:
    environment_plot_dir += '_hii_only'

if not os.path.exists(environment_plot_dir):
    os.makedirs(environment_plot_dir)

# We want to map out the centre (1), bar (2 and 3), spiral arms (5, 6), and disc (4, 7, 8, 9)

regions_of_interest = {'Centre': [1],
                       'Bar': [2, 3],
                       'Spiral Arms': [5, 6],
                       'Disc': [4, 7, 8, 9]}

x_kde = np.linspace(-0.25, 0.15, 1000)
x_lims = x_kde[0], x_kde[-1]

metallicity_vals = {}
# metallicity_errs = {}
weight_vals = {}
for key in regions_of_interest.keys():
    metallicity_vals[key] = []
    # metallicity_errs[key] = []
    weight_vals[key] = []

    metallicity_vals[key + '_bar'] = []
    # metallicity_errs[key + '_bar'] = []
    weight_vals[key + '_bar'] = []

    metallicity_vals[key + '_non_bar'] = []
    # metallicity_errs[key + '_non_bar'] = []
    weight_vals[key + '_non_bar'] = []

# Set up a .fits table to hold the median and spread in azimuthal variation

delta_o_h_tab = Table()
len_table = len(sl_table[sl_table['SIGNIFICANT'] == 1])
row = np.ones(len_table) * np.nan
names = sl_table['GALAXY'][sl_table['SIGNIFICANT'] == 1]

delta_o_h_tab.add_column(names, name='GALAXY')
delta_o_h_tab.add_column(np.zeros(len_table), name='BARRED?')

for region_of_interest in regions_of_interest:

    machine_name = region_of_interest.upper().replace(' ', '_')
    names = [machine_name, machine_name + '_ERR_UP', machine_name + '_ERR_DOWN']

    for name in names:
        delta_o_h_tab.add_column(row, name=name)

# Add a total column

names = ['TOTAL', 'TOTAL_ERR_UP', 'TOTAL_ERR_DOWN']
for name in names:
    delta_o_h_tab.add_column(row, name=name)

for row in sl_table:

    if row['SIGNIFICANT'] == 0:
        continue

    galaxy = row['GALAXY']

    print(galaxy)
    delta_o_h_idx = np.where(delta_o_h_tab['GALAXY'] == galaxy)[0][0]

    palette = itertools.cycle(sns.color_palette('bright'))

    # Read in the environmental mask

    mask_file_name = os.path.join(mask_dir, '%s_simple.fits' % galaxy)
    mask = fits.open(mask_file_name)[0]

    # Read in the Halpha map for optional

    h_alpha_hdu_file_name = os.path.join(muse_dir, muse_version, '%s_MAPS.fits' % galaxy)
    h_alpha_hdu = fits.open(h_alpha_hdu_file_name)['HA6562_FLUX']

    # Read in the azimuthal distribution

    hdu_file_name = os.path.join(metallicity_dir, muse_version, '%s_ext_curve' % extinction_curve, metallicity_calib,
                                 '%s_%s_gpr_pix' % (galaxy, metallicity_calib))
    if hii_only:
        hdu_file_name += '_hii_only'
    hdu_file_name = os.path.join(hdu_file_name, '%s_positions_linear_radial.fits' % galaxy)

    hdu = fits.open(hdu_file_name)[0]

    pred_file = os.path.join(metallicity_dir, muse_version, '%s_ext_curve' % extinction_curve, metallicity_calib,
                             '%s_%s_gpr_pix' % (galaxy, metallicity_calib))
    if hii_only:
        pred_file += '_hii_only'
    pred_file = os.path.join(pred_file, '%s_positions_linear_radial_predict.npy' % galaxy)

    metallicity_preds = np.load(pred_file)
    non_nan_mask = ~np.isnan(hdu.data)
    metallicity_radial_subtract = np.zeros_like(hdu.data)
    metallicity_radial_subtract[metallicity_radial_subtract == 0] = np.nan
    err = np.zeros_like(hdu.data)
    err[metallicity_radial_subtract == 0] = np.nan
    metallicity_radial_subtract[non_nan_mask] = metallicity_preds[0, :]
    err[non_nan_mask] = metallicity_preds[1, :]

    # Reproject the mask to the metallicity map

    mask_interp, _ = reproject_interp(mask, hdu.header, order='nearest-neighbor')

    # Match up values to regions of interest

    plt.figure(figsize=(5, 4))
    ax = plt.subplot(111)
    plot_name = os.path.join(environment_plot_dir, '%s_%s_environment' % (metallicity_calib, galaxy))

    total_region_values = np.zeros_like(x_kde)

    for region_of_interest in regions_of_interest.keys():

        c = next(palette)

        region_flags = regions_of_interest[region_of_interest]

        idx = get_mask_idx(mask_interp[non_nan_mask], region_flags)

        barred_galaxy = False

        for i in regions_of_interest['Bar']:

            if np.nansum(mask_interp[non_nan_mask] == i) > 0:
                barred_galaxy = True
                delta_o_h_tab['BARRED?'][delta_o_h_idx] = 1

        region_values = metallicity_radial_subtract[non_nan_mask][idx]
        region_errs = err[non_nan_mask][idx]

        # Include the error in the measurement
        region_values += np.random.normal(scale=region_errs)

        metallicity_vals[region_of_interest].extend(region_values)
        # metallicity_errs[region_of_interest].extend(region_errs)

        h_alpha_values = h_alpha_hdu.data[non_nan_mask][idx]
        weight_vals[region_of_interest].extend(h_alpha_values)

        if barred_galaxy:
            metallicity_vals[region_of_interest + '_bar'].extend(region_values)
            weight_vals[region_of_interest + '_bar'].extend(h_alpha_values)
        else:
            metallicity_vals[region_of_interest + '_non_bar'].extend(region_values)
            weight_vals[region_of_interest + '_non_bar'].extend(h_alpha_values)

        # Now plot these things

        if len(region_values) > 0:

            if weight_by_h_alpha:
                weights = h_alpha_values
            else:
                weights = [1] * len(region_values)

            y_kde = gaussian_kde(region_values, weights=weights, bw_method='silverman').evaluate(x_kde)

            total_region_values += y_kde

            # Calculate the median and spread of this distribution

            norm_cdf = np.cumsum(y_kde)
            norm_cdf /= np.max(norm_cdf)

            median = x_kde[(np.abs(norm_cdf - 0.5)).argmin()]
            upper_err = x_kde[np.abs(norm_cdf - 0.84).argmin()] - median
            lower_err = median - x_kde[np.abs(norm_cdf - 0.16).argmin()]

            machine_name = region_of_interest.upper().replace(' ', '_')

            delta_o_h_tab[machine_name][delta_o_h_idx] = median
            delta_o_h_tab[machine_name + '_ERR_UP'][delta_o_h_idx] = upper_err
            delta_o_h_tab[machine_name + '_ERR_DOWN'][delta_o_h_idx] = lower_err

            plt.plot(x_kde, y_kde, c=c, label=region_of_interest)

            # for draw in range(n_draws):
            #
            #     region_values_bs = region_values + np.random.normal(scale=region_errs)
            #     y_kde_bs = gaussian_kde(region_values_bs, weights=weights, bw_method='silverman').evaluate(x_kde)
            #     plt.plot(x_kde, y_kde_bs, c=c, alpha=0.5)

    # And overall distribution

    norm_cdf = np.cumsum(total_region_values)
    norm_cdf /= np.max(norm_cdf)

    median = x_kde[(np.abs(norm_cdf - 0.5)).argmin()]
    upper_err = x_kde[np.abs(norm_cdf - 0.84).argmin()] - median
    lower_err = median - x_kde[np.abs(norm_cdf - 0.16).argmin()]

    delta_o_h_tab['TOTAL'][delta_o_h_idx] = median
    delta_o_h_tab['TOTAL_ERR_UP'][delta_o_h_idx] = upper_err
    delta_o_h_tab['TOTAL_ERR_DOWN'][delta_o_h_idx] = lower_err

    plt.fill_between(x_kde, total_region_values, alpha=0.5, color='k', label='All')

    plt.legend(loc='upper left', frameon=False)
    plt.xlabel(r'$\Delta\log_{10}(\mathrm{O/H})$')
    plt.ylabel(r'Probability Density')

    ylims = plt.ylim()
    plt.ylim(0, ylims[-1])

    plt.xlim(x_kde[0], x_kde[-1])

    ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.01))
    ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())

    plt.grid()

    # plt.show()
    plt.savefig(plot_name + '.pdf', bbox_inches='tight')
    plt.savefig(plot_name + '.png', bbox_inches='tight')
    plt.close()

palette = itertools.cycle(sns.color_palette('bright'))

plot_name = os.path.join(environment_plot_dir, '%s_all_environment' % metallicity_calib)

fig = plt.figure(figsize=(5, 4))
ax = plt.subplot(111)

if not weight_by_h_alpha:
    weight_vals = None

y_vals = plot_normalised_kde(x_kde, metallicity_vals, regions_of_interest, weight_vals,
                             breakout_barred_sample=breakout_by_bar)

for y_val in y_vals:

    if y_val == 'All':
        plt.fill_between(x_kde, y_vals[y_val] / np.nanmax(y_vals[y_val]), alpha=0.5, color='k', label=y_val)
    else:

        c = next(palette)
        plt.plot(x_kde, y_vals[y_val] / np.nanmax(y_vals[y_val]), color=c, label=y_val)

plt.legend(loc='upper left', frameon=False)
plt.xlabel(r'$\Delta\log_{10}(\mathrm{O/H})$')
plt.ylabel(r'Probability Density')

plt.ylim(0, 1.1)

plt.xlim(x_lims)

ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.01))
# ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(1))

plt.grid()

plt.tight_layout()

# plt.show()
plt.savefig(plot_name + '.png', bbox_inches='tight')
plt.savefig(plot_name + '.pdf', bbox_inches='tight')
plt.close()

# Save out the offset values

delta_o_h_file_name = os.path.join(metallicity_dir, muse_version, '%s_ext_curve' % extinction_curve,
                                   'delta_o_h_%s' % metallicity_calib)
if hii_only:
    delta_o_h_file_name += '_hii_only'
delta_o_h_file_name += '.fits'

delta_o_h_tab.write(delta_o_h_file_name, overwrite=True)

print('Complete!')
