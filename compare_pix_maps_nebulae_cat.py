# -*- coding: utf-8 -*-
"""
Check to see the pixel-by-pixel maps are similar to the metallicity values from the nebulae catalogues

@author: Tom Williams
"""

import os
import socket

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.table import Table
from reproject import reproject_interp
from uncertainties import unumpy as unp
import seaborn as sns
from scipy.stats import mode

from vars import top_dir, muse_version, galaxies, metallicity_dir, hii_mask_dir, plot_dir, muse_dir, extinction_curve, \
    hii_only


def rolling_average(x_data, y_data, window=50, percentiles=None):

    if percentiles is None:
        percentiles = [16, 50, 84]
    if type(percentiles) != list:
        percentiles = [percentiles]

    # Filter NaNs
    idx = np.where((~np.isnan(x_data)) & (~np.isnan(y_data)))
    x_data, y_data = x_data[idx], y_data[idx]

    # Sort
    idx = np.argsort(x_data)
    x_data, y_data = x_data[idx], y_data[idx]

    rolling_y = np.zeros([len(x_data), len(percentiles)])

    for i in range(len(x_data)):

        start_idx = max(i - int(window / 2), 0)
        stop_idx = min(i + int(window / 2), len(x_data))
        rolling_y[i, :] = np.percentile(y_data[start_idx:stop_idx], percentiles)

    return x_data, rolling_y


if 'mac' in socket.gethostname():
    os.environ['PATH'] += ':/Library/TeX/texbin'

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['text.usetex'] = True
plt.rcParams.update({'text.latex.preamble': r'\usepackage{txfonts}'})

os.chdir(top_dir)
sns.set_color_codes()

metallicity_calib = 'pg16_scal'
weight_param = 'HA6562_FLUX'

lims = [8.105, 8.695]

nebulae_cat = Table.read(os.path.join(hii_mask_dir, 'Nebulae_catalogue_v2.fits'))

colours = iter(matplotlib.cm.rainbow(np.linspace(0, 1, len(galaxies))))

plot_name = os.path.join(plot_dir, muse_version, '%s_ext_curve' % extinction_curve, metallicity_calib)
if not os.path.exists(plot_name):
    os.makedirs(plot_name)
plot_name = os.path.join(plot_name, 'nebulae_pix_comparison')
if hii_only:
    plot_name += '_hii_only'

fig = plt.figure(figsize=(5, 4))
ax1 = fig.add_axes([0.1, 0.3, 0.8, 0.6])
ax2 = fig.add_axes([0.1, 0.1, 0.8, 0.2])

nebulae_mets = []
pix_mets = []

n_pix_per_nebulae = []
discrepant_met_values = []

for galaxy in galaxies:

    galaxy_nebulae_mets = []
    galaxy_nebulae_mets_err = []
    galaxy_pix_mets = []
    galaxy_pix_met_errs = []

    # Read in pixel metallicities

    pix_metal_file_name = os.path.join(metallicity_dir, 'pix_maps', muse_version, '%s_ext_curve' % extinction_curve,
                                       metallicity_calib)
    if hii_only:
        pix_metal_file_name += '_hii_only'
    pix_metal_file_name = os.path.join(pix_metal_file_name, '%s_metallicity_%s.fits' % (galaxy, metallicity_calib))

    pix_metal_hdu = fits.open(pix_metal_file_name)[0]
    pix_metal_err_hdu = fits.open(pix_metal_file_name.replace('.fits', '_err.fits'))[0]

    conv_file_name = os.path.join(muse_dir, muse_version, 'conv', '%s_MAPS.fits' % galaxy)
    conv_hdu = fits.open(conv_file_name)[weight_param]

    # Read in nebulae spatial mask

    nebulae_mask_file_name = os.path.join(hii_mask_dir, 'spatial_masks', '%s_nebulae_mask_V2.fits' % galaxy)
    nebulae_mask_hdu = fits.open(nebulae_mask_file_name)[0]

    nebulae_mask_reproj, _ = reproject_interp(nebulae_mask_hdu, pix_metal_hdu.header, order='nearest-neighbor')

    nebulae_rows = nebulae_cat[(nebulae_cat['gal_name'] == galaxy) & (nebulae_cat['HII_class'] == 1)]

    for row in nebulae_rows:

        galaxy_nebulae_mets.append(row['met_scal'])
        galaxy_nebulae_mets_err.append(row['met_scal_err'])

        idx = np.where(nebulae_mask_reproj == row['region_ID'])
        pix_met_values, pix_met_errs, weights = pix_metal_hdu.data[idx], pix_metal_err_hdu.data[idx], conv_hdu.data[idx]

        pix_met_diff = np.abs(pix_met_values - np.nanmean(pix_met_values))
        if len(pix_met_diff) > 1:
            max_met_diff = np.nanmax(pix_met_diff)
            discrepant_met_values.append(max_met_diff)

        nan_idx = np.where(~np.isnan(pix_met_values))
        if len(nan_idx[0]) > 0:
            n_pix_per_nebulae.append(len(nan_idx[0]))

        pix_met_values, pix_met_errs, weights = pix_met_values[nan_idx], pix_met_errs[nan_idx], weights[nan_idx]
        if len(nan_idx[0]) == 0:
            galaxy_pix_mets.append(np.nan)
            galaxy_pix_met_errs.append(np.nan)
        else:
            unc_arr = unp.uarray(pix_met_values, pix_met_errs)
            mean = sum(unc_arr * weights) / sum(weights)
            galaxy_pix_mets.append(unp.nominal_values(mean))
            galaxy_pix_met_errs.append(unp.std_devs(mean))

    nebulae_mets.extend(galaxy_nebulae_mets)
    pix_mets.extend(galaxy_pix_mets)

    galaxy_nebulae_mets = np.array(galaxy_nebulae_mets)
    galaxy_nebulae_mets_err = np.array(galaxy_nebulae_mets_err)
    galaxy_pix_mets = np.array(galaxy_pix_mets)
    galaxy_pix_met_errs = np.array(galaxy_pix_met_errs)

    c = next(colours)

    # Scatter the raw points

    ax1.errorbar(galaxy_nebulae_mets, galaxy_pix_mets,
                 xerr=galaxy_nebulae_mets_err, yerr=galaxy_pix_met_errs,
                 c='b', ls='none', marker='o', mfc='none', mew=0,
                 rasterized=True)

    # Scatter the residuals

    residual = galaxy_pix_mets - galaxy_nebulae_mets
    residual_err = np.sqrt(galaxy_nebulae_mets_err ** 2 + galaxy_pix_met_errs ** 2)

    ax2.errorbar(galaxy_nebulae_mets, residual,
                 xerr=galaxy_nebulae_mets_err, yerr=residual_err,
                 c='b', ls='none', marker='o', mfc='none', mew=0,
                 rasterized=True)

print('mode: %s, median: %s, mean: %s, max: %s'
      % (mode(n_pix_per_nebulae), np.median(n_pix_per_nebulae),
         np.mean(n_pix_per_nebulae), np.max(n_pix_per_nebulae))
      )
print(np.nanmax(discrepant_met_values))
no

nebulae_mets = np.array(nebulae_mets)
pix_mets = np.array(pix_mets)

residuals = pix_mets - nebulae_mets
residuals_percentiles = np.nanpercentile(residuals, [16, 50,  84])
residuals_std = np.nanstd(residuals)

roll_x, roll_y = rolling_average(nebulae_mets, residuals)
ax2.plot(roll_x, roll_y[:, 1], c='k', zorder=999)
ax2.fill_between(roll_x, roll_y[:, 0], roll_y[:, -1], color='k', zorder=998, alpha=0.5)

# lims = np.nanpercentile(nebulae_mets, [0.1, 99.9])

ax1.plot(lims, lims, c='k', ls='--', zorder=99)
ax2.axhline(0, c='k', ls='--', zorder=99)
# ax2.axhline(residuals_percentiles[1], c='powderblue', zorder=101)
# ax2.fill_between([8, 9], residuals_percentiles[0], residuals_percentiles[-1],
#                  color='powderblue', alpha=0.75, zorder=99)

plt.text(0.05, 0.95, r'$\Delta=%.2f,\,\sigma=%.2f$' % (residuals_percentiles[1], residuals_std),
         ha='left', va='top',
         transform=ax1.transAxes)

ax1.axes.set_xlim(lims)
ax1.axes.set_ylim(lims)

ax2.axes.set_xlim(lims)
ax2.axes.set_ylim([-0.075, 0.075])

ax1.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())

ax2.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
ax2.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())

ax1.grid()
ax2.grid()

ax1.set_ylabel(r'$12+\log_{10}(\mathrm{O/H})_\mathrm{pix}$')

ax2.set_xlabel(r'$12+\log_{10}(\mathrm{O/H})_\textsc{Hii}$')
ax2.set_ylabel(r'$\Delta$')

# plt.show()

plt.savefig(plot_name + '.png', bbox_inches='tight')
plt.savefig(plot_name + '.pdf', bbox_inches='tight')

print('Complete!')
