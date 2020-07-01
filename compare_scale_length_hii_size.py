# -*- coding: utf-8 -*-
"""
Compare GPR scale length with average HII region size

@author: Tom Williams
"""

import os

from astropy.table import Table
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from scipy.stats import kendalltau
from astropy.io import fits
from astropy.wcs import WCS
import dill

from vars import top_dir, metallicity_dir, hii_mask_dir, phangs_master_table, plot_dir, galaxies

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['font.size'] = 14

os.chdir(top_dir)

# Read in scale lengths

scale_length_f = open('metallicity/scale_lengths.txt', 'r')

nebula_table = Table.read(hii_mask_dir + 'Nebulae_Catalogue.fits')

nebula_sizes = []
scale_lengths = []
dists = []
sfrs = []
m_stars = []

use_hii_regions = True

for galaxy in galaxies:
    # Filter to galaxy, and only good metallicities

    nebula_rows = nebula_table[nebula_table['gal_name'] == galaxy]
    nebula_rows = nebula_rows[~np.isnan(nebula_rows['met_scal'])]

    avg_nebula_size = np.nanmedian(nebula_rows['region_circ_rad'])
    avg_nebula_size = avg_nebula_size

    # Convert this into a physical distance

    try:
        gal_edit = {'NGC628': 'NGC0628'}[galaxy]
    except KeyError:
        gal_edit = galaxy

    gal_edit = gal_edit.lower()

    row = phangs_master_table[phangs_master_table['name'] == gal_edit]
    dist, sfr, m_star = row['dist'][0], row['props_sfr'][0], row['props_mstar'][0]

    avg_nebula_size *= dist * 1e3 * np.pi / 180 / 3600

    nebula_sizes.append(avg_nebula_size)

    scale_length = np.nan

    for line in scale_length_f:
        if galaxy in line:
            line = line.strip()
            cols = line.split()
            scale_length = float(cols[-1])
            break

    scale_lengths.append(scale_length)

    dists.append(dist)
    sfrs.append(np.log10(sfr))
    m_stars.append(np.log10(m_star))

nebula_sizes = np.array(nebula_sizes)
scale_lengths = np.array(scale_lengths)

# KDE plot of scale length distributions

plt.figure(figsize=(6, 4))

sns.kdeplot(scale_lengths, color='k', shade=True)

plt.xlabel(r'$\sigma_l$ (kpc)')
plt.ylabel('Normalised Probability Density')

plt.tight_layout()

# plt.show()

plt.savefig(plot_dir + 'scale_length_kde.pdf',
            bbox_inches='tight')
plt.savefig(plot_dir + 'scale_length_kde.png',
            bbox_inches='tight')

plt.close()

plt.figure(figsize=(6, 4))
ax = plt.subplot(1, 1, 1)

tau = kendalltau(nebula_sizes, scale_lengths / nebula_sizes)

plt.scatter(nebula_sizes, scale_lengths / nebula_sizes, c=dists)

plt.text(0.05, 0.95, r'$\tau = %.2f, p = %.2f$' % tau,
         ha='left', va='top', transform=ax.transAxes)

plt.xlabel(r'$\langle R_\mathrm{nebula}\rangle$ (kpc)')
plt.ylabel(r'$\sigma_l/\langle R_\mathrm{nebula}\rangle$')

plt.colorbar(pad=0, label='Dist (Mpc)')

# plt.show()

plt.savefig(plot_dir + 'scale_length_region_size.pdf',
            bbox_inches='tight')
plt.savefig(plot_dir + 'scale_length_region_size.png',
            bbox_inches='tight')

plt.close()

# Some other scatter plots

plt.figure(figsize=(8, 3))

ax1 = plt.subplot(1, 2, 1)

plt.scatter(sfrs, scale_lengths, c=m_stars)

plt.xlabel(r'$\log10(\mathregular{SFR} [M_\odot\,\mathregular{yr}^{-1}])$')
plt.ylabel(r'$\sigma_l$ (kpc)')

tau = kendalltau(sfrs, scale_lengths)
plt.text(0.05, 0.95, r'$\tau = %.2f, p = %.2f$' % tau,
         ha='left', va='top', transform=ax1.transAxes)

plt.colorbar(pad=0, label=r'$\log10(M_\ast [M_\odot])$')

ax1 = plt.subplot(1, 2, 2)

plt.scatter(m_stars, scale_lengths, c=sfrs)

plt.xlabel(r'$\log10(M_\ast [M_\odot])$')
plt.ylabel(r'$\sigma_l$ (kpc)')

tau = kendalltau(m_stars, scale_lengths)
plt.text(0.05, 0.95, r'$\tau = %.2f, p = %.2f$' % tau,
         ha='left', va='top', transform=ax1.transAxes)

plt.colorbar(pad=0, label=r'$\log10(\mathregular{SFR} [M_\odot\,\mathregular{yr}^{-1}])$')

plt.tight_layout()

# plt.show()

plt.savefig(plot_dir + 'sfr_m_star_corr.pdf',
            bbox_inches='tight')
plt.savefig(plot_dir + 'sfr_m_star_corr.png',
            bbox_inches='tight')

plt.close()

# Also create a plot comparing Kathryn's results to mine. Put through (essentially) the same code.

k20_50percent_correlation_scales = []
k20_30percent_correlation_scales = []

my_50percent_correlation_scales = []

if not os.path.exists(os.path.join(metallicity_dir, 'correlations')):
    os.makedirs(os.path.join(metallicity_dir, 'correlations'))

scales = np.arange(0, 2.1, 0.1)

for galaxy in galaxies:

    try:
        correlation_scale = {'NGC628': 210,
                             'NGC1087': 370,
                             'NGC1672': 310,
                             'NGC2835': 280,
                             'NGC3627': 420,
                             'NGC4254': 350,
                             'NGC4535': 360,
                             'IC5332': 310}[galaxy]
    except KeyError:
        correlation_scale = np.nan

    k20_50percent_correlation_scales.append(correlation_scale * 1e-3)

    try:
        correlation_scale = {'NGC628': 320,
                             'NGC1087': 580,
                             'NGC1672': 460,
                             'NGC2835': 440,
                             'NGC3627': 1010,
                             'NGC4254': 620,
                             'NGC4535': 620,
                             'IC5332': 550}[galaxy]
    except KeyError:
        correlation_scale = np.nan

    k20_30percent_correlation_scales.append(correlation_scale * 1e-3)

    corr_file_name = os.path.join(metallicity_dir, 'correlations', galaxy + '_correlations')
    if use_hii_regions:
        corr_file_name += '_hii_reg'
    corr_file_name += '.txt'

    if not os.path.exists(corr_file_name):

        # Calculate two-point correlation function for our data

        hdu_file_name = os.path.join(metallicity_dir, galaxy + '_gpr', galaxy + '_regions_metallicity.fits')
        hdu = fits.open(hdu_file_name)[0]
        w = WCS(hdu_file_name)
        pix_size = np.abs(hdu.header['CD1_1']) * 3600

        # Calculate distances

        galaxy_edit = galaxy
        if galaxy == 'NGC628':
            galaxy_edit = 'NGC0628'

        row = phangs_master_table[phangs_master_table['name'] == galaxy_master_table.lower()]

        dist, ra, dec, pa, incl, r25 = row['dist'][0], row['orient_ra'][0], row['orient_dec'][0], \
            row['orient_posang'][0], row['orient_incl'][0], row['size_r25'][0]

        # From this, calculate a deprojected physical distance.

        x_cen, y_cen = w.all_world2pix(ra, dec, 1)

        # Let's try using HII regions instead of all the pixels

        if use_hii_regions:

            hii_rows = nebula_table[nebula_table['gal_name'] == galaxy]
            hii_rows = hii_rows[~np.isnan(hii_rows['met_scal'])]

            xi, yi = np.array(hii_rows['cen_ra']), np.array(hii_rows['cen_dec'])
            metallicities = np.array(hii_rows['met_scal'])

            xi -= ra
            yi -= dec

            xi *= np.pi / 180 * dist * 1e3
            yi *= np.pi / 180 * dist * 1e3

            nan_mask = []

        else:

            xi, yi = np.meshgrid((np.arange(hdu.data.shape[1]) - x_cen),
                                 (np.arange(hdu.data.shape[0]) - y_cen))

            xi *= pix_size / 3600 * np.pi / 180 * dist * 1e3
            yi *= pix_size / 3600 * np.pi / 180 * dist * 1e3

            metallicities = hdu.data
            nan_mask = np.where(np.isnan(hdu.data))

        # Convert these positions to physical positions (kpc), accounting for inclination and rotation

        r25 = r25 / 3600 * np.pi / 180 * dist * 1e3

        angle = pa * np.pi / 180

        x_rot = xi * cos_a - yi * sin_a
        y_rot = xi * sin_a + yi * cos_a

        # Account for inclination

        y_rot /= np.cos(incl * np.pi / 180)

        r_full = np.sqrt(x_rot ** 2 + y_rot ** 2)
        r_full[nan_mask] = np.nan

        # Read in radial gradient

        mcmc_file_name = os.path.join(metallicity_dir, galaxy + '_gpr', galaxy + '_mcmc.pkl')

        with open(mcmc_file_name, 'rb') as sampler_f:
            sampler = dill.load(sampler_f)

        n_walkers, n_steps = 500, 500
        flat_samples = sampler.get_chain(discard=int(n_steps / 2), flat=True)
        m_median, r_0_median, int_scatter_median = np.nanmedian(flat_samples, axis=0)

        data_radial_subtract = metallicities - (m_median * r_full / r25 + r_0_median)

        non_nan_mask = np.where(~np.isnan(data_radial_subtract))

        if use_hii_regions:
            step = 1
        else:
            step = 50

        data_flat = data_radial_subtract[non_nan_mask].flatten()[::step]
        r_flat = r_full[non_nan_mask].flatten()[::step]
        x_flat = x_rot[non_nan_mask].flatten()[::step]
        y_flat = y_rot[non_nan_mask].flatten()[::step]

        avg_metal = np.nanmean(data_flat)
        sigma_metal = np.nanmean((data_flat - avg_metal) ** 2)

        correlations = np.zeros_like(scales)

        for scale_idx, scale in enumerate(scales):

            print(scale)

            averages = np.zeros_like(data_flat)
            averages[averages == 0] = np.nan

            for i in range(len(data_flat)):
                pos_idx = np.where(np.sqrt((x_flat - x_flat[i]) ** 2 + (y_flat - y_flat[i]) ** 2) <= scale)
                averages[i] = np.nanmean(data_flat[i] * data_flat[pos_idx])

            correlations[scale_idx] = np.nanmean((averages - avg_metal ** 2) / sigma_metal)

        np.savetxt(corr_file_name,
                   np.c_[scales, correlations])

    else:

        correlations = np.loadtxt(corr_file_name,
                                  usecols=1, unpack=True)

    # Interpolate this to 10pc steps, rather than 100

    x_interp = np.arange(0, 2.01, 0.01)
    corr_interp = np.interp(x_interp, scales, correlations)

    try:
        my_50percent_correlation_scales.append(x_interp[np.where(corr_interp <= 0.5)[0][0]])
    except IndexError:
        my_50percent_correlation_scales.append(np.nan)

k20_50percent_correlation_scales = np.array(k20_50percent_correlation_scales)

plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)

plt.scatter(scale_lengths, k20_50percent_correlation_scales, c='k')

plt.xlabel(r'$\sigma_l$ (kpc)')
plt.ylabel(r'50% Correlation Scale (kpc; K20)')

plt.subplot(1, 2, 2)

xmin, xmax = 0, 2

plt.scatter(my_50percent_correlation_scales, k20_50percent_correlation_scales, c='k')
plt.plot([xmin, xmax], [xmin, xmax], c='k', ls='--')

plt.xlim([xmin, xmax])
plt.ylim([xmin, xmax])

plt.xlabel(r'50% Correlation Scale (kpc; this work)')
plt.ylabel(r'50% Correlation Scale (kpc; K20)')

# plt.legend(loc='upper left')
plt.tight_layout()

# plt.show()

plot_name = plot_dir + 'k20_comparison'
if use_hii_regions:
    plot_name += '_hii_regions'

plt.savefig(plot_name + '.pdf',
            bbox_inches='tight')
plt.savefig(plot_name + '.png',
            bbox_inches='tight')

plt.close()

print('Complete!')
