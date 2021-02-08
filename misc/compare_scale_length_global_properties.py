# -*- coding: utf-8 -*-
"""
Compare GPR scale length with various global properties (HII region size, SFR etc.)

@author: Tom Williams
"""

import os

import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from astropy.table import Table
from scipy.stats import kendalltau

from funcs import add_master_table_info
from vars import top_dir, hii_mask_dir, muse_version, phangs_master_table, plot_dir, galaxies


def coloured_errorbar_plot(ax, x_data, y_data, x_err, y_err, x_label, y_label, c, c_label, include_tau=True):
    plt.scatter(x_data, y_data, c=c)

    # Account for symmetrical errorbars

    if len(np.array(x_err).shape) == 1:
        x_err = [x_err, x_err]
    if len(np.array(y_err).shape) == 1:
        y_err = [y_err, y_err]

    norm = matplotlib.colors.Normalize(vmin=min(c), vmax=max(c), clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap='viridis')
    colours = np.array([(mapper.to_rgba(v)) for v in c])

    for i, colour in enumerate(colours):
        plt.errorbar(x_data[i], y_data[i],
                     xerr=np.array([[x_err[0][i], x_err[1][i]]]).T,
                     yerr=np.array([[y_err[0][i], y_err[1][i]]]).T,
                     c=colours[i])

    if include_tau:
        tau = kendalltau(x_data, y_data)
        tau = [tau[0], tau[1]]
        if tau[1] < 0.01:
            tau[1] = '<0.01'
        else:
            tau[1] = '=%.2f' % tau[1]
        plt.text(0.05, 0.95, r'$\tau = %.2f, p%s$' % (tau[0], tau[1]),
                 ha='left', va='top', transform=ax.transAxes,
                 bbox=dict(facecolor='white', edgecolor='none', alpha=0.75))

    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.colorbar(pad=0, label=c_label)


matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['font.size'] = 14

os.chdir(top_dir)

arcsec_to_kpc = 1e3 * np.pi / 180 / 3600

use_hii_regions = True

# Read in scale lengths TODO: Update this to the release version.

scale_length_table = Table.read(os.path.join('metallicity', 'scale_lengths_corrected.fits') )

nebula_table = Table.read(os.path.join(hii_mask_dir, 'Nebulae_Catalogue.fits'))

galaxy_info = {}

for galaxy in galaxies:

    idx = np.where(scale_length_table['GALAXY'] == galaxy)[0][0]

    galaxy_info[galaxy] = {'scale_length': scale_length_table['SIGMA_L'][idx],
                           'scale_length_err_up': scale_length_table['SIGMA_L_ERR_UP'][idx],
                           'scale_length_err_down': scale_length_table['SIGMA_L_ERR_DOWN'][idx]}

    idx = np.where(nebula_table['gal_name'] == galaxy)[0]
    nebula_rows = nebula_table[idx]
    nebula_rows = nebula_rows[~np.isnan(nebula_rows['met_scal'])]
    avg_nebula_size = np.nanmedian(nebula_rows['region_circ_rad'])

    galaxy_info[galaxy]['avg_nebula_size'] = avg_nebula_size

    # Pull info out from the master table

    gal_edit = galaxy.lower()
    if galaxy == 'NGC628':
        gal_edit = 'ngc0628'

    cols_to_include = ['dist', 'props_sfr', 'props_mstar', 'size_r25']
    galaxy_info = add_master_table_info(phangs_master_table, galaxy_info, gal_edit, cols_to_include,
                                        original_galaxy_name=galaxy)

    # Convert r_25 and nebula sizes to physical quantities

    galaxy_info[galaxy]['avg_nebula_size'] *= galaxy_info[galaxy]['dist'] * arcsec_to_kpc
    galaxy_info[galaxy]['size_r25'] *= galaxy_info[galaxy]['dist'] * arcsec_to_kpc

    # Roll in uncertainties in distances to nebula sizes and scale lengths

    dist = galaxy_info[galaxy]['dist']
    dist_err = galaxy_info[galaxy]['dist_err']

    dist_err_up = 10 ** (np.log10(dist) + dist_err) - dist
    dist_err_down = dist - 10 ** (np.log10(dist) - dist_err)

    galaxy_info[galaxy]['avg_nebula_size_err_up'] = galaxy_info[galaxy]['avg_nebula_size'] * dist_err_up / dist
    galaxy_info[galaxy]['avg_nebula_size_err_down'] = galaxy_info[galaxy]['avg_nebula_size'] * dist_err_down / dist

    scale_length = galaxy_info[galaxy]['scale_length']
    scale_length_err_up = galaxy_info[galaxy]['scale_length_err_up']
    scale_length_err_down = galaxy_info[galaxy]['scale_length_err_down']

    scale_length_err_up = scale_length * np.sqrt((scale_length_err_up / scale_length) ** 2 +
                                                 (dist_err_up / dist) ** 2)
    scale_length_err_down = scale_length * np.sqrt((scale_length_err_down / scale_length) ** 2 +
                                                   (dist_err_down / dist) ** 2)

    galaxy_info[galaxy]['scale_length_err_up'] = scale_length_err_up
    galaxy_info[galaxy]['scale_length_err_down'] = scale_length_err_down

    # Convert SFR and Mstar to surface densities

    galaxy_info[galaxy]['surface_density_sfr'] = np.log10(galaxy_info[galaxy]['props_sfr'] /
                                                          galaxy_info[galaxy]['size_r25'] ** 2)
    galaxy_info[galaxy]['surface_density_mstar'] = np.log10(galaxy_info[galaxy]['props_mstar'] /
                                                            galaxy_info[galaxy]['size_r25'] ** 2)

    # Also normalise nebula sizes and scale lengths by r_25, and also propagate errors through

    galaxy_info[galaxy]['normalised_nebula_size'] = galaxy_info[galaxy]['avg_nebula_size'] / \
                                                    galaxy_info[galaxy]['size_r25']
    galaxy_info[galaxy]['normalised_scale_length'] = galaxy_info[galaxy]['scale_length'] / \
                                                     galaxy_info[galaxy]['size_r25']

    galaxy_info[galaxy]['normalised_nebula_size_err_up'] = galaxy_info[galaxy]['avg_nebula_size_err_up'] / \
                                                           galaxy_info[galaxy]['size_r25']
    galaxy_info[galaxy]['normalised_scale_length_err_up'] = galaxy_info[galaxy]['scale_length_err_up'] / \
                                                            galaxy_info[galaxy]['size_r25']

    galaxy_info[galaxy]['normalised_nebula_size_err_down'] = galaxy_info[galaxy]['avg_nebula_size_err_down'] / \
                                                             galaxy_info[galaxy]['size_r25']
    galaxy_info[galaxy]['normalised_scale_length_err_down'] = galaxy_info[galaxy]['scale_length_err_down'] / \
                                                              galaxy_info[galaxy]['size_r25']

scale_lengths = np.array([galaxy_info[galaxy]['scale_length'] for galaxy in galaxies])

# KDE plot of scale length distributions

plot_name = os.path.join(plot_dir, muse_version, 'scale_length_kde')

plt.figure(figsize=(4, 3))

sns.kdeplot(scale_lengths, color='k', shade=True, legend=None)

plt.xlabel(r'$\sigma_l$ (kpc)')
plt.ylabel('Normalised Probability Density')

plt.tight_layout()

# plt.show()

plt.savefig(plot_name + '.pdf', bbox_inches='tight')
plt.savefig(plot_name + '.png', bbox_inches='tight')

plt.close()

# Plot scale lengths versus regions sizes

normalised_nebula_sizes = [galaxy_info[galaxy]['normalised_nebula_size'] for galaxy in galaxies]
normalised_scale_lengths = [galaxy_info[galaxy]['normalised_scale_length'] for galaxy in galaxies]

normalised_nebula_sizes_err_down = [galaxy_info[galaxy]['normalised_nebula_size_err_down'] for galaxy in galaxies]
normalised_nebula_sizes_err_up = [galaxy_info[galaxy]['normalised_nebula_size_err_up'] for galaxy in galaxies]

normalised_scale_lengths_err_down = [galaxy_info[galaxy]['normalised_scale_length_err_down'] for galaxy in galaxies]
normalised_scale_lengths_err_up = [galaxy_info[galaxy]['normalised_scale_length_err_up'] for galaxy in galaxies]

dists = [galaxy_info[galaxy]['dist'] for galaxy in galaxies]

plot_name = os.path.join(plot_dir, muse_version, 'scale_length_region_size')

fig = plt.figure(figsize=(4, 3))
ax = plt.subplot(1, 1, 1)

coloured_errorbar_plot(ax, x_data=normalised_nebula_sizes, y_data=normalised_scale_lengths,
                       x_err=[normalised_nebula_sizes_err_down, normalised_nebula_sizes_err_up],
                       y_err=[normalised_scale_lengths_err_down, normalised_scale_lengths_err_up],
                       x_label=r'$\langle R_\mathrm{nebula}\rangle / r_{25}$', y_label=r'$\sigma_l/r_{25}$',
                       c=dists, c_label='Dist (Mpc)', include_tau=True)

plt.tight_layout()

# plt.show()

plt.savefig(plot_name + '.pdf', bbox_inches='tight')
plt.savefig(plot_name + '.png', bbox_inches='tight')

plt.close()

# Some other scatter plots

plt.figure(figsize=(8, 3))

sfrs = [galaxy_info[galaxy]['surface_density_sfr'] for galaxy in galaxies]
sfr_errs = [galaxy_info[galaxy]['props_sfr_err'] for galaxy in galaxies]

m_stars = [galaxy_info[galaxy]['surface_density_mstar'] for galaxy in galaxies]
m_star_errs = [galaxy_info[galaxy]['props_mstar_err'] for galaxy in galaxies]

scale_lengths = [galaxy_info[galaxy]['scale_length'] for galaxy in galaxies]
scale_lengths_err_down = [galaxy_info[galaxy]['scale_length_err_down'] for galaxy in galaxies]
scale_lengths_err_up = [galaxy_info[galaxy]['scale_length_err_up'] for galaxy in galaxies]

ax1 = plt.subplot(1, 2, 1)

coloured_errorbar_plot(ax1, x_data=sfrs, y_data=scale_lengths,
                       x_err=sfr_errs, y_err=[scale_lengths_err_down, scale_lengths_err_up],
                       x_label=r'$\log10(\mathrm{SFR}/r_{25}^2 [M_\odot\,\mathrm{yr}^{-1}\,\mathrm{kpc}^{-2}])$',
                       y_label=r'$\sigma_l$ (kpc)',
                       c=m_stars, c_label=r'$\log10(M_\ast / r_{25}^2 [M_\odot\,\mathrm{kpc}^{-2}])$', include_tau=True)

ax2 = plt.subplot(1, 2, 2)

coloured_errorbar_plot(ax2, x_data=m_stars, y_data=scale_lengths,
                       x_err=m_star_errs, y_err=[scale_lengths_err_down, scale_lengths_err_up],
                       x_label=r'$\log10(M_\ast / r_{25}^2 [M_\odot\,\mathrm{kpc}^{-2}])$',
                       y_label=r'$\sigma_l$ (kpc)',
                       c=sfrs,
                       c_label=r'$\log10(\mathrm{SFR}/r_{25}^2 [M_\odot\,\mathrm{yr}^{-1}\,\mathrm{kpc}^{-2}])$',
                       include_tau=True)

plt.tight_layout()

# plt.show()

plt.savefig(plot_dir + 'sfr_m_star_corr.pdf',
            bbox_inches='tight')
plt.savefig(plot_dir + 'sfr_m_star_corr.png',
            bbox_inches='tight')

plt.close()

print('Complete!')
