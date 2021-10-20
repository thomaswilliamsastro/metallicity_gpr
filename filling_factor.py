# -*- coding: utf-8 -*-
"""
Look at filling factor for the metallicities.

@author: Tom Williams
"""

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from astropy.io import fits
from astropy.table import Table
from scipy.stats import gaussian_kde

from vars import top_dir, muse_dir, muse_version, galaxies, metallicity_dir, plot_dir, extinction_curve, hii_only


def normalise_kde(data, total_data_length):
    plt.figure(99)
    kde = sns.kdeplot(data, bw_method='silverman')
    line = kde.lines[0]
    x, y = line.get_data()
    plt.close(99)

    # Normalise by total number
    area = len(data) / total_data_length

    total_area = np.trapz(y, x)
    y *= total_area * area

    return x, y


matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['font.size'] = 14

sns.set_color_codes('muted')

os.chdir(top_dir)

metallicity_calib = 'pg16_scal'

filling_factors = {}

scale_length_file_name = os.path.join(metallicity_dir, muse_version, '%s_ext_curve' % extinction_curve,
                                      'scale_lengths_%s' % metallicity_calib)
if hii_only:
    scale_length_file_name += '_hii_only'
scale_length_file_name += '.fits'
scale_length_table = Table.read(scale_length_file_name)

n_pixs = []
n_zs = []

for galaxy in galaxies:

    pix_hdu_file_name = os.path.join('metallicity', 'pix_maps', muse_version, '%s_ext_curve' % extinction_curve,
                                     metallicity_calib)
    if hii_only:
        pix_hdu_file_name += '_hii_only'
    pix_hdu_file_name = os.path.join(pix_hdu_file_name, '%s_metallicity_%s.fits' % (galaxy, metallicity_calib))
    full_hdu_file_name = os.path.join(muse_dir, muse_version, 'conv', '%s_MAPS.fits' % galaxy)

    pix_hdu = fits.open(pix_hdu_file_name)[0]
    full_hdu = fits.open(full_hdu_file_name)[1]

    n_pix = len(np.where(~np.isnan(full_hdu.data))[0])
    n_z = len(np.where(~np.isnan(pix_hdu.data))[0])

    filling_factor = n_z / n_pix
    filling_factors[galaxy] = filling_factor * 100

    n_pixs.append(n_pix)
    n_zs.append(n_z)

filling_tab_file_name = os.path.join(metallicity_dir, muse_version, '%s_ext_curve' % extinction_curve,
                                     'filling_factors_%s' % metallicity_calib)
if hii_only:
    filling_tab_file_name += '_hii_only'
filling_tab_file_name += '.fits'

filling_tab = Table([galaxies, n_pixs, n_zs],
                    names=['GALAXY', 'N_PIX', 'N_Z'])
filling_tab.write(filling_tab_file_name, overwrite=True)

# Split out significant/non-significant

sig_scale_length_galaxies = [row['GALAXY'] for row in scale_length_table if row['SIGNIFICANT'] == 1]
non_sig_scale_length_galaxies = [row['GALAXY'] for row in scale_length_table if row['SIGNIFICANT'] == 0]

sig_filling_factors = [filling_factors[galaxy] for galaxy in sig_scale_length_galaxies]
non_sig_filling_factors = [filling_factors[galaxy] for galaxy in non_sig_scale_length_galaxies]

filling_factors_flat = [filling_factors[key] for key in filling_factors.keys()]

total_length = len(filling_factors_flat)

plot_name = os.path.join(plot_dir, muse_version, '%s_ext_curve' % extinction_curve,
                         '%s_filling_factor' % metallicity_calib)
if hii_only:
    plot_name += '_hii_only'

# x_kde = np.linspace(-1, 2.5, 1000)
x_kde = np.linspace(0, 30, 1000)

# sig_kde = gaussian_kde(np.log10(sig_filling_factors), bw_method='silverman').evaluate(x_kde)
sig_kde = gaussian_kde(sig_filling_factors, bw_method='silverman').evaluate(x_kde)

# non_sig_kde = gaussian_kde(np.log10(non_sig_filling_factors), bw_method='silverman').evaluate(x_kde)
non_sig_kde = gaussian_kde(non_sig_filling_factors, bw_method='silverman').evaluate(x_kde)

sig_kde *= len(sig_filling_factors) / np.trapz(sig_kde, x_kde)
non_sig_kde *= len(non_sig_filling_factors) / np.trapz(non_sig_kde, x_kde)

all_kde = sig_kde + non_sig_kde

final_scaling = np.trapz(all_kde, x_kde)

all_kde /= final_scaling
sig_kde /= final_scaling
non_sig_kde /= final_scaling

plt.figure(figsize=(5, 4))
ax = plt.subplot(111)

plt.grid()

plt.fill_between(x_kde, all_kde, alpha=0.5, color='k', label='All')
plt.fill_between(x_kde, sig_kde, alpha=0.5, color='b', label='Significant')
plt.fill_between(x_kde, non_sig_kde, alpha=0.5, color='r', label='Non-Significant')

ax.xaxis.set_minor_locator(matplotlib.ticker.AutoLocator())
ax.yaxis.set_minor_locator(matplotlib.ticker.AutoLocator())

# plt.xlabel(r'$\log_{10}$ [Metallicity filling factor (%)]')
plt.xlabel(r'Metallicity filling factor (%)')
plt.ylabel(r'Probability Density')

ylims = plt.ylim()
plt.ylim(0, ylims[-1])

plt.xlim(x_kde[0], x_kde[-1])

plt.legend(loc='upper right', frameon=False, framealpha=1)

plt.tight_layout()

# plt.show()
plt.savefig(plot_name + '.png', bbox_inches='tight')
plt.savefig(plot_name + '.pdf', bbox_inches='tight')
plt.close()

print('Complete!')
