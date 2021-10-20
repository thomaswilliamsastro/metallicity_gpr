# -*- coding: utf-8 -*-
"""
Check convergence of GPR, and also estimate scale length errors.

@author: Tom Williams
"""

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
from scipy.stats import kendalltau, pearsonr, spearmanr

from vars import top_dir, plot_dir, muse_version, metallicity_dir, extinction_curve, hii_only

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['font.size'] = 14

os.chdir(top_dir)

lims = [0, 40]

metallicity_calibs = ['pg16_scal', 'd16']
fancy_labels = {'pg16_scal': r'$\sigma_l$ (PG16; kpc)',
                'd16': r'$\sigma_l$ (D16; kpc)'}

scale_lengths_dict = {}

plot_name = os.path.join(plot_dir, muse_version, '%s_ext_curve' % extinction_curve, '%s_%s_scale_length_comparison' %
                         (metallicity_calibs[0], metallicity_calibs[1]))
if hii_only:
    plot_name += '_hii_only'

# Pull values from the tables

for metallicity_calib in metallicity_calibs:

    scale_lengths_dict[metallicity_calib] = {}

    scale_length_file_name = os.path.join(metallicity_dir, muse_version, '%s_ext_curve' % extinction_curve,
                                          'scale_lengths_%s' % metallicity_calib)
    if hii_only:
        scale_length_file_name += '_hii_only'
    scale_length_file_name += '.fits'
    scale_length_tab = Table.read(scale_length_file_name)

    sig_rows = scale_length_tab[scale_length_tab['SIGNIFICANT'] == 1]

    for sig_row in sig_rows:
        galaxy = sig_row['GALAXY']

        scale_lengths_dict[metallicity_calib][galaxy] = [sig_row['SIGMA_L'],
                                                         sig_row['SIGMA_L_ERR_UP'], sig_row['SIGMA_L_ERR_DOWN']]

# Only take overlaps

first_calib = []
first_calib_err_up = []
first_calib_err_down = []
second_calib = []
second_calib_err_up = []
second_calib_err_down = []

for key in scale_lengths_dict[metallicity_calibs[0]].keys():

    if key in scale_lengths_dict[metallicity_calibs[1]].keys():
        first_calib.append(scale_lengths_dict[metallicity_calibs[0]][key][0])
        first_calib_err_up.append(scale_lengths_dict[metallicity_calibs[0]][key][1])
        first_calib_err_down.append(scale_lengths_dict[metallicity_calibs[0]][key][2])
        second_calib.append(scale_lengths_dict[metallicity_calibs[1]][key][0])
        second_calib_err_up.append(scale_lengths_dict[metallicity_calibs[1]][key][1])
        second_calib_err_down.append(scale_lengths_dict[metallicity_calibs[1]][key][2])

first_calib = np.array(first_calib)
first_calib_err_up = np.array(first_calib_err_up)
first_calib_err_down = np.array(first_calib_err_down)
second_calib = np.array(second_calib)
second_calib_err_up = np.array(second_calib_err_up)
second_calib_err_down = np.array(second_calib_err_down)

# Errors on correlation

n_draws = 1000
tau_bs = np.zeros(n_draws)
pearson_bs = np.zeros(n_draws)
spearman_bs = np.zeros(n_draws)

# print(pearsonr(first_calib, second_calib))
# no

for i in range(n_draws):
    first_calib_bs = first_calib + np.random.normal(loc=0, scale=first_calib_err_up)
    second_calib_bs = second_calib + np.random.normal(loc=0, scale=second_calib_err_up)

    tau_bs[i], _ = kendalltau(first_calib_bs, second_calib_bs)
    pearson_bs[i], _ = pearsonr(first_calib_bs, second_calib_bs)
    spearman_bs[i], _ = spearmanr(first_calib_bs, second_calib_bs)

# plt.figure()
# plt.hist(tau_bs)
# plt.show()

tau = np.nanmedian(tau_bs)
tau_err = np.nanpercentile(tau_bs, 86) - tau

pearson = np.nanmedian(pearson_bs)
pearson_err = np.nanpercentile(pearson_bs, 86) - pearson

spearman = np.nanmedian(spearman_bs)
spearman_err = np.nanpercentile(spearman_bs, 86) - spearman

print(pearson, pearson_err)
print(spearman, spearman_err)

plt.figure(figsize=(5, 4))

ax = plt.subplot(111)

plt.errorbar(first_calib, second_calib,
             xerr=[first_calib_err_down, first_calib_err_up],
             yerr=[second_calib_err_down, second_calib_err_up],
             c='k', ls='none', marker='o')

xlims = plt.xlim()
ylims = plt.ylim()

plt.plot([0, xlims[-1]], [0, xlims[-1]], c='k', ls='--')

plt.xlim(lims)
plt.ylim(lims)

# plt.xscale('log')
# plt.yscale('log')

ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())

plt.grid()

plt.xlabel(fancy_labels[metallicity_calibs[0]])
plt.ylabel(fancy_labels[metallicity_calibs[1]])

plt.text(0.05, 0.95, r'$\tau=%.2f\pm%.2f$' % (tau, tau_err),
         ha='left', va='top',
         bbox=dict(facecolor='white', edgecolor='none', alpha=0.75),
         transform=ax.transAxes)

plt.tight_layout()

# plt.show()
plt.savefig(plot_name + '.png', bbox_inches='tight')
plt.savefig(plot_name + '.pdf', bbox_inches='tight')

print('Complete!')
