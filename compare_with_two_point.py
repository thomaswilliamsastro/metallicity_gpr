# -*- coding: utf-8 -*-
"""
Compare the corrected scale lengths with the two-point correlations from Kreckel+ (20)

@author: Tom Williams
"""

import os
import socket

import matplotlib
import matplotlib.pyplot as plt
from astropy.table import Table
import numpy as np
from astropy.io import fits
from scipy.stats import kendalltau
from tqdm import tqdm

from vars import top_dir, metallicity_dir, muse_version, phangs_master_table, gpr_version, plot_dir, extinction_curve, \
    hii_only
from funcs import get_info_from_master_table
from astronomical_gpr import AstronomicalGPR


def get_tau_err(x, x_err, y, y_err, n_draws=1000):
    tau_array = np.zeros(n_draws)

    for i in range(n_draws):
        x_perturb = x + np.random.normal(loc=0, scale=x_err)
        y_perturb = y + np.random.normal(loc=0, scale=y_err)

        tau_array[i] = kendalltau(x_perturb, y_perturb)[0]

    tau = np.nanmedian(tau_array)
    tau_err = np.nanpercentile(tau_array, 97.5) - tau

    return tau, tau_err


if 'mac' in socket.gethostname():
    os.environ['PATH'] += ':/Library/TeX/texbin'

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['text.usetex'] = True
plt.rcParams.update({'text.latex.preamble': r'\usepackage{txfonts}'})

np.random.seed(42)

os.chdir(top_dir)

metallicity_calib = 'pg16_scal'
ylims = [-0.1, 1.3]

sl_table_file_name = os.path.join(metallicity_dir, muse_version, '%s_ext_curve' % extinction_curve,
                                  'scale_lengths_%s' % metallicity_calib)
if hii_only:
    sl_table_file_name += '_hii_only'
sl_table_file_name += '.fits'
sl_table = Table.read(sl_table_file_name)

tp_corr_table_file_name = os.path.join(metallicity_dir, muse_version, '%s_ext_curve' % extinction_curve,
                                       'two_point_corr_%s' % metallicity_calib)
if hii_only:
    tp_corr_table_file_name += '_hii_only'
tp_corr_table_file_name += '.fits'
tp_corr_table = Table.read(tp_corr_table_file_name)

# We'll take the 50% correlation scale

correct_k20 = False

# k20_tp_corr_dict = {'NGC0628': [230, 8],
#                     'NGC1087': [380, 7],
#                     'NGC1672': [370, 6],
#                     'NGC2835': [290, 2],
#                     'NGC3627': [370, 5],
#                     'NGC4254': [340, 8],
#                     'NGC4535': [290, 2],
#                     'IC5332': [270, 8]}

# Updated, preliminary values

k20_tp_corr_dict = {'NGC0628': [212, 1],
                    'NGC1087': [347, 3],
                    'NGC1300': [476, 5],
                    'NGC1365': [2247, 29],
                    'NGC1385': [561, 10],
                    'NGC1433': [412, 3],
                    'NGC1512': [525, 5],
                    'NGC1566': [368, 3],
                    'NGC1672': [595, 3],
                    'NGC2835': [366, 4],
                    'NGC3351': [242, 2],
                    'NGC3627': [399, 2],
                    'NGC4254': [279, 1],
                    'NGC4303': [367, 2],
                    'NGC4321': [377, 1],
                    'NGC4535': [302, 4],
                    'NGC5068': [199, 1],
                    'NGC7496': [497, 7],
                    'IC5332': [218, 4],
                    }

k_20_dists = {'NGC0628': 9.77,
              'NGC1087': 14.4,
              'NGC1672': 11.9,
              'NGC2835': 10.1,
              'NGC3627': 10.6,
              'NGC4254': 16.8,
              'NGC4535': 15.8,
              'IC5332': 9.95}

k_20_incs = {'NGC0628': 8.7,
             'NGC1087': 41.3,
             'NGC1672': 37.5,
             'NGC2835': 47.8,
             'NGC3627': 55.0,
             'NGC4254': 37.8,
             'NGC4535': 40.7,
             'IC5332': 24}

scale_lengths = []
scale_lengths_err_up = []
scale_lengths_err_down = []

tp_corrs = []
tp_corrs_err_up = []
tp_corrs_err_down = []

tp_corrs_k = []
tp_corrs_err_k = []

galaxy_params = ['dist', 'orient_incl']

outlier_idx = np.where(tp_corr_table['GALAXY'] != 'NGC1365')

for row in tp_corr_table:

    galaxy = row['GALAXY']

    tp_corr, tp_corr_err_up, tp_corr_err_down = \
        row['TWO_POINT_CORR'], row['TWO_POINT_CORR_ERR_UP'], row['TWO_POINT_CORR_ERR_DOWN']
    sl_row = sl_table[sl_table['GALAXY'] == galaxy][0]
    sl, sl_err_up, sl_err_down = sl_row['SIGMA_L'], sl_row['SIGMA_L_ERR_UP'], sl_row['SIGMA_L_ERR_DOWN']

    tp_corrs.append(tp_corr)
    tp_corrs_err_up.append(tp_corr_err_up)
    tp_corrs_err_down.append(tp_corr_err_down)

    scale_lengths.append(sl)
    scale_lengths_err_up.append(sl_err_up)
    scale_lengths_err_down.append(sl_err_down)

    if galaxy in k20_tp_corr_dict.keys():

        dist, inc = get_info_from_master_table(phangs_master_table, galaxy.lower(), galaxy_params)

        if correct_k20:
            k_20_dist = k_20_dists[galaxy]
            inc_correction = np.cos(np.radians(k_20_incs[galaxy])) / np.cos(np.radians(inc))
        else:
            k_20_dist = dist
            inc_correction = 1

        two_point_corr_k = k20_tp_corr_dict[galaxy][0] * 1e-3 * dist / k_20_dist * inc_correction
        two_point_corr_err_k = k20_tp_corr_dict[galaxy][1] * 1e-3 * dist / k_20_dist * inc_correction

        tp_corrs_k.append(two_point_corr_k)
        tp_corrs_err_k.append(two_point_corr_err_k)

    else:

        tp_corrs_k.append(np.nan)
        tp_corrs_err_k.append(np.nan)

# Calculate the taus

tp_corrs = np.array(tp_corrs)
scale_lengths = np.array(scale_lengths)
tp_corrs_k = np.array(tp_corrs_k)

tau_draws = 1000
tau_bs_tp_sl = np.zeros(tau_draws)
tau_bs_tp_sl_no_outlier = np.zeros(tau_draws)
tau_bs_tp_tp_k = np.zeros(tau_draws)
tau_bs_tp_tp_k_no_outlier = np.zeros(tau_draws)

nan_idx = np.where(~np.isnan(tp_corrs_k))
nan_idx_no_outlier = np.where(~np.isnan(tp_corrs_k[outlier_idx]))

for draw in range(tau_draws):
    scatter_tp = np.random.normal(loc=0, scale=tp_corrs_err_up)
    scatter_sl = np.random.normal(loc=0, scale=scale_lengths_err_up)
    scatter_tp_k = np.random.normal(loc=0, scale=tp_corrs_err_k)

    tau_bs_tp_sl[draw], _ = kendalltau(tp_corrs + scatter_tp, scale_lengths + scatter_sl)
    tau_bs_tp_sl_no_outlier[draw], _ = kendalltau(tp_corrs[outlier_idx] + scatter_tp[outlier_idx],
                                                  scale_lengths[outlier_idx] + scatter_sl[outlier_idx])
    tau_bs_tp_tp_k[draw], _ = kendalltau(tp_corrs[nan_idx] + scatter_tp[nan_idx],
                                         tp_corrs_k[nan_idx] + scatter_tp_k[nan_idx])
    tau_bs_tp_tp_k_no_outlier[draw], _ = kendalltau(tp_corrs[outlier_idx][nan_idx_no_outlier] +
                                                    scatter_tp[outlier_idx][nan_idx_no_outlier],
                                                    tp_corrs_k[outlier_idx][nan_idx_no_outlier] +
                                                    scatter_tp_k[outlier_idx][nan_idx_no_outlier])

tau_tp_sl = np.median(tau_bs_tp_sl)
tau_tp_sl_err = np.percentile(tau_bs_tp_sl, 84) - tau_tp_sl
tau_tp_sl_no_outlier = np.median(tau_bs_tp_sl_no_outlier)
tau_tp_sl_err_no_outlier = np.percentile(tau_bs_tp_sl_no_outlier, 84) - tau_tp_sl_no_outlier
tau_tp_tp_k = np.median(tau_bs_tp_tp_k)
tau_tp_tp_k_err = tau_tp_tp_k - np.percentile(tau_bs_tp_tp_k, 16)
tau_tp_tp_k_no_outlier = np.median(tau_bs_tp_tp_k_no_outlier)
tau_tp_tp_k_err_no_outlier = tau_tp_tp_k_no_outlier - np.percentile(tau_bs_tp_tp_k_no_outlier, 16)

plot_name = os.path.join(plot_dir, muse_version, '%s_ext_curve' % extinction_curve,
                         '%s_two_point_corr_comparison' % metallicity_calib)
if hii_only:
    plot_name += '_hii_only'

plt.figure(figsize=(8, 4))

plt.subplots_adjust(hspace=0, wspace=0)

ax1 = plt.subplot(1, 2, 1)
plt.errorbar(scale_lengths, tp_corrs,
             xerr=[scale_lengths_err_down, scale_lengths_err_up],
             yerr=[tp_corrs_err_up, tp_corrs_err_down],
             c='k', ls='none', marker='o')

plt.grid()

ax1.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
ax1.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())

plt.ylim(ylims)

plt.xlabel(r'$\sigma_l$ (GPR; kpc)')
plt.ylabel(r'50\% Correlation Scale (GPR; kpc)')

plt.text(0.95, 0.05, # r'$\tau=%.2f\pm%.2f$'
                     # '\n'
                     r'$\tau=%.2f\pm%.2f$ (no NGC1365)' %
         (tau_tp_sl_no_outlier, tau_tp_sl_err_no_outlier),  # tau_tp_sl_k, tau_tp_sl_k_err
         ha='right', va='bottom',
         transform=ax1.transAxes)

ax2 = plt.subplot(1, 2, 2)
plt.errorbar(tp_corrs_k, tp_corrs,
             xerr=tp_corrs_err_k,
             yerr=[tp_corrs_err_up, tp_corrs_err_down],
             c='k', ls='none', marker='o')

plt.plot(ylims, ylims, c='k', ls='--')

plt.grid()

ax2.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
ax2.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())

plt.xlabel(r"50\% Correlation Scale (H{\sc ii} regions; kpc)")
plt.ylabel(r'50\% Correlation Scale (GPR; kpc)')

plt.text(0.95, 0.05, # r'$\tau=%.2f\pm%.2f$'
                     # '\n'
                     r'$\tau=%.2f\pm%.2f$ (no NGC1365)' %
         (tau_tp_tp_k_no_outlier, tau_tp_tp_k_err_no_outlier),  # tau_tp_tp_k, tau_tp_tp_k_err,
         ha='right', va='bottom',
         transform=ax2.transAxes)

plt.ylim(ylims)
plt.xlim(ylims)

ax2.yaxis.set_label_position('right')
ax2.yaxis.tick_right()

# plt.show()
plt.savefig(plot_name + '.png', bbox_inches='tight')
plt.savefig(plot_name + '.pdf', bbox_inches='tight')
plt.close()

print('Complete!')
