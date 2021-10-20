# -*- coding: utf-8 -*-
"""
Compare the scale lengths calculated from our fiducial setup to those where we restrict to only fitting within HII
regions

@author: Tom Williams
"""

import os
import socket

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
from scipy.stats import kendalltau

from vars import top_dir, muse_version, metallicity_dir, plot_dir, extinction_curve

if 'mac' in socket.gethostname():
    os.environ['PATH'] += ':/Library/TeX/texbin'

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['text.usetex'] = True
plt.rcParams.update({'text.latex.preamble': r'\usepackage{txfonts}'})

os.chdir(top_dir)

metallicity_calib = 'pg16_scal'

fiducial_tab_file_name = os.path.join(metallicity_dir, muse_version, '%s_ext_curve' % extinction_curve,
                                      'scale_lengths_%s_hii_only.fits' % metallicity_calib)
fiducial_tab = Table.read(fiducial_tab_file_name)

no_hii_mask_file_name = os.path.join(metallicity_dir, muse_version, '%s_ext_curve' % extinction_curve,
                                     'scale_lengths_%s.fits' % metallicity_calib)
no_hii_mask_tab = Table.read(no_hii_mask_file_name)

# Filter to only significant fits in both cases

sig_idx = np.where((fiducial_tab['SIGNIFICANT'] == 1) & (no_hii_mask_tab['SIGNIFICANT'] == 1))
fiducial_tab = fiducial_tab[sig_idx]
no_hii_mask_tab = no_hii_mask_tab[sig_idx]

# Calculate a correlation between these

n_draws = 2500
taus = np.zeros(n_draws)

for draw in range(n_draws):
    fid_l_bootstrap = fiducial_tab['SIGMA_L'] + np.random.normal(loc=0, scale=fiducial_tab['SIGMA_L_ERR_UP'])
    no_hii_l_boostrap = no_hii_mask_tab['SIGMA_L'] + np.random.normal(loc=0, scale=no_hii_mask_tab['SIGMA_L_ERR_UP'])

    tau, _ = kendalltau(fid_l_bootstrap, no_hii_l_boostrap)
    taus[draw] = tau

tau_down, tau_med, tau_up = np.percentile(taus, [16, 50, 84])
tau_up -= tau_med
tau_down = tau_med - tau_down

lims = [0, 1.25 * max(fiducial_tab['SIGMA_L'])]

plot_name = os.path.join(plot_dir, muse_version, '%s_ext_curve' % extinction_curve,
                         '%s_fiducial_no_hii_mask_comparison' % metallicity_calib)

plt.figure(figsize=(10, 4))
ax = plt.subplot(121)
plt.errorbar(fiducial_tab['SIGMA_L'], no_hii_mask_tab['SIGMA_L'],
             xerr=[fiducial_tab['SIGMA_L_ERR_DOWN'], fiducial_tab['SIGMA_L_ERR_UP']],
             yerr=[no_hii_mask_tab['SIGMA_L_ERR_DOWN'], no_hii_mask_tab['SIGMA_L_ERR_UP']],
             c='k', ls='none', marker='o')

plt.plot(lims, lims, c='k', ls='--')

plt.xlim(lims)
plt.ylim(lims)

ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())

plt.grid()

plt.xlabel(r'$\sigma_{l}$ (Fiducial; kpc)')
plt.ylabel(r'$\sigma_{l}$ (No H{\sc ii} mask; kpc)')
plt.text(0.95, 0.95, r'$\tau = %.2f\pm%.2f$' % (tau_med, np.max([tau_up, tau_down])),
         ha='right', va='top',
         transform=ax.transAxes,
         bbox=dict(facecolor='white', edgecolor='none', alpha=0.75))

# Also include the 50% correlation scales

corr_tab_fiducial_name = os.path.join(metallicity_dir, muse_version, '%s_ext_curve' % extinction_curve,
                                      'two_point_corr_%s_hii_only.fits' % metallicity_calib)
corr_tab_fiducial = Table.read(corr_tab_fiducial_name)

corr_tab_no_hii_name = os.path.join(metallicity_dir, muse_version, '%s_ext_curve' % extinction_curve,
                                    '50percent_two_point_corr_%s.fits' % metallicity_calib)
corr_tab_no_hii = Table.read(corr_tab_no_hii_name)

corr_fid = []
corr_fid_err_up = []
corr_fid_err_down = []

corr = []
corr_err_up = []
corr_err_down = []

for row in corr_tab_fiducial:

    idx = np.where(corr_tab_no_hii['GALAXY'] == row['GALAXY'])[0]
    if len(idx) == 0:
        continue

    corr_fid.append(row['TWO_POINT_CORR'])
    corr_fid_err_up.append(row['TWO_POINT_CORR_ERR_UP'])
    corr_fid_err_down.append(row['TWO_POINT_CORR_ERR_DOWN'])

    corr.append(corr_tab_no_hii['TWO_POINT_CORR'][idx][0])
    corr_err_up.append(corr_tab_no_hii['TWO_POINT_CORR_ERR_UP'][idx][0])
    corr_err_down.append(corr_tab_no_hii['TWO_POINT_CORR_ERR_DOWN'][idx][0])

# Calculate a correlation between these

taus = np.zeros(n_draws)

for draw in range(n_draws):
    corr_fid_bootstrap = corr_fid + np.random.normal(loc=0, scale=corr_fid_err_up)
    corr_boostrap = corr + np.random.normal(loc=0, scale=corr_err_up)

    tau, _ = kendalltau(corr_fid_bootstrap, corr_boostrap)
    taus[draw] = tau

tau_down, tau_med, tau_up = np.percentile(taus, [16, 50, 84])
tau_up -= tau_med
tau_down = tau_med - tau_down

lims = [0, 2]

ax = plt.subplot(1, 2, 2)

plt.errorbar(corr_fid, corr,
             xerr=[corr_fid_err_down, corr_fid_err_up],
             yerr=[corr_err_down, corr_err_up],
             c='k', ls='none', marker='o'
             )
plt.plot(lims, lims, c='k', ls='--')

plt.xlim(lims)
plt.ylim(lims)

ax.yaxis.set_label_position("right")
ax.yaxis.tick_right()

ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())

plt.grid()

plt.xlabel(r'50\% Correlation Scale (Fiducial; kpc)')
plt.ylabel(r'50\% Correlation Scale (No H{\sc ii} mask; kpc)')
plt.text(0.05, 0.95, r'$\tau = %.2f\pm%.2f$' % (tau_med, np.max([tau_up, tau_down])),
         ha='left', va='top',
         transform=ax.transAxes,
         bbox=dict(facecolor='white', edgecolor='none', alpha=0.75))

plt.subplots_adjust(hspace=0, wspace=0)

# plt.show()

plt.savefig(plot_name + '.png', bbox_inches='tight')
plt.savefig(plot_name + '.pdf', bbox_inches='tight')
plt.close()

print('Complete!')
