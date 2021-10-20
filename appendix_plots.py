# -*- coding: utf-8 -*-
"""
Python up the appendix plots

@author: Tom Williams
"""

import os

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from matplotlib.colors import Normalize

from vars import top_dir, metallicity_dir, plot_dir, galaxies, phangs_master_table
from funcs import get_info_from_master_table

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['font.size'] = 14

os.chdir(top_dir)

data_dir = os.path.join(metallicity_dir, 'data_for_app_plots')

# Start with the comparison

fig_name = os.path.join(plot_dir, 'plot_comp_kk20')

lims = [-0.1, 1.3]

k20, this_work, k20_err, this_work_err = np.loadtxt(os.path.join(data_dir, 'compare_KK20_30percent.txt'),
                                                    usecols=(1, 2, 3, 4), skiprows=1, unpack=True)

plt.figure(figsize=(8, 4))

ax = plt.subplot(1, 2, 1)

plt.errorbar(k20, this_work, xerr=k20_err, yerr=this_work_err, c='k', ls='none', marker='o')
plt.plot(lims, lims, c='k', ls='--')

plt.grid()

ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())

plt.xlim(lims)
plt.ylim(lims)

plt.xlabel('30% correlation scale (K+ 2020; kpc)')
plt.ylabel('30% correlation scale (kpc)')

k20, this_work, k20_err, this_work_err = np.loadtxt(os.path.join(data_dir, 'compare_KK20_50percent.txt'),
                                                    usecols=(1, 2, 3, 4), skiprows=1, unpack=True)

ax = plt.subplot(1, 2, 2)

plt.errorbar(k20, this_work, xerr=k20_err, yerr=this_work_err, c='k', ls='none', marker='o')
plt.plot(lims, lims, c='k', ls='--')

ax.yaxis.set_label_position("right")
ax.yaxis.tick_right()

plt.grid()

ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())

plt.xlim(lims)
plt.ylim(lims)

plt.xlabel('50% correlation scale (K+ 2020; kpc)')
plt.ylabel('50% correlation scale (kpc)')

plt.subplots_adjust(hspace=0, wspace=0)

# plt.show()
plt.savefig(fig_name + '.pdf', bbox_inches='tight')
plt.savefig(fig_name + '.png', bbox_inches='tight')
plt.close()

# Two-point correlation functions for everything

fig_name = os.path.join(plot_dir, 'plot_corr_scal_each')

fig = plt.figure(figsize=(12, 9))

lims = [0, 3.1]

masses = []
for galaxy in galaxies:
    mass = get_info_from_master_table(phangs_master_table, galaxy.lower(), 'props_mstar')
    masses.append(np.log10(mass))

cmap = matplotlib.cm.inferno
vmin, vmax = 0.95 * np.nanmin(masses), 1.05 * np.nanmax(masses)
cmap_norm = Normalize(vmin=vmin, vmax=vmax)

for i, galaxy in enumerate(galaxies):
    scale, corr, corr_err_up, corr_err_down, shuffle_up, shuffle_down = np.loadtxt(
        os.path.join(data_dir, '%s_plot.txt' % galaxy), skiprows=1, unpack=True)

    ax = plt.subplot(5, 4, i + 1)

    plt.fill_between(scale, corr_err_down, corr_err_up, color='k', alpha=0.75)
    plt.plot(scale, corr, c='k')

    plt.fill_between(scale, shuffle_down, shuffle_up, color='gray')

    plt.grid()

    ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())

    plt.xlim(lims)
    plt.ylim(-0.1, 1.1)

    if i not in [0, 4, 8, 12, 16]:
        ax.set_yticklabels([])
    if i not in [16, 17, 18, 19]:
        ax.set_xticklabels([])

    plt.text(0.95, 0.9, galaxy,
             ha='right', va='top', c=cmap(cmap_norm(masses[i])),
             transform=ax.transAxes)

ax = plt.subplot(5, 4, 20)
for i, galaxy in enumerate(galaxies):
    scale, corr, corr_err_up, corr_err_down, _, _ = np.loadtxt(
        os.path.join(data_dir, '%s_plot.txt' % galaxy), skiprows=1, unpack=True)
    plt.plot(scale, corr, c=cmap(cmap_norm(masses[i])))

plt.grid()

ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())

ax.set_yticklabels([])

plt.xlim(lims)
plt.ylim(-0.1, 1.1)

fig.text(0.5, 0.05, 'Scale (kpc)', ha='center')
fig.text(0.05, 0.5, 'Two point correlation of metals', va='center', rotation='vertical')

plt.subplots_adjust(hspace=0, wspace=0)

# plt.show()
plt.savefig(fig_name + '.pdf', bbox_inches='tight')
plt.savefig(fig_name + '.png', bbox_inches='tight')
plt.close()

print('Complete!')
