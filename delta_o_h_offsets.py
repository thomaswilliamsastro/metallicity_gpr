# -*- coding: utf-8 -*-
"""
Look at how azimuthal offsets vary with environment

@author: Tom Williams
"""

import os
import socket

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
from matplotlib.colors import Normalize
from matplotlib.ticker import MultipleLocator
from scipy.stats import kendalltau

from vars import top_dir, metallicity_dir, muse_version, phangs_master_table, plot_dir, extinction_curve, hii_only


def scatter_with_colour(axis, table, x_quantity, y_quantity, colour, subset=None, marker='o',
                        edgecolors='none'):

    if subset is None:

        subset = [True] * len(table[x_quantity])

    x_vals = table[x_quantity][subset]
    x_err_up = table[x_quantity + '_ERR_UP'][subset]
    x_err_down = table[x_quantity + '_ERR_DOWN'][subset]

    y_vals = table[y_quantity][subset]
    y_err_up = table[y_quantity + '_ERR_UP'][subset]
    y_err_down = table[y_quantity + '_ERR_DOWN'][subset]

    colour = colour[subset]

    idx = np.where((~np.isnan(x_vals)) & (~np.isnan(y_vals)))

    x_vals = x_vals[idx]
    x_err_up = x_err_up[idx]
    x_err_down = x_err_down[idx]

    y_vals = y_vals[idx]
    y_err_up = y_err_up[idx]
    y_err_down = y_err_down[idx]

    colour = colour[idx]

    axis.errorbar(x_vals, y_vals,
                  xerr=[x_err_down, x_err_up],
                  yerr=[y_err_down, y_err_up],
                  c=colour,
                  ls='none', zorder=98)
    axis.scatter(x_vals, y_vals,
                 c=colour, marker=marker, zorder=99, edgecolors=edgecolors)

    return axis


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

os.chdir(top_dir)

metallicity_calib = 'pg16_scal'

delta_o_h_file_name = os.path.join(metallicity_dir, muse_version, '%s_ext_curve' % extinction_curve,
                                   'delta_o_h_%s' % metallicity_calib)
if hii_only:
    delta_o_h_file_name += '_hii_only'
delta_o_h_file_name += '.fits'
delta_o_h_table = Table.read(delta_o_h_file_name)

environments_to_compare = ['Centre', 'Bar', 'Spiral Arms', 'Disc']
lims = [-0.08, 0.08]

xlim_default = -0.041, 0.061
ylim_default = -0.041, 0.061

# xlim_dict = {'[0, 0]': [-0.08, 0.08],
#              '[0, 1]': [-0.08, 0.08],
#              '[0, 2]': [-0.08, 0.08],
#              '[1, 0]': [-0.04, 0.04],
#              '[1, 1]': [-0.04, 0.04],
#              '[2, 0]': [-0.041, 0.041]}
# ylim_dict = {'[0, 0]': [-0.06, 0.06],
#              '[0, 1]': [-0.06, 0.06],
#              '[0, 2]': [-0.06, 0.06],
#              '[1, 0]': [-0.06, 0.06],
#              '[1, 1]': [-0.06, 0.06],
#              '[2, 0]': [-0.06, 0.06]}
xlim_dict = {}
ylim_dict = {}

# xlim_dict = {}
# ylim_dict = {}

breakout_barred = True
barred = delta_o_h_table['BARRED?'] == 1

colour_prop_val = []
colour_prop_err = []
colour_prop_name = 'props_mstar'
colour_prop_fancy_name = r'$\log_{10}(M_\ast\,[M_\odot])$'

for gal in delta_o_h_table['GALAXY']:
    samp_tab_row = phangs_master_table[phangs_master_table['name'] == gal.lower()]
    colour_prop_val.append(np.log10(samp_tab_row['props_mstar'][0]))
    colour_prop_err.append(samp_tab_row['props_mstar_unc'][0])

colour_prop_val = np.array(colour_prop_val)
colour_prop_err = 0.11 * np.ones_like(colour_prop_val)

# Calculate correlation between stellar mass and Delta O/H

centre_o_h = delta_o_h_table['CENTRE']
centre_o_h_err = delta_o_h_table['CENTRE_ERR_UP']

nan_idx = np.where(~np.isnan(centre_o_h))

cmap = matplotlib.cm.inferno
vmin, vmax = 0.95 * np.nanmin(colour_prop_val), 1.05 * np.nanmax(colour_prop_val)
cmap_norm = Normalize(vmin=vmin, vmax=vmax)

fig, axes = plt.subplots(len(environments_to_compare) - 1, len(environments_to_compare) - 1, figsize=(9, 8))
plt.subplots_adjust(hspace=0, wspace=0)

for ax in axes.flatten():
    ax.axis('off')

if breakout_barred:

    axes[0, 1].scatter(-99, -99, marker='x', c='k', label='Barred')
    axes[0, 1].scatter(-99, -99, marker='o', c='k', edgecolor='lime', label='Unbarred')

    axes[0, 1].set_xlim(lims)
    axes[0, 1].set_ylim(lims)

    axes[0, 1].legend(loc='center', frameon=False)

for i in range(len(environments_to_compare)):

    x_axis_name = environments_to_compare[i].upper().replace(' ', '_')

    environment_subset = environments_to_compare[i + 1:][::-1]

    for j in range(len(environment_subset)):
        y_axis_name = environment_subset[j].upper().replace(' ', '_')

        axes[-j - 1, i].axis('on')

        try:
            xlims = xlim_dict[str([i, j])]
        except KeyError:
            xlims = xlim_default
        try:
            ylims = ylim_dict[str([i, j])]
        except KeyError:
            ylims = ylim_default

        axes[-j - 1, i].set_xlim(xlims)
        axes[-j - 1, i].set_ylim(ylims)

        axes[-j - 1, i].yaxis.set_minor_locator(MultipleLocator(0.01))
        axes[-j - 1, i].xaxis.set_minor_locator(MultipleLocator(0.01))

        axes[-j - 1, i].grid(which='major')

        axes[-j - 1, i].plot(lims, lims, c='k', ls='--')

        colour = cmap(cmap_norm(colour_prop_val))

        if breakout_barred:

            scatter_with_colour(axes[-j - 1, i], delta_o_h_table, x_axis_name, y_axis_name,
                                colour, subset=barred, marker='x')

            scatter_with_colour(axes[-j - 1, i], delta_o_h_table, x_axis_name, y_axis_name,
                                colour, subset=~barred, marker='o', edgecolors='lime')

        else:

            scatter_with_colour(axes[-j - 1, i], delta_o_h_table, x_axis_name, y_axis_name,
                                colour)

        if j == 0:
            axes[-j - 1, i].set_xlabel(r'$\Delta \log_{10}(\mathrm{O/H})_\mathrm{H\alphaup, %s}$' %
                                       environments_to_compare[i].replace(' ', '\,'))
        else:
            axes[-j - 1, i].set_xticklabels([])
        if i == 0:
            axes[-j - 1, i].set_ylabel(r'$\Delta \log_{10}(\mathrm{O/H})_\mathrm{H\alphaup, %s}$' %
                                       environment_subset[j].replace(' ', '\,'))
        else:
            axes[-j - 1, i].set_yticklabels([])

nan_idx = np.where((~np.isnan(delta_o_h_table['CENTRE'])) & (~np.isnan(colour_prop_val)) )

print(get_tau_err(delta_o_h_table['CENTRE'][nan_idx], delta_o_h_table['CENTRE_ERR_UP'][nan_idx],
                  colour_prop_val[nan_idx], colour_prop_err[nan_idx]))

cax = fig.add_axes([0.91, 0.11, 0.02, 0.77])

plt.colorbar(matplotlib.cm.ScalarMappable(norm=cmap_norm, cmap=cmap),
             cax=cax,
             label=colour_prop_fancy_name)

fig_name = os.path.join(plot_dir, muse_version, '%s_ext_curve' % extinction_curve,
                        '%s_abundance_offsets' % metallicity_calib)
if hii_only:
    fig_name += '_hii_only'
# plt.show()
plt.savefig(fig_name + '.png', bbox_inches='tight')
plt.savefig(fig_name + '.pdf', bbox_inches='tight')
plt.close()

print('Complete!')
