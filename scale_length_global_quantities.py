# -*- coding: utf-8 -*-
"""
See how the mixing scale varies with various global parameters

@author: Tom Williams
"""

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
from matplotlib.colors import Normalize
from matplotlib.ticker import MultipleLocator
from scipy.stats import kendalltau
from astropy.io import fits
from scipy.stats import gaussian_kde

from vars import top_dir, metallicity_dir, muse_version, phangs_master_table, plot_dir, muse_dir, extinction_curve, \
    hii_only
from funcs import get_info_from_master_table


def errorbar_with_colour(axis, x, y, x_err_up, y_err_up, colour,
                         x_err_down=None, y_err_down=None,
                         marker='o'):
    if x_err_down is None:
        x_err_down = x_err_up
    if y_err_down is None:
        y_err_down = y_err_up

    idx = np.where((~np.isnan(x)) & (~np.isnan(y)))

    x, x_err_up, x_err_down = x[idx], x_err_up[idx], x_err_down[idx]
    y, y_err_up, y_err_down = y[idx], y_err_up[idx], y_err_down[idx]
    colour = colour[idx]

    axis.errorbar(x, y,
                  xerr=[x_err_down, x_err_up],
                  yerr=[y_err_down, y_err_up],
                  c=colour,
                  ls='none', zorder=98)
    axis.scatter(x, y,
                 c=colour, marker=marker, zorder=99)

    return axis


matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['font.size'] = 14

os.chdir(top_dir)

use_two_point_corr_table = '50percent'

x_co_kde = np.linspace(0, 35, 2500)
x_ha_kde = np.linspace(0, 100, 2500)

n_cols = 2

megatable_dir = os.path.join('mega_tables', 'v1p3')
megatable_ext = '_hexagon_stats_1kpc.fits'

alma_map_dir = os.path.join('alma', 'v39')

weight_vel_disps = True

# metallicity_calib = 'pg16_scal'
metallicity_calib = 'pg16_scal'

scale_length_file_name = os.path.join(metallicity_dir, muse_version, '%s_ext_curve' % extinction_curve,
                                      'scale_lengths_%s' % metallicity_calib)
if use_two_point_corr_table:
    scale_length_file_name = scale_length_file_name.replace('scale_lengths',
                                                            '%s_two_point_corr' % use_two_point_corr_table)
if hii_only:
    scale_length_file_name += '_hii_only'
scale_length_file_name += '.fits'
scale_length_table = Table.read(scale_length_file_name)

if use_two_point_corr_table:
    idx = scale_length_table['GALAXY'] != 'NGC1365'
    # idx = scale_length_table['TWO_POINT_CORR'] <= 0.6
    galaxies = scale_length_table['GALAXY'][idx]
    scale_lengths = scale_length_table['TWO_POINT_CORR'][idx]
    scale_lengths_err_up = scale_length_table['TWO_POINT_CORR_ERR_UP'][idx]
    scale_lengths_err_down = scale_length_table['TWO_POINT_CORR_ERR_DOWN'][idx]
else:
    idx = scale_length_table['SIGNIFICANT'] == 1

    galaxies = scale_length_table['GALAXY'][idx]
    scale_lengths = scale_length_table['SIGMA_L'][idx]
    scale_lengths_err_up = scale_length_table['SIGMA_L_ERR_UP'][idx]
    scale_lengths_err_down = scale_length_table['SIGMA_L_ERR_DOWN'][idx]

quantities = ['r25', 'morph_t', 'props_mstar', 'rotcur_v0', 'props_sfr', 'co_vel_disp']  # , 'ha_vel_disp'
quantities_to_log = ['props_mstar']
quantities_to_surf_dens = ['props_mstar', 'props_sfr']

megatable_mapping = {'co_vel_disp': 'F<vdisp_mol_pix_150pc>'}

fancy_labels = {'morph_t': r'$T$',
                'props_mstar': r'$\Sigma_{M_\ast}\,(M_\odot\,\mathrm{kpc}^{-2})$',
                'props_sfr': r'$\Sigma_\mathrm{SFR}\,(M_\odot\,\mathrm{yr}^{-1}\,\mathrm{kpc}^{-2})$',
                'co_vel_disp': r'$\tilde{\sigma}_{v, \mathrm{CO}}$ (km s$^{-1}$)',
                'ha_vel_disp': r'$\tilde{\sigma}_{v, \mathrm{H}\alpha}$ (km s$^{-1}$)',
                'rotcur_v0': r'$v_\mathrm{r, inf}$ (km s$^{-1}$)',
                # 'r25': r'$r_{25}$ (kpc)',
                'r25': r'$R_e$ (kpc)',
                }
colour_prop_fancy_name = r'$r_\mathrm{bar}/R_e$'  # /r_{25}$'

fig_name = os.path.join(plot_dir, muse_version, '%s_ext_curve' % extinction_curve,
                        '%s_scale_length_global_quantities' % metallicity_calib)
if use_two_point_corr_table:
    fig_name = fig_name.replace('scale_length', '%s_two_point_corr' % use_two_point_corr_table)
if hii_only:
    fig_name += '_hii_only'

fig = plt.figure(figsize=(8.25, 11.75))
plt.subplots_adjust(hspace=0, wspace=0)

xlim = {'pg16_scal': [-4, 41],
        'd16': [-4, 29]}
if use_two_point_corr_table == '30percent':
    xlim = {'pg16_scal': [0.1, 3.5]}
if use_two_point_corr_table == '50percent':
    xlim = {'pg16_scal': [0.1, 1.5]}

for i, quantity in enumerate(quantities):

    print('%s:' % quantity)

    quantity_vals = []
    quantity_errs = []
    colour_prop_val = []

    for j, galaxy in enumerate(galaxies):

        print('-%s' % galaxy)

        if quantity not in ['co_vel_disp', 'ha_vel_disp', 'r25']:

            # Pull values from sample table

            quantity_val, quantity_err, dist, r25 = get_info_from_master_table(
                phangs_master_table, galaxy.lower(),
                [quantity, quantity + '_unc', 'dist', 'size_reff'])  # 'size_r25'])

            r25_phys_kpc = dist * 1e3 * r25 / 3600 * np.pi / 180

            # if i == 0:
            #     scale_lengths[j] /= r25_phys_kpc
            #     scale_lengths_err_up[j] /= r25_phys_kpc
            #     scale_lengths_err_down[j] /= r25_phys_kpc

            if quantity in quantities_to_surf_dens:
                quantity_val /= np.pi * r25_phys_kpc ** 2
                quantity_err /= np.pi * r25_phys_kpc ** 2

        elif quantity == 'co_vel_disp':

            # Read in the mom0/mom2 maps and take the CO-weighted average

            mom0_hdu_filename = os.path.join(alma_map_dir, galaxy + '_12m+7m+tp_co21_strict_mom0.fits')
            if not os.path.exists(mom0_hdu_filename):
                quantity_val = np.nan
                quantity_err = np.nan
            else:
                mom0_hdu = fits.open(mom0_hdu_filename)[0]

                mom2_hdu_filename = mom0_hdu_filename.replace('mom0', 'mom2')
                mom2_hdu = fits.open(mom2_hdu_filename)[0]

                # Flatten, de-NaN

                mom0 = mom0_hdu.data.flatten()
                mom2 = mom2_hdu.data.flatten()
                nan_idx = np.where((~np.isnan(mom0)) & (~np.isnan(mom2)))
                mom0 = mom0[nan_idx]
                mom2 = mom2[nan_idx]

                if not weight_vel_disps:
                    mom0 = np.ones_like(mom2)

                y_kde = gaussian_kde(mom2, weights=mom0, bw_method='silverman').evaluate(x_co_kde)

                # Calculate the median and spread of this distribution

                norm_cdf = np.cumsum(y_kde)
                norm_cdf /= np.max(norm_cdf)

                quantity_val = x_co_kde[(np.abs(norm_cdf - 0.5)).argmin()]
                quantity_err = x_co_kde[np.abs(norm_cdf - 0.84).argmin()] - quantity_val
                # lower_err = median - x_kde[np.abs(norm_cdf - 0.16).argmin()]

        elif quantity == 'ha_vel_disp':

            muse_hdu_filename = os.path.join(muse_dir, muse_version, galaxy + '_MAPS.fits')
            ha = fits.open(muse_hdu_filename)['HA6562_FLUX'].data.flatten()
            ha_vel_disp = fits.open(muse_hdu_filename)['HA6562_SIGMA'].data.flatten()
            ha_vel_disp_corr = np.nanmax(fits.open(muse_hdu_filename)['HA6562_SIGMA_CORR'].data)

            hb_vel_disp = fits.open(muse_hdu_filename)['HB4861_SIGMA'].data.flatten()

            # Flatten, de-NaN

            nan_idx = np.where((~np.isnan(ha)) & (~np.isnan(ha_vel_disp)))
            ha = ha[nan_idx]
            ha_vel_disp = ha_vel_disp[nan_idx]
            ha_vel_disp = np.sqrt(ha_vel_disp ** 2 - ha_vel_disp_corr ** 2)

            if not weight_vel_disps:
                ha = np.ones_like(ha_vel_disp)

            y_kde = gaussian_kde(ha_vel_disp, weights=ha, bw_method='silverman').evaluate(x_ha_kde)

            # Calculate the median and spread of this distribution

            norm_cdf = np.cumsum(y_kde)
            norm_cdf /= np.max(norm_cdf)

            quantity_val = x_ha_kde[(np.abs(norm_cdf - 0.5)).argmin()]
            quantity_err = x_ha_kde[np.abs(norm_cdf - 0.84).argmin()] - quantity_val

        elif quantity == 'r25':

            dist, r25, r25_err = get_info_from_master_table(phangs_master_table, galaxy.lower(), ['dist', 'size_reff', 'size_reff_unc'])  #, 'size_r25'])

            quantity_val = dist * 1e3 * r25 / 3600 * np.pi / 180
            quantity_err = dist * 1e3 * r25_err / 3600 * np.pi / 180  # 0.1 * quantity_val

        else:

            raise Warning('I have no idea what a %s is' % quantity)

        quantity_vals.append(quantity_val)
        quantity_errs.append(quantity_err)

        # Pull out r_bar/r25.

        r25, r_bar = get_info_from_master_table(phangs_master_table, galaxy.lower(), ['size_r25', 'morph_bar_r'])

        if np.isnan(r_bar):
            colour_prop_val.append(0)
        else:
            colour_prop_val.append(r_bar / r25)

    quantity_vals = np.array(quantity_vals)
    quantity_errs = np.array(quantity_errs)

    colour_prop_val = np.array(colour_prop_val)

    cmap = matplotlib.cm.inferno
    vmin, vmax = 0.8 * np.nanmin(colour_prop_val[colour_prop_val != 0]), \
                 1.2 * np.nanmax(colour_prop_val[colour_prop_val != 0])
    cmap_norm = Normalize(vmin=vmin, vmax=vmax)

    bar_idx = colour_prop_val != 0

    colour = cmap(cmap_norm(colour_prop_val))

    # Calculate Tau correlation

    tau_draws = 1000
    tau_bs = np.zeros(tau_draws)
    tau_outlier_bs = np.zeros(tau_draws)

    nan_idx = np.where(~np.isnan(quantity_vals))
    outlier_idx = (galaxies[nan_idx] != 'NGC1300')

    for draw in range(tau_draws):
        scatter_scale = np.random.normal(loc=0, scale=scale_lengths_err_up[nan_idx])
        scatter_quantity = np.random.normal(loc=0, scale=quantity_errs[nan_idx])

        tau_bs[draw], _ = kendalltau(scale_lengths[nan_idx] + scatter_scale,
                                     quantity_vals[nan_idx] + scatter_quantity)
        tau_outlier_bs[draw], _ = kendalltau(scale_lengths[nan_idx][outlier_idx] + scatter_scale[outlier_idx],
                                             quantity_vals[nan_idx][outlier_idx] + scatter_quantity[outlier_idx])

    tau = np.nanmedian(tau_bs)
    tau_err = np.nanpercentile(tau_bs, 84) - tau

    tau_outlier = np.nanmedian(tau_outlier_bs)
    tau_outlier_err = np.nanpercentile(tau_outlier_bs, 84) - tau_outlier

    print(tau_outlier, tau_outlier_err)

    ax = plt.subplot(int(np.ceil(len(quantities) / n_cols)), n_cols, i + 1)

    errorbar_with_colour(ax, scale_lengths[bar_idx], quantity_vals[bar_idx],
                         x_err_up=scale_lengths_err_up[bar_idx], x_err_down=scale_lengths_err_down[bar_idx],
                         y_err_up=quantity_errs[bar_idx], colour=colour[bar_idx], marker='x')
    plt.errorbar(scale_lengths[~bar_idx], quantity_vals[~bar_idx],
                 xerr=[scale_lengths_err_down[~bar_idx], scale_lengths_err_up[~bar_idx]],
                 yerr=quantity_errs[~bar_idx],
                 marker='o', ls='none', c='k')

    if i in range(0, n_cols):
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top')

    if n_cols != 1 and (i + 1) % n_cols == 0:
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position('right')

    if quantity in quantities_to_log:
        plt.yscale('log')

    if i not in range(0, n_cols) and i not in range(len(quantities) - n_cols, len(quantities)):
        ax.xaxis.set_ticklabels([])
    if i in range(len(quantities) - n_cols, len(quantities)) or i in range(0, n_cols):
        if use_two_point_corr_table == '50percent':
            plt.xlabel(r'50% Correlation scale (kpc)')
        elif use_two_point_corr_table == '30percent':
            plt.xlabel(r'30% Correlation scale (kpc)')
        else:
            plt.xlabel(r'$\sigma_l$ (kpc)')  # /r_{25}

    plt.xlim(xlim[metallicity_calib])

    if quantity not in quantities_to_log:
        ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())

    plt.grid()

    plt.ylabel(fancy_labels[quantity])

    plt.text(0.95, 0.95,
             r'$\tau = %.2f\pm%.2f$' % (tau, tau_err),
             ha='right', va='top',
             transform=ax.transAxes,
             bbox=dict(facecolor='white', edgecolor='none', alpha=0.75))

cax = fig.add_axes([0.125, 0.03, 0.775, 0.02])

plt.colorbar(matplotlib.cm.ScalarMappable(norm=cmap_norm, cmap=cmap),
             cax=cax,
             label=colour_prop_fancy_name,
             orientation='horizontal')

# plt.show()

plt.savefig(fig_name + '.pdf', bbox_inches='tight')
plt.savefig(fig_name + '.png', bbox_inches='tight')
plt.close()

print('Complete!')
