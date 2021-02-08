# -*- coding: utf-8 -*-
"""
Run the GPR for metallicity maps

@author: Tom Williams
"""

import os
import socket
import time

import cmocean
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.table import Table

from astronomical_gpr import AstronomicalGPR
from funcs import get_info_from_master_table
from vars import top_dir, plot_dir, hii_mask_dir, muse_dir, muse_version, metallicity_dir, phangs_master_table, \
    use_pix_maps, include_radial_subtract, gpr_step_val, use_regions, simulation, metallicity_calib

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['font.size'] = 14

os.chdir(top_dir)

seed = np.random.seed(420)

if not os.path.exists(os.path.join(metallicity_dir, muse_version)):
    os.makedirs(os.path.join(metallicity_dir, muse_version))

start = time.time()

# Overwrite switches

overwrite_mcmc = False
overwrite_regressor = False
overwrite_predictions = False

emission_to_use = 'HA6562_FLUX'

hii_table = Table.read(os.path.join(hii_mask_dir, 'Nebulae_Catalogue.fits'))

galaxies = ['IC5332']
galaxy_params = ['dist', 'orient_ra', 'orient_dec', 'orient_posang', 'orient_incl', 'size_r25']

for null_hypothesis in [False]:  # TODO: needs True to, but for now just this

    radial_goodness_of_fit = []
    radial_and_azimuthal_goodness_of_fit = []

    for galaxy in galaxies:

        galaxy_metallicity_dir = os.path.join(metallicity_dir, muse_version, galaxy + '_gpr')
        if use_pix_maps:
            galaxy_metallicity_dir += '_pix'
        if null_hypothesis:
            galaxy_metallicity_dir += '_null'
        if not os.path.exists(galaxy_metallicity_dir):
            os.makedirs(galaxy_metallicity_dir)

        galaxy_plot_dir = os.path.join(plot_dir, muse_version, galaxy + '_gpr')
        if use_pix_maps:
            galaxy_plot_dir += '_pix'
        if null_hypothesis:
            galaxy_plot_dir += '_null'
        if not os.path.exists(galaxy_plot_dir):
            os.makedirs(galaxy_plot_dir)

        hdu_file_name = os.path.join(muse_dir, muse_version, galaxy + '_MAPS.fits')
        data_hdu = fits.open(hdu_file_name)[emission_to_use]

        region_hdu_file_name = os.path.join(hii_mask_dir, 'spatial_masks', galaxy + '_HIIreg_mask.fits')
        region_hdu = fits.open(region_hdu_file_name)[0]

        galaxy_edit = galaxy.lower()
        if galaxy == 'NGC628':
            galaxy_edit = 'ngc0628'

        dist, ra, dec, pa, inc, r25 = get_info_from_master_table(phangs_master_table, galaxy_edit, galaxy_params)

        mcmc_file_name = os.path.join(galaxy_metallicity_dir, galaxy)
        if simulation:
            mcmc_file_name += '_sim'
        mcmc_file_name += '_mcmc.pkl'
        step_plot_name = os.path.join(galaxy_plot_dir, galaxy)
        if simulation:
            step_plot_name += '_sim'
        step_plot_name += '_step'
        corner_plot_name = step_plot_name.replace('step', 'corner')
        fit_plot_name = step_plot_name.replace('step', 'fit')
        labels = ['m', r'$R_0$', r'$\sigma$']
        # else:
        #     radial_gpr_file_name = os.path.join(galaxy_metallicity_dir, galaxy)
        #     if simulation:
        #         radial_gpr_file_name += '_sim'
        #     radial_gpr_file_name += '_radial_gpr.pkl'
        #     fit_plot_name = os.path.join(galaxy_plot_dir, galaxy)
        #     if simulation:
        #         fit_plot_name += '_sim'
        #     fit_plot_name += '_fit_radial_gpr'

        print('Fitting %s, null hypothesis=%s, pix_maps=%s, regions=%s, include_radial_subtract=%s, '
              'simulation=%s '
              % (galaxy, null_hypothesis, use_pix_maps, use_regions, include_radial_subtract, simulation)
              )

        gpr_fitter = AstronomicalGPR(galaxy, data_hdu, dist, ra, dec, pa, inc, r25)
        gpr_fitter.calculate_pixel_positions()

        if not use_pix_maps:
            gpr_fitter.calculate_region_parameters(hii_table, 'met_scal')
        else:
            pix_map_file_name = os.path.join(metallicity_dir, 'pix_maps', muse_version,
                                             galaxy + '_metallicity_' + metallicity_calib + '.fits')
            pix_map = fits.open(pix_map_file_name)[0].data
            pix_map_err = fits.open(pix_map_file_name.replace('.fits', '_err.fits'))[0].data

            # Calculate a step value to get us below 100,000 points
            gpr_step_val = int(np.ceil(len(np.where(~np.isnan(pix_map))[0]) / 1e5))

            gpr_fitter.calculate_pixel_parameters(pix_map, pix_map_err, step_val=gpr_step_val)

        if null_hypothesis:
            gpr_fitter.shuffle_parameter()

        # Fit the radial distribution

        # if use_radial_gpr:
        #     gpr_fitter.fit_radial_distribution_gpr(gpr_file=radial_gpr_file_name, verbose=True)
        #     gpr_fitter.subtract_radial_fit_gpr(radial_gpr_file_name, n_batch=100)
        #     gpr_fitter.plot_radial_fit_gpr(radial_gpr_file_name, fit_plot_name)
        # else:
        gpr_fitter.fit_radial_distribution(mcmc_file_name=mcmc_file_name, overwrite_mcmc=overwrite_mcmc,
                                           verbose=True)
        gpr_fitter.subtract_radial_fit()
        if not os.path.exists(step_plot_name) or overwrite_mcmc:
            gpr_fitter.plot_step(step_plot_name, labels)
            gpr_fitter.plot_corner(corner_plot_name, labels)
            gpr_fitter.plot_radial_fit(fit_plot_name)

        radial_goodness_of_fit.append(gpr_fitter.radial_goodness_of_fit)

        # Set up pixels to fit

        if use_regions:
            gpr_fitter.calc_xy_to_fit_regions(region_hdu, step_val=gpr_step_val, use_radial=include_radial_subtract)
        else:
            gpr_fitter.calc_xy_to_fit_positions(use_radial=include_radial_subtract)

        gpr_file = os.path.join(galaxy_metallicity_dir, galaxy)
        if simulation:
            gpr_file += '_sim'
        if use_regions:
            gpr_file += '_regions'
        else:
            gpr_file += '_positions'
        if include_radial_subtract:
            # if use_radial_gpr:
            #     gpr_file += '_gpr_radial'
            # else:
            gpr_file += '_linear_radial'
        else:
            gpr_file += '_no_radial'
        gpr_file += '_gpr.pkl'

        # Throw into GPR and create also create predictions

        pred_file = gpr_file.replace('_gpr.pkl', '_predict.npy')
        pred_file_central = gpr_file.replace('_gpr.pkl', '_predict_central.npy')
        pred_file_radial_gpr = gpr_file.replace('_gpr.pkl', '_predict_radial_gpr.npy')
        realisations_cube_file = gpr_file.replace('_gpr.pkl', '_cube.npy')

        # Skip reading in/generating GPR files if running on laptop

        if socket.gethostname() in ['astro-node4']:
            gpr_fitter.fit_gpr_regressor(gpr_file, overwrite_regressor=overwrite_regressor, verbose=True)
            radial_and_azimuthal_goodness_of_fit.append(gpr_fitter.gp_goodness_of_fit)

        gpr_fitter.make_predictions(pred_file, gpr_fitter.xy_full, name='full', n_batch=100,
                                    overwrite_predictions=overwrite_predictions, verbose=True)
        gpr_fitter.make_predictions(pred_file_central, gpr_fitter.xy_regions, name='central',
                                    overwrite_predictions=overwrite_predictions, verbose=True)

        # if use_radial_gpr:
        #     gpr_fitter.make_predictions(pred_file_radial_gpr, gpr_fitter.r_full[gpr_fitter.nan_mask].flatten(),
        #                                 gpr_file=radial_gpr_file_name, name='radial_gpr', n_batch=100,
        #                                 overwrite_predictions=overwrite_predictions, verbose=True)

        # Create metallicity maps

        metallicity_file_name = os.path.join(galaxy_metallicity_dir, galaxy)
        if use_regions:
            metallicity_file_name += '_regions'
        else:
            metallicity_file_name += '_positions'
        if include_radial_subtract:
            # if use_radial_gpr:
            #     metallicity_file_name += '_gpr_radial'
            # else:
            metallicity_file_name += '_linear_radial'
        else:
            gpr_file += '_no_radial'

        metallicity_map, metallicity_map_err, metallicity_map_radial_subtract = \
            gpr_fitter.create_map(metallicity_file_name, pred_name='full', use_radial=include_radial_subtract)

        x_region_pix = gpr_fitter.x_region_pix
        y_region_pix = gpr_fitter.y_region_pix
        r_metallicity = gpr_fitter.r_regions
        metallicity = gpr_fitter.parameter
        metallicity_radial_subtract = gpr_fitter.parameter_radial_subtract
        metallicity_err = gpr_fitter.parameter_err
        intrinsic_scatter = gpr_fitter.intrinsic_scatter

        z_pred_central = np.load(pred_file_central)

        z_percentiles = np.nanpercentile(metallicity_map, [2, 98])
        z_residual_percentiles = np.max(np.abs(np.nanpercentile(metallicity_radial_subtract, [2, 98])))
        z_err_percentiles = np.nanpercentile(metallicity_map_err, [2, 98])

        # Produce a big ol' plot with a whole bunch of useful maps on it

        v_min = np.nanpercentile(data_hdu.data, 2)
        v_max = np.nanpercentile(data_hdu.data, 98)

        fig = plt.figure(figsize=(8, 6))

        if use_pix_maps:
            s = 0.25
        else:
            s = 1

        ax1 = fig.add_axes([0.125, 0.5, 0.5 - 0.125, 0.4])

        # Overlay positions on the Halpha image
        ax1_im = ax1.imshow(data_hdu.data, cmap='gray_r', vmin=v_min, vmax=v_max, origin='lower')
        ax1_scatter = ax1.scatter(x_region_pix, y_region_pix, c=metallicity, cmap=cmocean.cm.haline,
                                  vmin=z_percentiles[0], vmax=z_percentiles[1], s=s, alpha=0.25)

        ax1.axis('off')

        if include_radial_subtract:
            ax2 = fig.add_axes([0.5, 0.5, 0.5 - 0.125, 0.4])
            ax2_im = ax2.imshow(metallicity_map_radial_subtract, origin='lower', cmap=cmocean.cm.balance,
                                vmin=-z_residual_percentiles, vmax=z_residual_percentiles)

            ax2_scatter = ax2.scatter(x_region_pix, y_region_pix, c=metallicity_radial_subtract,
                                      cmap=cmocean.cm.balance,
                                      vmin=-z_residual_percentiles, vmax=z_residual_percentiles, alpha=0.25, s=s)

            ax2_cbarax = fig.add_axes([0.875, 0.5, 0.025, 0.4])
            ax2_cbar = plt.colorbar(ax2_im, cax=ax2_cbarax)
            ax2_cbar.set_label(r'$\Delta$(log(O/H))')

            ax2.axis('off')

        ax3 = fig.add_axes([0.1, 0.1, 0.5 - 0.125, 0.4])
        ax3_im = ax3.imshow(metallicity_map, origin='lower', cmap=cmocean.cm.haline, vmin=z_percentiles[0],
                            vmax=z_percentiles[1])
        ax3_scatter = ax3.scatter(x_region_pix, y_region_pix, c=metallicity, cmap=cmocean.cm.haline,
                                  vmin=z_percentiles[0], vmax=z_percentiles[1], alpha=0.25, s=s)

        ax3_cbarax = fig.add_axes([0.1, 0.1, 0.025, 0.8])
        ax3_cbar = plt.colorbar(ax3_im, cax=ax3_cbarax)
        ax3_cbar.set_label(r'12 + log(O/H)')
        ax3_cbarax.yaxis.set_ticks_position('left')
        ax3_cbarax.yaxis.set_label_position('left')

        ax3.axis('off')

        ax4 = fig.add_axes([0.5, 0.1, 0.5 - 0.125, 0.4])
        ax4_im = ax4.imshow(metallicity_map_err, origin='lower', cmap=cmocean.cm.thermal, vmin=z_err_percentiles[0],
                            vmax=z_err_percentiles[1])

        ax4_cbarax = fig.add_axes([0.875, 0.1, 0.025, 0.4])
        ax4_cbar = plt.colorbar(ax4_im, cax=ax4_cbarax)
        ax4_cbar.set_label(r'$\sigma_\mathrm{log(O/H)}$')

        ax4.axis('off')

        map_plot_name = os.path.join(galaxy_plot_dir, galaxy)

        if simulation:
            map_plot_name += '_sim'

        if use_regions:
            map_plot_name += '_regions'
        else:
            map_plot_name += '_positions'

        map_plot_name += '_gpr'

        # plt.show()

        plt.savefig(map_plot_name + '.pdf', bbox_inches='tight')
        plt.savefig(map_plot_name + '.png', bbox_inches='tight')

        plt.close()

        comparison_plot_name = map_plot_name[:-4] + '_radial_comparison'

        metallicity_azimuthal_subtract = metallicity_radial_subtract - z_pred_central[0]

        radial_scatter = np.nanstd(metallicity_radial_subtract)
        azimuthal_scatter = np.nanstd(metallicity_azimuthal_subtract)

        metallicity_gpr_central_err = z_pred_central[1]

        # Calculate chi-square

        radial_abs_residual = np.nansum(
            (metallicity_radial_subtract / metallicity_err) ** 2
        )
        azimuthal_abs_residual = np.nansum(
            (metallicity_azimuthal_subtract / metallicity_err) ** 2
        )

        residual_ratios.append(azimuthal_abs_residual / radial_abs_residual)

        # Produce a plot to see if we've reduced the scatter

        x_min, x_max = 0, 1.05 * np.nanmax(r_metallicity)
        y_min, y_max = np.nanpercentile(metallicity_radial_subtract, 0.1), np.nanpercentile(metallicity_radial_subtract,
                                                                                            99.9)

        plt.figure(figsize=(8, 6))

        plt.errorbar(r_metallicity, metallicity_radial_subtract, yerr=metallicity_err, c='k', marker='o', ls='none',
                     alpha=0.5, label=r'Radial, $\sigma=%.2f$' % radial_scatter)
        plt.errorbar(r_metallicity, metallicity_azimuthal_subtract, yerr=metallicity_err, c='r', marker='o', ls='none',
                     alpha=0.5, label=r'Radial and Azimuthal, $\sigma=%.2f$' % azimuthal_scatter)
        plt.axhline(0, c='k', ls='--')

        plt.xlim([x_min, x_max])
        plt.ylim([y_min, y_max])

        plt.xlabel(r'$R/r_{25}$')
        plt.ylabel(r'[12 + log(O/H)]$_\mathrm{obs}$ - [12 + log(O/H)]$_\mathrm{pred}$')

        plt.legend(loc='upper right', framealpha=0.5)

        plt.tight_layout()

        plt.savefig(comparison_plot_name + '.pdf', bbox_inches='tight')
        plt.savefig(comparison_plot_name + '.png', bbox_inches='tight')

        plt.close()

        # And also look at the radial variation in the residuals

        v_max = np.max(np.abs(np.nanpercentile(metallicity_radial_subtract, [2, 98])))
        im_vmin, im_vmax = np.nanpercentile(data_hdu.data, 0.5), np.nanpercentile(data_hdu.data, 99.5)

        plt.figure(figsize=(9, 6))

        if use_pix_maps:
            s = 1
        else:
            s = 3

        plt.subplot(1, 2, 1)
        plt.imshow(data_hdu.data,
                   cmap='gray_r',
                   vmin=im_vmin, vmax=im_vmax,
                   origin='lower')
        plt.scatter(x_region_pix, y_region_pix, c=metallicity_radial_subtract,
                    vmin=-v_max, vmax=v_max, s=s, cmap=cmocean.cm.balance)
        plt.title('Radial')

        plt.colorbar(pad=0.03, label=r'$\Delta$(O/H)', orientation='horizontal')

        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(data_hdu.data,
                   cmap='gray_r',
                   vmin=im_vmin, vmax=im_vmax,
                   origin='lower')
        plt.scatter(x_region_pix, y_region_pix, c=metallicity_azimuthal_subtract,
                    vmin=-v_max, vmax=v_max, s=s, cmap=cmocean.cm.balance)
        plt.title('Radial and azimuthal')

        plt.colorbar(pad=0.03, label=r'$\Delta$(O/H)', orientation='horizontal')

        plt.tight_layout()

        plt.axis('off')

        residual_position_plot_name = map_plot_name[:-4] + '_residual_positions'

        plt.savefig(residual_position_plot_name + '.pdf', bbox_inches='tight')
        plt.savefig(residual_position_plot_name + '.png', bbox_inches='tight')

        plt.close()

    # Produce plot of the improvement in residuals, and also the (ln) Bayes factor.

    bayes_factor = np.array(scale_length_table['gp_ln_l'] - scale_length_table['radial_ln_l'])

    plot_name = os.path.join(plot_dir, muse_version, 'model_comparisons')
    if use_pix_maps:
        plot_name += '_pix_maps'
        ylims = [10 ** -1.6, 10 ** 1.1]
    else:
        ylims = [10 ** -2.6, 10 ** 2.1]

    fig, axes = plt.subplots(2, 1, sharex='all', figsize=(12, 6))

    ax1 = axes[0]

    ax1.axhline(1, c='k', ls='--')

    ax1.text(0, 1.2, 'Radial fit better describes data', ha='left', va='bottom')
    ax1.text(0, 0.8, 'Radial and azimuthal fit better describes data', ha='left', va='top')

    ax1.scatter(np.arange(len(galaxies)), residual_ratios, c='k')

    ax1.set_xticks(np.arange(len(galaxies)))
    ax1.set_xticklabels(galaxies, rotation=45)
    ax1.xaxis.tick_top()
    ax1.tick_params(labeltop=True)
    ax1.set_yscale('log')

    ax1.set_ylabel(r'$\frac{\chi^2_\mathrm{Radial+Azimuthal}}{\chi^2_\mathrm{Radial}}$')
    ax1.set_ylim(ylims)

    # plt.show()

    ax2 = axes[1]

    ax2.axhline(0, c='k', ls='--')

    ax2.text(0, 1, 'Radial and azimuthal model preferred', ha='left', va='bottom')
    ax2.text(0, -1, 'Radial model preferred', ha='left', va='top')

    ax2.scatter(np.arange(len(galaxies)), bayes_factor, c='k')

    ax2.set_yscale('symlog')

    ax2.set_xticks(np.arange(len(galaxies)))
    ax2.set_xticklabels(galaxies, rotation=45)

    ax2.set_ylabel(r'$\ln(\mathcal{L}_\mathrm{Radial+azimuthal}) - \ln(\mathcal{L}_\mathrm{Radial})$')
    ax2.set_ylim([-10 ** 6.1, 10 ** 6.1])
    ax2.set_yticks([-10 ** 6, -10 ** 4, -10 ** 2, 0, 10 ** 2, 10 ** 4, 10 ** 6])

    plt.subplots_adjust(wspace=0, hspace=0.1)

    # plt.show()

    plt.savefig(plot_name + '.pdf', bbox_inches='tight')
    plt.savefig(plot_name + '.png', bbox_inches='tight')

    plt.close()

print('Complete! Took %.2fm' % ((time.time() - start) / 60))
