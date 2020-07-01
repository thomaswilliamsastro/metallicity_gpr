# -*- coding: utf-8 -*-
"""
Run the GPR for metallicity maps

@author: Tom Williams
"""

import os
import socket
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.table import Table
import cmocean

from vars import top_dir, plot_dir, hii_mask_dir, muse_dir, muse_version, metallicity_dir, galaxies, phangs_master_table
from astronomical_gpr import AstronomicalGPR

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['font.size'] = 14

os.chdir(top_dir)

seed = np.random.seed(420)

if not os.path.exists('metallicity'):
    os.makedirs('metallicity')

start = time.time()

overwrite_mcmc = False
overwrite_regressor = False
overwrite_predictions = False
use_regions_bools = [True]  # [False, True]
simulation = False

emission_to_use = 'HA6562_FLUX'

if socket.gethostname() in ['astro-node4']:
    write_scale_lengths = True
else:
    write_scale_lengths = False

# Step value for filtering down the GPR points
step_val = 3

chisq_ratios = []

hii_table = Table.read(hii_mask_dir + 'Nebulae_Catalogue.fits')

if write_scale_lengths:
    scale_length_file = open('metallicity/scale_lengths.txt', 'w+')

for galaxy in galaxies:

    galaxy_plot_dir = os.path.join(plot_dir, galaxy + '_gpr')
    galaxy_metallicity_dir = os.path.join(metallicity_dir, galaxy + '_gpr')

    if not os.path.exists(galaxy_plot_dir):
        os.makedirs(galaxy_plot_dir)
    if not os.path.exists(galaxy_metallicity_dir):
        os.makedirs(galaxy_metallicity_dir)

    hdu_file_name = os.path.join(muse_dir, muse_version, galaxy + '_MAPS.fits')
    data_hdu = fits.open(hdu_file_name)[emission_to_use]

    region_hdu_file_name = os.path.join(hii_mask_dir, 'spatial_masks', galaxy + '_HIIreg_mask.fits')
    region_hdu = fits.open(region_hdu_file_name)[0]

    galaxy_edit = galaxy.lower()
    if galaxy == 'NGC628':
        galaxy_edit = 'ngc0628'

    row = phangs_master_table[phangs_master_table['name'] == galaxy_edit]
    dist, ra, dec, pa, inc, r25 = row['dist'][0], row['orient_ra'][0], row['orient_dec'][0], row['orient_posang'][0], \
        row['orient_incl'][0], row['size_r25'][0]

    for use_regions in use_regions_bools:

        mcmc_file_name = os.path.join(metallicity_dir, galaxy + '_gpr', galaxy)
        if simulation:
            mcmc_file_name += '_sim'
        mcmc_file_name += '_mcmc.pkl'

        step_plot_name = os.path.join(plot_dir, galaxy + '_gpr', galaxy)
        if simulation:
            step_plot_name += '_sim'
        step_plot_name += '_step'
        corner_plot_name = step_plot_name.replace('step', 'corner')
        fit_plot_name = step_plot_name.replace('step', 'fit')

        labels = ['m', r'$R_0$', r'$\sigma$']

        print('Fitting %s with regions=%s' % (galaxy, use_regions))

        gpr_fitter = AstronomicalGPR(galaxy, data_hdu, dist, ra, dec, pa, inc, r25)
        gpr_fitter.calculate_pixel_positions()
        gpr_fitter.calculate_region_parameters(hii_table, 'met_scal')

        # Fit the radial distribution

        gpr_fitter.fit_radial_distribution(mcmc_file_name=mcmc_file_name, overwrite_mcmc=overwrite_mcmc, verbose=True)
        if not os.path.exists(step_plot_name + '.png') or overwrite_mcmc:
            gpr_fitter.plot_step(step_plot_name, labels)
            gpr_fitter.plot_corner(corner_plot_name, labels)
            gpr_fitter.plot_radial_fit(fit_plot_name)

        gpr_fitter.subtract_radial_fit()

        # Set up pixels to fit

        if use_regions:
            gpr_fitter.calc_xy_to_fit_regions(region_hdu, step_val=step_val)
        else:
            gpr_fitter.calc_xy_to_fit_positions()

        gpr_file = metallicity_dir + galaxy + '_gpr/' + galaxy
        if simulation:
            gpr_file += '_sim'
        if use_regions:
            gpr_file += '_regions'
        else:
            gpr_file += '_positions'
        gpr_file += '_gpr.pkl'

        # Throw into GPR and create also create predictions

        pred_file = gpr_file.replace('_gpr.pkl', '_predict.npy')
        pred_file_central = gpr_file.replace('_gpr.pkl', '_predict_central.npy')
        realisations_cube_file = gpr_file.replace('_gpr.pkl', '_cube.npy')

        # Skip reading in/generating GPR files if running on laptop

        if socket.gethostname() in ['astro-node4']:
            gpr_fitter.fit_gpr_regressor(gpr_file, overwrite_regressor=overwrite_regressor, verbose=True)

            print(gpr_fitter.scale_length)
            if write_scale_lengths:
                scale_length_file.write('%s, %s, %.2f\n' % (galaxy, use_regions, gpr_fitter.scale_length))

        gpr_fitter.make_predictions(pred_file, gpr_fitter.xy_full, name='full', n_batch=100,
                                    overwrite_predictions=overwrite_predictions, verbose=True)
        gpr_fitter.make_predictions(pred_file_central, gpr_fitter.xy_regions, name='central',
                                    overwrite_predictions=overwrite_predictions, verbose=True)

        # Create metallicity maps

        metallicity_file_name = os.path.join(metallicity_dir, galaxy + '_gpr', galaxy)
        if use_regions:
            metallicity_file_name += '_regions'
        else:
            metallicity_file_name += '_positions'

        metallicity_map, metallicity_map_err, metallicity_map_radial_subtract = \
            gpr_fitter.create_map(metallicity_file_name, pred_name='full')

        x_region_pix = gpr_fitter.x_region_pix
        y_region_pix = gpr_fitter.y_region_pix
        r_metallicity = gpr_fitter.r_regions
        metallicity = gpr_fitter.parameter
        metallicity_radial_subtract = gpr_fitter.parameter_radial_subtract
        metallicity_err = gpr_fitter.parameter_err

        z_pred_central = np.load(pred_file_central)

        z_percentiles = np.nanpercentile(metallicity_map, [2, 98])
        z_residual_percentiles = np.max(np.abs(np.nanpercentile(metallicity_radial_subtract, [2, 98])))
        z_err_percentiles = np.nanpercentile(metallicity_map_err, [2, 98])

        # Produce a big ol' plot with a whole bunch of useful maps on it

        v_min = np.nanpercentile(data_hdu.data, 2)
        v_max = np.nanpercentile(data_hdu.data, 98)

        fig = plt.figure(figsize=(8, 6))

        ax1 = fig.add_axes([0.125, 0.5, 0.5 - 0.125, 0.4])

        # Overlay positions on the Halpha image
        ax1_im = ax1.imshow(data_hdu.data,
                            cmap='gray_r',
                            vmin=v_min, vmax=v_max,
                            origin='lower')

        ax1_scatter = ax1.scatter(x_region_pix, y_region_pix,
                                  c=metallicity, cmap=cmocean.cm.haline,
                                  vmin=z_percentiles[0], vmax=z_percentiles[1],
                                  s=1, alpha=0.25)

        ax1.axis('off')

        ax2 = fig.add_axes([0.5, 0.5, 0.5 - 0.125, 0.4])
        ax2_im = ax2.imshow(metallicity_map_radial_subtract,
                            origin='lower',
                            cmap=cmocean.cm.balance,
                            vmin=-z_residual_percentiles, vmax=z_residual_percentiles)
        ax2_scatter = ax2.scatter(x_region_pix, y_region_pix,
                                  c=gpr_fitter.parameter_radial_subtract, cmap=cmocean.cm.balance,
                                  vmin=-z_residual_percentiles, vmax=z_residual_percentiles,
                                  alpha=0.25, s=1)

        ax2_cbarax = fig.add_axes([0.875, 0.5, 0.025, 0.4])
        ax2_cbar = plt.colorbar(ax2_im, cax=ax2_cbarax)
        ax2_cbar.set_label(r'$\Delta$(log(O/H))')

        ax2.axis('off')

        ax3 = fig.add_axes([0.1, 0.1, 0.5 - 0.125, 0.4])
        ax3_im = ax3.imshow(metallicity_map,
                            origin='lower',
                            cmap=cmocean.cm.haline,
                            vmin=z_percentiles[0], vmax=z_percentiles[1])
        ax3_scatter = ax3.scatter(x_region_pix, y_region_pix,
                                  c=metallicity, cmap=cmocean.cm.haline,
                                  vmin=z_percentiles[0], vmax=z_percentiles[1],
                                  alpha=0.25, s=1)

        ax3_cbarax = fig.add_axes([0.1, 0.1, 0.025, 0.8])
        ax3_cbar = plt.colorbar(ax3_im, cax=ax3_cbarax)
        ax3_cbar.set_label(r'12 + log(O/H)')
        ax3_cbarax.yaxis.set_ticks_position('left')
        ax3_cbarax.yaxis.set_label_position('left')

        ax3.axis('off')

        ax4 = fig.add_axes([0.5, 0.1, 0.5 - 0.125, 0.4])
        ax4_im = ax4.imshow(metallicity_map_err,
                            origin='lower',
                            cmap=cmocean.cm.thermal,
                            vmin=z_err_percentiles[0], vmax=z_err_percentiles[1])

        ax4_cbarax = fig.add_axes([0.875, 0.1, 0.025, 0.4])
        ax4_cbar = plt.colorbar(ax4_im, cax=ax4_cbarax)
        ax4_cbar.set_label(r'$\sigma_\mathrm{log(O/H)}$')

        ax4.axis('off')

        map_plot_name = plot_dir + galaxy + '_gpr/' + galaxy

        if simulation:
            map_plot_name += '_sim'

        if use_regions:
            map_plot_name += '_regions'
        else:
            map_plot_name += '_positions'

        map_plot_name += '_gpr'

        # plt.show()

        plt.savefig(map_plot_name + '.pdf',
                    bbox_inches='tight')
        plt.savefig(map_plot_name + '.png',
                    bbox_inches='tight')

        plt.close()

        metallicity_azimuthal_subtract = metallicity_radial_subtract - z_pred_central[0]
        comparison_plot_name = map_plot_name[:-4] + '_radial_comparison'

        radial_scatter = np.nanstd(metallicity_radial_subtract)
        azimuthal_scatter = np.nanstd(metallicity_azimuthal_subtract)

        # Calculate a chi-square to show the model improves the predictions

        radial_chisq = np.nansum(metallicity_radial_subtract ** 2 / metallicity_err ** 2)
        azimuthal_chisq = np.nansum(metallicity_azimuthal_subtract ** 2 / metallicity_err ** 2)

        chisq_ratios.append(azimuthal_chisq / radial_chisq)

        # Produce a plot to see if we've reduced the scatter

        x_min, x_max = 0, 1.05 * np.nanmax(r_metallicity)
        y_min, y_max = np.nanpercentile(metallicity_radial_subtract, 0.1), np.nanpercentile(metallicity_radial_subtract,
                                                                                            99.9)

        plt.figure(figsize=(8, 6))

        plt.errorbar(r_metallicity, metallicity_radial_subtract, yerr=metallicity_err, c='k', marker='o', ls='none',
                     alpha=0.5, label=r'Radial, $\sigma=%.2f$' % radial_scatter)
        plt.errorbar(r_metallicity, metallicity_azimuthal_subtract, yerr=metallicity_err, c='r', marker='o', ls='none',
                     alpha=0.5, label=r'Azimuthal, $\sigma=%.2f$' % azimuthal_scatter)
        plt.axhline(0, c='k', ls='--')

        plt.xlim([x_min, x_max])
        plt.ylim([y_min, y_max])

        plt.xlabel(r'$R/r_{25}$')
        plt.ylabel(r'[12 + log(O/H)]$_\mathrm{obs}$ - [12 + log(O/H)]$_\mathrm{pred}$')

        plt.legend(loc='upper right', framealpha=0.5)

        plt.tight_layout()

        plt.savefig(comparison_plot_name + '.pdf',
                    bbox_inches='tight')
        plt.savefig(comparison_plot_name + '.png',
                    bbox_inches='tight')

        plt.close()

        # And also look at the radial variation in the residuals

        v_max = np.max(np.abs(np.nanpercentile(metallicity_radial_subtract, [2, 98])))
        im_vmin, im_vmax = np.nanpercentile(data_hdu.data, 0.5), np.nanpercentile(data_hdu.data, 99.5)

        plt.figure(figsize=(9, 6))

        plt.subplot(1, 2, 1)
        plt.imshow(data_hdu.data,
                   cmap='gray_r',
                   vmin=im_vmin, vmax=im_vmax,
                   origin='lower')
        plt.scatter(x_region_pix, y_region_pix, c=metallicity_radial_subtract,
                    vmin=-v_max, vmax=v_max, s=3, cmap=cmocean.cm.balance)
        plt.title('Radial')

        plt.colorbar(pad=0.03, label=r'$\Delta$(O/H)', orientation='horizontal')

        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(data_hdu.data,
                   cmap='gray_r',
                   vmin=im_vmin, vmax=im_vmax,
                   origin='lower')
        plt.scatter(x_region_pix, y_region_pix, c=metallicity_azimuthal_subtract,
                    vmin=-v_max, vmax=v_max, s=3, cmap=cmocean.cm.balance)
        plt.title('GPR')

        plt.colorbar(pad=0.03, label=r'$\Delta$(O/H)', orientation='horizontal')

        plt.tight_layout()

        plt.axis('off')

        residual_position_plot_name = map_plot_name[:-4] + '_residual_positions'

        plt.savefig(residual_position_plot_name + '.pdf',
                    bbox_inches='tight')
        plt.savefig(residual_position_plot_name + '.png',
                    bbox_inches='tight')

        plt.close()

if write_scale_lengths:
    scale_length_file.close()

# Produce plot of the improvement in scatter

plot_name = os.path.join(plot_dir, 'chisq_comparison')

plt.figure(figsize=(12, 4))

plt.axhline(1, c='k', ls='--')
plt.scatter(np.arange(len(galaxies)), chisq_ratios, c='k')

plt.xticks(np.arange(len(galaxies)), galaxies, rotation=45)

plt.ylabel(r'$\frac{\chi^2_\mathrm{azimuthal}}{\chi^2_\mathrm{radial}}$')

plt.tight_layout()

plt.savefig(plot_name + '.pdf', bbox_inches='tight')
plt.savefig(plot_name + '.png', bbox_inches='tight')

plt.close()

print('Complete! Took %.2fm' % ((time.time() - start) / 60))

print('Complete!')
