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
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from uncertainties import ufloat
from uncertainties import unumpy as unp

from astronomical_gpr import AstronomicalGPR, get_pix_size
from funcs import get_info_from_master_table
from vars import top_dir, plot_dir, hii_mask_dir, muse_dir, muse_version, metallicity_dir, phangs_master_table, \
    use_pix_maps, include_radial_subtract, gpr_step_val, use_regions, simulation, metallicity_calibs, use_conv, \
    galaxies, hii_only, extinction_curve

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['font.size'] = 14

os.chdir(top_dir)

seed = np.random.seed(420)

if not os.path.exists(os.path.join(metallicity_dir, muse_version, '%s_ext_curve' % extinction_curve)):
    os.makedirs(os.path.join(metallicity_dir, muse_version, '%s_ext_curve' % extinction_curve))

start = time.time()

# Overwrite switches

overwrite_mcmc = False
overwrite_regressor = False
overwrite_predictions = False

emission_to_use = 'HA6562_FLUX'

if use_regions:
    hii_table = Table.read(os.path.join(hii_mask_dir, 'Nebulae_Catalogue.fits'))

galaxy_params = ['dist', 'orient_ra', 'orient_dec', 'orient_posang', 'orient_incl', 'size_r25']

for metallicity_calib in metallicity_calibs:

    for null_hypothesis in [False]:  # [False, True]:

        radial_goodness_of_fit = []
        radial_and_azimuthal_goodness_of_fit = []
        chisq_ratios = []

        for galaxy in galaxies:

            galaxy_metallicity_dir = os.path.join(metallicity_dir, muse_version, '%s_ext_curve' % extinction_curve,
                                                  metallicity_calib, galaxy + '_' + metallicity_calib + '_gpr')
            if use_pix_maps:
                galaxy_metallicity_dir += '_pix'
            if hii_only:
                galaxy_metallicity_dir += '_hii_only'
            if null_hypothesis:
                galaxy_metallicity_dir += '_null'
            if not os.path.exists(galaxy_metallicity_dir):
                os.makedirs(galaxy_metallicity_dir)

            galaxy_plot_dir = os.path.join(plot_dir, muse_version, '%s_ext_curve' % extinction_curve,
                                           metallicity_calib, galaxy + '_' + metallicity_calib + '_gpr')
            if use_pix_maps:
                galaxy_plot_dir += '_pix'
            if hii_only:
                galaxy_plot_dir += '_hii_only'
            if null_hypothesis:
                galaxy_plot_dir += '_null'
            if not os.path.exists(galaxy_plot_dir):
                os.makedirs(galaxy_plot_dir)

            # Native resolution MUSE maps
            hdu_file_name = os.path.join(muse_dir, muse_version, galaxy + '_MAPS.fits')
            data_hdu = fits.open(hdu_file_name)[emission_to_use]
            wcs = WCS(data_hdu.header)

            # Get a simple (x, y, z) to contour to show data coverage
            coverage_hdu = fits.open(hdu_file_name)[1]
            x_grid, y_grid = np.meshgrid(np.arange(coverage_hdu.data.shape[1]),
                                         np.arange(coverage_hdu.data.shape[0]))
            z_grid = np.zeros_like(coverage_hdu.data)
            z_grid[~np.isnan(coverage_hdu.data)] = 1

            if use_regions:
                region_hdu_file_name = os.path.join(hii_mask_dir, 'spatial_masks', galaxy + '_HIIreg_mask.fits')
                region_hdu = fits.open(region_hdu_file_name)[0]

            dist, ra, dec, pa, inc, r25 = get_info_from_master_table(phangs_master_table, galaxy.lower(), galaxy_params)

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

            print('Fitting %s, metallicity calib %s, null hypothesis=%s, pix_maps=%s, regions=%s, '
                  'include_radial_subtract=%s, simulation=%s, hii_only=%s '
                  % (galaxy, metallicity_calib, null_hypothesis, use_pix_maps, use_regions, include_radial_subtract,
                     simulation, hii_only)
                  )

            gpr_fitter = AstronomicalGPR(galaxy, data_hdu, dist, ra, dec, pa, inc, r25)
            gpr_fitter.calculate_pixel_positions()

            if not use_pix_maps:
                gpr_fitter.calculate_region_parameters(hii_table, 'met_scal')
            else:
                pix_map_file_name = os.path.join(metallicity_dir, 'pix_maps', muse_version,
                                                 '%s_ext_curve' % extinction_curve, metallicity_calib)
                if hii_only:
                    pix_map_file_name += '_hii_only'
                pix_map_file_name = os.path.join(pix_map_file_name,
                                                 galaxy + '_metallicity_' + metallicity_calib + '.fits')
                pix_map = fits.open(pix_map_file_name)[0]
                pix_map_err = fits.open(pix_map_file_name.replace('.fits', '_err.fits'))[0]
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

            # Throw into GPR

            pred_file = gpr_file.replace('_gpr.pkl', '_predict.npy')
            pred_file_central = gpr_file.replace('_gpr.pkl', '_predict_central.npy')
            # pred_file_radial_gpr = gpr_file.replace('_gpr.pkl', '_predict_radial_gpr.npy')
            # realisations_cube_file = gpr_file.replace('_gpr.pkl', '_cube.npy')

            # Skip reading in/generating GPR files if running on laptop

            if 'astro-node' in socket.gethostname():
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

            r0 = ufloat(gpr_fitter.r_0, gpr_fitter.r_0_err)
            m = ufloat(gpr_fitter.m, gpr_fitter.m_err)

            z_pred_central = np.load(pred_file_central)

            z_percentiles = np.nanpercentile(metallicity_map, [2, 98])
            z_residual_percentiles = np.max(np.abs(np.nanpercentile(metallicity_radial_subtract, [2, 98])))
            z_err_percentiles = np.nanpercentile(metallicity_map_err, [2, 98])

            # Produce a big ol' plot with a whole bunch of useful maps on it

            v_min = np.nanpercentile(data_hdu.data, 2)
            v_max = np.nanpercentile(data_hdu.data, 98)

            # Set the Fig to match the shape of the HDU so there aren't gaps between subplots

            data_ratio = data_hdu.data.shape[0] / data_hdu.data.shape[1]

            # kpc in pixels

            pix_size = get_pix_size(data_hdu)
            arcsec_to_kpc = 1 / (dist * 1e3) * 180 / np.pi * 3600
            kpc_length = arcsec_to_kpc / pix_size

            background = data_hdu.data.copy()
            background[~np.isnan(background)] = 1

            # fig = plt.figure(figsize=(8.25, 11.75))
            fig = plt.figure(figsize=(8.25, 8.25 * data_ratio))

            if use_pix_maps:
                s = 0.25
            else:
                s = 1

            ax1 = fig.add_axes([0.125, 0.5, 0.5 - 0.125, 0.5 - 0.125], projection=wcs)

            # Overlay positions on the Halpha image
            ax1.contour(x_grid, y_grid, z_grid, colors='k', linewidths=0.5, alpha=0.75)
            ax1_im = ax1.imshow(background, cmap='gray_r', vmin=0, vmax=1, origin='lower',
                                alpha=0.5)
            ax1_scatter = ax1.scatter(x_region_pix, y_region_pix, c=metallicity, cmap=cmocean.cm.haline,
                                      vmin=z_percentiles[0], vmax=z_percentiles[1], s=s, alpha=1)

            scalebar = AnchoredSizeBar(ax1.transData,
                                       kpc_length, '1 kpc', 'lower left',
                                       pad=0.5,
                                       borderpad=0.5,
                                       sep=3,
                                       color='black',
                                       frameon=True,
                                       size_vertical=1,)

            ax1.add_artist(scalebar)

            # ax1.axis('off')
            ax1.tick_params(direction='in')

            ax_x = ax1.coords[0]
            ax_y = ax1.coords[1]

            ax_x.grid()
            ax_y.grid()

            ax_x.set_axislabel('RA (J2000)')
            ax_y.set_axislabel('Dec (J2000)')

            ax_x.display_minor_ticks(True)
            ax_y.display_minor_ticks(True)

            ax_x.set_ticklabel_position('t')
            ax_x.set_axislabel_position('t')

            if include_radial_subtract:
                ax2 = fig.add_axes([0.5, 0.5, 0.5 - 0.125, 0.5 - 0.125], projection=wcs)
                ax2.contour(x_grid, y_grid, z_grid, colors='k', linewidths=0.5, alpha=0.75)
                ax2_im = ax2.imshow(metallicity_map_radial_subtract, origin='lower', cmap=cmocean.cm.balance,
                                    vmin=-z_residual_percentiles, vmax=z_residual_percentiles)

                ax2_scatter = ax2.scatter(x_region_pix, y_region_pix, c=metallicity_radial_subtract,
                                          cmap=cmocean.cm.balance,
                                          vmin=-z_residual_percentiles, vmax=z_residual_percentiles, alpha=0.5, s=s)

                scalebar = AnchoredSizeBar(ax2.transData,
                                           kpc_length, '1 kpc', 'lower left',
                                           pad=0.5,
                                           borderpad=0.5,
                                           sep=3,
                                           color='black',
                                           frameon=True,
                                           size_vertical=1, )

                ax2.add_artist(scalebar)

                ax2.tick_params(direction='in')

                ax2_cbarax = fig.add_axes([1.05, 0.5, 0.025, 0.5-0.125])
                ax2_cbar = plt.colorbar(ax2_im, cax=ax2_cbarax)
                ax2_cbar.set_label(r'$\Delta$log$_{10}$(O/H)')

                # ax2.axis('off')

                ax_x = ax2.coords[0]
                ax_y = ax2.coords[1]

                ax_x.grid()
                ax_y.grid()

                ax_x.set_axislabel('RA (J2000)')
                ax_y.set_axislabel('Dec (J2000)')

                ax_x.display_minor_ticks(True)
                ax_y.display_minor_ticks(True)

                ax_x.set_ticklabel_position('t')
                ax_x.set_axislabel_position('t')

                ax_y.set_ticklabel_position('r')
                ax_y.set_axislabel_position('r')

            ax3 = fig.add_axes([0.125, 0.125, 0.5 - 0.125, 0.5 - 0.125], projection=wcs)
            ax3.contour(x_grid, y_grid, z_grid, colors='k', linewidths=0.5, alpha=0.75)
            ax3_im = ax3.imshow(metallicity_map, origin='lower', cmap=cmocean.cm.haline, vmin=z_percentiles[0],
                                vmax=z_percentiles[1])
            ax3_scatter = ax3.scatter(x_region_pix, y_region_pix, c=metallicity, cmap=cmocean.cm.haline,
                                      vmin=z_percentiles[0], vmax=z_percentiles[1], alpha=0.5, s=s)

            scalebar = AnchoredSizeBar(ax3.transData,
                                       kpc_length, '1 kpc', 'lower left',
                                       pad=0.5,
                                       borderpad=0.5,
                                       sep=3,
                                       color='black',
                                       frameon=True,
                                       size_vertical=1, )

            ax3.add_artist(scalebar)

            ax3.tick_params(direction='in')

            ax3_cbarax = fig.add_axes([-0.075, 0.125, 0.025, 1 - 0.25])
            ax3_cbar = plt.colorbar(ax3_im, cax=ax3_cbarax)
            ax3_cbar.set_label(r'12 + log$_{10}$(O/H)')
            ax3_cbarax.yaxis.set_ticks_position('left')
            ax3_cbarax.yaxis.set_label_position('left')

            # ax3.axis('off')

            ax_x = ax3.coords[0]
            ax_y = ax3.coords[1]

            ax_x.grid()
            ax_y.grid()

            ax_x.set_axislabel('RA (J2000)')
            ax_y.set_axislabel('Dec (J2000)')

            ax_x.display_minor_ticks(True)
            ax_y.display_minor_ticks(True)

            ax4 = fig.add_axes([0.5, 0.125, 0.5 - 0.125, 0.5 - 0.125], projection=wcs)
            ax4.contour(x_grid, y_grid, z_grid, colors='k', linewidths=0.5, alpha=0.75)
            ax4_im = ax4.imshow(metallicity_map_err, origin='lower', cmap=cmocean.cm.thermal, vmin=z_err_percentiles[0],
                                vmax=z_err_percentiles[1])

            scalebar = AnchoredSizeBar(ax4.transData,
                                       kpc_length, '1 kpc', 'lower left',
                                       pad=0.5,
                                       borderpad=0.5,
                                       sep=3,
                                       color='black',
                                       frameon=True,
                                       size_vertical=1, )

            ax4.add_artist(scalebar)

            ax4.tick_params(direction='in')

            ax4_cbarax = fig.add_axes([1.05, 0.125, 0.025, 0.5 - 0.125])
            ax4_cbar = plt.colorbar(ax4_im, cax=ax4_cbarax)
            ax4_cbar.set_label(r'$\sigma_\mathrm{log_{10}(O/H)}$')

            # ax4.axis('off')
            ax_x = ax4.coords[0]
            ax_y = ax4.coords[1]

            ax_x.grid()
            ax_y.grid()

            ax_x.set_axislabel('RA (J2000)')
            ax_y.set_axislabel('Dec (J2000)')

            ax_x.display_minor_ticks(True)
            ax_y.display_minor_ticks(True)

            ax_y.set_ticklabel_position('r')
            ax_y.set_axislabel_position('r')

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

            radial_chisq = np.nansum(
                (metallicity_radial_subtract / metallicity_err) ** 2
            )
            azimuthal_chisq = np.nansum(
                (metallicity_azimuthal_subtract / metallicity_err) ** 2
            )

            chisq_ratios.append(azimuthal_chisq / radial_chisq)

            # Produce a plot to see if we've reduced the scatter

            x_min, x_max = 0, 1.05 * np.nanmax(r_metallicity)
            y_min, y_max = np.nanpercentile(metallicity_radial_subtract, 0.1), np.nanpercentile(
                metallicity_radial_subtract,
                99.9)

            plt.figure(figsize=(8, 6))

            plt.errorbar(r_metallicity, metallicity_radial_subtract, yerr=metallicity_err, c='k', marker='o', ls='none',
                         alpha=0.5, label=r'Radial, $\sigma=%.2f$' % radial_scatter)
            plt.errorbar(r_metallicity, metallicity_azimuthal_subtract, yerr=metallicity_err, c='r', marker='o',
                         ls='none',
                         alpha=0.5, label=r'Radial and Higher Order, $\sigma=%.2f$' % azimuthal_scatter)
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

            fig = plt.figure(figsize=(9, 6))

            if use_pix_maps:
                s = 2
            else:
                s = 3

            # ax1 = plt.subplot(1, 2, 1, projection=wcs)
            ax1 = fig.add_axes([0.125, 0.125, 0.5 - 0.125, 0.8], projection=wcs)
            ax1.contour(x_grid, y_grid, z_grid, colors='k', linewidths=0.5, alpha=0.75)
            plt.imshow(data_hdu.data,
                       cmap='gray_r',
                       vmin=im_vmin, vmax=im_vmax,
                       origin='lower')
            scatter = ax1.scatter(x_region_pix, y_region_pix, c=metallicity_radial_subtract,
                                  vmin=-v_max, vmax=v_max, s=s, cmap=cmocean.cm.balance)

            plt.text(0.05, 0.95, r'Radial',
                     ha='left', va='top',
                     bbox=dict(facecolor='white', edgecolor='black', alpha=0.8),
                     transform=ax1.transAxes)

            ax_x = ax1.coords[0]
            ax_y = ax1.coords[1]

            ax_x.grid()
            ax_y.grid()

            ax_x.set_axislabel('RA (J2000)')
            ax_y.set_axislabel('Dec (J2000)')

            ax_x.display_minor_ticks(True)
            ax_y.display_minor_ticks(True)

            ax2 = fig.add_axes([0.5, 0.125, 0.5 - 0.125, 0.8], projection=wcs)
            ax2.contour(x_grid, y_grid, z_grid, colors='k', linewidths=0.5, alpha=0.75)
            plt.imshow(data_hdu.data,
                       cmap='gray_r',
                       vmin=im_vmin, vmax=im_vmax,
                       origin='lower')
            plt.scatter(x_region_pix, y_region_pix, c=metallicity_azimuthal_subtract,
                        vmin=-v_max, vmax=v_max, s=s, cmap=cmocean.cm.balance)
            plt.text(0.05, 0.95, r'Radial and Higher Order',
                     ha='left', va='top',
                     bbox=dict(facecolor='white', edgecolor='black', alpha=0.8),
                     transform=ax2.transAxes)

            # plt.colorbar(pad=0.125, label=r'$\Delta$log$_{10}$(O/H)', orientation='horizontal')
            cbarax = fig.add_axes([0.125, 0, 0.775, 0.025])
            cbar = plt.colorbar(scatter, cax=cbarax, orientation='horizontal')
            cbar.set_label(r'$\log_{10}\mathrm{(O/H)}_\mathrm{obs} - \log_{10}\mathrm{(O/H)}_\mathrm{pred}$')

            # plt.tight_layout()

            # plt.axis('off')
            ax_x = ax2.coords[0]
            ax_y = ax2.coords[1]

            ax_x.grid()
            ax_y.grid()

            ax_x.set_axislabel('RA (J2000)')
            ax_y.set_axislabel('Dec (J2000)')

            ax_x.display_minor_ticks(True)
            ax_y.display_minor_ticks(True)

            ax_y.set_ticklabel_position('r')
            ax_y.set_axislabel_position('r')

            residual_position_plot_name = map_plot_name[:-4] + '_residual_positions'

            plt.savefig(residual_position_plot_name + '.pdf', bbox_inches='tight')
            plt.savefig(residual_position_plot_name + '.png', bbox_inches='tight')

            plt.close()

            # Also residuals but as sigma outliers

            v_min, v_max = -3, 3

            radial_fit_err = unp.std_devs(r_metallicity * m + r0)

            radial_sigma = metallicity_radial_subtract / np.sqrt(radial_fit_err ** 2 + metallicity_err ** 2)
            azimuthal_sigma = metallicity_azimuthal_subtract / \
                              np.sqrt(metallicity_gpr_central_err ** 2 + metallicity_err ** 2)

            fig = plt.figure(figsize=(9, 9 * data_ratio / 2))

            if use_pix_maps:
                s = 2
            else:
                s = 3

            ax1 = fig.add_axes([0.125, 0.125, 0.5 - 0.125, 1 - 0.125], projection=wcs)
            ax1.contour(x_grid, y_grid, z_grid, colors='k', linewidths=0.5, alpha=0.75)
            # plt.imshow(data_hdu.data,
            #            cmap='gray_r',
            #            vmin=im_vmin, vmax=im_vmax,
            #            origin='lower')
            plt.imshow(background,
                       cmap='gray_r',
                       vmin=0, vmax=1,
                       origin='lower',
                       alpha=0.5)
            scatter = ax1.scatter(x_region_pix, y_region_pix, c=radial_sigma,
                                  vmin=v_min, vmax=v_max, s=s, cmap=cmocean.cm.balance)

            plt.text(0.05, 0.95, r'Radial',
                     ha='left', va='top',
                     bbox=dict(facecolor='white', edgecolor='black', alpha=0.8),
                     transform=ax1.transAxes)

            scalebar = AnchoredSizeBar(ax1.transData,
                                       kpc_length, '1 kpc', 'lower left',
                                       pad=0.5,
                                       borderpad=0.5,
                                       sep=3,
                                       color='black',
                                       frameon=True,
                                       size_vertical=1, )

            ax1.add_artist(scalebar)

            ax1.tick_params(direction='in')

            ax_x = ax1.coords[0]
            ax_y = ax1.coords[1]

            ax_x.grid()
            ax_y.grid()

            ax_x.set_axislabel('RA (J2000)')
            ax_y.set_axislabel('Dec (J2000)')

            ax_x.display_minor_ticks(True)
            ax_y.display_minor_ticks(True)

            ax2 = fig.add_axes([0.5, 0.125, 0.5 - 0.125, 1 - 0.125], projection=wcs)
            ax2.contour(x_grid, y_grid, z_grid, colors='k', linewidths=0.5, alpha=0.75)
            # plt.imshow(data_hdu.data,
            #            cmap='gray_r',
            #            vmin=im_vmin, vmax=im_vmax,
            #            origin='lower',
            #            alpha=0.75)
            plt.imshow(background,
                       cmap='gray_r',
                       vmin=0, vmax=1,
                       origin='lower',
                       alpha=0.5)
            plt.scatter(x_region_pix, y_region_pix, c=azimuthal_sigma,
                        vmin=v_min, vmax=v_max, s=s, cmap=cmocean.cm.balance)
            plt.text(0.05, 0.95, r'Radial and Higher Order',
                     ha='left', va='top',
                     bbox=dict(facecolor='white', edgecolor='black', alpha=0.8),
                     transform=ax2.transAxes)

            scalebar = AnchoredSizeBar(ax2.transData,
                                       kpc_length, '1 kpc', 'lower left',
                                       pad=0.5,
                                       borderpad=0.5,
                                       sep=3,
                                       color='black',
                                       frameon=True,
                                       size_vertical=1, )

            ax2.add_artist(scalebar)

            cbarax = fig.add_axes([0.125, 0.075, 0.75, 0.025])
            cbar = plt.colorbar(scatter, cax=cbarax, orientation='horizontal')

            cbar.set_label(r'$\frac{\log_{10}(\mathrm{O/H})_{\rm obs} - \log_{10}(\mathrm{O/H})_{\rm pred}}'
                           r'{\sigma[\log_{10}(\mathrm{O/H})_{\rm obs} - \log_{10}(\mathrm{O/H})_{\rm pred}]}$')

            ax2.tick_params(direction='in')

            ax_x = ax2.coords[0]
            ax_y = ax2.coords[1]

            ax_x.grid()
            ax_y.grid()

            ax_x.set_axislabel('RA (J2000)')
            ax_y.set_axislabel('Dec (J2000)')

            ax_x.display_minor_ticks(True)
            ax_y.display_minor_ticks(True)

            ax_y.set_ticklabel_position('r')
            ax_y.set_axislabel_position('r')

            residual_position_plot_name = map_plot_name[:-4] + '_residual_positions_sigma'

            plt.savefig(residual_position_plot_name + '.pdf', bbox_inches='tight')
            plt.savefig(residual_position_plot_name + '.png', bbox_inches='tight')

            plt.close()

        if null_hypothesis:
            continue

        # Look at the difference in R^2 values

        r_factor_diff = np.array(radial_and_azimuthal_goodness_of_fit) - np.array(radial_goodness_of_fit)

        plot_name = os.path.join(plot_dir, muse_version, '%s_ext_curve' % extinction_curve, metallicity_calib,
                                 'model_comparisons_' + metallicity_calib)
        if use_pix_maps:
            plot_name += '_pix_maps'
            ylims = [10 ** -1.3, 10 ** 1.3]
        else:
            ylims = [10 ** -2.6, 10 ** 2.1]

        if hii_only:
            plot_name += '_hii_only'

        fig, axes = plt.subplots(2, 1, sharex='all', figsize=(12, 6))

        ax1 = axes[0]

        ax1.axhline(1, c='k', ls='--')

        ax1.text(0, 1e1, 'Radial model better describes data', ha='left', va='bottom')
        ax1.text(0, 1e-1, 'Radial and higher order model better describes data', ha='left', va='top')

        ax1.scatter(np.arange(len(galaxies)), chisq_ratios, c='k')

        ax1.set_xticks(np.arange(len(galaxies)))
        ax1.set_xticklabels(galaxies, rotation=45, ha='left')
        ax1.xaxis.tick_top()
        ax1.tick_params(labeltop=True)
        ax1.set_yscale('log')

        ax1.set_ylabel(r'$\frac{\chi^2_\mathrm{Radial+Higher Order}}{\chi^2_\mathrm{Radial}}$')
        ax1.set_ylim(ylims)

        ax1.grid(axis='both')

        plt.grid()

        # plt.show()

        ax2 = axes[1]

        ax2.axhline(0, c='k', ls='--')

        ax2.text(0, 0.8, 'Radial and higher order model better describes variability', ha='left', va='bottom')
        ax2.text(0, -0.8, 'Radial model better describes variability', ha='left', va='top')

        ax2.scatter(np.arange(len(galaxies)), r_factor_diff, c='k')

        ax2.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.1))
        ax2.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.5))

        # ax2.set_yscale('symlog')

        ax2.set_xticks(np.arange(len(galaxies)))
        ax2.set_xticklabels(galaxies, rotation=45, ha='right')

        ax2.set_ylabel(r'$R^2_\mathrm{Radial+higher order} - R^2_\mathrm{Radial}$')
        ax2.set_ylim([-1.1, 1.1])
        # ax2.set_yticks([-10 ** 6, -10 ** 4, -10 ** 2, 0, 10 ** 2, 10 ** 4, 10 ** 6])

        ax2.grid(b=True, axis='both')

        plt.subplots_adjust(wspace=0, hspace=0)

        # plt.show()

        plt.savefig(plot_name + '.pdf', bbox_inches='tight')
        plt.savefig(plot_name + '.png', bbox_inches='tight')

        plt.close()

print('Complete! Took %.2fm' % ((time.time() - start) / 60))
