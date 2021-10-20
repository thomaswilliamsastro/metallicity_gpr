# -*- coding: utf-8 -*-
"""
Create pixel-by-pixel metallicity maps, rather than from nebulae catalogues

@author: Tom Williams
"""

import os
import socket
import time
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pyneb as pn
import seaborn as sns
from astropy.io import fits
from uncertainties import unumpy as unp
from astropy.table import Table
from reproject import reproject_interp
from scipy.stats import gaussian_kde

from vars import top_dir, muse_dir, muse_version, galaxies, metallicity_dir, metallicity_calibs, all_lines, \
    strong_lines, use_conv, plot_dir, extinction_curve, hii_only, hii_mask_dir


def get_contour_z(x, y, X, Y):
    k = gaussian_kde(np.vstack([x, y]))
    Z = k(np.vstack([X.flatten(), Y.flatten()]))
    Z = (Z - Z.min()) / (Z.max() - Z.min())
    Z = Z.reshape(X.shape)

    return Z


if 'mac' in socket.gethostname():
    os.environ['PATH'] += ':/Library/TeX/texbin'

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['text.usetex'] = True
plt.rcParams.update({'text.latex.preamble': r'\usepackage{txfonts}'})

warnings.simplefilter('ignore')
sns.set_color_codes()

os.chdir(top_dir)

overwrite = False

nebulae_cat = Table.read(os.path.join(hii_mask_dir, 'Nebulae_catalogue_v2.fits'))

k_lambda_themis = {'HB4861': 3.886,
                   'OIII4958': 3.797,
                   'OIII5006': 3.755,
                   'NII6548': 2.728,
                   'HA6562': 2.720,
                   'NII6583': 2.710,
                   'SII6716': 2.644,
                   'SII6730': 2.637}

s_n_ratio = 5  # 3
vel_cutoff = 100

flux_to_use = 'HA6562_FLUX'

plot_out_dir = os.path.join(plot_dir, muse_version, '%s_ext_curve' % extinction_curve, 'bpt_diagnostics')
if hii_only:
    plot_out_dir += '_hii_only'
if not os.path.exists(plot_out_dir):
    os.makedirs(plot_out_dir)

for galaxy in galaxies:

    for metallicity_calib in metallicity_calibs:

        hdu_out_dir = os.path.join(metallicity_dir, 'pix_maps', muse_version, '%s_ext_curve' % extinction_curve,
                                   metallicity_calib)
        if hii_only:
            hdu_out_dir += '_hii_only'
        if not os.path.exists(hdu_out_dir):
            os.makedirs(hdu_out_dir)

        hdu_out_name = os.path.join(hdu_out_dir, galaxy + '_metallicity_' + metallicity_calib + '.fits')

        if not os.path.exists(hdu_out_name) or overwrite:

            print('Beginning %s, %s, hii only=%s' % (galaxy, metallicity_calib, hii_only))

            start = time.time()

            maps = {}

            muse_hdu_file_name = os.path.join(muse_dir, muse_version)
            if use_conv:
                muse_hdu_file_name += '/conv/'
            muse_hdu_file_name += galaxy + '_MAPS.fits'
            muse_hdu = fits.open(muse_hdu_file_name)

            # Also load in the native HDU for the velocity dispersion correction
            muse_native_hdu = fits.open(muse_hdu_file_name.replace('/conv', ''))

            # We begin the mask. First up, any NaNs in the original image. 1 for pixels we exclude, 0 for ones we
            # include

            mask = np.zeros_like(muse_hdu[flux_to_use].data)
            mask[np.isnan(muse_hdu[flux_to_use].data)] = 1

            for line in all_lines:

                if line in ['OIII4958', 'NII6548']:
                    stronger_line = {'OIII4958': 'OIII5006',
                                     'NII6548': 'NII6583'}[line]
                    maps[line + '_uncorr'] = maps[stronger_line + '_uncorr'] / 3
                else:

                    data = muse_hdu[line + '_FLUX'].data
                    err = muse_hdu[line + '_FLUX_ERR'].data

                    idx = np.where(data == 0)
                    data[idx] = np.nan
                    err[idx] = np.nan

                    maps[line + '_uncorr'] = unp.uarray(data, err)

            print('%.1f: Initial arrays created' % (time.time() - start))

            if extinction_curve == 'themis':
                denom = -0.466
            elif extinction_curve == 'odonnell94':
                rc = pn.RedCorr(R_V=3.1, law='CCM89 oD94')
                k_ha, k_hb = rc._CCM89_oD94([6562, 4861])
                denom = 0.4 * (k_ha - k_hb)
            else:
                raise Exception('Extinction curve type %s not recognised' % extinction_curve)

            c_hbeta = (np.log10(2.86) - unp.log10(maps['HA6562_uncorr'] / maps['HB4861_uncorr'])) / denom

            c_hbeta[c_hbeta <= 0] = 0
            c_hbeta[c_hbeta > 3] = np.nan

            print('%.1f: Balmer decrement calculated' % (time.time() - start))

            # Correct all the lines

            for line in all_lines:

                if extinction_curve == 'themis':
                    k = k_lambda_themis[line]
                elif extinction_curve == 'odonnell94':
                    wavelength = float(line[-4:])
                    k = rc._CCM89_oD94(wavelength)
                else:
                    raise Exception('Extinction curve type %s not recognised' % extinction_curve)

                if line in ['OIII4958', 'NII6548']:
                    stronger_line = {'OIII4958': 'OIII5006',
                                     'NII6548': 'NII6583'}[line]
                    maps[line] = maps[stronger_line] / 3
                else:
                    maps[line] = maps[line + '_uncorr'] * 10 ** (0.4 * c_hbeta * k)

            print('%.1f: Lines corrected' % (time.time() - start))

            # Take only the lines with S/N > cutoff, and low velocity dispersions

            for line in strong_lines:
                s_n_mask = np.where(unp.nominal_values(maps[line]) < s_n_ratio * unp.std_devs(maps[line]))
                mask[s_n_mask] = 1

                # The velocity dispersions should be corrected for the line spread function
                vel_disp_factor = np.nanmax(muse_native_hdu[line + '_SIGMA_CORR'].data)
                vel_disp_corr = np.sqrt(muse_hdu[line + '_SIGMA'].data ** 2 - vel_disp_factor ** 2)

                vel_disp_mask = np.where(vel_disp_corr >= vel_cutoff)
                mask[vel_disp_mask] = 1

            print('%.1f: S/N and vel disp mask created' % (time.time() - start))

            # If only using HII regions, mask here

            if hii_only:

                region_ids = nebulae_cat[(nebulae_cat['gal_name'] == galaxy) & (nebulae_cat['HII_class'] == 1)][
                    'region_ID']
                nebulae_mask_file_name = os.path.join(hii_mask_dir, 'spatial_masks', '%s_nebulae_mask_V2.fits' % galaxy)
                nebulae_mask_hdu = fits.open(nebulae_mask_file_name)[0]

                hii_regions = np.ones_like(nebulae_mask_hdu.data)

                for region_id in region_ids:
                    idx = np.where(nebulae_mask_hdu.data == region_id)
                    hii_regions[idx] = 0

                nebulae_mask_hdu.data = hii_regions

                hii_region_mask, _ = reproject_interp(nebulae_mask_hdu, muse_hdu[1].header,
                                                      order='nearest-neighbor')
                hii_region_mask[np.isnan(hii_region_mask)] = 1

                mask[np.where(hii_region_mask == 1)] = 1

                print('%.1f: HII regions selected' % (time.time() - start))

            # Now correct via BPT plots. First, [NII]/Halpha versus [OIII]/Hbeta

            bpt_plot_mask = np.ones_like(mask)

            n2_ha = unp.log10(maps['NII6583'] / maps['HA6562'])
            oiii_hb = unp.log10(maps['OIII5006'] / maps['HB4861'])

            bpt_mask = np.where((oiii_hb >= 0.61 / (n2_ha - 0.05) + 1.3) | (n2_ha >= 0.05))
            mask[bpt_mask] = 1
            bpt_plot_mask[bpt_mask] = 0

            # Second, [SII]/Halpha versus [OIII]/Hbeta

            s2_ha = unp.log10((maps['SII6716'] + maps['SII6730']) / maps['HA6562'])

            bpt_mask = (oiii_hb >= 0.72 / (s2_ha - 0.32) + 1.3) | (s2_ha >= 0.32)
            mask[bpt_mask] = 1
            bpt_plot_mask[bpt_mask] = 0

            # BPT plots

            n2_ha_plot = unp.nominal_values(n2_ha)
            s2_ha_plot = unp.nominal_values(s2_ha)
            oiii_hb_plot = unp.nominal_values(oiii_hb)

            n2_ha_lims = [-1, 0.1]
            s2_ha_lims = [-1.3, 0.4]
            oiii_hb_lims = [-2, 1.1]

            kauffman_x = np.linspace(n2_ha_lims[0], 0.05)
            kauffman_y = 0.61 / (kauffman_x - 0.05) + 1.3

            kewley_x = np.linspace(s2_ha_lims[0], 0.32)
            kewley_y = 0.72 / (kewley_x - 0.32) + 1.3

            plot_name = os.path.join(plot_out_dir, galaxy + '_bpt')

            plt.figure(figsize=(8, 4))

            # We plot all pixels, then all pixels that pass BPT, then all pixels that satisfy all constraints.

            # levels = [0.1, 0.5, 0.9]
            #
            # x = np.linspace(*n2_ha_lims, 100)
            # y = np.linspace(*oiii_hb_lims, 100)
            # X, Y = np.meshgrid(x, y)
            #
            # nan_idx = np.where((~np.isnan(n2_ha_plot)) & (~np.isnan(oiii_hb_plot)))
            # mask_idx = mask[nan_idx]
            # bpt_mask_idx = bpt_plot_mask[nan_idx]
            #
            # Z_all = get_contour_z(n2_ha_plot[nan_idx][mask_idx == 1], oiii_hb_plot[nan_idx][mask_idx == 1],
            #                       X, Y)
            # Z_bpt = get_contour_z(n2_ha_plot[nan_idx][bpt_mask_idx == 1], oiii_hb_plot[nan_idx][bpt_mask_idx == 1],
            #                       X, Y)
            # Z_final = get_contour_z(n2_ha_plot[nan_idx][mask_idx == 0], oiii_hb_plot[nan_idx][mask_idx == 0],
            #                         X, Y)

            ax = plt.subplot(1, 2, 1)
            plt.scatter(n2_ha_plot[mask == 1], oiii_hb_plot[mask == 1], c='k', s=2, alpha=0.1, label='All pixels',
                        rasterized=True)
            plt.scatter(n2_ha_plot[bpt_plot_mask == 1], oiii_hb_plot[bpt_plot_mask == 1], c='orange', s=2, alpha=0.1,
                        label='BPT constraints',
                        rasterized=True)
            plt.scatter(n2_ha_plot[mask == 0], oiii_hb_plot[mask == 0], c='b', s=2, alpha=0.25, label='All constraints',
                        rasterized=True)
            # plt.contour(X, Y, Z_all, levels, colors='k', linewidths=1, origin='lower', label='All pixels')
            # plt.contour(X, Y, Z_bpt, levels, colors='orange', linewidths=1, origin='lower', label='BPT constraints')
            # plt.contour(X, Y, Z_final, levels, colors='b', linewidths=1, origin='lower', label='All constraints')

            plt.plot(kauffman_x, kauffman_y, c='k', lw=1)

            plt.xlim(n2_ha_lims)
            plt.ylim(oiii_hb_lims)

            plt.minorticks_on()

            plt.grid(c='k', alpha=0.5)

            plt.xlabel(r'$\log_{10}$([N{\sc ii}]$\lambdaup 6583$/H$\alphaup$)')
            plt.ylabel(r'$\log_{10}$([O{\sc iii}]$\lambdaup 5006$/H$\betaup$)')
            # plt.text(0.05, 0.05, r'{\bf %s}' % galaxy,
            #          ha='left', va='bottom',
            #          bbox=dict(facecolor='white', edgecolor='black', alpha=0.8),
            #          transform=ax.transAxes)
            # plt.legend(loc='upper left', edgecolor='k', handletextpad=0, fancybox=False)

            ax = plt.subplot(1, 2, 2)

            plt.scatter(s2_ha_plot[mask == 1], oiii_hb_plot[mask == 1], c='k', s=2, alpha=0.1,
                        rasterized=True)
            plt.scatter(s2_ha_plot[bpt_plot_mask == 1], oiii_hb_plot[bpt_plot_mask == 1], c='orange', s=2, alpha=0.1,
                        rasterized=True)
            plt.scatter(s2_ha_plot[mask == 0], oiii_hb_plot[mask == 0], c='b', alpha=0.25, s=2,
                        rasterized=True)

            plt.scatter(-99, -99, c='k', label='All pixels')
            plt.scatter(-99, -99, c='orange', label='BPT constraints')
            plt.scatter(-99, -99, c='b', label='All constraints')

            plt.plot(kewley_x, kewley_y, c='k', lw=1)

            plt.xlim(s2_ha_lims)
            plt.ylim(oiii_hb_lims)

            ax.yaxis.set_label_position("right")
            ax.yaxis.tick_right()

            plt.minorticks_on()

            plt.grid(c='k', alpha=0.5)

            plt.xlabel(r'$\log_{10}$([S{\sc ii}]$\lambdaup\lambdaup 6716,30$/H$\alphaup$)')
            plt.ylabel(r'$\log_{10}$([O{\sc iii}]$\lambdaup 5006$/H$\betaup$)')
            plt.legend(loc='lower right', edgecolor='k', handletextpad=0, fancybox=False)

            plt.tight_layout()

            plt.subplots_adjust(wspace=0, hspace=0)

            # plt.show()

            plt.savefig(plot_name + '.png', bbox_inches='tight')
            plt.savefig(plot_name + '.pdf', bbox_inches='tight')

            plt.close()

            print('%.1f: BPT mask created' % (time.time() - start))

            # We're now good to calculate metallicities.

            metallicity = np.zeros_like(muse_hdu[flux_to_use].data)
            metallicity[metallicity == 0] = np.nan
            metallicity_err = metallicity.copy()

            if metallicity_calib == 'pg16_scal':
                # PG '16 Scal metallicity

                n2 = (maps['NII6583'] + maps['NII6548']) / maps['HB4861']
                s2 = (maps['SII6716'] + maps['SII6730']) / maps['HB4861']
                r3 = (maps['OIII5006'] + maps['OIII4958']) / maps['HB4861']

                # Calculate upper and lower branches

                upper = np.where(unp.log10(n2) >= -0.6)
                lower = np.where(unp.log10(n2) < -0.6)

                lower_metallicity = 8.072 + 0.789 * unp.log10(r3[lower] / s2[lower]) + 0.726 * unp.log10(n2[lower]) + \
                                    (1.069 - 0.17 * unp.log10(r3[lower] / s2[lower]) + 0.022 * unp.log10(
                                        n2[lower])) * unp.log10(s2[lower])

                metallicity[lower] = unp.nominal_values(lower_metallicity)
                metallicity_err[lower] = unp.std_devs(lower_metallicity)

                upper_metallicity = 8.424 + 0.03 * unp.log10(r3[upper] / s2[upper]) + 0.751 * unp.log10(n2[upper]) + \
                                    (-0.349 + 0.182 * unp.log10(r3[upper] / s2[upper]) + 0.508 * unp.log10(
                                        n2[upper])) * unp.log10(s2[upper])

                metallicity[upper] = unp.nominal_values(upper_metallicity)
                metallicity_err[upper] = unp.std_devs(upper_metallicity)

            elif metallicity_calib == 'd16':
                # Dopita '16 metallicity
                z_sol = 8.77

                n2_s2 = unp.log10(maps['NII6583'] / (maps['SII6716'] + maps['SII6730']))
                n2_ha = unp.log10(maps['NII6583'] / maps['HA6562'])

                y = n2_s2 + 0.264 * n2_ha
                Z = z_sol + y + 0.45 * (y + 0.3) ** 5
                metallicity = unp.nominal_values(Z)
                metallicity_err = unp.std_devs(Z)

            else:
                raise Warning('Metallicity calibration %s not recognised' % metallicity_calib)

            metallicity[mask == 1] = np.nan
            metallicity_err[mask == 1] = np.nan

            idx = np.argsort(metallicity.flatten())

            vmin, vmax = np.nanpercentile(metallicity, [5, 95])

            plot_name = os.path.join(plot_out_dir, galaxy + '_bpt_metallicity_coloured')

            fig = plt.figure(figsize=(8, 4))

            # We plot all pixels, then all pixels that pass BPT, then all pixels that satisfy all constraints

            ax = plt.subplot(1, 2, 1)
            ax.scatter(n2_ha_plot.flatten()[idx], oiii_hb_plot.flatten()[idx], c=metallicity.flatten()[idx], s=2,
                       vmin=vmin, vmax=vmax)

            plt.plot(kauffman_x, kauffman_y, c='k', lw=1)

            plt.xlim(n2_ha_lims)
            plt.ylim(oiii_hb_lims)

            plt.minorticks_on()

            plt.grid(c='k', alpha=0.5)

            plt.xlabel(r'$\log_{10}$([N{\sc ii}]$\lambdaup 6583$/H$\alphaup$)')
            plt.ylabel(r'$\log_{10}$([O{\sc iii}]$\lambdaup 5006$/H$\betaup$)')

            ax = plt.subplot(1, 2, 2)

            ax_scatter = ax.scatter(s2_ha_plot.flatten()[idx], oiii_hb_plot.flatten()[idx],
                                    c=metallicity.flatten()[idx], s=2, vmin=vmin, vmax=vmax)

            plt.plot(kewley_x, kewley_y, c='k', lw=1)

            plt.xlim(s2_ha_lims)
            plt.ylim(oiii_hb_lims)

            ax.yaxis.set_label_position("right")
            ax.yaxis.tick_right()

            plt.minorticks_on()

            plt.grid(c='k', alpha=0.5)

            plt.xlabel(r'$\log_{10}$([S{\sc ii}]$\lambdaup\lambdaup 6716,30$/H$\alphaup$)')
            plt.ylabel(r'$\log_{10}$([O{\sc iii}]$\lambdaup 5006$/H$\betaup$)')

            # plt.tight_layout()
            cbarax = fig.add_axes([0.125, -0.05, 0.775, 0.025])
            cbar = plt.colorbar(ax_scatter, cax=cbarax, orientation='horizontal')
            cbar.set_label('$12+\log_{10}(\mathrm{O/H})$')

            plt.subplots_adjust(wspace=0, hspace=0)

            # plt.show()

            plt.savefig(plot_name + '.png', bbox_inches='tight')
            plt.savefig(plot_name + '.pdf', bbox_inches='tight')

            plt.close()

            print('%.1f: Metallicity map calculated' % (time.time() - start))

            header = muse_hdu[flux_to_use].header
            fits.writeto(hdu_out_name, metallicity, header, overwrite=True)
            fits.writeto(hdu_out_name.replace('.fits', '_err.fits'), metallicity_err, header, overwrite=True)

print('Complete!')
