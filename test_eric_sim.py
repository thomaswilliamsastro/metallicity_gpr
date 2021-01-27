# -*- coding: utf-8 -*-
"""
Run the GPR on simulated metallicity map

@author: Tom Williams
"""

import os
import socket
import warnings

import cmocean
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.table import Table
from astropy.wcs import WCS
from photutils import make_source_mask
from scipy.stats import gaussian_kde

from astronomical_gpr import AstronomicalGPR, get_pix_size
from vars import top_dir, metallicity_dir, muse_dir, muse_version, hii_mask_dir, phangs_master_table, plot_dir


def sigma_clip(data, sigma=3, n_pixels=5, max_iterations=20):
    """TODO: Write docstring"""

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        mask = make_source_mask(data, nsigma=sigma, npixels=n_pixels)
        mean, median, std_dev = sigma_clipped_stats(data, mask=mask, sigma=sigma, maxiters=max_iterations)

    return [mean, median, std_dev]


def density_estimation(m1, m2):
    X, Y = np.mgrid[np.nanmin(m1):np.nanmax(m1):100j, np.nanmin(m2):np.nanmax(m2):100j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([m1, m2])
    kernel = gaussian_kde(values)
    Z = np.reshape(kernel(positions).T, X.shape)
    return X, Y, Z


os.chdir(top_dir)

if not os.path.exists(os.path.join(metallicity_dir, 'eric_sim_gpr')):
    os.makedirs(os.path.join(metallicity_dir, 'eric_sim_gpr'))
if not os.path.exists(os.path.join(plot_dir, 'eric_sim_gpr')):
    os.makedirs(os.path.join(plot_dir, 'eric_sim_gpr'))

hdu_file_name = os.path.join(metallicity_dir, 'eric_sim', 'O_00188_l15_N628_70hr.fits')
sim_hdu = fits.open(hdu_file_name)[0]

row = phangs_master_table[phangs_master_table['name'] == 'ngc1512']
dist, r25 = row['dist'][0], row['size_r25'][0]

inc = 0
pa = 0
step_val = 3

# Put this into a fits file we already have

hdu_file_name = os.path.join(muse_dir, muse_version, 'NGC1512_MAPS.fits')
real_hdu = fits.open(hdu_file_name)['HA6562_FLUX']

region_hdu_file_name = os.path.join(hii_mask_dir, 'spatial_masks', 'NGC1512_HIIreg_mask.fits')
region_hdu = fits.open(region_hdu_file_name)[0]

hii_table = Table.read(hii_mask_dir + 'Nebulae_Catalogue.fits')

final_hdu = real_hdu.copy()
final_hdu.data = sim_hdu.data[
                 int(sim_hdu.data.shape[0] / 2) - int(real_hdu.data.shape[0] / 2):
                 int(sim_hdu.data.shape[0] / 2) + int(real_hdu.data.shape[0] / 2),
                 int(sim_hdu.data.shape[1] / 2) - int(real_hdu.data.shape[1] / 2):
                 int(sim_hdu.data.shape[1] / 2) + int(real_hdu.data.shape[1] / 2)]
final_hdu.data[np.isnan(real_hdu.data)] = np.nan

w = WCS(final_hdu)
ra, dec = w.all_pix2world(final_hdu.data.shape[1] / 2, final_hdu.data.shape[0] / 2, 0)

rows = hii_table[hii_table['gal_name'] == 'NGC1512']
region_ids = rows['region_ID']
x_cens = rows['cen_x']
y_cens = rows['cen_y']

metallicities = []
metallicities_errs = []
gal_names = []

for i in range(int(np.nanmax(region_hdu.data)) + 1):
    idx = np.where(region_hdu.data == i)
    metallicities.append(np.nanmean(final_hdu.data[idx]))
    metallicities_errs.append(np.nanstd(final_hdu.data[idx]))
    gal_names.append('sim')

sim_hii_table = Table([gal_names, region_ids, x_cens, y_cens, metallicities, metallicities_errs],
                      names=('gal_name', 'region_ID', 'cen_x', 'cen_y', 'metallicity', 'metallicity_err'))

mcmc_file_name = os.path.join(metallicity_dir, 'eric_sim_gpr', 'sim_mcmc.pkl')

step_plot_name = os.path.join(plot_dir, 'eric_sim_gpr', 'sim_step')
corner_plot_name = step_plot_name.replace('step', 'corner')
fit_plot_name = step_plot_name.replace('step', 'fit')

labels = ['m', r'$R_0$', r'$\sigma$']

gpr_fitter = AstronomicalGPR('sim', final_hdu, dist, ra, dec, pa, inc, r25)
gpr_fitter.calculate_pixel_positions()
gpr_fitter.calculate_region_parameters(sim_hii_table, 'metallicity')

gpr_fitter.fit_radial_distribution(mcmc_file_name=mcmc_file_name, verbose=True)
if not os.path.exists(step_plot_name + '.png'):
    gpr_fitter.plot_step(step_plot_name, labels)
    gpr_fitter.plot_corner(corner_plot_name, labels)
    gpr_fitter.plot_radial_fit(fit_plot_name)

gpr_fitter.subtract_radial_fit()

gpr_fitter.calc_xy_to_fit_regions(region_hdu, step_val=step_val)

gpr_file = os.path.join(metallicity_dir, 'eric_sim_gpr', 'sim_gpr.pkl')

# Throw into GPR and create also create predictions

pred_file = gpr_file.replace('_gpr.pkl', '_predict.npy')

# Skip reading in/generating GPR files if running on laptop

if socket.gethostname() in ['astro-node4']:
    gpr_fitter.fit_gpr_regressor(gpr_file, verbose=True)

gpr_fitter.make_predictions(pred_file, gpr_fitter.xy_full, name='full', n_batch=100, verbose=True)
gpr_predictions_arrays = np.load(pred_file)

gpr_predictions = np.zeros_like(final_hdu.data)
gpr_predictions[gpr_predictions == 0] = np.nan
gpr_predictions[~np.isnan(final_hdu.data)] = gpr_predictions_arrays[0]

r0 = gpr_fitter.r_0
m = gpr_fitter.m

ii, ji = np.meshgrid(np.arange(final_hdu.data.shape[1]), np.arange(final_hdu.data.shape[0]))

i_cen, j_cen = final_hdu.data.shape[0] / 2, final_hdu.data.shape[1] / 2
pix_size = get_pix_size(final_hdu)

ii = (ii - i_cen) * pix_size
ji = (ji - j_cen) * pix_size

r = np.sqrt(ii ** 2 + ji ** 2)
metallicity_radial = m * (r / r25) + r0
residual_radial = final_hdu.data - metallicity_radial

metallicity_gpr = metallicity_radial + gpr_predictions
residual_gpr = residual_radial - gpr_predictions

comparison_plot_name = os.path.join(plot_dir, 'eric_sim_gpr', 'sim_scatter_comparison')
x_min, x_max = 8.925, 9

input_data_flat = final_hdu.data.flatten()
gpr_data_flat = metallicity_gpr.flatten()

nan_mask = np.where(~np.isnan(input_data_flat) & ~np.isnan(gpr_data_flat))
input_data_mask = input_data_flat[nan_mask]
gpr_data_mask = gpr_data_flat[nan_mask]

x_gpr, y_gpr, z_gpr = density_estimation(input_data_mask,
                                         gpr_data_mask)

radial_data_flat = metallicity_radial.flatten()

nan_mask = np.where(~np.isnan(input_data_flat) & ~np.isnan(radial_data_flat))
input_data_mask = input_data_flat[nan_mask]
radial_data_mask = radial_data_flat[nan_mask]

x_radial, y_radial, z_radial = density_estimation(input_data_mask,
                                                  radial_data_mask)

plt.figure(figsize=(6, 4))

plt.plot([-1,  -1], [-1, -1], c='b', label='GPR')
plt.plot([-1,  -1], [-1, -1], c='r', label='Radial')

plt.contour(x_gpr, y_gpr, z_gpr, levels=3, colors=['b'])
plt.contour(x_radial, y_radial, z_radial, levels=3, colors=['r'])

plt.plot([x_min, x_max], [x_min, x_max], c='k')

plt.xlim(x_min, x_max)
plt.ylim(x_min, x_max)

plt.xlabel('12 + log(O/H) [True]')
plt.ylabel('12 + log(O/H) [Predicted]')

plt.legend(loc='upper left', scatterpoints=1, frameon=False)

# plt.show()

plt.savefig(comparison_plot_name + '.png', bbox_inches='tight')
plt.savefig(comparison_plot_name + '.pdf', bbox_inches='tight')
plt.close()

vmin_fit, vmax_fit = np.nanpercentile(final_hdu.data, 2), np.nanpercentile(final_hdu.data, 98)
vmin_residual = np.nanpercentile(residual_radial, 16)

comparison_plot_name = os.path.join(plot_dir, 'eric_sim_gpr', 'sim_position_comparison')

plt.figure(figsize=(8, 4))

ax = plt.subplot(2, 3, 1)
plt.imshow(final_hdu.data, origin='lower', vmin=vmin_fit, vmax=vmax_fit)
plt.title('Original')
ax.xaxis.set_visible(False)
ax.tick_params(left=False, labelleft=False)
plt.setp(ax.spines.values(), visible=False)
plt.ylabel('Radial', rotation=0, labelpad=20)
# plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(metallicity_radial, origin='lower', vmin=vmin_fit, vmax=vmax_fit)
plt.title('Fit')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(residual_radial, origin='lower', cmap=cmocean.cm.balance, vmin=vmin_residual, vmax=-vmin_residual)
plt.title('Residual')
plt.axis('off')

ax = plt.subplot(2, 3, 4)
plt.imshow(final_hdu.data, origin='lower', vmin=vmin_fit, vmax=vmax_fit)
# plt.axis('off')
ax.xaxis.set_visible(False)
ax.tick_params(left=False, labelleft=False)
plt.setp(ax.spines.values(), visible=False)
plt.ylabel('GPR', rotation=0, labelpad=20)

plt.subplot(2, 3, 5)
plt.imshow(metallicity_gpr, origin='lower', vmin=vmin_fit, vmax=vmax_fit)
plt.axis('off')

plt.subplot(2, 3, 6)
plt.imshow(residual_gpr, origin='lower', cmap=cmocean.cm.balance, vmin=vmin_residual, vmax=-vmin_residual)
plt.axis('off')

plt.subplots_adjust(wspace=0, hspace=0)

# plt.show()

plt.savefig(comparison_plot_name + '.png', bbox_inches='tight')
plt.savefig(comparison_plot_name + '.pdf', bbox_inches='tight')
plt.close()

print('Complete!')
