# -*- coding: utf-8 -*-
"""
Just some tests

@author: Tom Williams
"""

import os

from astropy.table import Table
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
import dill
from astropy.wcs import WCS
from astropy.convolution import convolve_fft, Gaussian2DKernel, Gaussian1DKernel
import matplotlib
import seaborn as sns
from scipy.stats import kendalltau, spearmanr

from vars import top_dir, muse_dir, muse_version, hii_mask_dir, metallicity_dir, phangs_master_table, plot_dir, \
    extinction_curve, hii_only, galaxies
from funcs import get_info_from_master_table


def calculate_two_point_correlation(x, y, z, tp_scales):
    avg_z = np.nanmean(z)
    sigma_z = np.nanmean((z - avg_z) ** 2)

    tp_corr = np.zeros_like(tp_scales)

    for scale_idx, scale in enumerate(tp_scales):

        averages = np.zeros_like(z)
        averages[averages == 0] = np.nan

        for i in range(len(z)):
            pos_idx = np.where(np.sqrt((x - x[i]) ** 2 + (y - y[i]) ** 2) <= scale)
            averages[i] = np.nanmean(z[i] * z[pos_idx])

        tp_corr[scale_idx] = np.nanmean((averages - avg_z ** 2) / sigma_z)

    return tp_corr


matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['font.size'] = 14

os.chdir(top_dir)

sigma_l = 10
sigma = 1

x = np.linspace(-50, 50, 25)
x_tp = np.linspace(0, 20, 100)

X, Y = np.meshgrid(x, x)

xy_to_fit = np.array([[X.flatten()[i], Y.flatten()[i]] for i in range(len(X.flatten()))])
r = np.sqrt(X ** 2 + Y ** 2)

matern = sigma ** 2 * (1 + np.sqrt(3) * r / sigma_l) * np.exp(- np.sqrt(3) * r / sigma_l)

kernel = Matern(length_scale=sigma_l, nu=1.5)
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=25)
gp_sample = gp.sample_y(xy_to_fit).reshape(X.shape)

matern_tp = calculate_two_point_correlation(X.flatten(), Y.flatten(), gp_sample.flatten(), x_tp)

plt.figure()
plt.subplot(2, 2, 1)
plt.imshow(matern, origin='lower', vmin=0, vmax=1)
plt.subplot(2, 2, 2)
plt.imshow(gp_sample, origin='lower')
plt.subplot(2,2,3)
plt.plot(x_tp, matern_tp)
plt.axvline(sigma_l, c='k', ls='--')
plt.show()

y = sigma ** 2 * (1 + np.sqrt(3) * x / sigma_l) * np.exp(- np.sqrt(3) * x / sigma_l)

x_two_point_corr = np.random.uniform(0, 10, 1000)
y_two_point_corr = sigma ** 2 * (1 + np.sqrt(3) * x_two_point_corr / sigma_l) * \
                   np.exp(- np.sqrt(3) * x_two_point_corr / sigma_l)
two_point_corr_y = two_point_correlation(x_two_point_corr, y_two_point_corr, x)
two_point_corr_y /= np.nanmax(two_point_corr_y)
two_point_corr_y[np.isnan(two_point_corr_y)] = 1

kernel = Matern(length_scale=sigma_l, length_scale_bounds=(0.1, 10),
                nu=1.5)
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=25)

y_gp_sample = gp.sample_y(x_two_point_corr.reshape(-1, 1)).flatten()
two_point_corr_y_gp = two_point_correlation(x_two_point_corr, y_gp_sample, x)
# two_point_corr_y_gp /= np.nanmax(two_point_corr_y_gp)
# two_point_corr_y_gp[np.isnan(two_point_corr_y_gp)] = 1

plt.figure()
plt.plot(x, y)
plt.plot(x, two_point_corr_y)
plt.plot(x, two_point_corr_y_gp)
plt.show()

hdu_1 = fits.open(
    '/Users/williams/Documents/phangs/metallicity/pix_maps/DR2.1/odonnell94_ext_curve/pg16_scal_hii_only/NGC3627_metallicity_pg16_scal.fits')[
    0]
hdu_2 = fits.open(
    '/Users/williams/Documents/phangs/metallicity/pix_maps/DR2.1/odonnell94_ext_curve/d16_hii_only/NGC3627_metallicity_d16.fits')[
    0]

data_1 = hdu_1.data
data_2 = hdu_2.data

plt.figure()
plt.hist(data_1.flatten())
plt.hist(data_2.flatten())

plt.show()

data_sub = data_1 - data_2
print(np.nanstd(data_sub.flatten()))
print(np.nanmedian(data_sub.flatten()))

# Look at relationship between number of HII regions and galaxy mass

nebulae_catalogue_file_name = os.path.join('nebulae_catalogue', 'v2', 'Nebulae_catalogue_v2.fits')
nebulae_catalogue = Table.read(nebulae_catalogue_file_name)

print(nebulae_catalogue.colnames)

masses = []
n_nebula = []

for galaxy in galaxies:
    n_nebulae = len(np.where(nebulae_catalogue['gal_name'] == galaxy)[0])
    print(galaxy, n_nebulae)
    n_nebula.append(n_nebulae)
    mass = get_info_from_master_table(phangs_master_table, galaxy.lower(), 'props_mstar')
    masses.append(mass)

print(spearmanr(np.log10(masses), n_nebula))

plt.figure()
plt.scatter(n_nebula, np.log10(masses))
plt.show()

no

metallicity_calib = 'pg16_scal'

plot_name = os.path.join(plot_dir, muse_version, '%s_ext_curve' % extinction_curve,
                         '%s_scale_length_r25' % metallicity_calib)

scale_length_file_name = os.path.join(metallicity_dir, muse_version, '%s_ext_curve' % extinction_curve,
                                      'scale_lengths_%s' % metallicity_calib)
if hii_only:
    scale_length_file_name += '_hii_only'
scale_length_file_name += '.fits'
scale_length_table = Table.read(scale_length_file_name)

idx = scale_length_table['SIGNIFICANT'] == 1

galaxies = scale_length_table['GALAXY'][idx]
scale_lengths = scale_length_table['SIGMA_L'][idx]
scale_lengths_err_up = scale_length_table['SIGMA_L_ERR_UP'][idx]
scale_lengths_err_down = scale_length_table['SIGMA_L_ERR_DOWN'][idx]

r25s = []

for galaxy in galaxies:
    dist, r25 = get_info_from_master_table(phangs_master_table, galaxy.lower(), ['dist', 'size_r25'])
    r25_phys_kpc = dist * 1e3 * r25 / 3600 * np.pi / 180
    r25s.append(r25_phys_kpc)

tau_draws = 1000
tau_bs = np.zeros(tau_draws)

for draw in range(tau_draws):
    scatter_scale = np.random.normal(loc=0, scale=scale_lengths_err_up)

    tau_bs[draw], _ = kendalltau(scale_lengths + scatter_scale,
                                 r25s)

tau = np.nanmedian(tau_bs)
tau_err = np.nanpercentile(tau_bs, 84) - tau

plt.figure(figsize=(6, 4))
ax = plt.subplot(111)
plt.errorbar(scale_lengths, r25s,
             xerr=[scale_lengths_err_down, scale_lengths_err_up],
             c='k', ls='none', marker='o')

plt.text(0.95, 0.95, r'$\tau = %.2f \pm %.2f$' % (tau, tau_err),
         ha='right', va='top',
         transform=ax.transAxes,
         bbox=dict(facecolor='white', edgecolor='none', alpha=0.75))

plt.xlabel(r'$\sigma_l$ (kpc)')
plt.ylabel(r'$r_{25}$ (kpc)')

plt.savefig(plot_name + '.png', bbox_inches='tight')
plt.savefig(plot_name + '.pdf', bbox_inches='tight')

plt.close()
no

galaxy_pa = phangs_master_table['orient_posang']
bar_pa = phangs_master_table['morph_bar_pa']

# bar_pa -= 90

plt.figure()
plt.hist(np.abs(np.cos(np.radians(bar_pa - galaxy_pa))))
plt.xlabel(r'cos($\Delta$PA)')
plt.ylabel('N')
plt.show()

metallicity_calib = 'pg16_scal'

scale_length_file_name = os.path.join(metallicity_dir, 'DR2.0', 'scale_lengths_%s.fits' % metallicity_calib)
scale_length_table_dr2 = Table.read(scale_length_file_name)

scale_length_file_name = os.path.join(metallicity_dir, 'DR2.1', 'scale_lengths_%s.fits' % metallicity_calib)
scale_length_table_dr2_1 = Table.read(scale_length_file_name)

idx = np.where((scale_length_table_dr2['SIGNIFICANT'] == 1) & (scale_length_table_dr2_1['SIGNIFICANT'] == 1))

lim = [0, 50]

plt.figure(figsize=(5, 4))

plt.errorbar(scale_length_table_dr2['SIGMA_L'][idx], scale_length_table_dr2_1['SIGMA_L'][idx],
             xerr=[scale_length_table_dr2['SIGMA_L_ERR_DOWN'][idx], scale_length_table_dr2['SIGMA_L_ERR_UP'][idx]],
             yerr=[scale_length_table_dr2_1['SIGMA_L_ERR_DOWN'][idx], scale_length_table_dr2_1['SIGMA_L_ERR_UP'][idx]],
             c='k', marker='o', ls='none')

plt.plot(lim, lim, c='k', ls='--')

plt.xlim(lim)
plt.ylim(lim)

plt.xlabel(r'$\sigma_l$ (DR2.0; kpc)')
plt.ylabel(r'$\sigma_l$ (DR2.1; kpc)')

plt.grid()

plt.tight_layout()

plt.show()

# Let's say we have a little gaussian, stddev 1, which will be our "error"

start, stop, step = -5, 5, 1000

pix_size = (stop - start) / step

x = np.linspace(start, stop, step)
y = 1 / np.sqrt(2 * np.pi) * np.exp(-0.5 * x ** 2)

kernel = Gaussian1DKernel(2 / pix_size)

# Case one: error with kernel
y_1 = convolve_fft(y, kernel)

# Case two: variance with kernel ** 2
kernel_2 = kernel.array ** 2 / np.sum(kernel.array ** 2)
y_2 = np.sqrt(convolve_fft(y ** 2, kernel_2))

print(y_1 / y_2)

plt.figure()
plt.plot(x, y, c='r')
plt.plot(x, y_1, c='g')
plt.plot(x, y_2, c='b')
plt.show()

# OK, let's look at this convolution nonsense. We have an original grid with some noise
size = 100

original_noise = np.random.normal(scale=5, size=(size, size))

# Say each pixel is 0.2arcsec, convolve to 0.798arcsec resolution
pix_size = 0.2
fwhm = 0.798 / pix_size
sigma = fwhm / 2.355
orig_kernel = Gaussian2DKernel(sigma)
original_noise = convolve_fft(original_noise, orig_kernel)
original_sigma = np.nanstd(original_noise)
estimated_original_sigma = original_sigma * 2 * np.sqrt(np.pi) * sigma

# Now, convolve this with a Gaussian of 0.35 (i.e. convolving up to 0.87arcsec res)
fwhm = 0.35 / pix_size
sigma = fwhm / 2.355
kernel = Gaussian2DKernel(sigma)
conv_noise = convolve_fft(original_noise, kernel, normalize_kernel=False)
conv_sigma = np.nanstd(conv_noise)

# OK so we have the sigma, let's try the MPDAF way
original_var = estimated_original_sigma ** 2
orig_variance_map = np.ones([size, size]) * original_var
fwhm = 0.87 / pix_size
sigma = fwhm / 2.355
kernel = Gaussian2DKernel(sigma)

conv_variance_map = convolve_fft(orig_variance_map, kernel.array ** 2,
                                 normalize_kernel=False)
conv_variance = np.nanmax(conv_variance_map)

x = np.sqrt(conv_variance) / original_sigma

print(x)
print(conv_sigma / origin)
print(conv_sigma / original_sigma / x)
no

# And this agrees with what was going on, but this seems totally ludicrous. Let's MCMC it

# n_draws = 2000
# realisations = np.zeros([size, size, n_draws])
#
# for i in range(n_draws):
#     realisation = np.random.normal(scale=5, size=(size, size))
#     realisation_conv = convolve_fft(realisation, orig_kernel)
#     realisations[:, :, i] = convolve_fft(realisation_conv, kernel)
#
#     # realisation = np.random.normal(scale=original_sigma, size=(size, size))
#     # realisations[:, :, i] = convolve_fft(realisation, kernel)
#
# plt.figure()
# plt.hist(realisations[50, 50, :])
#
# mcmc_sigma = np.nanstd(realisations, axis=-1)
# print(np.nanmedian(mcmc_sigma) / original_sigma)
# plt.figure()
# plt.imshow(mcmc_sigma, origin='lower')
# # plt.show()
#
# plt.figure()
# plt.subplot(1, 2, 1)
# plt.imshow(orig_variance_map, origin='lower')
# plt.subplot(1, 2, 2)
# plt.imshow(conv_variance_map, origin='lower')
# # plt.show()

plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(original_noise, origin='lower')
plt.title(r'$\sigma=%.3f$' % original_sigma)
plt.subplot(1, 2, 2)
plt.imshow(conv_noise, origin='lower')
plt.title(r'$\sigma=%.3f$' % conv_sigma)
plt.show()

x = np.linspace(-10, 10, 250)
x = x.reshape(-1, 1)

kernel = Matern(length_scale=3, length_scale_bounds=(0.1, 10),
                nu=1.5)  # + ConstantKernel(0) + WhiteKernel(0.1, noise_level_bounds=(1e-9, 10))
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=25)

y = gp.sample_y(x)  # + np.random.normal(scale=0.1, size=x.shape)

# Fit this

gp.fit(x, y)
y_pred = gp.sample_y(x).flatten()
print(gp.kernel_)

# https://arxiv.org/pdf/1408.5810.pdf, p4
det_cov = np.linalg.det(gp.kernel_(gp.X_train_))

log_likelihood = gp.log_marginal_likelihood()
print(log_likelihood)
no

plt.figure()
plt.plot(x, y, 'rx')
plt.plot(x, y_pred, 'bo')
plt.show()
