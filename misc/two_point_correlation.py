# -*- coding: utf-8 -*-
"""
Calculate and compare two-point correlation functions

@author: Tom Williams
"""

import os

import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
import dill
from tqdm import tqdm
import matplotlib.pyplot as plt
import emcee
import corner
import matplotlib
import pickle

from funcs import add_master_table_info
from vars import top_dir, metallicity_dir, plot_dir, galaxies, muse_version, phangs_master_table, hii_mask_dir, \
    gpr_version
from astronomical_gpr import project


def calculate_two_point_correlation(x, y, z, two_point_scales):
    avg_z = np.nanmean(z)
    sigma_z = np.nanmean((z - avg_z) ** 2)

    two_point_corr = np.zeros_like(two_point_scales)

    for scale_idx, scale in enumerate(two_point_scales):

        averages = np.zeros_like(z)
        averages[averages == 0] = np.nan

        for i in range(len(z)):
            pos_idx = np.where(np.sqrt((x - x[i]) ** 2 + (y - y[i]) ** 2) <= scale)
            averages[i] = np.nanmean(z[i] * z[pos_idx])

        two_point_corr[scale_idx] = np.nanmean((averages - avg_z ** 2) / sigma_z)

    return two_point_corr


def half_cauchy(x, gamma_cauchy):
    y = 1 / (np.pi * gamma_cauchy * (1 + (x / gamma_cauchy) ** 2))

    # Normalise to be 1 at 0
    y /= y[0]

    return y


def students_t(x, nu):
    y = (1 + x ** 2 / nu) ** (- (nu + 1) / 2)

    # Normalise to be 1 at 0
    y /= y[0]

    return y


def gaussian(x, sigma):
    y = np.exp(-0.5 * (x / sigma) ** 2)

    # Normalise to be 1 at 0
    y /= y[0]

    return y


def logistic(x, s):
    y = np.cosh(x / (2 * s)) ** -2

    # Normalise to be 1 at 0
    y /= y[0]

    return y


def matern_1_2(x, sigma, rho):
    y = sigma ** 2 * np.exp(- x / rho)

    # Normalise to be 1 at 0
    y /= y[0]

    return y


def matern_3_2(x, sigma, rho):
    y = sigma ** 2 * (1 + (np.sqrt(3) * x) / rho) * np.exp(- (np.sqrt(3) * x) / rho)

    # Normalise to be 1 at 0
    y /= y[0]

    return y

def exp_square(x, sigma):
    y = np.exp(- (x / sigma) ** 2)

    # Normalise to be 1 at 0
    y /= y[0]

    return y


def ln_like(theta, x, y, y_err, method):
    if method == 'cauchy':
        model = half_cauchy
    elif method == 'stud_t':
        model = students_t
    elif method == 'gauss':
        model = gaussian
    elif method == 'logistic':
        model = logistic
    elif method == 'matern_1_2':
        model = matern_1_2
    elif method == 'matern_3_2':
        model = matern_3_2
    elif method == 'exp_square':
        model = exp_square
    else:
        raise Warning('method %s not understood' % method)

    y_model = model(x, *theta)

    return -0.5 * np.nansum((y - y_model) ** 2 / y_err ** 2)


def ln_prob(theta, x, y, y_err, method='cauchy'):
    lp = ln_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + ln_like(theta, x, y, y_err, method)


def ln_prior(theta):
    param_max = 50
    if np.all([theta >= 0, theta <= param_max]):
        return 0.0
    return -np.inf


matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['font.size'] = 14

os.chdir(top_dir)

# TODO: Null hypothesis, bootstrapping in errors.

correlation_dir = os.path.join(metallicity_dir, 'correlations', muse_version)
nebula_table = Table.read(hii_mask_dir + 'Nebulae_Catalogue.fits')

if not os.path.exists(correlation_dir):
    os.makedirs(correlation_dir)

# Calculate the two-point correlation function for these galaxies (and optionally, the null hypothesis) for a number of
# scales TODO: rerun with the Matern function

scales = np.linspace(0, 2, 11, endpoint=True)
n_bootstraps = 100
use_hii_regions = False
use_null_hypotheses = [False, True]
overwrite_correlations = False
curve_type = 'exp_square'

arcsec_to_kpc = 1e3 * np.pi / 180 / 3600

# Distances assumed in Kathryn's paper

k20_dists = {'NGC628': 9.77, 'NGC1087': 14.4, 'NGC1672': 11.9, 'NGC2835': 10.1, 'NGC3627': 10.6, 'NGC4254': 16.8,
             'NGC4535': 15.8, 'IC5332': 9.95}

galaxy_info = {}

for galaxy in galaxies:

    print('Beginning %s' % galaxy)

    galaxy_correlation_dir = os.path.join(correlation_dir, galaxy)
    if not os.path.exists(galaxy_correlation_dir):
        os.makedirs(galaxy_correlation_dir)

    galaxy_plot_dir = os.path.join(plot_dir, 'correlations', galaxy)
    if not os.path.exists(galaxy_plot_dir):
        os.makedirs(galaxy_plot_dir)

    galaxy_info[galaxy] = {}

    cols_to_include = ['dist', 'orient_ra', 'orient_dec', 'orient_posang', 'orient_incl', 'size_r25']

    gal_edit = galaxy.lower()
    if galaxy == 'NGC628':
        gal_edit = 'ngc0628'

    galaxy_info = add_master_table_info(phangs_master_table, galaxy_info, gal_edit, cols_to_include,
                                        original_galaxy_name=galaxy)

    dist = galaxy_info[galaxy]['dist']
    dist_err = galaxy_info[galaxy]['dist_err']

    dist_err_up = 10 ** (np.log10(dist) + dist_err) - dist
    dist_err_down = dist - 10 ** (np.log10(dist) - dist_err)

    # Include the correlation scales from K20

    if galaxy in k20_dists.keys():
        k20_dist = k20_dists[galaxy]

        correlation_scale = {'NGC628': 210,
                             'NGC1087': 370,
                             'NGC1672': 310,
                             'NGC2835': 280,
                             'NGC3627': 420,
                             'NGC4254': 350,
                             'NGC4535': 360,
                             'IC5332': 310}[galaxy]
        correlation_scale_err = {'NGC628': 8,
                                 'NGC1087': 7,
                                 'NGC1672': 6,
                                 'NGC2835': 2,
                                 'NGC3627': 5,
                                 'NGC4254': 8,
                                 'NGC4535': 2,
                                 'IC5332': 8}[galaxy]

        # Account for any distance differences between correlation scale lengths

        correlation_scale *= dist / k20_dist

        correlation_scale_err_up = correlation_scale * np.sqrt((dist_err_up / dist) ** 2 +
                                                               (correlation_scale_err / correlation_scale) ** 2)
        correlation_scale_err_down = correlation_scale * np.sqrt((dist_err_down / dist) ** 2 +
                                                                 (correlation_scale_err / correlation_scale) ** 2)

        galaxy_info[galaxy]['k20_50percent'] = correlation_scale
        galaxy_info[galaxy]['k20_50percent_err_up'] = correlation_scale_err_up
        galaxy_info[galaxy]['k20_50percent_err_down'] = correlation_scale_err_down

        # Here are the 30% scales but I don't think we'll need them

        # correlation_scale = {'NGC628': 320,
        #                      'NGC1087': 580,
        #                      'NGC1672': 460,
        #                      'NGC2835': 440,
        #                      'NGC3627': 1010,
        #                      'NGC4254': 620,
        #                      'NGC4535': 620,
        #                      'IC5332': 550}[galaxy]

    correlation_dict = {galaxy: {}}

    x_fit = np.linspace(0, 2.1, 100)

    for use_null_hypothesis in use_null_hypotheses:

        corr_file_name = os.path.join(galaxy_correlation_dir, galaxy + '_correlations')
        if use_hii_regions:
            corr_file_name += '_hii_reg'
        if use_null_hypothesis:
            corr_file_name += '_null'
        corr_file_name += '.txt'

        if not os.path.exists(corr_file_name) or overwrite_correlations:

            # Calculate two-point correlation function for our data

            if use_null_hypothesis:
                hdu_file_name = os.path.join(metallicity_dir, 'metallicity_gpr_releases', gpr_version,
                                             'null_hypothesis', galaxy + '_metallicity_gpr.fits')
                hdu_err_file_name = os.path.join(metallicity_dir, 'metallicity_gpr_releases', gpr_version,
                                                 'null_hypothesis', galaxy + '_metallicity_gpr_err.fits')
            else:
                hdu_file_name = os.path.join(metallicity_dir, 'metallicity_gpr_releases', gpr_version,
                                             galaxy + '_metallicity_gpr.fits')
                hdu_err_file_name = os.path.join(metallicity_dir, 'metallicity_gpr_releases', gpr_version,
                                                 galaxy + '_metallicity_gpr_err.fits')

            hdu = fits.open(hdu_file_name)[0]
            err = fits.open(hdu_err_file_name)[0].data
            w = WCS(hdu_file_name)
            pix_size = np.abs(hdu.header['CD1_1']) * 3600

            # Calculate a deprojected physical distance.

            x_cen, y_cen = w.all_world2pix(galaxy_info[galaxy]['orient_ra'], galaxy_info[galaxy]['orient_dec'], 1)

            # For testing, use HII regions instead of all the pixels

            if use_hii_regions:

                hii_rows = nebula_table[nebula_table['gal_name'] == galaxy]
                hii_rows = hii_rows[~np.isnan(hii_rows['met_scal'])]

                xi, yi = np.array(hii_rows['cen_ra']), np.array(hii_rows['cen_dec'])
                metallicities = np.array(hii_rows['met_scal'])

                xi -= ra
                yi -= dec

                xi *= arcsec_to_kpc * 3600 * dist
                yi *= arcsec_to_kpc * 3600 * dist

                nan_mask = []

            else:

                xi, yi = np.meshgrid((np.arange(hdu.data.shape[1]) - x_cen),
                                     (np.arange(hdu.data.shape[0]) - y_cen))

                xi *= pix_size * arcsec_to_kpc * dist
                yi *= pix_size * arcsec_to_kpc * dist

                metallicities = hdu.data
                nan_mask = np.where(np.isnan(hdu.data))

            # Convert these positions to physical positions (kpc), accounting for inclination and rotation

            r25 = galaxy_info[galaxy]['size_r25'] * arcsec_to_kpc * dist

            x_rot, y_rot = project(xi, yi, galaxy_info[galaxy]['orient_posang'], galaxy_info[galaxy]['orient_incl'])

            r_full = np.sqrt(x_rot ** 2 + y_rot ** 2)
            r_full[nan_mask] = np.nan

            # Read in radial gradient

            mcmc_file_name = os.path.join(metallicity_dir, galaxy + '_gpr')
            if use_null_hypothesis:
                mcmc_file_name += '_null'
            mcmc_file_name = os.path.join(mcmc_file_name, galaxy + '_mcmc.pkl')

            with open(mcmc_file_name, 'rb') as sampler_f:
                sampler = dill.load(sampler_f)

            n_walkers, n_steps = 500, 500
            flat_samples = sampler.get_chain(discard=int(n_steps / 2), flat=True)
            m_median, r_0_median, int_scatter_median = np.nanmedian(flat_samples, axis=0)

            data_radial_subtract = metallicities - (m_median * r_full / r25 + r_0_median)

            non_nan_mask = np.where(~np.isnan(data_radial_subtract))

            if use_hii_regions:
                step = 1
            else:
                step = 50

            data_flat = data_radial_subtract[non_nan_mask].flatten()[::step]
            if use_hii_regions:
                data_err_flat = 0  # Obviously not right but hey
            else:
                data_err_flat = err[non_nan_mask].flatten()[::step]
            r_flat = r_full[non_nan_mask].flatten()[::step]
            x_flat = x_rot[non_nan_mask].flatten()[::step]
            y_flat = y_rot[non_nan_mask].flatten()[::step]

            correlations = np.zeros([len(scales), n_bootstraps])

            for i in tqdm(range(n_bootstraps)):
                data_bootstrap = data_flat + np.random.normal(loc=0, scale=data_err_flat, size=len(data_err_flat))
                correlations[:, i] = calculate_two_point_correlation(x_flat, y_flat, data_bootstrap, scales)

            np.savetxt(corr_file_name,
                       np.c_[scales, correlations])

        else:

            correlations = np.loadtxt(corr_file_name)
            correlations = correlations[:, 1:]

        # From these, calculate an error bar that we'll put into fitting

        correlations_mcmc = np.nanpercentile(correlations, [16, 50, 84], axis=-1)

        corr_med = correlations_mcmc[1, :]
        corr_err = correlations_mcmc[-1, :] - corr_med

        # Fit a half-Cauchy distribution to the data

        mcmc_file_name = os.path.join(galaxy_correlation_dir, galaxy + '_two_point_' + curve_type)
        if use_null_hypothesis:
            mcmc_file_name += '_null'
        mcmc_file_name += '_mcmc.pkl'

        n_walkers = 500
        n_steps = 1000
        n_plot_samples = 500

        if curve_type in ['cauchy', 'stud_t', 'gauss', 'logistic', 'exp_square']:
            n_dim = 1
        elif curve_type in ['matern_1_2', 'matern_3_2']:
            n_dim = 2
        else:
            raise Warning('ndims unknown for curve type %s' % curve_type)

        if not os.path.exists(mcmc_file_name):

            # Do a quick MCMC run

            pos = np.ones(n_dim) + 1e-1 * np.random.randn(n_walkers, n_dim)

            sampler = emcee.EnsembleSampler(n_walkers, n_dim, ln_prob,
                                            args=(scales, corr_med, corr_err, curve_type))
            sampler.run_mcmc(pos, n_steps, progress=True)

            with open(mcmc_file_name, 'wb') as f:
                pickle.dump(sampler, f)

        else:

            with open(mcmc_file_name, 'rb') as f:
                sampler = pickle.load(f)

        # Produce a step plot

        plot_name = os.path.join(galaxy_plot_dir, galaxy + '_step_' + curve_type)
        if use_null_hypothesis:
            plot_name += '_null'

        if curve_type == 'cauchy':
            labels = [r"$\gamma$"]
        elif curve_type == 'stud_t':
            labels = [r'$\nu$']
        elif curve_type == 'gauss':
            labels = [r'$\sigma$']
        elif curve_type == 'logistic':
            labels = [r'$s$']
        elif curve_type== 'exp_square':
            labels = [r'$\lambda$']
        elif curve_type in ['matern_1_2', 'matern_3_2']:
            labels = [r'$\sigma$', r'$\rho$']
        else:
            raise Warning('labels unknown for curve type %s' % curve_type)

        samples = sampler.get_chain()
        fig, axes = plt.subplots(n_dim, figsize=(4, 3), sharex='all')
        for i in range(n_dim):

            if n_dim == 1:
                ax = axes
            else:
                ax = axes[i]
            ax.plot(samples[:, :, i], "k", alpha=0.3)
            ax.set_xlim(0, len(samples))
            ax.set_ylabel(labels[i])
            ax.yaxis.set_label_coords(-0.1, 0.5)

        ax.set_xlabel("step number")

        # plt.show()
        plt.savefig(plot_name + '.png', bbox_inches='tight')
        plt.savefig(plot_name + '.pdf', bbox_inches='tight')
        plt.close()

        flat_samples = sampler.get_chain(discard=int(n_steps / 2), flat=True)
        samples_percentiles = np.nanpercentile(flat_samples, [16, 50, 84], axis=0)

        # Produce corner plot

        plot_name = os.path.join(galaxy_plot_dir, galaxy + '_corner_' + curve_type)

        if use_null_hypothesis:
            plot_name += '_null'

        fig = corner.corner(flat_samples, labels=labels, show_titles=True, quantiles=[0.16, 0.5, 0.84])

        # plt.show()
        plt.savefig(plot_name + '.png', bbox_inches='tight')
        plt.savefig(plot_name + '.pdf', bbox_inches='tight')
        plt.close()

        # Pull out the average fit curve from the samples

        if curve_type == 'cauchy':
            curve = half_cauchy
        elif curve_type == 'stud_t':
            curve = students_t
        elif curve_type == 'gauss':
            curve = gaussian
        elif curve_type == 'logistic':
            curve = logistic
        elif curve_type == 'exp_square':
            curve = exp_square
        elif curve_type == 'matern_1_2':
            curve = matern_1_2
        elif curve_type == 'matern_3_2':
            curve = matern_3_2
        else:
            raise Warning('curve type %s unknown' % curve_type)

        y_fit = np.zeros([len(x_fit), n_plot_samples])

        plot_samples = np.random.randint(len(flat_samples), size=n_plot_samples)
        for i, plot_sample in enumerate(plot_samples):
            sample = flat_samples[plot_sample]

            y_fit[:, i] = curve(x_fit, *sample)

        y_fit_lower, y_fit_median, y_fit_upper = np.nanpercentile(y_fit, [16, 50, 84], axis=-1)

        if use_null_hypothesis:
            key_name = 'null'
        else:
            key_name = 'data'

        if curve_type in ['cauchy', 'stud_t', 'gauss', 'logistic', 'exp_square']:
            width = samples_percentiles[1][0]
        elif curve_type in ['matern_1_2', 'matern_3_2']:
            width = samples_percentiles[1][1]
        else:
            raise Warning('No idea where to find the distribution width for curve type %s' % curve_type)

        correlation_dict[galaxy][key_name] = {'med': corr_med, 'err': corr_err,
                                              'fit': y_fit_median, 'fit_lower': y_fit_lower, 'fit_upper': y_fit_upper,
                                              'width': width}

    y_data = correlation_dict[galaxy]['data']['med']
    y_data_err = correlation_dict[galaxy]['data']['err']
    y_data_fit = correlation_dict[galaxy]['data']['fit']
    y_data_width = correlation_dict[galaxy]['data']['width']

    y_null = correlation_dict[galaxy]['null']['med']
    y_null_err = correlation_dict[galaxy]['null']['err']
    y_null_fit = correlation_dict[galaxy]['null']['fit']
    y_null_width = correlation_dict[galaxy]['null']['width']

    plot_name = os.path.join(galaxy_plot_dir, galaxy + '_fit_' + curve_type)

    plt.figure(figsize=(4, 3))
    ax = plt.subplot(1, 1, 1)

    plt.errorbar(scales, y_null, yerr=y_null_err, c='k', ls='none', marker='o', label='Null')
    plt.plot(x_fit, y_null_fit, c='k', alpha=0.75)

    plt.errorbar(scales, y_data, yerr=y_data_err, c='r', ls='none', marker='o', label='Data')
    plt.plot(x_fit, y_data_fit, c='r', alpha=0.75)

    plt.xlim([0, 2])

    plt.legend(loc='upper right', frameon=False)

    plt.text(0.95, 0.25, r'$\sigma_\mathrm{data}/\sigma_\mathrm{null} = %.2f$' % (y_data_width / y_null_width),
             ha='right', va='center',
             transform=ax.transAxes)

    # plt.show()
    plt.savefig(plot_name + '.png', bbox_inches='tight')
    plt.savefig(plot_name + '.pdf', bbox_inches='tight')
    plt.close()

print('Complete!')
