# -*- coding: utf-8 -*-
"""
Check convergence of GPR, and also estimate scale length errors.

@author: Tom Williams
"""

import os
import pickle
import socket
import time

import astropy.units as u
import corner
import emcee
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.table import Table
from joblib import load
from tqdm import tqdm

from astronomical_gpr import AstronomicalGPR
from vars import top_dir, plot_dir, hii_mask_dir, muse_dir, muse_version, metallicity_dir, phangs_master_table, galaxies


def negative_exponential(x, amp, scale, offset):
    """Negative exponential with an offset.

    Negative exponential with a horizontal offset. Given by

    .. math::
        y = A \exp(-\lambda x) + c
    if :math:`y \geq 0`, or 0 otherwise.

    Args:
        x (float or numpy.ndarray): x-values to calculate y for
        amp (float): the amplitude, A.
        scale (float): The exponential scale, :math:`\lambda`.
        offset (float): The horizontal offset, c.

    Returns:
        float or numpy.ndarray: Calculated y-values

    """

    y = np.zeros_like(x)
    y[x >= 0] = amp * np.exp(-scale * x) + offset

    return y


def ln_like(theta, x, y, y_err):
    amp, scale, offset = theta
    model = negative_exponential(x, amp, scale, offset)

    return -0.5 * np.nansum((y - model) ** 2 / y_err ** 2)


def ln_prob(theta, x, y, y_err):
    lp = ln_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + ln_like(theta, x, y, y_err)


def ln_prior(theta):
    param_max = 50

    amp, scale, offset = theta
    if 0 < amp < param_max and 0 < scale < param_max and 0 < offset < param_max:
        return 0.0
    return -np.inf


matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['font.size'] = 14

os.chdir(top_dir)

seed = np.random.seed(420)

if not os.path.exists('metallicity'):
    os.makedirs('metallicity')

start = time.time()

null_hypothesis = False

n_jackknifes = 100
fractions = np.arange(0.1, 1.1, 0.1)
x_fit = np.linspace(0, 1.1, 100)

emission_to_use = 'HA6562_FLUX'

# Step value for filtering down the GPR points
step_val = 3

hii_table = Table.read(os.path.join(hii_mask_dir, 'Nebulae_Catalogue.fits'))

galaxies = ['IC5332', 'NGC1512']

scale_length_corr_file_name = os.path.join(metallicity_dir, muse_version, 'scale_lengths_corrected.fits')

dists = {}

scale_length_corr = {True: [], False: []}
scale_length_corr_err_up = {True: [], False: []}
scale_length_corr_err_down = {True: [], False: []}

for galaxy in galaxies:

    print('Jackknifing %s' % galaxy)

    scale_lengths_median_dict = {}
    scale_lengths_err_up_dict = {}
    scale_lengths_err_down_dict = {}

    y_fits_median = {}
    y_fits_lower = {}
    y_fits_upper = {}

    sigma_median = {}
    sigma_lower = {}
    sigma_upper = {}

    for null_hypothesis in [True, False]:

        scale_lengths_median = []
        scale_lengths_err_up = []
        scale_lengths_err_down = []

        galaxy_plot_dir = os.path.join(plot_dir, muse_version, galaxy + '_gpr', 'errors')
        if null_hypothesis:
            galaxy_plot_dir += '_null'
        galaxy_metallicity_dir = os.path.join(metallicity_dir, muse_version, galaxy + '_gpr')
        if null_hypothesis:
            galaxy_metallicity_dir += '_null'
        galaxy_err_dir = os.path.join(metallicity_dir, muse_version, galaxy + '_gpr_errors')
        if null_hypothesis:
            galaxy_err_dir += '_null'

        if not os.path.exists(galaxy_plot_dir):
            os.makedirs(galaxy_plot_dir)
        if not os.path.exists(os.path.join(galaxy_err_dir, 'jackknifes')):
            os.makedirs(os.path.join(galaxy_err_dir, 'jackknifes'))

        galaxy_edit = galaxy.lower()
        if galaxy == 'NGC628':
            galaxy_edit = 'ngc0628'

        row = phangs_master_table[phangs_master_table['name'] == galaxy_edit]
        dist, ra, dec, pa, inc, r25 = row['dist'][0], row['orient_ra'][0], row['orient_dec'][0], \
                                      row['orient_posang'][0], row['orient_incl'][0], row['size_r25'][0]

        dists[galaxy] = dist

        if socket.gethostname() in ['astro-node4']:
            # Read in the fitted GPR kernel as a start for the fit.

            gpr_file = os.path.join(galaxy_metallicity_dir, galaxy + '_regions_gpr.pkl')
            kernel = load(gpr_file).kernel_

            region_hdu_file_name = os.path.join(hii_mask_dir, 'spatial_masks', galaxy + '_HIIreg_mask.fits')
            region_hdu = fits.open(region_hdu_file_name)[0]

            mcmc_file_name = os.path.join(galaxy_metallicity_dir, galaxy + '_mcmc.pkl')

            hdu_file_name = os.path.join(muse_dir, muse_version, galaxy + '_MAPS.fits')
            data_hdu = fits.open(hdu_file_name)[emission_to_use]

        plot_name = os.path.join(galaxy_plot_dir, galaxy + '_jackknife_hist')

        fig, axes = plt.subplots(nrows=2, ncols=5, sharey='all', figsize=(8, 4))
        i = 0
        j = 0

        for fraction in fractions:

            output_file_name = os.path.join(galaxy_err_dir, 'jackknifes',
                                            galaxy + '_%.1f_jacknifes.txt' % fraction)

            if socket.gethostname() in ['astro-node4']:

                scale_lengths = []

                for jackknife_iteration in tqdm(range(n_jackknifes), desc='Fraction: %.1f' % fraction):

                    # We'll just save out the scale length, rather than the regressor, to save on time.

                    gpr_file = os.path.join(galaxy_err_dir, '%s_%.1f_%d_scale_length.npy'
                                            % (galaxy, fraction, jackknife_iteration))

                    if not os.path.exists(gpr_file):

                        gpr_fitter = AstronomicalGPR(galaxy, data_hdu, dist, ra, dec, pa, inc, r25)
                        gpr_fitter.calculate_pixel_positions()
                        gpr_fitter.calculate_region_parameters(hii_table, 'met_scal')

                        # If we're using the null hypothesis, randomize metallicities
                        if null_hypothesis:
                            gpr_fitter.shuffle_parameter()

                        gpr_fitter.remove_fraction_of_regions(fraction=fraction)
                        gpr_fitter.fit_radial_distribution(mcmc_file_name=mcmc_file_name)
                        gpr_fitter.subtract_radial_fit()
                        gpr_fitter.calc_xy_to_fit_regions(region_hdu, step_val=step_val)
                        gpr_fitter.perturb_pixels_to_fit()  # TODO: This won't work if we use GPR radial fitting.

                        # Throw into GPR. We'll only use one run from the initial parameters, assuming they're pretty
                        # good already
                        gpr_fitter.fit_gpr_regressor(kernel=kernel, n_restarts_optimizer=0)
                        scale_length = gpr_fitter.scale_length
                        np.save(gpr_file, scale_length)

                    else:

                        scale_length = float(np.load(gpr_file))

                    scale_lengths.append(scale_length)

                    # gpr_fitter.make_predictions(pred_file, gpr_fitter.xy_full, name='full', n_batch=100,
                    #                             overwrite_predictions=overwrite_predictions, verbose=True)
                    # gpr_fitter.make_predictions(pred_file_central, gpr_fitter.xy_regions, name='central',
                    #                             overwrite_predictions=overwrite_predictions, verbose=True)

                np.savetxt(output_file_name, scale_lengths)

            # At this point, produce the histogram for each fraction

            ax = axes[i, j]

            try:
                scale_lengths = np.loadtxt(output_file_name, unpack=True)

                scale_length_median = np.nanmedian(scale_lengths)
                scale_length_err_up = np.nanpercentile(scale_lengths, 84) - scale_length_median
                scale_length_err_down = scale_length_median - np.nanpercentile(scale_lengths, 16)

                ax.hist(scale_lengths, color='k', histtype='step')
                ax.axvline(scale_length_median, c='k', ls='-')
                ax.axvline(scale_length_median + scale_length_err_up, c='k', ls='--')
                ax.axvline(scale_length_median - scale_length_err_up, c='k', ls='--')

            except OSError:
                scale_length_median, scale_length_err_up, scale_length_err_down = np.nan, np.nan, np.nan

            ax.set_xlabel(r'$\sigma_l$ (kpc)')

            if j == 0:
                ax.set_ylabel(r'$N$')

            ax.set_title(r'$f_\mathrm{nebulae}$: %.1f' % fraction)

            j += 1
            if j > 4:
                j = 0
                i += 1

            scale_lengths_median.append(scale_length_median)
            scale_lengths_err_up.append(scale_length_err_up)
            scale_lengths_err_down.append(scale_length_err_down)

        scale_lengths_median = np.array(scale_lengths_median)
        scale_lengths_err_up = np.array(scale_lengths_err_up)
        scale_lengths_err_down = np.array(scale_lengths_err_down)

        scale_lengths_median_dict[null_hypothesis] = scale_lengths_median
        scale_lengths_err_up_dict[null_hypothesis] = scale_lengths_err_up
        scale_lengths_err_down_dict[null_hypothesis] = scale_lengths_err_down

        plt.tight_layout()

        plt.subplots_adjust(wspace=0)

        # plt.show()
        plt.savefig(plot_name + '.png', bbox_inches='tight')
        plt.savefig(plot_name + '.pdf', bbox_inches='tight')
        plt.close()

        if np.any(np.isnan(scale_lengths_median)):
            print('Jackknifing incomplete. Skipping fitting')
            scale_length_corr[null_hypothesis].append(scale_length)
            scale_length_corr_err_up[null_hypothesis].append(0)
            scale_length_corr_err_down[null_hypothesis].append(0)
            continue

        # Look at the convergence, and correct for incomplete coverage. Start by fitting exponential to the data.

        n_walkers = 500
        n_steps = 1000
        n_dim = 3
        n_plot_samples = 500

        mcmc_file_name = os.path.join(galaxy_err_dir, galaxy + '_mcmc.pkl')

        if not os.path.exists(mcmc_file_name):

            # Start with a quick burn-in run

            pos = np.array([scale_lengths_median[0], 1, scale_lengths_median[-1]]) + 1e-1 * np.random.randn(n_walkers,
                                                                                                            n_dim)

            sampler = emcee.EnsembleSampler(n_walkers, n_dim, ln_prob,
                                            args=(fractions, scale_lengths_median, scale_lengths_err_up))
            sampler.run_mcmc(pos, int(n_steps / 2), progress=True)
            flat_samples = sampler.get_chain(discard=int(n_steps / 4), flat=True)

            # Now for the proper run

            pos = np.median(flat_samples, axis=0) + 1e-1 * np.random.randn(n_walkers, n_dim)

            sampler = emcee.EnsembleSampler(n_walkers, n_dim, ln_prob,
                                            args=(fractions, scale_lengths_median, scale_lengths_err_up))
            sampler.run_mcmc(pos, n_steps, progress=True)

            with open(mcmc_file_name, 'wb') as f:
                pickle.dump(sampler, f)

        else:

            with open(mcmc_file_name, 'rb') as f:
                sampler = pickle.load(f)

        # Produce a step plot

        plot_name = os.path.join(galaxy_plot_dir, galaxy + '_step')

        labels = [r"$A$", r"$\lambda$", r"$c$"]
        samples = sampler.get_chain()
        fig, axes = plt.subplots(n_dim, figsize=(10, 7), sharex='all')
        for i in range(n_dim):
            ax = axes[i]
            ax.plot(samples[:, :, i], "k", alpha=0.3)
            ax.set_xlim(0, len(samples))
            ax.set_ylabel(labels[i])
            ax.yaxis.set_label_coords(-0.1, 0.5)

        axes[-1].set_xlabel("step number")

        # plt.show()
        plt.savefig(plot_name + '.png', bbox_inches='tight')
        plt.savefig(plot_name + '.pdf', bbox_inches='tight')
        plt.close()

        flat_samples = sampler.get_chain(discard=int(n_steps / 2), flat=True)
        samples_percentiles = np.nanpercentile(flat_samples, [16, 50, 84], axis=0)

        c_lower, c_median, c_upper = samples_percentiles[:, -1]

        sigma_median[null_hypothesis] = c_median
        sigma_upper[null_hypothesis] = c_upper
        sigma_lower[null_hypothesis] = c_lower

        scale_length_corr[null_hypothesis].append(c_median)
        scale_length_corr_err_up[null_hypothesis].append(c_upper - c_median)
        scale_length_corr_err_down[null_hypothesis].append(c_median - c_lower)

        # Produce corner plot

        plot_name = os.path.join(galaxy_plot_dir, galaxy + '_corner')

        fig = corner.corner(flat_samples, labels=labels, show_titles=True, quantiles=[0.16, 0.5, 0.84])

        # plt.show()
        plt.savefig(plot_name + '.png', bbox_inches='tight')
        plt.savefig(plot_name + '.pdf', bbox_inches='tight')
        plt.close()

        # Pull out the average fit curve from the samples

        y_fit = np.zeros([len(x_fit), n_plot_samples])

        plot_samples = np.random.randint(len(flat_samples), size=n_plot_samples)
        for i, plot_sample in enumerate(plot_samples):
            sample = flat_samples[plot_sample]

            y_fit[:, i] = negative_exponential(x_fit, *sample)

        y_fit_lower, y_fit_median, y_fit_upper = np.nanpercentile(y_fit, [16, 50, 84], axis=-1)

        y_fits_lower[null_hypothesis] = y_fit_lower
        y_fits_median[null_hypothesis] = y_fit_median
        y_fits_upper[null_hypothesis] = y_fit_upper

    # Calculate ratio of scale length to null hypothesis, with errors

    scale_length_ratio = sigma_median[False]/sigma_median[True]
    scale_length_err = [np.abs(sigma_lower[idx] - sigma_median[idx])/sigma_median[idx] for idx in [True, False]]
    scale_length_ratio_err = scale_length_ratio * np.sqrt(np.sum(scale_length_err) ** 2)

    plot_name = os.path.join(plot_dir, muse_version, galaxy + '_gpr', galaxy + '_convergence')

    plt.figure(figsize=(4, 3))
    ax = plt.subplot(1, 1, 1)

    plt.errorbar(fractions, scale_lengths_median_dict[False],
                 yerr=[scale_lengths_err_down_dict[False], scale_lengths_err_up_dict[False]],
                 marker='o', ls='none', c='r')
    plt.errorbar(fractions, scale_lengths_median_dict[True],
                 yerr=[scale_lengths_err_down_dict[True], scale_lengths_err_up_dict[True]],
                 marker='o', ls='none', c='k')

    plt.xlabel(r'$f_\mathrm{nebulae}$')
    plt.ylabel(r'$\sigma_l$ (kpc)')

    plt.xlim([0, 1.05])
    ylims = plt.ylim()
    plt.ylim([0, ylims[-1]])

    plt.fill_between(x_fit, y_fits_upper[False], y_fits_lower[False], color='r', alpha=0.5)
    plt.plot(x_fit, y_fits_median[False], c='r', ls='-', zorder=99)

    plt.fill_between(x_fit, y_fits_upper[True], y_fits_lower[True], color='k', alpha=0.5)
    plt.plot(x_fit, y_fits_median[True], c='k', ls='-', zorder=99)

    plt.axhline(sigma_median[False], color='r', ls='-')
    plt.axhspan(sigma_lower[False], sigma_upper[False], color='r', alpha=0.5)

    plt.axhline(sigma_median[True], color='k', ls='-')
    plt.axhspan(sigma_lower[True], sigma_upper[True], color='k', alpha=0.5)

    plt.text(0.95, 0.95,
             '$\sigma_l/\sigma_{l, \mathrm{null}} = %.2f \pm %.2f$' % (scale_length_ratio, scale_length_ratio_err),
             ha='right', va='top',
             transform=ax.transAxes)

    plt.tight_layout()

    # plt.show()
    plt.savefig(plot_name + '.png', bbox_inches='tight', dpi=200)
    plt.savefig(plot_name + '.pdf', bbox_inches='tight')
    plt.close()
no

# TODO: Sort this out

dists = np.array(dists) * u.Mpc
scale_length_corr = np.array(scale_length_corr) * u.kpc
scale_length_corr_err_up = np.array(scale_length_corr_err_up) * u.kpc
scale_length_corr_err_down = np.array(scale_length_corr_err_down) * u.kpc

# Finally, write out a table that includes useful information

scale_length_corr_table = Table([galaxies, dists,
                                 scale_length_corr, scale_length_corr_err_up, scale_length_corr_err_down],
                                names=['GALAXY', 'DIST', 'SIGMA_L', 'SIGMA_L_ERR_UP', 'SIGMA_L_ERR_DOWN'])

scale_length_corr_table.write(scale_length_corr_file_name, overwrite=True)

print('Complete! Took %.2fm' % ((time.time() - start) / 60))
