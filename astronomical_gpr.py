# -*- coding: utf-8 -*-
import multiprocessing as mp
import os
import time
import warnings
from multiprocessing import Pool

import corner
import dill
import emcee
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from joblib import dump, load
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.utils import shuffle
from tqdm import tqdm

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['font.size'] = 14

PIXEL_SCALE_NAMES = ['XPIXSIZE', 'CDELT1', 'CD1_1']
N_THREADS = min([8, mp.cpu_count()])


def ln_prior(theta):
    """Prior function for straight line fitting."""
    m, c, scatter = theta
    if -5.0 < m < 5 and 0.0 < c < 10.0 and scatter >= 0:
        return 0.0
    return -np.inf


def ln_prob(theta, x, y, y_err):
    """Generic function for emcee fitting."""
    lp = ln_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + ln_like(theta, x, y, y_err)


def ln_like(theta, x, y, y_err):
    """Likelihood function for emcee straight line fitting, including intrinsic scatter."""
    m, c, scatter = theta
    model = m * x + c
    total_err2 = y_err ** 2 + scatter ** 2

    return -0.5 * np.sum((y - model) ** 2 / total_err2 + np.log(2 * np.pi * total_err2))


def get_pix_size(hdu):
    """Get pixel scale from header.

    Checks HDU header for pixel scale keywords, and returns a pixel scale in arcsec. If no suitable keyword is found,
    will throw up an error.

    Args:
        hdu (astropy.fits.PrimaryHDU): HDU to get pixel scale for.

    Returns:
        pix_scale (float): Pixel scale in arcsec.

    Raises:
        Warning: If no suitable pixel scale keyword is found in header.

    """
    for pixel_keyword in PIXEL_SCALE_NAMES:
        try:
            try:
                pix_scale = np.abs(float(hdu.header[pixel_keyword]))
            except ValueError:
                continue
            if pixel_keyword in ['CDELT1', 'CD1_1']:
                pix_scale *= 3600
            return pix_scale
        except KeyError:
            pass

    raise Warning('No pixel scale found')


def project(x, y, pa, inc):
    """General rotation/projection routine.

    Given coordinates (x, y), will rotate and project given position angle (counter-clockwise from N), and
    inclination. Assumes centre is at (0, 0).

    Args:
        x (float or numpy.ndarray): x-coordinate(s)
        y (float or numpy.ndarray): y-coordinates(s)
        pa (float): Position angle (degrees)
        inc (float): Inclination (degrees)

    Returns:
        x_proj, y_proj: The rotated, projected (x, y) coordinates.

    """

    angle = np.radians(pa)
    cos_a, sin_a = np.cos(angle), np.sin(angle)

    x_proj = x * cos_a + y * sin_a
    y_proj = - x * sin_a + y * cos_a

    # Account for inclination

    x_proj /= np.cos(np.radians(inc))

    return x_proj, y_proj


class AstronomicalGPR:
    """Apply Gaussian Process Regression (GPR) to astronomical data.

    Can be used to interpolate, using GPR, of stochastically sampled parameters to a smooth map. Usually, these
    parameters will also have an underlying gradient, so also allows to subtract this off before fitting, then add it
    back in to create a final map.

    The basic flow goes:
        * Set up the pixel grid to eventually interpolate on to (set by the input `data_hdu`).
        * Pull the parameter in question out for defined regions.
        * OPTIONAL: Simulate some underlying GPR (only for testing).
        * OPTIONAL: Fit and subtract radial distribution of the parameters.
        * OPTIONAL: Produce plots (fit, corner, step) for the radial fitting).
        * OPTIONAL: Account for region size when performing the fit by putting the parameter into input region masks.
        * Fit the GPR.
        * Make predictions based on the fitted GPR.
        * Celebrate! You've just made a map that is ~significantly better than a radial gradient alone (YMMV).

    Args:
        galaxy (str): Name of the galaxy. This will be used to query the table to pull out the parameter later.
        data_hdu (astropy.fits.PrimaryHDU): HDU that defines the pixel grid that we will eventually interpolate the GPR
            onto.
        dist (float): Distance to galaxy (Mpc).
        ra (float): Central RA of galaxy (degrees).
        dec (float): Central Dec of galaxy (degrees).
        pa (float): Position angle of galaxy (counter-clockwise from N; degrees).
        inc (float): Galaxy inclination (degrees).
        r25 (float, optional): If provided, will produce plots and do fits in terms of r25. Defaults to None, at which
            point fits will instead be done in terms of kpc. This can be useful to normalise multiple galaxies (arcsec).

    Attributes:

        Data properties:

        data_hdu (astropy.fits.PrimaryHDU): HDU that defines the pixel grid that we will eventually interpolate the GPR
            onto.
        nan_mask (numpy.ndarray): Mask defining NaNs in `data_hdu`.
        pix_size (float): pixel size for `data_hdu` (arcsec).
        wcs: WCS for `data_hdu`.

        Galaxy properties:

        dist (float): Distance to galaxy (Mpc).
        ra (float): Central RA of galaxy (degrees).
        dec (float): Central Dec of galaxy (degrees).
        pa (float): Position angle of galaxy (counter-clockwise from N; degrees).
        inc (float): Galaxy inclination (degrees).
        r25 (float, optional): If provided, will produce plots and do fits in terms of r25. Defaults to None, at which
            point fits will instead be done in terms of kpc. This can be useful to normalise multiple galaxies (arcsec).

        Radial fitting properties:

        n_dim (int): Dimensionality for radial fitting.
        n_walkers (int): Number of MCMC walkers.
        n_steps (int): Number of steps per walker.
        sampler: The emcee sampler.
        m (float): Median radial gradient of fit.
        r_0 (float): Central amplitude of radial fit.
        m_err (float): Gradient error.
        r_0_err (float): Central amplitude error.
        intrinsic_scatter (float): Modelled intrinsic scatter of the data.
        radial_goodness_of_fit (float): R^2 value for the radial fit.

        HDU properties:

        x_full (np.ndarray): Array of physical x positions in `data_hdu`.
        y_full (np.ndarray): Array of physical y-positions in `data_hdu`.
        xy_full (np.ndarray): Concatenated, flattened, NaN removed `x_full` and `y_full` for scikit-learn.
        r_full (np.ndarray): Radial distance array for every pixel in `data_hdu`.

        Parameter properties:

        region_numbers (np.ndarray): If using regions, these will be mapped to a spatial mask via these numbers.
        x_region_pix (np.ndarray): Array of physical x-positions for each position.
        y_region_pix (np.ndarray): Array of physical y-positions for each position.
        parameter (np.ndarray): Array of parameter values for each position.
        parameter_radial_subtract (np.ndarray): If subtracting a radial component, this is the residual parameter.
        parameter_err (np.ndarray): Associated error for each parameter value.
        xy_regions (np.ndarray): Combined (x, y) position coordinates.
        r_regions (np.ndarray): Radial distance for every position with a parameter value.

        xy_to_fit (np.ndarray): Concatenated, flattened, NaN removed parameter coordinates for scikit-learn.
        parameter_to_fit (np.ndarray): Parameter array transformed to something suitable for scikit-learn.
        parameter_err_to_fit (np.ndarray): Parameter error array transformed to something suitable for scikit-learn.

        GPR Properties:

        scale_length (float): Measured kernel scale length.
        gp: The scikit-learn GaussianProcessRegressor.
        gp_goodness_of_fit (float): R^2 value for the GPR fit. If radial components considered, will include those too.

        predictions (dict): Dictionary containing predictions for a set of coordinates.


    TODO:
        * Include option for not radially subtracting

    """

    def __init__(self, galaxy, data_hdu, dist, ra, dec, pa, inc, r25=None):

        self.galaxy = galaxy

        # Calculate various data properties from data_hdu

        self.data_hdu = data_hdu
        self.nan_mask = np.where(~np.isnan(data_hdu.data))
        self.pix_size = get_pix_size(data_hdu)
        self.wcs = WCS(data_hdu)

        # Orientation parameters of the galaxy

        self.dist = dist
        self.ra = ra
        self.dec = dec
        self.pa = pa
        self.inc = inc
        self.r25 = r25

        # Parameters for straight line radial fitting

        self.n_dim = None
        self.n_walkers = None
        self.n_steps = None
        self.sampler = None
        self.m = None
        self.r_0 = None
        self.m_err = None
        self.r_0_err = None
        self.intrinsic_scatter = None
        self.radial_goodness_of_fit = None

        # Positions and parameters to feed into the GPR

        self.x_full = None
        self.y_full = None
        self.xy_full = None
        self.r_full = None

        self.region_numbers = None
        self.x_region_pix = None
        self.y_region_pix = None
        self.parameter = None
        self.parameter_err = None
        self.parameter_radial_subtract = None
        self.parameter_radial_subtract_err = None

        self.xy_regions = None
        self.r_regions = None

        self.xy_to_fit = None
        self.r_to_fit = None
        self.parameter_to_fit = None
        self.parameter_err_to_fit = None

        self.scale_length = None
        self.gp = None
        self.gp_goodness_of_fit = None

        self.predictions = {}

    def calculate_pixel_positions(self):
        """Calculate coordinates of pixels in the image, for later map predictions.

        Using pix_size, PA, inclination, and (optionally) r25, will convert pixels into rotated, projected physical
        distances, and will mask out any NaNs in the original image.

        """

        # Calculate a projected physical distance.

        x_cen, y_cen = self.wcs.all_world2pix(self.ra, self.dec, 1)

        xi, yi = np.meshgrid((np.arange(self.data_hdu.data.shape[1]) - x_cen),
                             (np.arange(self.data_hdu.data.shape[0]) - y_cen))

        # Convert these positions to physical positions (kpc), accounting for inclination and rotation

        xi *= self.pix_size / 3600 * np.pi / 180 * self.dist * 1e3
        yi *= self.pix_size / 3600 * np.pi / 180 * self.dist * 1e3

        x_all, y_all = project(xi, yi, self.pa, self.inc)

        if self.r25 is not None:
            r25_phys = self.r25 / 3600 * np.pi / 180 * self.dist * 1e3
            x_all /= r25_phys
            y_all /= r25_phys

        # We won't fit any NaNs

        x_all_flat = x_all[self.nan_mask].flatten()
        y_all_flat = y_all[self.nan_mask].flatten()

        self.x_full = x_all
        self.y_full = y_all
        self.r_full = np.sqrt(x_all ** 2 + y_all ** 2)
        self.xy_full = np.array([[x_all_flat[i], y_all_flat[i]] for i in range(len(x_all_flat))])

    def calculate_region_parameters(self, parameter_table, parameter_name, parameter_name_err=None,
                                    parameter_lower_limit=7.8, galaxy_colname='gal_name', region_id_colname='region_ID',
                                    x_pos_colname='cen_x', y_pos_colname='cen_y'):
        """Pull the values and central coordinates for parameters from an input .fits table.

        Using an input parameter table, will pull out `parameter_name` for each region where it is defined (and above a
        lower limit).

        Args:
            parameter_table (astropy.table.Table): Input table to pull region values from.
            parameter_name (str): Parameter column name to pull from `parameter_table`.
            parameter_name_err (str, optional): Column name for the parameter error. Defaults to None, which will be
                "`parameter`_err".
            parameter_lower_limit (float, optional): Optional lower bound to exclude values below. Defaults to 7.8
                (set up for metallicity catalogues).
            galaxy_colname (str, optional): Column name that lists galaxy. Should match with `galaxy` parameter.
                Defaults to 'gal_name'.
            region_id_colname (str, optional): Column name for region ID. Defaults to 'region_ID'.
            x_pos_colname (str, optional): Column name for x-centre of region in PIXELS (as defined by `data_hdu`).
                Defaults to 'cen_x'.
            y_pos_colname (str, optional): Column name for y-centre of region in PIXELS (as defined by `data_hdu`).
                Defaults to 'cen_y'.

        """

        if not parameter_name_err:
            parameter_name_err = parameter_name + '_err'

        x_cen, y_cen = self.wcs.all_world2pix(self.ra, self.dec, 1)

        # Find the central coordinates and parameters

        rows = parameter_table[parameter_table[galaxy_colname] == self.galaxy]
        rows = rows[~np.isnan(rows[parameter_name])]

        # For anomalously low values (set by parameter_lower_limit), filter these out
        rows = rows[rows[parameter_name] >= parameter_lower_limit]

        self.region_numbers = np.array(rows[region_id_colname])
        self.x_region_pix = np.array(rows[x_pos_colname])
        self.y_region_pix = np.array(rows[y_pos_colname])
        self.parameter = np.array(rows[parameter_name])
        self.parameter_err = np.array(rows[parameter_name_err])

        # Calculate projected x/y coordinates

        x_region = self.x_region_pix - x_cen
        y_region = self.y_region_pix - y_cen

        x_region *= self.pix_size / 3600 * np.pi / 180 * self.dist * 1e3
        y_region *= self.pix_size / 3600 * np.pi / 180 * self.dist * 1e3

        x_region_phys, y_region_phys = project(x_region, y_region, self.pa, self.inc)

        if self.r25 is not None:
            r25_phys = self.r25 / 3600 * np.pi / 180 * self.dist * 1e3
            x_region_phys /= r25_phys
            y_region_phys /= r25_phys

        self.xy_regions = np.c_[x_region_phys, y_region_phys]
        self.r_regions = np.sqrt(x_region_phys ** 2 + y_region_phys ** 2)

    def calculate_pixel_parameters(self, parameter_map, parameter_err_map, step_val=1,
                                   parameter_lower_limit=7.9):
        """Pull parameter values and physical (x, y) coordinates from an input .fits file.

        Takes an input .fits map, and uses the RA, Dec, PA and inclination to calculate deprojected coordinates for
        each pixel within the map. Also returns pixel coordinates for each value, but in the frame of `self.data_hdu`.

        Args:
             parameter_map (astropy.fits.PrimaryHDU): Input parameter map
             parameter_err_map (astropy.fits.PrimaryHDU): Input error map for the parameter
             step_val (int, optional): If there are many pixels, can use this to only select every `step_val`-th pixel.
                Defaults to 1.
            parameter_lower_limit (float, optional): If you have reason to distrust values below a certain limit, can
                input this here. Defaults to 7.9, which is what is recommended in the PHANGS-MUSE HII region catalogues.

        """

        wcs = WCS(parameter_map)
        pix_size = get_pix_size(parameter_map)

        parameter_map = parameter_map.data
        parameter_err_map = parameter_err_map.data

        x_cen, y_cen = wcs.all_world2pix(self.ra, self.dec, 1)

        # Pull out pixel coordinates

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            coordinates = np.where((~np.isnan(parameter_map)) & (parameter_map > parameter_lower_limit))

        x_region_pix = coordinates[1][::step_val]
        y_region_pix = coordinates[0][::step_val]
        self.parameter = parameter_map[coordinates][::step_val]
        self.parameter_err = parameter_err_map[coordinates][::step_val]

        # Calculate projected x/y coordinates

        x_region = x_region_pix - x_cen
        y_region = y_region_pix - y_cen

        x_region *= pix_size / 3600 * np.pi / 180 * self.dist * 1e3
        y_region *= pix_size / 3600 * np.pi / 180 * self.dist * 1e3

        # Put x_region_pix and y_region_pix into the frame of the original input HDU.

        ra, dec = wcs.all_pix2world(x_region_pix, y_region_pix, 1)
        self.x_region_pix, self.y_region_pix = self.wcs.all_world2pix(ra, dec, 1)

        x_region_phys, y_region_phys = project(x_region, y_region, self.pa, self.inc)

        if self.r25 is not None:
            r25_phys = self.r25 / 3600 * np.pi / 180 * self.dist * 1e3
            x_region_phys /= r25_phys
            y_region_phys /= r25_phys

        self.xy_regions = np.c_[x_region_phys, y_region_phys]
        self.r_regions = np.sqrt(x_region_phys ** 2 + y_region_phys ** 2)

    def shuffle_parameter(self):
        """Randomly shuffle the parameter values (and associated errors) around. Useful for testing the null hypothesis.

        """

        self.parameter, self.parameter_err = shuffle(self.parameter, self.parameter_err)

    def simulate_data(self, length_scale=0.05, scaling_factor=0.01, intrinsic_scatter=0.0005,
                      random_scatter=None, include_radial=True, r_0=8.5, m=-0.2):
        """Simulate some data, for testing purposes.

        Takes an underlying GPR, adds some intrinsic and random scatter, and an optional radial gradient.

        Args:
            length_scale (float, optional): Length scale for the GPR. Defaults to 0.05.
            scaling_factor (float, optional): Scale up or down the GPR. Defaults to 0.01.
            intrinsic_scatter (float, optional): Scale of intrinsic scatter to add to simulated data. Defaults to
                0.0005.
            random_scatter (list, optional): List of [lower, upper] bounds for random scatter. Defaults to None, which
                will use [0.005, 0.05].
            include_radial (bool, optional): Whether to add in a radial component. Defaults to True.
            r_0 (float, optional): Central amplitude of radial gradient. Defaults to 8.5.
            m (float, optional): Radial gradient. Defaults to -0.2.

        """

        if random_scatter is None:
            random_scatter = [0.005, 0.05]

        kernel_sim = scaling_factor * Matern(length_scale=length_scale)
        gpr_sim = GaussianProcessRegressor(kernel=kernel_sim)

        # Use positions to generate a central metallicity for each region
        parameter = gpr_sim.sample_y(self.xy_regions).T[0]

        # Add some noise to the data

        random_scatter_draw = np.random.uniform(low=random_scatter[0], high=random_scatter[1],
                                                size=parameter.shape)
        random_err_draw = np.random.normal(scale=random_scatter_draw)
        intrinsic_scatter_draw = np.random.normal(scale=intrinsic_scatter, size=parameter.shape)

        parameter += random_err_draw + intrinsic_scatter_draw
        parameter_err = random_scatter_draw

        if include_radial:
            # Into this, stick a fake gradient that we'll fit and subtract off later
            parameter = parameter + m * r_metallicity + r_0

        self.parameter = parameter
        self.parameter_err = parameter_err

    def fit_radial_distribution(self, p0=None, n_walkers=500, n_steps=500, mcmc_file_name=None,
                                overwrite_mcmc=False, verbose=False):
        """Fit a radial distribution, including intrinsic scatter.

        A pretty loose wrapper around emcee fitting for a radial gradient that includes intrinsic scatter. Will fit,
        then save out the sampler.

        Args:

            p0 (list, optional): Initial guesses for [m, r_0, intrinsic_scatter]. Defaults to None, which will use
                [-0.05, 8.5, 0.05]
            n_walkers (int, optional): Number of MCMC walkers to use. Defaults to 500.
            n_steps (int, optional): Number of steps for each walker to take. Defaults to 500.
            mcmc_file_name (str, optional): Name to save out sampler to. Defaults to None, which will not save out.
            overwrite_mcmc (bool, optional): Whether to overwrite `mcmc_file_name`, even if it exists. Defaults to
                False.
            verbose (bool, optional): Whether to print out progress for the emcee fitting. Defaults to False.

        """

        # Fit radial distribution of the parameter

        if p0 is None:
            p0 = [-0.05, 8.5, 0.05]
        p0 = np.array(p0)
        n_dim = len(p0)

        self.n_walkers = n_walkers
        self.n_steps = n_steps
        self.n_dim = n_dim

        pos = p0 + 1e-2 * p0 * np.random.randn(n_walkers, n_dim)

        if not os.path.exists(mcmc_file_name) or overwrite_mcmc:
            with Pool(N_THREADS) as pool:
                sampler = emcee.EnsembleSampler(n_walkers, n_dim, ln_prob,
                                                args=(self.r_regions, self.parameter, self.parameter_err),
                                                pool=pool)
                sampler.run_mcmc(pos, n_steps, progress=verbose)

            # Save out the sampler

            if mcmc_file_name is not None:
                with open(mcmc_file_name, 'wb') as sampler_f:
                    dill.dump(sampler, sampler_f)
        else:
            with open(mcmc_file_name, 'rb') as sampler_f:
                sampler = dill.load(sampler_f)

        self.sampler = sampler

        # Pull out the fit parameters

        flat_samples = sampler.get_chain(discard=int(self.n_steps / 2), flat=True)
        self.m, self.r_0, self.intrinsic_scatter = np.nanmedian(flat_samples, axis=0)
        self.m_err, self.r_0_err, _ = np.nanpercentile(flat_samples, 84, axis=0)
        self.m_err -= self.m
        self.r_0_err -= self.r_0

        # Calculate the R^2 statistic
        u = np.nansum((self.parameter - (self.m * self.r_regions + self.r_0)) ** 2)
        v = np.nansum((self.parameter - np.nanmean(self.parameter)) ** 2)
        r_square = 1 - u / v
        self.radial_goodness_of_fit = r_square

    def plot_step(self, step_plot_name, labels):
        """Create a step plot from the MCMC sampler.

        Plots the steps for the various MCMC walkers. Can be used to check convergence.

        Args:
            step_plot_name (str): Filename to save the step plot to. Should not include file extension.
            labels (list, optional): List of parameters to label axes with. Should be len(n_dim).

        """

        samples = self.sampler.get_chain()

        fig, axes = plt.subplots(self.n_dim, figsize=(10, 7), sharex='all')

        for i in range(self.n_dim):
            ax = axes[i]
            ax.plot(samples[:, :, i], 'k', alpha=0.3)
            ax.set_xlim(0, len(samples))
            ax.set_ylabel(labels[i])
            ax.yaxis.set_label_coords(-0.1, 0.5)

        axes[-1].set_xlabel("step number")

        plt.savefig(step_plot_name + '.png',
                    bbox_inches='tight')
        plt.savefig(step_plot_name + '.pdf',
                    bbox_inches='tight')

        plt.close()

    def plot_corner(self, corner_plot_name, labels):
        """Producer MCMC corner plot.

        A loose wrapper around corner, whereby we thin by half the chain and plot the corner from that.

        Args:
            corner_plot_name (str): Name to save out corner plot to. Should not include file extension.
            labels (list): List of axis labels for the corner. Should be len(n_dim).

        """

        flat_samples = self.sampler.get_chain(discard=int(self.n_steps / 2), flat=True)

        corner.corner(flat_samples, labels=labels, show_titles=True, title_fmt='.3f')

        plt.savefig(corner_plot_name + '.png', bbox_inches='tight')
        plt.savefig(corner_plot_name + '.pdf', bbox_inches='tight')

        plt.close()

    def plot_radial_fit(self, fit_plot_name, x_label=r'$R/r_{25}$', y_label='12 + log$_{10}$(O/H)',
                        n_points=1000, n_draws=100, figsize=(10, 4)):
        """Plot the radial fit to the data.

        Takes the MCMC sampler and plots the fit to the data, along with errors in the fit and data itself.

        Args:
            fit_plot_name (str): File to save the fit to. Should not include file extension.
            x_label (str, optional): x label for the plot. Defaults to r'$R/r_{25}$'.
            y_label (str, optional): y label for the plot. Defaults to '12 + log_10(O/H)' (can you tell this was built
                for metallicities?).
            n_points (int, optional): Number of points in the MCMC line. Defaults to 1000, which is pretty quick.
            n_draws (int, optional): Number of draws to calculate the error in the MCMC line. Defaults to 1000.
            figsize (tuple, optional): Figure size. Defaults to (10, 4), which is suitable for a two-column figure.

        """

        flat_samples = self.sampler.get_chain(discard=int(self.n_steps / 2), flat=True)

        x_min, x_max = 0, 1.05 * np.nanmax(self.r_regions)
        y_min, y_max = np.nanpercentile(self.parameter, 0.1), np.nanpercentile(self.parameter, 99.9)

        x_plot_mcmc = np.linspace(x_min, x_max, n_points)

        line_mcmc = np.zeros([n_points, n_draws])
        line_uncertainty_mcmc = np.zeros_like(line_mcmc)

        for draw in range(n_draws):
            sample = np.random.randint(low=0, high=flat_samples.shape[0])

            line_mcmc[:, draw] = flat_samples[sample, 0] * x_plot_mcmc + flat_samples[sample, 1] + \
                                 np.random.normal(scale=flat_samples[sample, 2])
            line_uncertainty_mcmc[:, draw] = flat_samples[sample, 0] * x_plot_mcmc + flat_samples[sample, 1]

        line_percentiles = np.nanpercentile(line_mcmc, [50, 16, 84], axis=1)
        line_uncertainty_percentiles = np.nanpercentile(line_uncertainty_mcmc, [50, 16, 84], axis=1)

        plt.figure(figsize=figsize)
        ax = plt.subplot(111)

        plt.errorbar(self.r_regions, self.parameter, yerr=self.parameter_err, c='k', marker='x', ls='none',
                     markersize=3)
        plt.plot(x_plot_mcmc, line_percentiles[0, :], c='powderblue', zorder=98)
        plt.fill_between(x_plot_mcmc, line_percentiles[1, :], line_percentiles[2, :],
                         color='powderblue', alpha=0.75, zorder=98)
        plt.fill_between(x_plot_mcmc, line_uncertainty_percentiles[1, :], line_uncertainty_percentiles[2, :],
                         color='r', alpha=0.75, zorder=99)

        plt.xlim([x_min, x_max])
        plt.ylim([y_min, y_max])

        ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.025))
        ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.01))

        plt.grid()

        plt.xlabel(x_label)
        plt.ylabel(y_label)

        plt.text(0.05, 0.95, r'$12+\log_{10}(\mathrm{O/H}) = %.3f(\pm%.3f)R/r_{25} + %.3f(\pm%.3f)$, '
                             '\n'
                             r'$\sigma_\mathrm{int}=%.2f$' %
                 (self.m, self.m_err, self.r_0, self.r_0_err, self.intrinsic_scatter),
                 ha='left', va='top',
                 transform=ax.transAxes,
                 bbox=dict(facecolor='white', edgecolor='none', alpha=0.75))

        plt.tight_layout()

        plt.savefig(fit_plot_name + '.png', bbox_inches='tight')
        plt.savefig(fit_plot_name + '.pdf', bbox_inches='tight')

        plt.close()

    def subtract_radial_fit(self):
        """Subtract radial gradient from data

        """

        self.parameter_radial_subtract = self.parameter - (self.m * self.r_regions + self.r_0)

    def remove_fraction_of_regions(self, fraction):
        """Remove a certain fraction of regions, for jackknife testing scale lengths.

        Args:
            fraction (float): Fraction of regions to retain

        """

        total_size = len(self.region_numbers)

        idx = np.random.choice(total_size, size=int(fraction * total_size), replace=False)

        self.region_numbers = self.region_numbers[idx]
        self.x_region_pix = self.x_region_pix[idx]
        self.y_region_pix = self.y_region_pix[idx]
        self.parameter = self.parameter[idx]
        self.parameter_radial_subtract = self.parameter_radial_subtract[idx]
        self.parameter_err = self.parameter_err[idx]
        self.xy_regions = self.xy_regions[idx]
        self.r_regions = self.r_regions[idx]

    def remove_fraction_of_pixels(self, fraction):
        """Remove a certain fraction of pixels, for jackknife testing scale lengths.

        Args:
            fraction (float): Fraction of pixels to retain

        """

        total_size = len(self.parameter)

        idx = np.random.choice(total_size, size=int(fraction * total_size), replace=False)

        self.x_region_pix = self.x_region_pix[idx]
        self.y_region_pix = self.y_region_pix[idx]
        self.parameter = self.parameter[idx]
        self.parameter_radial_subtract = self.parameter_radial_subtract[idx]
        self.parameter_err = self.parameter_err[idx]
        self.xy_regions = self.xy_regions[idx]
        self.r_regions = self.r_regions[idx]

    def calc_xy_to_fit_regions(self, region_hdu, step_val=1, use_radial=True):
        """Include region information for the fitting.

        Will take regions from input region mask and put parameter values into those regions. In this way, we can
        account for irregular region shapes.

        Args:
            region_hdu (astropy.fits.PrimaryHDU): .fits file defining regions. The regions here should match up with
                `region_numbers`, and this HDU should be the same projection as `data_hdu`.
            step_val (int, optional): For large data sets, we may need to prune down the total number of pixels. This
                will step through uniformly and take a representative subset of the data (i.e. step_val=2 removes half
                the data). Defaults to 1, which keeps everything.
            use_radial (bool, optional): Whether to use a radially subtracted parameter. Defaults to True.

        """

        if use_radial:
            parameter = self.parameter_radial_subtract
        else:
            parameter = self.parameter

        # Make regions using the masks

        parameter_map = np.zeros_like(self.data_hdu.data)
        parameter_map[parameter_map == 0] = np.nan
        parameter_map_original = parameter_map.copy()
        parameter_map_err = parameter_map.copy()

        # Loop over the regions, put them into an array to fit.
        for i, region_number in enumerate(self.region_numbers):
            parameter_map_original[region_hdu.data == region_number] = self.parameter[i]
            parameter_map[region_hdu.data == region_number] = parameter[i]
            parameter_map_err[region_hdu.data == region_number] = self.parameter_err[i]

        # Filter down to points we can fit, and sort into arrays suitable for scikit-learn

        pixels_to_fit = np.where(~np.isnan(parameter_map))

        x_to_fit = self.x_full[pixels_to_fit]
        y_to_fit = self.y_full[pixels_to_fit]
        parameter_to_fit = parameter_map[pixels_to_fit]
        parameter_err_to_fit = parameter_map_err[pixels_to_fit]

        x_to_fit_flat = x_to_fit.flatten()
        y_to_fit_flat = y_to_fit.flatten()
        parameter_to_fit_flat = parameter_to_fit.flatten()
        parameter_err_to_fit_flat = parameter_err_to_fit.flatten()

        # Take every step_val-th value.

        x_to_fit_flat = x_to_fit_flat[::step_val]
        y_to_fit_flat = y_to_fit_flat[::step_val]
        self.xy_to_fit = np.array([[x_to_fit_flat[i], y_to_fit_flat[i]] for i in range(len(x_to_fit_flat))])
        self.r_to_fit = np.sqrt(x_to_fit_flat ** 2 + y_to_fit_flat ** 2)

        self.parameter_to_fit = parameter_to_fit_flat[::step_val]
        self.parameter_err_to_fit = parameter_err_to_fit_flat[::step_val]

    def calc_xy_to_fit_positions(self, use_radial=True):
        """Prepare xy positions and parameter values if we're not using region information.

        This should be used on a pixel-by-pixel parameter map, rather than drawing from some extended object catalogue.

        Args:
            use_radial (bool, optional): Whether to use a radially subtracted parameter. Defaults to True.

        """

        if use_radial:
            parameter = self.parameter_radial_subtract
        else:
            parameter = self.parameter

        self.xy_to_fit = self.xy_regions.copy()
        self.r_to_fit = self.r_regions.copy()

        self.parameter_to_fit = parameter.flatten()
        self.parameter_err_to_fit = self.parameter_err.flatten()

    def perturb_pixels_to_fit(self):
        """Perturb the parameters by their errors.

        Take into account the errors both in the radial fit and the measurement uncertainties in the parameter itself.
        Firstly, add back in the radial component, sample from the MCMC and subtract off again, then perturb by the
        measured uncertainties in the parameter measurements.

        """

        flat_samples = self.sampler.get_chain(discard=int(self.n_steps / 2), flat=True)
        self.parameter_to_fit += (self.m * self.r_to_fit + self.r_0)
        idx = np.random.randint(low=0, high=flat_samples.shape[0])
        m_idx = flat_samples[idx, 0]
        r0_idx = flat_samples[idx, 1]
        self.parameter_to_fit -= (m_idx * self.r_to_fit + r0_idx)

        self.parameter_to_fit += np.random.normal(loc=0, scale=self.parameter_err_to_fit)

    def fit_gpr_regressor(self, gpr_file=None,
                          matern_length_scale=0.1, matern_length_scale_bounds=(0.001, 5), matern_nu=1.5,
                          n_restarts_optimizer=25, kernel=None,
                          overwrite_regressor=False, verbose=False):
        """Fit the Gaussian processor regressor.

        Ah, we're finally here. This is where the bulk of the processing time occurs, as we fit the azimuthal variations
        in the data. The kernel itself is a sum of a Matern kernel (to describe the underlying process), a constant
        kernel (to model the mean of the data), and a White kernel (to model the intrinsic scatter). After fitting,
        this also pulls out the scale length and converts into kpc for later analysis.

        Args:
            gpr_file (str, optional): Filename to save GPR to. Defaults to None, which will not save out the regressor
                at the end (not recommended for big ones!).
            matern_length_scale (float, optional): Length scale for initial guess at Matern kernel. Defaults to 0.1.
            matern_length_scale_bounds (tuple, optional): Length scale bounds for Matern kernel. Defaults to (0.001, 2).
            matern_nu (float, optional): Smoothness of the Matern kernel. Unless you know what you're doing, don't
                change this. Defaults to 1.5.
            n_restarts_optimizer (int, optional): n_restarts_optimizer in the GPR fitter. Defaults to 25.
            kernel (optional): Can optionally input a kernel from another fit here. This can help with convergence.
                Defaults to None, which will set up the kernel as default.
            overwrite_regressor (bool, optional): Whether to overwrite GPR if it already exists. Defaults to False.
            verbose (bool, optional): If True, prints out how long fitting took, as well as initial and final kernel
                parameters. Defaults to False.

        """

        if not gpr_file:
            gpr_file_exists = False
        elif gpr_file and not os.path.exists(gpr_file):
            gpr_file_exists = False
        else:
            gpr_file_exists = True

        int_scatter = np.nanstd(self.parameter_to_fit)

        if not kernel:
            kernel_matern = Matern(length_scale=matern_length_scale,
                                   length_scale_bounds=matern_length_scale_bounds,
                                   nu=matern_nu,
                                   )
            kernel_constant_mean = ConstantKernel(0.0)
            kernel_noise = WhiteKernel(noise_level=int_scatter,
                                       noise_level_bounds=(0.01 * int_scatter, int_scatter),
                                       )
            kernel = kernel_matern + kernel_constant_mean + kernel_noise

        # Put all of this into the regressor.
        gp = GaussianProcessRegressor(kernel=kernel, alpha=self.parameter_err_to_fit,
                                      n_restarts_optimizer=n_restarts_optimizer)

        if verbose:
            print('Initial kernel: %s' % gp.kernel)

        if not gpr_file_exists or overwrite_regressor:

            gp_time = time.time()

            # Fit the GPR. Since this is computationally expensive, save out the regressor after the fitting is done

            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                gp.fit(self.xy_to_fit, self.parameter_to_fit)

            if gpr_file:
                dump(gp, gpr_file)

            if verbose:
                print('GPR fitted: took %.2fm' % ((time.time() - gp_time) / 60))

        else:

            gp = load(gpr_file)

        if verbose:
            print('Optimised kernel: %s ' % gp.kernel_)

        scale_length = gp.kernel_.get_params()['k1__k1__length_scale']

        if self.r25 is not None:
            r25_phys = self.r25 / 3600 * np.pi / 180 * self.dist * 1e3
            scale_length *= r25_phys

        self.scale_length = scale_length
        self.gp = gp

        # TODO: Just a quick test looking at the BIC

        # theta = (self.m, self.r_0, self.intrinsic_scatter)
        # sl_n_par = 3
        # sl_likelihood = ln_like(theta, self.r_to_fit, self.parameter, self.parameter_err_to_fit)
        # print(sl_likelihood)
        # sl_bic = sl_n_par * np.log(len(self.r_to_fit)) - 2 * sl_likelihood
        # print(sl_bic)
        #
        # gpr_bic = gp.log_marginal_likelihood()
        # print(gpr_bic)
        # no

        # Calculate the goodness of fit

        if self.radial_goodness_of_fit is not None:

            azimuthal_predictions = gp.predict(self.xy_to_fit)

            # Calculate the R^2 statistic including the radial fit too
            u = np.nansum((self.parameter - (self.m * self.r_regions + self.r_0) - azimuthal_predictions) ** 2)
            v = np.nansum((self.parameter - np.nanmean(self.parameter)) ** 2)
            r_square = 1 - u / v
            self.gp_goodness_of_fit = r_square

        else:
            self.gp_goodness_of_fit = gp.score(self.xy_to_fit, self.parameter_to_fit)

    def make_predictions(self, pred_file, xy_positions, name='predictions', gpr_file=None, overwrite_predictions=False,
                         n_batch=None, verbose=False):
        """Predict the parameter for each given coordinate pair (x, y).

        Using the fitted GPR, calculate predictions for values at given coordinates. Given that this can be
        computationally expensive, we can also batch the predictions if needs be.

        Args:
            pred_file (str): File to save the predictions to.
            gpr_file (str): Optional pickled GPR file. Defaults to None, which will use self.gp
            xy_positions (np.ndarray): Physical positions to estimate the parameter at.
            name (str, optional): Key for `predictions` dict. Defaults to 'predictions'.
            overwrite_predictions (bool, optional): Overwrite `pred_file` if already exists. Defaults to False.
            n_batch (int, optional): If not None, will batch in sets of n_batch. Usually I use 100, but whatever to
                not run out of RAM. Defaults to None, which tries to predict everything simultaneously.
            verbose (bool, optional): Print out statement of how long predictions took? Defaults to False

        """

        if not gpr_file:
            gp = self.gp
        else:
            gp = load(gpr_file)

        if not os.path.exists(pred_file) or overwrite_predictions:

            predict_time = time.time()

            if n_batch is not None:

                n_batches = np.ceil(len(xy_positions) / n_batch)
                start_batch = 0

                # Batch predictions to avoid out of memory issues

                pred = [np.zeros(len(xy_positions)), np.zeros(len(xy_positions))]

                for batch in tqdm(range(int(n_batches)), desc='Batch Predictions'):

                    if batch == n_batches - 1:
                        xy_batch = xy_positions[start_batch:]
                    else:
                        xy_batch = xy_positions[start_batch:start_batch + n_batch]

                    if len(xy_batch.shape) == 1:
                        xy_batch = xy_batch.reshape(-1, 1)

                    gp_pred = gp.predict(xy_batch, return_std=True)

                    for i in range(len(gp_pred[0])):
                        pred[0][i + start_batch] = gp_pred[0][i]
                        pred[1][i + start_batch] = gp_pred[1][i]

                    start_batch += n_batch

            else:

                if xy_positions.shape == 1:
                    xy_positions = xy_positions.reshape(-1, 1)

                pred = gp.predict(xy_positions, return_std=True)

            np.save(pred_file, pred)

            if verbose:
                print('Predictions completed: took %.2fm' % ((time.time() - predict_time) / 60))

        self.predictions[name] = np.load(pred_file)

    def create_map(self, map_name, map_err_name=None, pred_name='predictions', use_radial=True):
        """Create parameter map and associated error.

        Takes the GPR predictions, combines with the radial fit and produces a final map of the parameter and its
        associated errors.

        Args:
            map_name (str): Name to save the parameter map to. Should not include the .fits extension.
            map_err_name (str, optional): Name to save the parameter error map to. Defaults to None, which will use
                `map_name`_err'.
            pred_name  (str, optional): Which predictions to use. Defaults to 'predictions'.
            use_radial (bool, optional): Include the radial fitting in the final map. Defaults to True.

        """

        if not map_err_name:
            map_err_name = map_name + '_err'

        pred = self.predictions[pred_name]

        pred_map = np.zeros_like(self.data_hdu.data)
        pred_map[pred_map == 0] = np.nan

        pred_map_err = pred_map.copy()
        pred_map_err[self.nan_mask] = pred[1]

        if use_radial:
            pred_map_radial_subtract = pred_map.copy()
            pred_map_radial_subtract[self.nan_mask] = pred[0]

            # Calculate the final parameter map by adding the radial contribution back in

            # if self.r_0 is not None:
            radial_contribution = self.m * self.r_full[self.nan_mask] + self.r_0
            pred_map = pred_map_radial_subtract.copy()
            pred_map[self.nan_mask] = pred_map_radial_subtract[self.nan_mask] + radial_contribution

            # Calculate an error map. We have the error from the GPR already, but we also need to include the error
            # from uncertainty in the radial fit.

            slope_error = np.zeros_like(self.data_hdu.data)
            slope_error[slope_error == 0] = np.nan
            slope_error[self.nan_mask] = self.m_err * self.r_full[self.nan_mask] + self.r_0_err
            pred_map_err = np.sqrt(pred_map_err ** 2 + slope_error ** 2)

        else:
            pred_map[self.nan_mask] = pred[0]
            pred_map_radial_subtract = None

        fits.writeto(map_name + '.fits',
                     pred_map, self.data_hdu.header, overwrite=True)
        fits.writeto(map_err_name + '.fits',
                     pred_map_err, self.data_hdu.header, overwrite=True)

        return pred_map, pred_map_err, pred_map_radial_subtract
