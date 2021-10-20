import os
import socket

from astropy.io import fits
from astropy.wcs import WCS
import numpy as np

from astronomical_gpr import AstronomicalGPR
from funcs import get_info_from_master_table
from vars import top_dir, plot_dir, hii_mask_dir, muse_dir, muse_version, metallicity_dir, phangs_master_table, \
    use_pix_maps, include_radial_subtract, gpr_step_val, use_regions, simulation, hii_only, extinction_curve

os.chdir(top_dir)

overwrite_regressor = False
overwrite_predictions = False
null_hypothesis = False

galaxy = 'NGC3627'
metallicity_calib = 'pg16_scal'
emission_to_use = 'HA6562_FLUX'

galaxy_params = ['dist', 'orient_ra', 'orient_dec', 'orient_posang', 'orient_incl', 'size_r25']

galaxy_metallicity_dir = os.path.join(metallicity_dir, muse_version, '%s_ext_curve' % extinction_curve,
                                      metallicity_calib, galaxy + '_' + metallicity_calib + '_gpr')
if use_pix_maps:
    galaxy_metallicity_dir += '_pix'
if hii_only:
    galaxy_metallicity_dir += '_hii_only'
if not os.path.exists(galaxy_metallicity_dir):
    os.makedirs(galaxy_metallicity_dir)

galaxy_plot_dir = os.path.join(plot_dir, muse_version, '%s_ext_curve' % extinction_curve,
                               metallicity_calib, galaxy + '_' + metallicity_calib + '_gpr')
if use_pix_maps:
    galaxy_plot_dir += '_pix'
if hii_only:
    galaxy_plot_dir += '_hii_only'
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

gpr_fitter.fit_radial_distribution(mcmc_file_name=mcmc_file_name, overwrite_mcmc=False,
                                   verbose=True)
gpr_fitter.subtract_radial_fit()

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
gpr_file += '_low_nu_gpr.pkl'

# Throw into GPR

pred_file = gpr_file.replace('_gpr.pkl', '_predict.npy')
pred_file_central = gpr_file.replace('_gpr.pkl', '_predict_central.npy')

# Skip reading in/generating GPR files if running on laptop

if 'astro-node' in socket.gethostname():
    gpr_fitter.fit_gpr_regressor(gpr_file, matern_nu=0.5, overwrite_regressor=overwrite_regressor, verbose=True)

gpr_fitter.make_predictions(pred_file, gpr_fitter.xy_full, name='full', n_batch=100,
                            overwrite_predictions=overwrite_predictions, verbose=True)
gpr_fitter.make_predictions(pred_file_central, gpr_fitter.xy_regions, name='central',
                            overwrite_predictions=overwrite_predictions, verbose=True)

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
    metallicity_file_name += '_no_radial'
metallicity_file_name += '_low_nu'

metallicity_map, metallicity_map_err, metallicity_map_radial_subtract = \
    gpr_fitter.create_map(metallicity_file_name, pred_name='full', use_radial=include_radial_subtract)

print('Complete!')
