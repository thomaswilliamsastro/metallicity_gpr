import os
import socket

from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt

from astronomical_gpr import AstronomicalGPR
from funcs import get_info_from_master_table
from vars import top_dir, phangs_master_table

os.chdir(top_dir)

overwrite_mcmc = False
overwrite_regressor = False
use_radial = True
overwrite_predictions = True
use_ppak = True
res = '1arcsec'

galaxy_params = ['dist', 'orient_ra', 'orient_dec', 'orient_posang', 'orient_incl', 'size_r25']
galaxy = 'NGC5457'
metallicity_calib = 'scal'
# metallicity_calib = 'direct'

metallicity_dir = os.path.join('m101_metallicity', 'metallicity_gpr')
if use_ppak:
    metallicity_dir += '_ppak'
if res is not None:
    metallicity_dir += '_%s' % res
if not os.path.exists(metallicity_dir):
    os.makedirs(metallicity_dir)
plot_dir = os.path.join('m101_metallicity', 'metallicity_gpr_plots')
if use_ppak:
    plot_dir += '_ppak'
if res is not None:
    plot_dir += '_%s' % res
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

if use_ppak:
    hdu_file_name = os.path.join('m101_metallicity')
    if res is not None:
        hdu_file_name = os.path.join(hdu_file_name, res)
    hdu_file_name = os.path.join(hdu_file_name, 'M101_%s_PPAK_CHAOS.fits' % metallicity_calib)
else:
    hdu_file_name = os.path.join('m101_metallicity', 'CHAOS_M101_%s.fits' % metallicity_calib)
hdu = fits.open(hdu_file_name)[0]
hdu_err = fits.open(hdu_file_name.replace('.fits', '_err.fits'))[0]

errant_nan_idx = np.where((np.isnan(hdu_err.data)) & (~np.isnan(hdu.data)))
hdu.data[errant_nan_idx] = np.nan
hdu_err.data[errant_nan_idx] = np.nan

# We don't have a footprint (yet), so just put in an HDU where mostly everything is included. Trim percentiles for speed
# for now

hdu_no_nans = hdu.copy()

if not use_ppak:
    hdu_no_nans.data[~np.isnan(hdu_no_nans.data)] = np.nan

    measurement_idx = np.where(~np.isnan(hdu.data))
    i_min, i_max = int(np.percentile(measurement_idx[0], 5)), int(np.percentile(measurement_idx[0], 95))
    j_min, j_max = int(np.percentile(measurement_idx[1], 5)), int(np.percentile(measurement_idx[1], 95))

    hdu_no_nans.data[i_min:i_max, j_min:j_max] = 1

if res:

    hdu_no_nans.data[~np.isnan(hdu_no_nans.data)] = np.nan
    measurement_idx = np.where(~np.isnan(hdu.data))
    # i_min, i_max = np.min(measurement_idx[0]), np.max(measurement_idx[0])
    # j_min, j_max = np.min(measurement_idx[1]), np.max(measurement_idx[1])
    i_min, i_max = int(np.percentile(measurement_idx[0], 1)), int(np.percentile(measurement_idx[0], 99))
    j_min, j_max = int(np.percentile(measurement_idx[1], 1)), int(np.percentile(measurement_idx[1], 99))
    hdu_no_nans.data[i_min:i_max, j_min:j_max] = 1

dist, ra, dec, pa, inc, r25 = get_info_from_master_table(phangs_master_table, galaxy.lower(), galaxy_params)

mcmc_file_name = os.path.join(metallicity_dir, 'radial_fit_%s.pkl' % metallicity_calib)

step_plot_name = os.path.join(plot_dir, 'step_%s' % metallicity_calib)
corner_plot_name = step_plot_name.replace('step', 'corner')
fit_plot_name = step_plot_name.replace('step', 'fit')
labels = ['m', r'$R_0$', r'$\sigma$']

gpr_fitter = AstronomicalGPR(galaxy, hdu_no_nans, dist, ra, dec, pa, inc, r25)
gpr_fitter.calculate_pixel_positions()
gpr_fitter.calculate_pixel_parameters(hdu, hdu_err, parameter_lower_limit=7)
gpr_fitter.fit_radial_distribution(mcmc_file_name=mcmc_file_name, overwrite_mcmc=overwrite_mcmc,
                                   verbose=True)
gpr_fitter.subtract_radial_fit()
if not os.path.exists(step_plot_name) or overwrite_mcmc:
    gpr_fitter.plot_step(step_plot_name, labels)
    gpr_fitter.plot_corner(corner_plot_name, labels)
    gpr_fitter.plot_radial_fit(fit_plot_name)
gpr_fitter.calc_xy_to_fit_positions(use_radial=use_radial)

gpr_file = os.path.join(metallicity_dir, 'm101_%s_gpr.pkl' % metallicity_calib)
pred_file = gpr_file.replace('_gpr.pkl', '_predict.npy')
pred_file_central = gpr_file.replace('_gpr.pkl', '_predict_central.npy')

if 'astro-node' in socket.gethostname():
    gpr_fitter.fit_gpr_regressor(gpr_file, overwrite_regressor=overwrite_regressor, verbose=True)

gpr_fitter.make_predictions(pred_file, gpr_fitter.xy_full, name='full', n_batch=100,
                            overwrite_predictions=overwrite_predictions, verbose=True)
gpr_fitter.make_predictions(pred_file_central, gpr_fitter.xy_regions, name='central',
                            overwrite_predictions=overwrite_predictions, verbose=True)

if use_ppak:
    metallicity_file_name = os.path.join(metallicity_dir, 'M101_%s_PPAK_CHAOS_metallicity' % metallicity_calib)
else:
    metallicity_file_name = os.path.join(metallicity_dir, 'CHAOS_M101_%s_metallicity' % metallicity_calib)

metallicity_map, metallicity_map_err, _ = gpr_fitter.create_map(metallicity_file_name, pred_name='full',
                                                                use_radial=use_radial)

print('Complete!')
