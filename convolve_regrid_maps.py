# -*- coding: utf-8 -*-
"""
Convolve maps to 150pc resolution, then regrid to pixels of 150pc

@author: Tom Williams
"""

import glob
import os

import astropy.units as u
import numpy as np
from astropy.convolution import convolve as convolve
from astropy.io import fits
from astropy.wcs import WCS
from radio_beam import Beam
from reproject import reproject_exact as reproject

from funcs import get_info_from_master_table
from vars import top_dir, muse_dir, muse_version, phangs_master_table, all_lines

PIXEL_SCALE_NAMES = ['XPIXSIZE', 'CDELT1', 'CD1_1']


def convolve_regrid(original_data, convolution_kernel, header, reproj_header, flux_density=True, normalize_kernel=True):
    data_convolved = convolve(original_data, convolution_kernel, preserve_nan=True, nan_treatment='interpolate',
                              normalize_kernel=normalize_kernel)

    data_convolved_hdu = fits.PrimaryHDU(data=data_convolved, header=header)
    data_reprojected, _ = reproject(data_convolved_hdu, reproj_header)

    if not flux_density:
        # Rescale by the pixel sizes
        scale_factor = get_pix_size(reproj_header) ** 2 / get_pix_size(header) ** 2
        data_reprojected *= scale_factor

    return data_reprojected


def get_pix_size(header):
    for pixel_keyword in PIXEL_SCALE_NAMES:
        try:
            try:
                pix_scale = np.abs(float(header[pixel_keyword]))
            except ValueError:
                continue
            if pixel_keyword in ['CDELT1', 'CD1_1']:
                pix_scale *= 3600
            return pix_scale
        except KeyError:
            pass

    raise Warning('No pixel scale found')


os.chdir(os.path.join(top_dir, muse_dir, muse_version))

target_phys_res = 120 * u.pc

hdu_names = glob.glob(os.path.join('copt', '*_MAPS.fits'))
hdu_names = sorted(hdu_names)

line_exts = ['FLUX', 'SIGMA']

if not os.path.exists('conv'):
    os.makedirs('conv')

for hdu_name in hdu_names:

    hdu = fits.open(hdu_name)

    pix_scale = get_pix_size(hdu[1].header) * u.arcsec

    # Pull galaxy and resolution from the filename

    galaxy = hdu_name.split('/')[-1].split('-')[0].lower()
    copt_res = float(hdu_name.split('/')[-1].split('-')[-1].split('asec')[0]) * u.arcsec

    # Figure out the angular resolution for chosen physical resolution
    dist = get_info_from_master_table(phangs_master_table, galaxy, ['dist']) * u.Mpc
    conv_res = (target_phys_res / dist).decompose() * 180 / np.pi * 3600 * u.arcsec

    if conv_res < copt_res:
        raise Warning('%s onvolved resolution %s smaller than current resolution %s!' % (galaxy, conv_res, copt_res))

    # Create the convolution kernel

    print('Convolving %s from %s to %s' % (galaxy, copt_res, conv_res))

    copt_psf = Beam(major=copt_res, minor=copt_res, pa=0.0 * u.deg)
    conv_psf = Beam(major=conv_res, minor=conv_res, pa=0.0 * u.deg)
    err_kernel = conv_psf.as_kernel(pix_scale)
    err_kernel_square_norm = err_kernel.array ** 2 / np.sum(err_kernel.array ** 2)
    conv_kernel = conv_psf.deconvolve(copt_psf).as_kernel(pix_scale)

    # We project to Nyquist sample the maps, i.e. 2 pixels across the PSF

    new_pix_scale = conv_res / 2

    # Create a header to reproject to. This is a little fiddly.

    base_hdu = hdu['%s_%s' % (all_lines[0], line_exts[0])]

    wcs = WCS(base_hdu)

    # Find the WCS coords at the central pixel
    cen_y, cen_x = np.array(base_hdu.data.shape) / 2
    crval_1, crval_2 = wcs.all_pix2world(cen_x, cen_y, 1)

    # Figure out the new extent of the image
    extent_y, extent_x = np.ceil(np.array(base_hdu.data.shape) * pix_scale / new_pix_scale)
    extent_y, extent_x = int(extent_y), int(extent_x)

    # CRPIX are just the extent/2
    crpix_1, crpix_2 = extent_x / 2, extent_y / 2

    new_wcs = WCS(naxis=2)
    new_wcs.wcs.crpix = [crpix_1, crpix_2]
    new_wcs.wcs.cdelt = np.array([-new_pix_scale.to(u.deg).value,
                                  new_pix_scale.to(u.deg).value])
    new_wcs.wcs.crval = [crval_1, crval_2]
    new_wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    new_wcs.wcs.equinox = 2000
    new_hdr = new_wcs.to_header()

    reproj_hdu = fits.PrimaryHDU(data=np.zeros([extent_y, extent_x]), header=new_hdr)
    reproj_header = reproj_hdu.header

    # Set up for a multi-extension FITS file

    hdu_reproj = fits.HDUList(hdu[0])

    for line in all_lines:

        print('%s' % line)

        for line_ext in line_exts:
            data = hdu['%s_%s' % (line, line_ext)].data
            error = hdu['%s_%s_ERR' % (line, line_ext)].data
            header = hdu['%s_%s' % (line, line_ext)].header

            if line_ext == 'FLUX':
                flux_density = False
            else:
                flux_density = True

                # For velocity dispersion, want to take the intensity weighted convolution, and also propagate through
                # the errors like that

                error_from_sigma = (error/data) ** 2
                error_from_flux = (hdu['%s_FLUX_ERR' % line].data / hdu['%s_FLUX' % line].data) ** 2
                # Avoid any divide by zero errors here
                error_from_flux[~np.isfinite(error_from_flux)] = 0

                error = data * np.sqrt(error_from_sigma + error_from_flux)

                data *= hdu['%s_%s' % (line, 'FLUX')].data

            # Convolve and reproject
            data_reproj = convolve_regrid(data, conv_kernel, header, reproj_header, flux_density=flux_density)

            # Create a mask of all pixels that are less than the convolved resolution away from a NaN (to avoid
            # issues in the convolution). We only need to do this once.

            if line == all_lines[0] and line_ext == line_exts[0]:

                nan_mask = np.zeros_like(data_reproj)

                idx = np.where(np.isnan(data_reproj))
                r_cutoff = conv_res / new_pix_scale

                for i in range(data_reproj.shape[0]):
                    for j in range(data_reproj.shape[1]):

                        i_nearest = i - idx[0]
                        j_nearest = j - idx[1]
                        r_nearest = np.nanmin(np.sqrt(i_nearest ** 2 + j_nearest ** 2))
                        if r_nearest <= r_cutoff:
                            nan_mask[i, j] = 1

            data_reproj[nan_mask == 1] = np.nan

            if line_ext == 'SIGMA':
                # Divide through by the convolved flux
                flux_conv = convolve_regrid(hdu['%s_%s' % (line, 'FLUX')].data, conv_kernel, header, reproj_header,
                                            flux_density=flux_density)
                data_reproj /= flux_conv

            reproj_header_edit = reproj_header.copy()
            reproj_header_edit['EXTNAME'] = '%s_%s' % (line, line_ext)
            hdu_reproj.append(fits.ImageHDU(data=data_reproj, header=reproj_header_edit))

            # Deal with the error maps differently, as the pixels are correlated. First, estimate the per-pixel error

            error_per_pix = error * 2 * np.sqrt(np.pi) * copt_res.value / 2.355 / pix_scale.value
            variance = error_per_pix ** 2

            # Convolve this with the target res beam.

            err_reproj = np.sqrt(convolve_regrid(variance, err_kernel_square_norm, header,
                                                 reproj_header, flux_density=flux_density))

            err_reproj[nan_mask == 1] = np.nan
            reproj_header_edit = reproj_header.copy()
            reproj_header_edit['EXTNAME'] = '%s_%s_ERR' % (line, line_ext)
            hdu_reproj.append(fits.ImageHDU(data=err_reproj, header=reproj_header_edit))

    hdu_reproj.writeto(os.path.join('conv', '%s_MAPS.fits' % galaxy.upper()), overwrite=True)

print('Complete!')
