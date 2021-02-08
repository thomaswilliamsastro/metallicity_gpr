# -*- coding: utf-8 -*-
"""
Convolve maps to 150pc resolution, then regrid to pixels of 150pc

@author: Tom Williams
"""

import glob
import os

from astropy.io import fits
import astropy.units as u
import numpy as np
from radio_beam import Beam
from astropy.convolution import convolve_fft
import matplotlib.pyplot as plt
from MontagePy.main import mImgtbl, mMakeHdr
from reproject import reproject_exact as reproject
from astropy.wcs import WCS

from vars import top_dir, muse_dir, muse_version, phangs_master_table, all_lines
from funcs import get_info_from_master_table


def convolve_regrid(data, conv_kernel, header, reproj_header):
    data_conv = convolve_fft(data, conv_kernel, preserve_nan=True, nan_treatment='interpolate')
    data_conv_hdu = fits.PrimaryHDU(data=data_conv, header=header)
    data_reproj, _ = reproject(data_conv_hdu, reproj_header)

    return data_reproj


os.chdir(os.path.join(top_dir, muse_dir, muse_version))

target_phys_res = 150 * u.pc

hdu_names = glob.glob(os.path.join('copt', '*_MAPS.fits'))

line_exts = ['FLUX', 'VEL', 'SIGMA']

if not os.path.exists(os.path.join('copt', 'temp')):
    os.makedirs(os.path.join('copt', 'temp'))
if not os.path.exists('conv'):
    os.makedirs('conv')

for hdu_name in hdu_names:

    hdu = fits.open(hdu_name)

    pix_scale = np.abs(hdu[1].header['CD1_1']) * 3600 * u.arcsec

    # Pull galaxy and resolution from the filename

    galaxy = hdu_name.split('/')[-1].split('-')[0].lower()
    copt_res = float(hdu_name.split('/')[-1].split('-')[-1].split('asec')[0]) * u.arcsec

    # Figure out the angular resolution for 150pc, which will also be the output pixel scale
    dist = get_info_from_master_table(phangs_master_table, galaxy, ['dist']) * u.Mpc
    conv_res = (target_phys_res / dist).decompose() * 180 / np.pi * 3600 * u.arcsec

    # Create the convolution kernel

    print('Will be convolving %s from %s to %s res' % (galaxy, copt_res, conv_res))

    copt_psf = Beam(major=copt_res, minor=copt_res, pa=0.0 * u.deg)
    conv_psf = Beam(major=conv_res, minor=conv_res, pa=0.0 * u.deg)
    conv_kernel = conv_psf.deconvolve(copt_psf).as_kernel(pix_scale)

    # Create a header to reproject to. This is a little fiddly, but use Montage

    hdu['%s_%s' % (all_lines[0], line_exts[0])].writeto(os.path.join('copt', 'temp', 'hdu.fits'), overwrite=True)
    mImgtbl(os.path.join('copt', 'temp'), tblname=os.path.join('copt', 'img.tbl'))
    mMakeHdr(tblfile=os.path.join('copt', 'img.tbl'), template=os.path.join('copt', 'hdr.hdr'),
             pixelScale=conv_res.to(u.deg).value)
    reproj_header = fits.Header.fromfile(os.path.join('copt', 'hdr.hdr'), sep='\n', endcard=False, padding=False)

    # Set up for a multi-extension FITS file

    hdu_reproj = fits.HDUList(hdu[0])

    for line in all_lines:

        print('Starting %s' % line)

        for line_ext in line_exts:
            data = hdu['%s_%s' % (line, line_ext)].data
            error = hdu['%s_%s_ERR' % (line, line_ext)].data
            header = hdu['%s_%s' % (line, line_ext)].header

            # Convolve and reproject
            data_reproj = convolve_regrid(data, conv_kernel, header, reproj_header)

            reproj_header_edit = reproj_header.copy()
            reproj_header_edit['EXTNAME'] = '%s_%s' % (line, line_ext)
            hdu_reproj.append(fits.ImageHDU(data=data_reproj, header=reproj_header_edit))

            # Convolve and reproject errors maps
            err_reproj = np.sqrt(convolve_regrid(error, conv_kernel, header, reproj_header))
            reproj_header_edit = reproj_header.copy()
            reproj_header_edit['EXTNAME'] = '%s_%s_ERR' % (line, line_ext)
            hdu_reproj.append(fits.ImageHDU(data=err_reproj, header=reproj_header_edit))

    hdu_reproj.writeto(os.path.join('conv', '%s_MAPS.fits' % galaxy.upper()), overwrite=True)

    # Clear out any hanging around Montage files
    os.system('rm -f %s' % os.path.join('copt', 'img.tbl'))
    os.system('rm -f %s' % os.path.join('copt', 'hdr.hdr'))
    os.system('rm -f %s' % os.path.join('copt', 'temp', 'hdu.fits'))

os.system('rm -rf %s' % os.path.join('copt', 'temp/'))

print('Complete!')
