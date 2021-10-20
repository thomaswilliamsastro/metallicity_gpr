# -*- coding: utf-8 -*-
"""
Compare metallicity prescriptions on the pixel-by-pixel maps

@author: Tom Williams
"""

import os

import cmocean
import numpy as np
from astropy.io import fits
import matplotlib
import matplotlib.pyplot as plt

from vars import top_dir, muse_version, metallicity_dir, plot_dir, muse_version

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['font.size'] = 14

os.chdir(top_dir)

hdu_dir = os.path.join(metallicity_dir, 'pix_maps', muse_version)

metallicity_calibs = ['pg16_scal', 'd16']
if len(metallicity_calibs) > 2:
    raise Warning('Can only compare two calibrations')

galaxies = ['IC5332']

calib_name_dict = {'pg16_scal': "PG16, Scal",
                   'd16': "D16"}

for galaxy in galaxies:

    galaxy_plot_dir = os.path.join(plot_dir, muse_version, galaxy)
    if not os.path.exists(galaxy_plot_dir):
        os.makedirs(galaxy_plot_dir)
    plot_name = os.path.join(galaxy_plot_dir,
                             galaxy + '_' + metallicity_calibs[0] + '_' + metallicity_calibs[1] + '_comparison')

    maps = {}
    maps_err = {}

    for metallicity_calib in metallicity_calibs:
        hdu_name = os.path.join(hdu_dir, galaxy + '_metallicity_' + metallicity_calib + '.fits')
        maps[metallicity_calib] = fits.open(hdu_name)[0].data
        maps_err[metallicity_calib] = fits.open(hdu_name.replace('.fits', '_err.fits'))[0].data

    map_ratio = maps[metallicity_calibs[0]] - maps[metallicity_calibs[1]]
    vmin = np.nanpercentile(map_ratio, 5)
    xmin, xmax = np.nanmin(maps[metallicity_calibs[0]]), np.nanmax(maps[metallicity_calibs[1]]) # np.nanpercentile(maps[metallicity_calibs[0]], [0.1, 99.9])

    calib_fancy_names = [calib_name_dict[name] for name in metallicity_calibs]

    plt.figure(figsize=(8, 4))
    plt.suptitle(galaxy)

    ax = plt.subplot(1, 2, 1)
    plt.imshow(map_ratio, cmap=cmocean.cm.balance, origin='lower', vmin=vmin, vmax=-vmin)
    plt.colorbar(ax=[ax], location='left', pad=0,
                 label=r'$\log_{10}\frac{\mathrm{O/H}_\mathrm{%s}}{\mathrm{O/H}_\mathrm{%s}}$'
                       % (calib_fancy_names[0], calib_fancy_names[1]))
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.errorbar(maps[metallicity_calibs[0]].flatten(), maps[metallicity_calibs[1]].flatten(),
                 xerr=maps_err[metallicity_calibs[0]].flatten(), yerr=maps_err[metallicity_calibs[1]].flatten(),
                 ls='none', marker='o', c='k')
    plt.plot([xmin, xmax], [xmin, xmax], c='r', ls='--', zorder=99)
    plt.xlim(xmin, xmax)
    plt.ylim(xmin, xmax)
    plt.xlabel(r'12 + log(O/H)$_\mathrm{%s}$' % calib_fancy_names[0])
    plt.ylabel(r'12 + log(O/H)$_\mathrm{%s}$' % calib_fancy_names[1])

    # plt.tight_layout()
    plt.show()

    plt.savefig(plot_name + '.png', bbox_inches='tight')
    plt.savefig(plot_name + '.pdf', bbox_inches='tight')

    plt.close()
