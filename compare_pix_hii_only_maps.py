# -*- coding: utf-8 -*-
"""
Look at how different the all pixels/HII regions only maps are

@author: Tom Williams
"""

import os
import socket

from astropy.io import fits
import matplotlib.pyplot as plt
from astropy.table import Table
import numpy as np
import cmocean
import matplotlib

from vars import top_dir, muse_version, metallicity_dir, plot_dir, extinction_curve

if 'mac' in socket.gethostname():
    os.environ['PATH'] += ':/Library/TeX/texbin'

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['text.usetex'] = True
plt.rcParams.update({'text.latex.preamble': r'\usepackage{txfonts}'})

os.chdir(top_dir)

metallicity_calib = 'pg16_scal'

map_dir = os.path.join(metallicity_dir, muse_version, metallicity_calib)

scale_length_file_name = os.path.join(metallicity_dir, muse_version, '%s_ext_curve' % extinction_curve,
                                      'scale_lengths_%s.fits' % metallicity_calib)
scale_length_table = Table.read(scale_length_file_name)

galaxies = scale_length_table[scale_length_table['SIGNIFICANT'] == 1]['GALAXY']

plot_name = os.path.join(plot_dir, muse_version, '%s_ext_curve' % extinction_curve,
                         '%s_pix_hii_only_sigma_maps' % metallicity_calib)

fig = plt.figure(figsize=(12, 6))

vmin, vmax = -2.5, 2.5  # 0.95, 1.05

for i, galaxy in enumerate(galaxies):

    pix_map_file = os.path.join(map_dir, '%s_%s_gpr_pix' % (galaxy, metallicity_calib),
                                '%s_positions_linear_radial.fits' % galaxy)

    pix_map = fits.open(pix_map_file)[0].data
    pix_err = fits.open(pix_map_file.replace('.fits', '_err.fits'))[0].data

    hii_map_file = os.path.join(map_dir, '%s_%s_gpr_pix_hii_only' % (galaxy, metallicity_calib),
                                '%s_positions_linear_radial.fits' % galaxy)

    hii_map = fits.open(hii_map_file)[0].data
    hii_err = fits.open(hii_map_file.replace('.fits', '_err.fits'))[0].data

    # residual = (pix_map - 12) / (hii_map - 12) # / (pix_map - 12) * 100

    # Look at the sigma difference
    residual = (pix_map - hii_map) / np.sqrt(pix_err ** 2 + hii_err ** 2)

    ax = plt.subplot(3, 5, i + 1)
    ax_im = ax.imshow(residual, cmap=cmocean.cm.balance, vmin=vmin, vmax=vmax, origin='lower')

    plt.axis('off')

    plt.text(0.05, 0.95, galaxy,
             ha='left', va='top',
             bbox=dict(facecolor='white', edgecolor='black', alpha=0.8),
             transform=ax.transAxes)

    if i == 0:
        cbarax = fig.add_axes([0.1, 0.025, 0.8, 0.025])
        cbar = plt.colorbar(ax_im, cax=cbarax, orientation='horizontal')
        # cbar.set_label(r'$\frac{\log_{10}(\mathrm{O/H})_{\rm pix} - \log_{10}(\mathrm{O/H})_{\rm HII}}'
        #                r'{\log_{10}(\mathrm{O/H})_\mathrm{pix}}$ (\%)')
        # cbar.set_label(r'$\frac{\log_{10}(\mathrm{O/H})_{\rm pix}}{\log_{10}(\mathrm{O/H})_{\rm HII}}$')
        cbar.set_label(r'$\frac{\log_{10}(\mathrm{O/H})_{\rm pix} - \log_{10}(\mathrm{O/H})_{\rm HII}}'
                       r'{\sqrt{\sigma_{\log_{10}(\mathrm{O/H})_{\rm pix}}^2 + \sigma_{\log_{10}(\mathrm{O/H})_{\rm HII}}^2}}$')

# plt.show()

plt.savefig(plot_name + '.pdf', bbox_inches='tight')
plt.savefig(plot_name + '.png', bbox_inches='tight')

plt.close()

print('Complete!')
