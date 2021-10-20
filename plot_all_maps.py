# -*- coding: utf-8 -*-
"""
Nice plot of all the metallicity maps

@author: Tom Williams
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
import cmocean
import matplotlib

from vars import top_dir, metallicity_dir, galaxies, gpr_version, plot_dir

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['font.size'] = 14

os.chdir(top_dir)

use_regions = True

n_col = int(np.ceil(np.sqrt(len(galaxies))))

plot_name = os.path.join(plot_dir, 'all_maps')

plt.figure(figsize=(12, 8))

for n_subplot, galaxy in enumerate(galaxies):

    hdu_file_name = os.path.join(metallicity_dir, 'metallicity_gpr_releases', gpr_version,
                                 galaxy + '_metallicity_gpr.fits')

    try:
        hdu = fits.open(hdu_file_name)[0]
    except FileNotFoundError:
        print('%s not found. Skipping' % galaxy)
        continue

    v_min = np.nanpercentile(hdu.data, 5)
    v_max = np.nanpercentile(hdu.data, 95)

    ax = plt.subplot(n_col, n_col, n_subplot + 1)
    plt.imshow(hdu.data, origin='lower', cmap=cmocean.cm.haline, vmin=v_min, vmax=v_max)
    plt.colorbar(label='12 + log(O/H)')
    plt.axis('off')

    plt.text(0.95, 0.95, galaxy, va='top', ha='right', fontweight='bold', transform=ax.transAxes,
             bbox=dict(facecolor='white', edgecolor='black', alpha=0.75))

plt.subplots_adjust(wspace=0, hspace=0)
plt.tight_layout()

# plt.show()

plt.savefig(plot_name + '.png', bbox_inches='tight')
plt.savefig(plot_name + '.pdf', bbox_inches='tight')

plt.close()
