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

from vars import top_dir, metallicity_dir, galaxies, gpr_version

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['font.size'] = 14

os.chdir(top_dir)

use_regions = True

n_col = int(np.ceil(np.sqrt(len(galaxies))))

plt.figure(figsize=(n_col * 2, n_col * 2))

for n_subplot, galaxy in enumerate(galaxies):

    hdu_file_name = os.path.join(metallicity_dir, 'metallicity_gpr_releases', gpr_version,
                                 galaxy + '_metallicity_gpr.fits')

    try:
        hdu = fits.open(hdu_file_name)[0]
    except FileNotFoundError:
        print('%s not found. Skipping' % galaxy)
        continue

    v_min = np.nanpercentile(hdu.data, 2)
    v_max = np.nanpercentile(hdu.data, 98)

    ax = plt.subplot(n_col, n_col, n_subplot + 1)
    plt.imshow(hdu.data, origin='lower', cmap=cmocean.cm.haline, vmin=v_min, vmax=v_max)
    plt.axis('off')

    plt.text(0.95, 0.95, galaxy, va='top', ha='right', fontweight='bold', transform=ax.transAxes,
             bbox=dict(facecolor='white', edgecolor='black', alpha=0.75))

plt.tight_layout()

plt.show()
