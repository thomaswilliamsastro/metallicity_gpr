# -*- coding: utf-8 -*-
"""
Useful variables used throughout code

@author: Tom Williams
"""

import socket

from astropy.table import Table

if socket.gethostname() == 'astro-node4':
    top_dir = '/data/beegfs/astro-storage/groups/schinnerer/williams/phangs'
else:
    top_dir = '/Users/williams/Documents/phangs'

muse_dir = 'muse/'
metallicity_dir = 'metallicity/'
plot_dir = 'plots/metallicity_gpr/'

nebulae_ver = 'v1'
hii_mask_dir = 'nebulae_catalogue/' + nebulae_ver + '/'

muse_version = 'DR1.0'
phangs_version = 'v1p5'
gpr_version = 'v0p2'

phangs_master_table = Table.read(top_dir + '/documents/phangs_sample_table_' + phangs_version + '.fits')

galaxies = ['IC5332', 'NGC1087', 'NGC1365', 'NGC1512', 'NGC1566', 'NGC1672', 'NGC2835', 'NGC3351', 'NGC3627', 'NGC4254',
            'NGC4535', 'NGC5068', 'NGC628']
