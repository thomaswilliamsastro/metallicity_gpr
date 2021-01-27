# -*- coding: utf-8 -*-
"""
Useful variables used throughout code

@author: Tom Williams
"""

import os
import socket

from astropy.table import Table

if socket.gethostname() == 'astro-node4':
    top_dir = '/data/beegfs/astro-storage/groups/schinnerer/williams/phangs'
else:
    top_dir = '/Users/williams/Documents/phangs'

muse_dir = 'muse'
metallicity_dir = 'metallicity'
plot_dir = os.path.join('plots', 'metallicity_gpr')

nebulae_ver = 'v1'
hii_mask_dir = os.path.join('nebulae_catalogue/', nebulae_ver)

muse_version = 'DR1.0'
phangs_version = 'v1p5'
gpr_version = 'v0p2'

phangs_table_name = os.path.join(top_dir, 'documents', 'phangs_sample_table_' + phangs_version + '.fits')
phangs_master_table = Table.read(phangs_table_name)

if muse_version == 'DR1.0':
    galaxies = ['IC5332', 'NGC1087', 'NGC1365', 'NGC1512', 'NGC1566', 'NGC1672', 'NGC2835', 'NGC3351', 'NGC3627',
                'NGC4254', 'NGC4535', 'NGC5068', 'NGC628']
elif muse_version == 'DR2.0':
    galaxies = ['IC5332', 'NGC0628', 'NGC1087', 'NGC1300', 'NGC1365', 'NGC1385', 'NGC1433', 'NGC1512', 'NGC1566',
                'NGC1672', 'NGC2835', 'NGC3351', 'NGC3627', 'NGC4254', 'NGC4303', 'NGC4321', 'NGC4535', 'NGC5068',
                'NGC7496']
else:
    raise Warning('Unknown MUSE version %s' % muse_version)

galaxies = sorted(galaxies)

# Calibrations available are pg16_scal, d16

metallicity_calib = 'pg16_scal'

# Various settings for type of GPR

gpr_step_val = 1
# use_radial_gpr = True
use_pix_maps = True
use_regions = False
include_radial_subtract = True
simulation = False
