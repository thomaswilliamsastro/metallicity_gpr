# -*- coding: utf-8 -*-
"""
TeX up the galaxy parameters

@author: Tom Williams
"""

import os

from astropy.table import Table
import numpy as np

from vars import top_dir, metallicity_dir, phangs_master_table, galaxies, muse_version, extinction_curve, hii_only
from funcs import get_info_from_master_table

os.chdir(top_dir)

metallicity_calib = 'pg16_scal'

filling_tab_file_name = os.path.join(metallicity_dir, muse_version, '%s_ext_curve' % extinction_curve,
                                     'filling_factors_%s' % metallicity_calib)
if hii_only:
    filling_tab_file_name += '_hii_only'
filling_tab_file_name += '.fits'
filling_tab = Table.read(filling_tab_file_name)

tex_table = open(os.path.join(metallicity_dir, 'galaxy_params.tex'), 'w+')

params = ['dist', 'orient_posang', 'orient_incl', 'size_r25', 'size_reff', 'props_mstar', 'props_mstar_unc', 'props_sfr',
          'props_sfr_unc', 'morph_bar_r']

tex_table.write('\\begin{table*}\n')
tex_table.write('\\caption{Key parameters for the 19 galaxies. Galaxies with a bar are marked with an asterisk. '
                'Distances are taken from \\citet{2021Anand}, PA and inclination from \\citet{2020Lang}, and sizes, '
                'global stellar masses and global SFRs from \\citet{2021Leroy}. The uncertainty in stellar mass is '
                'dominated by calibration uncertainty, so is always 0.11~dex for these galaxies. We also include '
                'the number of pixels in our convolved and regridded maps, and the number of pixels that have '
                'metallicity measurements after various cuts (see Sect. \\ref{sec:metallicity_calculation}).}\n')
tex_table.write('\\label{tab:galaxy_params}\n')
tex_table.write('\\begin{tabular}{lccccccccr}\n')
tex_table.write('\\hline\\hline\n')
tex_table.write('Galaxy & Dist (Mpc) & PA (deg) & $i$ (deg) & $r_{25}$ (arcmin) & $R_e$ (arcmin) &  '
                '$\log_{10}(M_\\ast [M_\\odot])$ & SFR ($M_\\odot\\,{\\rm yr^{-1}}$) & '
                '$N_{\\rm pix}$ & $N_Z$ \\\\\n')
tex_table.write('\\hline\n')

for galaxy in galaxies:

    dist, pa, inc, r25, re, mstar, mstar_unc, sfr, sfr_unc, r_bar = get_info_from_master_table(phangs_master_table,
                                                                                           galaxy.lower(),
                                                                                           params)

    idx = np.where(filling_tab['GALAXY'] == galaxy)[0]
    n_pix = filling_tab['N_PIX'][idx][0]
    n_z = filling_tab['N_Z'][idx][0]

    if ~np.isnan(r_bar):
        galaxy += '$^\\ast$'

    tex_table.write('%s & %s & %.1f & %.2f & %.2f & %.2f & $%.2f\pm%.2f$ & $%.2f\pm%.2f$ & %d & %d\\\\\n' %
                    (galaxy, dist, pa, inc, r25/60, re/60, np.log10(mstar), 0.434 * (mstar_unc/mstar), sfr, sfr_unc,
                     n_pix, n_z))

tex_table.write('\\hline\n')
tex_table.write('\\end{tabular}\n')
tex_table.write('\\end{table*}')

tex_table.close()

print('Complete!')
