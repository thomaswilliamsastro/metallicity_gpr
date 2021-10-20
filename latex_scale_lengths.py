# -*- coding: utf-8 -*-
"""
TeX up the scale length table

@author: Tom Williams
"""

import os

from astropy.table import Table

from vars import top_dir, metallicity_dir, muse_version, extinction_curve, hii_only

os.chdir(top_dir)

metallicity_calib = 'pg16_scal'

scale_length_file_name = os.path.join(metallicity_dir, muse_version, '%s_ext_curve' % extinction_curve,
                                      'scale_lengths_%s' % metallicity_calib)
if hii_only:
    scale_length_file_name += '_hii_only'
scale_length_file_name += '.fits'
scale_length_table = Table.read(scale_length_file_name)

tp_file_name = os.path.join(metallicity_dir, muse_version, '%s_ext_curve' % extinction_curve,
                            'two_point_corr_%s' % metallicity_calib)
if hii_only:
    tp_file_name += '_hii_only'
tp_file_name += '.fits'
tp_table = Table.read(tp_file_name)

rows = scale_length_table[scale_length_table['SIGNIFICANT'] == 1]
n_galaxies = len(rows)

tex_table = open(os.path.join(metallicity_dir, muse_version, '%s_ext_curve', 'scale_lengths_%s.tex') %
                 (extinction_curve, metallicity_calib), 'w+')

tex_table.write('\\begin{table}\n')
tex_table.write('\\caption{Corrected kernel scale lengths ($\\sigma_l$), and 50~per~cent two-point correlation scale for the ' +
                str(n_galaxies) +
                ' galaxies with significant azimuthal variation.}\n')
tex_table.write('\\label{tab:scale_lengths_corrected}\n')
tex_table.write('\\begin{tabular}{lcr}\n')
tex_table.write('\\hline\\hline\n')
tex_table.write('Galaxy & $\sigma_l$ (kpc) & 50~per~cent Correlation Scale (kpc) \\\\\n')
tex_table.write('\\hline\n')

for row in rows:

    tp_row = tp_table[tp_table['GALAXY'] == row['GALAXY']]
    # print(tp_row)

    tex_table.write('%s & $%.2f^{+%.2f}_{-%.2f}$ & $%.2f^{+%.2f}_{-%.2f}$\\\\\n' %
                    (row['GALAXY'],
                     row['SIGMA_L'], row['SIGMA_L_ERR_UP'], row['SIGMA_L_ERR_DOWN'],
                     tp_row['TWO_POINT_CORR'], tp_row['TWO_POINT_CORR_ERR_UP'], tp_row['TWO_POINT_CORR_ERR_DOWN']))

tex_table.write('\\hline\n')
tex_table.write('\\end{tabular}\n')
tex_table.write('\\end{table}')

tex_table.close()

print('Complete!')
