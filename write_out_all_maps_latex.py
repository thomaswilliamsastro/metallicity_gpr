# -*- coding: utf-8 -*-
"""
Put all the maps into an appendix

@author: Tom Williams
"""

import os

from vars import top_dir, metallicity_dir, plot_dir, phangs_version, muse_version, galaxies, extinction_curve, hii_only

os.chdir(top_dir)

metallicity_calib = 'pg16_scal'
output_dir = os.path.join(plot_dir, 'appendix_plots')

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for galaxy in galaxies:

    in_plot = os.path.join(plot_dir, muse_version, '%s_ext_curve' % extinction_curve, metallicity_calib,
                           '%s_%s_gpr_pix' % (galaxy, metallicity_calib))
    if hii_only:
        in_plot += '_hii_only'
    in_plot = os.path.join(in_plot, galaxy + '_positions_gpr.pdf')
    out_plot = os.path.join(output_dir, galaxy + '_fit_overview.pdf')

    os.system('cp ' + in_plot + ' ' + out_plot)

f = open(os.path.join(output_dir, 'fit_overview.tex'), 'w+')

for galaxy in galaxies:

    f.write('\\begin{figure*}\n')
    f.write('\\includegraphics[width=2\columnwidth]{overview_figs/%s_fit_overview.pdf}\n' % galaxy)
    f.write('\\caption{As Fig. \\ref{fig:gpr_fit}, but for %s}\n' % galaxy)
    f.write('\\end{figure*}\n')
    f.write('\\newpage\n')

f.close()

print('Complete!')
