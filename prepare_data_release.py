# -*- coding: utf-8 -*-
"""
Prepare files for a data release

@author: Tom Williams
"""

import os

from astropy.io import fits

from vars import top_dir, metallicity_dir, galaxies, nebulae_ver, muse_version, phangs_version, gpr_version, plot_dir, \
    hii_only, extinction_curve

os.chdir(os.path.join(top_dir, metallicity_dir))

metallicity_calib = 'pg16_scal'

output_dir = os.path.join('metallicity_gpr_releases', gpr_version)

if not os.path.exists(os.path.join(output_dir, 'diagnostics')):
    os.makedirs(os.path.join(output_dir, 'diagnostics'))

for galaxy in galaxies:

    if not os.path.exists(os.path.join(output_dir, 'diagnostics', galaxy)):
        os.makedirs(os.path.join(output_dir, 'diagnostics', galaxy))

    # TODO: Swap out switches

    hdu_in_name = os.path.join(muse_version, '%s_ext_curve' % extinction_curve, metallicity_calib,
                               '%s_%s_gpr_pix' % (galaxy, metallicity_calib))
    if hii_only:
        hdu_in_name += '_hii_only'
    hdu_in_name = os.path.join(hdu_in_name, '%s_positions_linear_radial.fits' % galaxy)

    if not os.path.exists(hdu_in_name):
        print('%s not found. Skipping' % galaxy)
        continue

    for hdu_name in [hdu_in_name, hdu_in_name.replace('.fits', '_err.fits')]:

        hdu = fits.open(hdu_name)[0]
        hdu.header['BUNIT'] = '12 + log(O/H)'
        hdu.header['AUTHOR'] = 'Thomas Williams <williams@mpia.de>'

        del hdu.header['EXTNAME']

        hdu_out_name = os.path.join(output_dir, galaxy + '_metallicity.fits')
        if '_err' in hdu_name:
            hdu_out_name = hdu_out_name.replace('.fits', '_err.fits')

        hdu.writeto(hdu_out_name, overwrite=True)

    # Also move diagnostic plots to a folder

    plot_in_name = os.path.join('..', plot_dir, muse_version, '%s_ext_curve' % extinction_curve, metallicity_calib,
                                '%s_%s_gpr_pix' % (galaxy, metallicity_calib))
    if hii_only:
        plot_in_name += '_hii_only'
    plot_in_name = os.path.join(plot_in_name, galaxy + '_positions_gpr.png')
    plot_out_name = os.path.join(output_dir, 'diagnostics', galaxy, galaxy + '_fit_overview.png')
    os.system('cp ' + plot_in_name + ' ' + plot_out_name)

    plot_in_name = plot_in_name.replace('_positions_gpr.png', '_positions_radial_comparison.png')
    plot_out_name = os.path.join(output_dir, 'diagnostics', galaxy, galaxy + '_scatter_radial.png')
    os.system('cp ' + plot_in_name + ' ' + plot_out_name)

    plot_in_name = plot_in_name.replace('_positions_radial_comparison.png', '_positions_residual_positions_sigma.png')
    plot_out_name = os.path.join(output_dir, 'diagnostics', galaxy, galaxy + '_residual_positions.png')
    os.system('cp ' + plot_in_name + ' ' + plot_out_name)

plot_in_name = os.path.join('..', plot_dir, muse_version, '%s_ext_curve' % extinction_curve, metallicity_calib,
                            'model_comparisons_%s_pix_maps' % metallicity_calib)
if hii_only:
    plot_in_name += '_hii_only'
plot_in_name += '.png'

plot_out_name = os.path.join(output_dir, 'diagnostics', 'model_comparisons.png')
os.system('cp ' + plot_in_name + ' ' + plot_out_name)

# Scale length table.

scale_length_in_name = os.path.join(muse_version, '%s_ext_curve' % extinction_curve,
                                    'scale_lengths_%s' % metallicity_calib)
if hii_only:
    scale_length_in_name += '_hii_only'
scale_length_in_name += '.fits'

scale_length_out_name = os.path.join(output_dir, 'scale_lengths.fits')
os.system('cp ' + scale_length_in_name + ' ' + scale_length_out_name)

# Finally, write a quick readme

readme = open(os.path.join(output_dir, 'readme.txt'), 'w+')

readme.write("This directory contains Gaussian Process Regression (GPR) smoothed metallicity maps (and associated\n"
             "errors), %s. These maps use the %s metallicity calibration. The maps themselves are named as\n"
             "'GALAXY_metallicity.fits', and the errors as 'GALAXY_metallicity_err.fits'. The units are 12+log(O/H).\n"
             "These maps have been generated using MUSE %s, and sample table %s. There are also some diagnostic plots\n"
             "for each galaxy that show the change in scatter radially with either a radial gradient, or the GPR\n"
             "fitting, a map of the residuals, and an overall overview of the map and error. There is also a summary\n"
             "plot for all galaxies that shows the model improvement (if any) of including the azimuthal term. If\n "
             "there are any questions, please feel free to email me at williams@mpia.de.\n" %
             (gpr_version, metallicity_calib, muse_version, phangs_version))

readme.write('\nCHANGES\n')
readme.write("v0.5: Moved to O'Donnell '94 extinction curve. Moved to restricting pixels within HII regions from the "
             "nebulae catalogues. (2021/07/22)\n")
readme.write('v0.4: Fixed double Galactic extinction correction. (2021/06/17)\n')
readme.write('v0.3: Updated to v1.6 of sample table. Updated to MUSE DR2.0. Uses common spatial maps for the GPR '
             'fitting. (2021/09/03)\n')
readme.write('v0.2: Updated to v1.5 of sample table (which means IC5332 is now included). Fixed important\n'
             'deprojection error. Includes some diagnostic plots. (2020/07/27)\n')
readme.write('v0.1: Initial release (2020/06/24)\n')

readme.close()

print('Complete!')
