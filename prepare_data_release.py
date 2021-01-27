# -*- coding: utf-8 -*-
"""
Prepare files for a data release

@author: Tom Williams
"""

import os

from astropy.io import fits

from vars import top_dir, metallicity_dir, galaxies, nebulae_ver, muse_version, phangs_version, gpr_version, plot_dir

include_null_hypothesis = True

os.chdir(os.path.join(top_dir, metallicity_dir))

output_dir = os.path.join('metallicity_gpr_releases', gpr_version)

if not os.path.exists(os.path.join(output_dir, 'diagnostics')):
    os.makedirs(os.path.join(output_dir, 'diagnostics'))

if include_null_hypothesis and not os.path.exists(os.path.join(output_dir, 'null_hypothesis')):
    os.makedirs(os.path.join(output_dir, 'null_hypothesis'))

for galaxy in galaxies:

    hdu_in_name = os.path.join(muse_version, galaxy + '_gpr', galaxy + '_regions.fits')

    if not os.path.exists(hdu_in_name):
        print('%s not found. Skipping' % galaxy)
        continue

    for hdu_name in [hdu_in_name, hdu_in_name.replace('.fits', '_err.fits')]:

        hdu = fits.open(hdu_name)[0]
        hdu.header['BUNIT'] = '12 + log(O/H)'
        hdu.header['AUTHOR'] = 'Thomas Williams <williams@mpia.de>'

        del hdu.header['EXTNAME']

        hdu_out_name = os.path.join(output_dir, galaxy + '_metallicity_gpr.fits')
        if '_err' in hdu_name:
            hdu_out_name = hdu_out_name.replace('.fits', '_err.fits')

        hdu.writeto(hdu_out_name, overwrite=True)

    if include_null_hypothesis:
        hdu_in_name = os.path.join(muse_version, galaxy + '_gpr_null', galaxy + '_regions.fits')

        if not os.path.exists(hdu_in_name):
            print('%s null hypothesis not found. Skipping' % galaxy)
            continue

        for hdu_name in [hdu_in_name, hdu_in_name.replace('.fits', '_err.fits')]:

            hdu = fits.open(hdu_name)[0]
            hdu.header['BUNIT'] = '12 + log(O/H)'
            hdu.header['AUTHOR'] = 'Thomas Williams <williams@mpia.de>'

            del hdu.header['EXTNAME']

            hdu_out_name = os.path.join(output_dir, 'null_hypothesis', galaxy + '_metallicity_gpr.fits')
            if '_err' in hdu_name:
                hdu_out_name = hdu_out_name.replace('.fits', '_err.fits')

            hdu.writeto(hdu_out_name, overwrite=True)

    # Also move diagnostic plots to a folder

    plot_in_name = os.path.join('..', plot_dir, muse_version, galaxy + '_gpr',
                                galaxy + '_regions_radial_comparison.png')
    plot_out_name = os.path.join(output_dir, 'diagnostics', galaxy + '_radial.png')
    os.system('cp ' + plot_in_name + ' ' + plot_out_name)
    plot_in_name = os.path.join('..', plot_dir, muse_version, galaxy + '_gpr',
                                galaxy + '_regions_residual_positions.png')
    plot_out_name = os.path.join(output_dir, 'diagnostics', galaxy + '_position.png')
    os.system('cp ' + plot_in_name + ' ' + plot_out_name)

    plot_in_name = os.path.join('..', plot_dir, muse_version, 'model_comparisons.png')
    plot_out_name = os.path.join(output_dir, 'diagnostics', 'model_comparisons.png')
    os.system('cp ' + plot_in_name + ' ' + plot_out_name)

# Finally, write a quick readme

readme = open(os.path.join(output_dir, 'readme.txt'), 'w+')

readme.write("This directory contains Gaussian Process Regression (GPR) smoothed metallicity maps (and associated "
             "errors), %s. The maps themselves are named as 'galaxy_metallicity_gpr', and the errors and "
             "'galaxy_metallicity_gpr_err'. The units are 12 + log(O/H). These maps have been generated using "
             "MUSE %s, sample table %s and nebulae catalogue %s. There are also some diagnostic plots for each galaxy "
             "that show the change in scatter radially with either a radial gradient, or the GPR fitting, and a map of "
             "the residuals, as well as an overall summary plot for all galaxies with the ratio of chi-squares and "
             "Bayes factors. If there are any questions, please feel free to email me at williams@mpia.de.\n" %
             (gpr_version, muse_version, phangs_version, nebulae_ver))

readme.write('\nCHANGES\n')
readme.write('v0.2: Updated to v1.5 of sample table (which means IC5332 is now included). Fixed important deprojection '
             'error. Includes some diagnostic plots. (2020/07/27)\n')
readme.write('v0.1: Initial release (2020/06/24)\n')

readme.close()

print('Complete!')
