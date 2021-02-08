# -*- coding: utf-8 -*-
"""
Create pixel-by-pixel metallicity maps, rather than from nebulae catalogues

@author: Tom Williams
"""

import os
import warnings
import time

from astropy.io import fits
from astroquery.irsa_dust import IrsaDust
import matplotlib.pyplot as plt
import numpy as np
from uncertainties import ufloat
from uncertainties import unumpy as unp

from vars import top_dir, muse_dir, muse_version, galaxies, metallicity_dir, metallicity_calib, all_lines, strong_lines, \
    use_conv

warnings.simplefilter('ignore')

os.chdir(top_dir)

overwrite = True

hdu_out_dir = os.path.join(metallicity_dir, 'pix_maps', muse_version)
if not os.path.exists(hdu_out_dir):
    os.makedirs(hdu_out_dir)

k_lambda = {'HB4861': 5.252,
            'OIII4958': 3.886,
            'OIII5006': 3.755,
            'NII6548': 2.728,
            'HA6562': 2.720,
            'NII6583': 2.710,
            'SII6716': 2.644,
            'SII6730': 2.637}

s_n_ratio = 5  # 3
vel_cutoff = 100

galaxies = ['IC5332']
flux_to_use = 'HA6562_FLUX'

for galaxy in galaxies:

    hdu_out_name = os.path.join(hdu_out_dir, galaxy + '_metallicity_' + metallicity_calib + '.fits')

    if not os.path.exists(hdu_out_name) or overwrite:

        print('Beginning %s' % galaxy)

        start = time.time()

        maps = {}

        muse_hdu_file_name = os.path.join(muse_dir, muse_version)
        if use_conv:
            muse_hdu_file_name += '/conv/'
        muse_hdu_file_name += galaxy + '_MAPS.fits'
        muse_hdu = fits.open(muse_hdu_file_name)

        # for hdu in muse_hdu[1:]:
        #     print(hdu.header['EXTNAME'])
        # no

        # We begin the mask. First up, any NaNs in the original image. 1 for pixels we include, 0 for ones we exclude

        mask = np.zeros_like(muse_hdu[flux_to_use].data)
        mask[np.isnan(muse_hdu[flux_to_use].data)] = 1

        # Start by calculating an extinction correction. We first want an E(B-V) to correct for Galactic extinction

        query = IrsaDust.get_query_table(galaxy, section='ebv')
        ebv = ufloat(float(query['ext SandF mean']), float(query['ext SandF std']))

        for line in all_lines:

            if line in ['OIII4958', 'NII6548']:
                stronger_line = {'OIII4958': 'OIII5006',
                                 'NII6548': 'NII6583'}[line]
                maps[line + '_uncorr'] = maps[stronger_line + '_uncorr'] / 3
            else:

                data = muse_hdu[line + '_FLUX'].data
                err = muse_hdu[line + '_FLUX_ERR'].data

                idx = np.where(data == 0)
                data[idx] = np.nan
                err[idx] = np.nan

                maps[line + '_uncorr'] = unp.uarray(data, err)

        print('%.1f: Initial arrays created' % (time.time() - start))

        c_hbeta = (np.log10(2.87) - unp.log10(maps['HA6562_uncorr'] / maps['HB4861_uncorr'])) / -0.466
        c_hbeta[c_hbeta <= 0] = 0
        c_hbeta[c_hbeta > 3] = np.nan

        print('%.1f: Balmer decrement calculated' % (time.time() - start))

        # Correct all the lines

        for line in all_lines:

            k = k_lambda[line]

            if line in ['OIII4958', 'NII6548']:
                stronger_line = {'OIII4958': 'OIII5006',
                                 'NII6548': 'NII6583'}[line]
                maps[line] = maps[stronger_line] / 3
                # maps[line + '_err'] = maps[stronger_line + '_err'] / 3
            else:

                maps[line] = maps[line + '_uncorr'] * 10 ** (0.4 * (ebv + c_hbeta) * k)

                # maps[line] = muse_hdu[line + '_FLUX'].data * 10 ** (0.4 * (ebv + c_hbeta) * k)
                # maps[line + '_err'] = muse_hdu[line + '_FLUX_ERR'].data * 10 ** (0.4 * (ebv + c_hbeta) * k)

        print('%.1f: Lines corrected' % (time.time() - start))

        # Take only the lines with S/N > cutoff, and low velocity dispersions

        for line in strong_lines:
            # s_n_mask = np.where(maps[line] < s_n_ratio * maps[line + '_err'])
            s_n_mask = np.where(unp.nominal_values(maps[line]) < s_n_ratio * unp.std_devs(maps[line]))
            mask[s_n_mask] = 1

            vel_disp_mask = np.where(muse_hdu[line + '_SIGMA'].data >= vel_cutoff)
            mask[vel_disp_mask] = 1

        print('%.1f: S/N and vel disp mask created' % (time.time() - start))

        # Now correct via BPT plots. First, [NII]/Halpha versus [OIII]/Hbeta

        n2_ha = unp.log10((maps['NII6548'] + maps['NII6583']) / maps['HA6562'])
        oiii_hb = unp.log10((maps['OIII4958'] + maps['OIII5006']) / maps['HB4861'])

        bpt_mask = np.where((oiii_hb >= 0.61 / (n2_ha - 0.05) + 1.3) | (n2_ha >= 0.05))
        mask[bpt_mask] = 1

        # Second, [SII]/Halpha versus [OIII]/Hbeta

        s2_ha = unp.log10((maps['SII6716'] + maps['SII6730']) / maps['HA6562'])
        oiii_hb = unp.log10((maps['OIII4958'] + maps['OIII5006']) / maps['HB4861'])

        bpt_mask = np.where((oiii_hb >= 0.72 / (s2_ha - 0.32) + 1.3) | (s2_ha >= 0.32))
        mask[bpt_mask] = 1

        print('%.1f: BPT mask created' % (time.time() - start))

        # We're now good to calculate metallicities.

        metallicity = np.zeros_like(muse_hdu['HB4861_FLUX'].data)
        metallicity[metallicity == 0] = np.nan
        metallicity_err = metallicity.copy()

        if metallicity_calib == 'pg16_scal':
            # PG '16 Scal metallicity

            n2 = (maps['NII6583'] + maps['NII6548']) / maps['HB4861']
            s2 = (maps['SII6716'] + maps['SII6730']) / maps['HB4861']
            r3 = (maps['OIII5006'] + maps['OIII4958']) / maps['HB4861']

            # Calculate upper and lower branches

            upper = np.where(unp.log10(n2) >= -0.6)
            lower = np.where(unp.log10(n2) < -0.6)

            lower_metallicity = 8.072 + 0.789 * unp.log10(r3[lower] / s2[lower]) + 0.726 * unp.log10(n2[lower]) + \
                                (1.069 - 0.17 * unp.log10(r3[lower] / s2[lower]) + 0.022 * unp.log10(
                                    n2[lower])) * unp.log10(s2[lower])

            metallicity[lower] = unp.nominal_values(lower_metallicity)
            metallicity_err[lower] = unp.std_devs(lower_metallicity)

            upper_metallicity = 8.424 + 0.03 * unp.log10(r3[upper] / s2[upper]) + 0.751 * unp.log10(n2[upper]) + \
                                (-0.349 + 0.182 * unp.log10(r3[upper] / s2[upper]) + 0.508 * unp.log10(
                                    n2[upper])) * unp.log10(s2[upper])

            metallicity[upper] = unp.nominal_values(upper_metallicity)
            metallicity_err[upper] = unp.std_devs(upper_metallicity)

        elif metallicity_calib == 'd16':
            # Dopita '16 metallicity, but using solar metallicity of 12+log(O/H) = 8.69, as opposed to the 8.77 used in
            # the paper
            z_sol = 8.69

            n2_s2 = unp.log10((maps['NII6583'] + maps['NII6548']) / (maps['SII6716'] + maps['SII6730']))
            n2_ha = unp.log10((maps['NII6583'] + maps['NII6548']) / maps['HA6562'])

            y = n2_s2 + 0.264 * n2_ha
            Z = z_sol + y + 0.45 * (y + 0.3) ** 5
            metallicity = unp.nominal_values(Z)
            metallicity_err = unp.std_devs(Z)

        else:
            raise Warning('Metallicity calibration %s not recognised' % metallicity_calib)

        metallicity[mask == 1] = np.nan
        metallicity_err[mask == 1] = np.nan

        print('%.1f: Metallicity map calculated' % (time.time() - start))

        header = muse_hdu[flux_to_use].header
        fits.writeto(hdu_out_name, metallicity, header, overwrite=True)
        fits.writeto(hdu_out_name.replace('.fits', '_err.fits'), metallicity_err, header, overwrite=True)

print('Complete!')
