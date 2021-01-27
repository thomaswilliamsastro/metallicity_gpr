# -*- coding: utf-8 -*-
"""
Useful functions

@author: Tom Williams
"""

import numpy as np


def add_master_table_info(master_table, galaxy_dict, galaxy, cols_to_include, original_galaxy_name=None):
    if not original_galaxy_name:
        original_galaxy_name = galaxy

    for col_to_include in cols_to_include:
        idx = np.where(master_table['name'] == galaxy)[0][0]

        galaxy_dict[original_galaxy_name][col_to_include] = master_table[col_to_include][idx]

        # Also include errors if available

        if col_to_include + '_unc' in master_table.colnames:
            galaxy_dict[original_galaxy_name][col_to_include + '_err'] = \
                master_table[col_to_include + '_unc'][idx]

    return galaxy_dict


def get_info_from_master_table(table, galaxy, params):
    row = table[table['name'] == galaxy]

    return [row[param][0] for param in params]
