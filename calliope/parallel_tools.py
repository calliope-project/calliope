"""
Copyright (C) 2013 Stefan Pfenninger.
Licensed under the Apache 2.0 License (see LICENSE file).

parallel_tools.py
~~~~~~~~~~~~~~~~~

Functions to process results from parallel runs created by parallel.py.

"""

from __future__ import print_function
from __future__ import division

import os

import pandas as pd

from . import utils


def read_dir(directory, files_to_read=['costs', 'system_variables',
                                       'node_parameters',
                                       'node_variables_e_power']):
    """Combines output files from `directory` and return an AttrDict
    containing them all.

    """
    results = utils.AttrDict()
    results.iterations = pd.read_csv(os.path.join(directory, 'iterations.csv'),
                                     index_col=0)
    for f in files_to_read:
        results[f] = utils.AttrDict()
        panel_items = {}
        for i in results.iterations.index:
            iteration_dir = '{:0>4d}'.format(i)
            src = os.path.join(directory, iteration_dir, f + '.csv')
            try:
                df = pd.read_csv(src, index_col=0, parse_dates=True)
            except IOError:
                results.iterations.at[i, 'IOError'] = 1
                continue
            # If 'minor' is in columns, we have a flattened panel!
            if 'minor' in df.columns:
                df['major'] = df.index
                df = df.set_index(['major', 'minor']).to_panel()
            panel_items[i] = df
        if panel_items:
            try:
                results[f] = pd.Panel(panel_items)
            except AttributeError:  # trying to make a panel of panels
                results[f] = pd.Panel4D(panel_items)
    return results


def reshape_results(results, table, iterations, column, row):
    """
    Reshape results

    NB: does not work for node_parameters table, as it is a Panel4D rather
    than a panel
    """
    return results['table'].loc[iterations, column, row]
