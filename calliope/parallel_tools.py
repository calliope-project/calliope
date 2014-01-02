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


def read_dir(directory, files_to_read=['costs', 'overall', 'plant_parameters',
                                       'plants_output']):
    """Combines output files from `directory` and return an AttrDict containing
    them all.

    """
    results = utils.AttrDict()
    results.iterations = pd.read_csv(os.path.join(directory, 'iterations.csv'),
                                     index_col=0)
    for f in files_to_read:
        results[f] = utils.AttrDict()
        for i in results.iterations.index:
            iteration_dir = '{:0>4d}'.format(i)
            src = os.path.join(directory, iteration_dir, f + '.csv')
            results[f][i] = pd.read_csv(src, index_col=0)
            # TODO if 'minor' and 'major' in columns it was a panel, convert
            # it back to panel?
    return results


def _get_index_lookup(results, x, y, z=None):
    """
    If x, y given, returns DataFrame with x rows and y columns
    If x, y, z given, returns Panel with z items, x on the major_axis and
    y on the minor_axis

    """
    index_lookup = results.iterations
    index_lookup['idx'] = index_lookup.index
    if z:
        zs = results.iterations[z].unique()
        il = {zi: index_lookup[index_lookup[z] == zi].pivot_table(values='idx',
              rows=x, cols=y) for zi in zs}
        index_lookup = pd.Panel(il)
    else:
        index_lookup = index_lookup.pivot_table(values='idx', rows=x, cols=y)
    return index_lookup


def reshape_results(results, table='costs', x_axis='override.noncsp_avail',
                    y_axis='input.demand', items=None, value='lcoe'):
    """Reshape `results`, returning either a DataFrame with `x_axis` and
    `y_axis`, or a panel with the same axes as well as `items`.

    Currently only properly works for table=='costs'.

    Other args:
        value : can either be 'lcoe' or 'cf'

    """
    x_axis_values = results.iterations[x_axis].unique()
    y_axis_values = results.iterations[y_axis].unique()
    if not items:
        items_values = [0]
    else:
        items_values = results.iterations[items].unique()
    p = {}
    index_lookup = _get_index_lookup(results, x=x_axis, y=y_axis, z=items)
    for s in items_values:
        df = {}
        for x in x_axis_values:
            df[x] = {}
            for l in y_axis_values:
                if not items:
                    i = index_lookup[l][x]
                else:
                    i = index_lookup[s][l][x]
                df[x][l] = results[table][i][value]['total']
        if not items:
            return pd.DataFrame(df).T
        else:
            p[s] = pd.DataFrame(df).T
    return pd.Panel(p)
