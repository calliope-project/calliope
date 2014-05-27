"""
Copyright (C) 2013 Stefan Pfenninger.
Licensed under the Apache 2.0 License (see LICENSE file).

parallel_tools.py
~~~~~~~~~~~~~~~~~

Functions to process results from parallel runs created by parallel.py.

"""

from __future__ import print_function
from __future__ import division

import glob
import os

import pandas as pd

from . import utils


def read_hdf(hdf_file, tables_to_read=None):
    """Read model solution from HDF file"""
    store = pd.HDFStore(hdf_file, mode='r')
    solution = utils.AttrDict()
    if not tables_to_read:
        # Make sure leading/trailing '/' are removed from keys
        tables_to_read = [k.strip('/') for k in store.keys()]
    for k in tables_to_read:
        solution[k] = store.get(k)
    store.close()
    # Also add model and run config to the solution object
    solution['config_run'] = utils.AttrDict.from_yaml(hdf_file
                                                      + '.config_run.yaml')
    solution['config_model'] = utils.AttrDict.from_yaml(hdf_file
                                                        + '.config_model.yaml')
    return solution


def read_csv(directory, tables_to_read=None):
    solution = utils.AttrDict()
    if not tables_to_read:
        tables_to_read = glob.glob(directory + '/*.csv')
        # Only keep basenames without extension
        tables_to_read = [os.path.splitext(os.path.basename(f))[0]
                          for f in tables_to_read]
    for f in tables_to_read:
        src = os.path.join(directory, f + '.csv')
        df = pd.read_csv(src, index_col=0, parse_dates=True)
        # If 'minor' is in columns, we have a flattened panel!
        if 'minor' in df.columns:
            df['major'] = df.index
            df = df.set_index(['major', 'minor']).to_panel()
        solution[f] = df
    return solution


def _detect_format(directory):
    if os.path.exists(os.path.join(directory, 'solution.hdf')):
        return 'hdf'
    else:
        return 'csv'


def read_dir(directory, tables_to_read=None, verbose=False):
    """Combines output files from `directory` and return an AttrDict
    containing them all.

    """
    results = utils.AttrDict()
    results.iterations = pd.read_csv(os.path.join(directory, 'iterations.csv'),
                                     index_col=0)
    results.solutions = utils.AttrDict()
    for i in results.iterations.index.tolist():
        iteration_dir = os.path.join(directory, '{:0>4d}'.format(i))
        fmt = _detect_format(iteration_dir)
        if verbose:
            print('Iteration: {}, Format detected: {}'.format(i, fmt))
        try:
            if fmt == 'hdf':
                hdf_file = os.path.join(iteration_dir, 'solution.hdf')
                if verbose:
                    print('Reading: {}'.format(hdf_file))
                results.solutions[i] = read_hdf(hdf_file, tables_to_read)
            else:
                if verbose:
                    print('Reading: {}'.format(iteration_dir))
                results.solutions[i] = read_csv(iteration_dir, tables_to_read)
        except IOError:
            if verbose:
                print('I/O error at iteration: {}'.format(i))
            results.iterations.at[i, 'IOError'] = 1
            continue
    return results


def reshape_results(results, table, iterations, column, row):
    """
    Reshape results

    NB: does not work for node_parameters table, as it is a Panel4D rather
    than a panel
    """
    return results['table'].loc[iterations, column, row]
