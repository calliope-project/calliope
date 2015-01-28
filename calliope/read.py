"""
Copyright (C) 2013-2015 Stefan Pfenninger.
Licensed under the Apache 2.0 License (see LICENSE file).

read.py
~~~~~~~

Functions to read saved model results.

"""

import glob
import logging
import os

import pandas as pd

from . import utils


REQUIRED_KEYS = ['capacity_factor', 'costs', 'levelized_cost',
                 'metadata', 'node', 'parameters',
                 'shares', 'summary', 'time_res', 'totals', 'config_model',
                 'config_run']


def read_hdf(hdf_file, tables_to_read=None):
    """Read model solution from HDF file"""
    store = pd.HDFStore(hdf_file, mode='r')
    solution = utils.AttrDict()
    if not tables_to_read:
        # Make sure leading/trailing '/' are removed from keys
        tables_to_read = [k.strip('/') for k in store.keys()]
        # And don't read the 'config' key (yet)
        tables_to_read.remove('config')
    for k in tables_to_read:
        solution[k] = store.get(k)
    # Also add model and run config to the solution object, which are stored
    # as strings in a Series in the 'config' key
    for k in ['config_model', 'config_run']:
        solution[k] = utils.AttrDict.from_yaml_string(store.get('config')[k])
    # Check if any keys are missing
    missing_keys = set(REQUIRED_KEYS) - set(solution.keys())
    if len(missing_keys) > 0:
        raise IOError('HDF file missing keys: {}'.format(missing_keys))
    store.close()
    return solution


def read_csv(directory, tables_to_read=None):
    solution = utils.AttrDict()
    if not tables_to_read:
        tables_to_read = glob.glob(directory + '/*.csv')
        if len(tables_to_read) == 0:
            raise IOError('No CSV files found')
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
    """Autodetects format, falling back to CSV if it can't find HDF"""
    if os.path.exists(os.path.join(directory, 'solution.hdf')):
        return 'hdf'
    else:
        return 'csv'


def read_dir(directory, tables_to_read=None):
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
        logging.debug('Iteration: {}, Format detected: {}'.format(i, fmt))
        try:
            if fmt == 'hdf':
                hdf_file = os.path.join(iteration_dir, 'solution.hdf')
                logging.debug('Read HDF: {}'.format(hdf_file))
                results.solutions[i] = read_hdf(hdf_file, tables_to_read)
            else:
                results.solutions[i] = read_csv(iteration_dir, tables_to_read)
                logging.debug('Read CSV: {}'.format(iteration_dir))
        except IOError as err:
            logging.warning('I/O error in `{}` at iteration `{}`'
                            ': {}'.format(iteration_dir, i, err))
            results.solutions[i] = utils.AttrDict()  # add an empty entry
            continue
    return results
