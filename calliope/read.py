"""
Copyright (C) 2013-2016 Stefan Pfenninger.
Licensed under the Apache 2.0 License (see LICENSE file).

read.py
~~~~~~~

Functions to read saved model results.

"""

import glob
import logging
import os

import pandas as pd
import xarray as xr

from .utils import AttrDict


REQUIRED_TABLES = ['capacity_factor', 'levelized_cost',
                   'metadata', 'groups', 'shares', 'summary',
                   'time_res']


def _check(path, solution):
    # Superficial check if some key tables are missing
    missing_keys = set(REQUIRED_TABLES) - set(solution.data_vars)
    if len(missing_keys) > 0:
        logging.warning('Solution {} missing tables: '
                        '{}'.format(path, missing_keys))


def read_netcdf(path):
    """Read model solution from NetCDF4 file"""
    solution = xr.open_dataset(path)

    # Deserialize YAML attributes
    for k in ['config_model', 'config_run']:
        solution.attrs[k] = AttrDict.from_yaml_string(solution.attrs[k])

    _check(path, solution)

    return solution


def read_csv(directory):
    solution = AttrDict()
    tables_to_read = glob.glob(directory + '/*.csv')
    if len(tables_to_read) == 0:
        raise IOError('No CSV files found')
    # Only keep basenames without extension
    tables_to_read = [os.path.splitext(os.path.basename(f))[0]
                      for f in tables_to_read]
    arrays = {}
    for f in tables_to_read:
        src = os.path.join(directory, f + '.csv')
        cols = pd.read_csv(src, nrows=1).columns
        series = pd.read_csv(src, index_col=list(range(len(cols) - 1)), parse_dates=True, squeeze=True)
        # Make everything except the last column a MultiIndex
        arrays[f] = xr.DataArray.from_series(series)
    solution = xr.Dataset(arrays)

    # Add YAML attributes
    for k in ['config_model', 'config_run']:
        file_path = os.path.join(directory, '{}.yaml'.format(k))
        solution.attrs[k] = AttrDict.from_yaml(file_path)

    _check(directory, solution)

    return solution


def _detect_format(directory):
    """Detects format, falling back to CSV if it can't find NetCDF4"""
    if os.path.exists(os.path.join(directory, 'solution.nc')):
        return 'netcdf'
    else:
        return 'csv'


def read_dir(directory):
    """Combines output files from `directory` and return an AttrDict
    containing them all.

    If a solution is missing or there is an error reading it, an empty
    AttrDict is added to the results in its stead and the error is logged.

    """
    results = AttrDict()
    results.iterations = pd.read_csv(os.path.join(directory, 'iterations.csv'),
                                     index_col=0)
    results.solutions = AttrDict()
    for i in results.iterations.index.tolist():
        iteration_dir = os.path.join(directory, '{:0>4d}'.format(i))
        fmt = _detect_format(iteration_dir)
        logging.debug('Iteration: {}, Format detected: {}'.format(i, fmt))
        try:
            if fmt == 'netcdf':
                sol_path = os.path.join(iteration_dir, 'solution.nc')
                results.solutions[i] = read_netcdf(sol_path)
            else:
                sol_path = iteration_dir
                results.solutions[i] = read_csv(sol_path)
            logging.debug('Read as {}: {}'.format(fmt, sol_path))
        except IOError as err:
            logging.warning('I/O error in `{}` at iteration `{}`'
                            ': {}'.format(iteration_dir, i, err))
            # results.solutions[i] = AttrDict()  # add an empty entry
            continue
    return results
