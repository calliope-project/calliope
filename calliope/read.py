"""
Copyright (C) 2013-2017 Calliope contributors listed in AUTHORS.
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
    with xr.open_dataset(path) as solution:
        solution.load()

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

    # Restore metadata from YAML
    md = AttrDict.from_yaml(os.path.join(directory, 'metadata.yaml'))
    for k in md.keys():
        solution.attrs[k] = md[k]

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

##
# Functionality to post-process parallel runs into aggregated NetCDF files
##


def union_of_indexes(indexes):
    idx = indexes[0]
    for i in range(len(indexes) - 1):
        idx = idx.union(indexes[i + 1])
    return idx


def get_overarching_time_index(datasets):
    all_time_dims = [v.coords['t'].to_index() for k, v in datasets.items()]
    return union_of_indexes(all_time_dims)


def get_longest_time_index_length(datasets):
    all_time_dims = [len(v.coords['t'].to_index()) for k, v in datasets.items()]
    return max(all_time_dims)


def results_to_dataset(results, run_name, reset_time_index=False):
    # Add 'run' dimension
    for k in results.solutions:
        results.solutions[k].coords['run'] = k
        results.solutions[k]['run'] = k

    # Reindex all datasets with reset list of integeres or
    # union of all time indices
    # In the first case, all loaded datasets should have roughly the
    # same length of dimension t (e.g. multiple runs across a single year)
    if reset_time_index:
        max_len_idx = list(range(get_longest_time_index_length(results.solutions)))
        for k in results.solutions:
            results.solutions[k]['t'] = list(range(len(results.solutions[k]['t'])))
            results.solutions[k] = results.solutions[k].reindex(dict(t=max_len_idx))
    else:
        new_idx = get_overarching_time_index(results.solutions)
        for k in results.solutions:
            results.solutions[k] = results.solutions[k].reindex(dict(t=new_idx))

    ds_results = xr.concat(results.solutions.values(), dim='run')

    # Remove metadata
    for k in list(ds_results.attrs.keys()):
        if k != 'calliope_version':
            del ds_results.attrs[k]

    results.iterations.index.name = 'run'
    results.iterations.columns.name = 'cols_iterations'
    ds_results['iterations'] = results.iterations
    ds_results['iterations'] = ds_results.iterations.astype(str)  # Force to str

    # Add run name dimension
    ds_results['run_name'] = xr.DataArray([run_name], coords={'run_name': [run_name]})
    ds_results = ds_results.set_coords('run_name')

    return ds_results


def dir_to_dataset(in_dir, run_name, reset_time_index=False):
    results = read_dir(in_dir)
    return results_to_dataset(results, run_name, reset_time_index)


def convert_run_dir_to_netcdf(in_dir, out_file, reset_time_index=False):
    this_dir = os.path.join(in_dir, 'Output')
    run_name = this_dir.split('/')[-1]
    ds = dir_to_dataset(this_dir, run_name, reset_time_index)

    encoding = {k: {'zlib': True, 'complevel': 4} for k in ds.data_vars}
    ds.to_netcdf(out_file, format='netCDF4', encoding=encoding)
    ds.close()  # Force-close NetCDF file after writing


def convert_subdirs_to_netcdfs(in_dir, out_dir, reset_time_index_for_subdirs=None):
    if reset_time_index_for_subdirs is None:
        reset_time_index_for_subdirs = []

    subdirs = [
        i for i in os.listdir(in_dir)
        if os.path.isdir(os.path.join(in_dir, i))
        and not i.startswith('.')
    ]

    os.makedirs(out_dir, exist_ok=True)

    for s in subdirs:
        if s in reset_time_index_for_subdirs:
            reset_time_index = True
        else:
            reset_time_index = False
        this_path = os.path.join(in_dir, s)
        out_file = os.path.join(out_dir, s + '.nc')
        if os.path.exists(out_file):
            print('File exists, skipping: {}'.format(out_file))
        else:
            print('Processing {}'.format(this_path))
            convert_run_dir_to_netcdf(this_path, out_file, reset_time_index)


def combine_subdir_netcdfs(in_dir, out_file):
    in_files = glob.glob(os.path.join(in_dir, '*.nc'))
    datasets = [xr.open_dataset(i) for i in in_files]

    t_idx = get_overarching_time_index(datasets)
    for i in range(len(datasets)):
        datasets[i] = datasets[i].reindex(dict(t=t_idx))

    ds = xr.concat(datasets, dim='run_name')

    encoding = {k: {'zlib': True, 'complevel': 4} for k in ds.data_vars}
    ds.to_netcdf(out_file, format='netCDF4', encoding=encoding)
    ds.close()  # Force-close NetCDF file after writing
    for d in datasets:
        d.close()
