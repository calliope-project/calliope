"""
Copyright (C) 2013-2017 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

io.py
~~~~~

Functions to read and save model results.

"""

import os

import xarray as xr

from calliope.core.util.dataset import split_loc_techs


def read_netcdf(path):
    """Read model_data from NetCDF file"""
    with xr.open_dataset(path) as model_data:
        model_data.load()

    # FIXME some checks for consistency
    # use check_dataset from the checks module
    # also check the old checking from 0.5.x

    return model_data


def save_netcdf(model_data, path):
    #FIXME
    return None


def save_csv(model_data, path):
    for var in model_data.data_vars:
        in_out = 'results' if model_data[var].attrs['is_result'] else 'inputs'
        out_path = os.path.join(path, '{}_{}.csv'.format(in_out, var))
        series = split_loc_techs(model_data[var], as_='Series')
        series.to_csv(out_path)
