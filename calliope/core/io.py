"""
Copyright (C) 2013-2019 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

io.py
~~~~~

Functions to read and save model results.

"""

import os

import xarray as xr

from calliope._version import __version__
from calliope import exceptions
from calliope.core.util.dataset import split_loc_techs


def read_netcdf(path):
    """Read model_data from NetCDF file"""
    with xr.open_dataset(path) as model_data:
        model_data.load()

    calliope_version = model_data.attrs.get('calliope_version', False)
    if calliope_version:
        if not str(calliope_version) in __version__:
            exceptions.warn(
                'This model data was created with Calliope version {}, '
                'but you are running {}. Proceed with caution!'.format(calliope_version, __version__)
            )

    # FIXME some checks for consistency
    # use check_dataset from the checks module
    # also check the old checking from 0.5.x

    return model_data


def save_netcdf(model_data, path, model=None):
    encoding = {k: {'zlib': True, 'complevel': 4} for k in model_data.data_vars}

    original_model_data_attrs = model_data.attrs
    model_data_attrs = model_data.attrs.copy()

    if model is not None and hasattr(model, '_model_run'):
        # Attach _model_run and _debug_data to _model_data
        model_run_to_save = model._model_run.copy()
        if 'timeseries_data' in model_run_to_save:
            del model_run_to_save['timeseries_data']  # Can't be serialised!
        model_data_attrs['_model_run'] = model_run_to_save.to_yaml()
        model_data_attrs['_debug_data'] = model._debug_data.to_yaml()

    # Convert boolean attrs to ints
    bool_attrs = [
        k for k, v in model_data_attrs.items()
        if isinstance(v, bool)
    ]
    for k in bool_attrs:
        model_data_attrs[k] = int(model_data_attrs[k])

    # Convert None attrs to 'None'
    none_attrs = [
        k for k, v in model_data_attrs.items()
        if v is None
    ]
    for k in none_attrs:
        model_data_attrs[k] = 'None'

    # Convert `object` dtype coords to string
    # FIXME: remove once xarray issue https://github.com/pydata/xarray/issues/2404 is resolved
    for k, v in model_data.coords.items():
        if v.dtype == 'O':
            model_data[k] = v.astype('<U{}'.format(max([len(i.item()) for i in v])))

    try:
        model_data.attrs = model_data_attrs
        model_data.to_netcdf(path, format='netCDF4', encoding=encoding)
        model_data.close()  # Force-close NetCDF file after writing
    finally:  # Revert model_data.attrs back
        model_data.attrs = original_model_data_attrs


def save_csv(model_data, path, dropna=True):
    """
    If termination condition was not optimal, filters inputs only, and
    warns that results will not be saved.

    """
    os.makedirs(path, exist_ok=False)

    # a MILP model which optimises to within the MIP gap, but does not fully
    # converge on the LP relaxation, may return as 'feasible', not 'optimal'
    if ('termination_condition' not in model_data.attrs or
            model_data.attrs['termination_condition'] in ['optimal', 'feasible']):
        data_vars = model_data.data_vars
    else:
        data_vars = model_data.filter_by_attrs(is_result=0).data_vars
        exceptions.warn(
            'Model termination condition was not optimal, saving inputs only.'
        )

    for var in data_vars:
        in_out = 'results' if model_data[var].attrs['is_result'] else 'inputs'
        out_path = os.path.join(path, '{}_{}.csv'.format(in_out, var))
        series = split_loc_techs(model_data[var], return_as='Series')
        if dropna:
            series = series.dropna()
        series.to_csv(out_path, header=True)


def save_lp(model, path):
    if not model.run_config['backend'] == 'pyomo':
        raise IOError('Only the pyomo backend can save to LP.')
    if not hasattr(model, '_backend_model'):
        model.run(build_only=True)
    model._backend_model.write(path, format='lp', io_options={'symbolic_solver_labels': True})
