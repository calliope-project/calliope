"""
Copyright (C) 2013-2017 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

"""

import numpy as np
import pandas as pd
import xarray as xr

from calliope.core.util.tools import memoize
from calliope import exceptions


#FIXME: change to get_param
@memoize
def param_getter(backend_model, var, dims):
    """
    Params
    ------
    backend_model : Pyomo model instance
    var : str
    dims : single value or tuple

    """
    try:
        return backend_model.__calliope_model_data__['data'][var][dims]
    except KeyError:  # Try without time dimension, which is always last
        try:
            if len(dims) > 2:
                return backend_model.__calliope_model_data__['data'][var][dims[:-1]]
            else:
                return backend_model.__calliope_model_data__['data'][var][dims[0]]
        except KeyError:  # Static default value
            return backend_model.__calliope_defaults__[var]


def get_previous_timestep(backend_model, timestep):
    # order_dict starts numbering at zero, timesteps is one-indexed, so we do not need
    # to subtract 1 to get to previous_step -- it happens "automagically"
    return backend_model.timesteps[backend_model.timesteps.order_dict[timestep]]


@memoize
def get_loc_tech_carriers(backend_model, loc_carrier):

    lookup = backend_model.__calliope_model_data__['data']['lookup_loc_carriers']
    loc_tech_carriers = lookup[loc_carrier].split(',')

    loc_tech_carriers_prod = [
        i for i in loc_tech_carriers if i in backend_model.loc_tech_carriers_prod
    ]
    loc_tech_carriers_con = [
        i for i in loc_tech_carriers if i in backend_model.loc_tech_carriers_con
    ]

    if hasattr(backend_model, 'loc_tech_carriers_export'):
        loc_tech_carriers_export = [
            i for i in loc_tech_carriers
            if i in backend_model.loc_tech_carriers_export
        ]
    else:
        loc_tech_carriers_export = []

    return (
        loc_tech_carriers_prod,
        loc_tech_carriers_con,
        loc_tech_carriers_export
    )


@memoize
def get_loc_tech(loc_tech_carrier):
    return loc_tech_carrier.rsplit('::', 1)[0]


@memoize
def get_timestep_weight(backend_model):
    model_data_dict = backend_model.__calliope_model_data__
    time_res = list(model_data_dict['data']['timestep_resolution'].values())
    weights = list(model_data_dict['data']['timestep_weights'].values())
    return sum(np.multiply(time_res, weights)) / 8760


def get_var(backend_model, var, dims=None):
    """
    Return output for variable `var` as a pandas.Series (1d),
    pandas.Dataframe (2d), or xarray.DataArray (3d and higher).

    Parameters
    ----------
    var : variable name as string, e.g. 'resource'
    dims : list, optional
        indices as strings, e.g. ('loc_techs', 'timesteps');
        if not given, they are auto-detected

    """
    try:
        var_container = getattr(backend_model, var)
    except AttributeError:
        raise exceptions.BackendError('Variable {} inexistent.'.format(var))

    if not dims:
        if var + '_index' == var_container.index_set().name:
            dims = [i.name for i in var_container.index_set().set_tuple]
        else:
            dims = [var_container.index_set().name]

    result = pd.DataFrame.from_dict(var_container.get_values(), orient='index')

    if result.empty:
        raise exceptions.BackendError('Variable {} has no data.'.format(var))

    result = result[0]  # Get the only column in the dataframe

    if len(dims) > 1:
        result.index = pd.MultiIndex.from_tuples(result.index, names=dims)

    if len(result.index.names) == 1:
        result = result.sort_index()
        result.index.name = dims[0]
    elif len(result.index.names) == 2:
        # if len(dims) is 2, we already have a well-formed DataFrame
        result = result.unstack(level=0)
        result = result.sort_index()
    else:  # len(dims) >= 3
        result = xr.DataArray.from_series(result)

    return result
