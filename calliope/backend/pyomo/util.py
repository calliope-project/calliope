"""
Copyright (C) 2013-2018 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

"""

import logging

import numpy as np
import pandas as pd
import xarray as xr

from calliope.core.util.tools import memoize
from calliope.core.util.logging import logger
from calliope import exceptions


@memoize
def get_param(backend_model, var, dims):
    """
    Get an input parameter held in a Pyomo object, or held in the defaults
    dictionary if that Pyomo object doesn't exist.

    Parameters
    ----------
    backend_model : Pyomo model instance
    var : str
    dims : single value or tuple

    """
    try:
        return getattr(backend_model, var)[dims]
    except AttributeError:  # i.e. parameter doesn't exist at all
        logger.debug('get_param: var {} and dims {} leading to default lookup'.format(var, dims))
        return backend_model.__calliope_defaults__[var]
    except KeyError:  # try removing timestep
        try:
            if len(dims) > 2:
                return getattr(backend_model, var)[dims[:-1]]
            else:
                return getattr(backend_model, var)[dims[0]]
        except KeyError:  # Static default value
            logger.debug('get_param: var {} and dims {} leading to default lookup'.format(var, dims))
            return backend_model.__calliope_defaults__[var]


def get_previous_timestep(backend_model, timestep):
    """Get the timestamp for the timestep previous to the input timestep"""
    # order_dict starts numbering at zero, timesteps is one-indexed, so we do not need
    # to subtract 1 to get to previous_step -- it happens "automagically"
    return backend_model.timesteps[backend_model.timesteps.order_dict[timestep]]


@memoize
def get_loc_tech_carriers(backend_model, loc_carrier):
    """
    For a given loc_carrier concatenation, get lists of the relevant
    loc_tech_carriers which produce energy (loc_tech_carriers_prod), consume
    energy (loc_tech_carriers_con) and export energy (loc_tech_carriers_export)
    """
    lookup = backend_model.__calliope_model_data__['data']['lookup_loc_carriers']
    loc_tech_carriers = split_comma_list(lookup[loc_carrier])

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
    """
    Split the string of a loc_tech_carrier (e.g. `region1::ccgt::power`) to get
    just the loc_tech (e.g. `region1::ccgt`)
    """
    return loc_tech_carrier.rsplit('::', 1)[0]


@memoize
def get_timestep_weight(backend_model):
    """
    Get the total number of years this model considers, by summing all
    timestep resolution with timestep weight (a weight/resolution of 1 = 1 hour)
    and divide it by number of hours in the year. Weight/resolution will almost
    always be 1 per step, unless time clustering/masking/resampling has taken place.
    """
    model_data_dict = backend_model.__calliope_model_data__
    time_res = list(model_data_dict['data']['timestep_resolution'].values())
    weights = list(model_data_dict['data']['timestep_weights'].values())
    return sum(np.multiply(time_res, weights)) / 8760


@memoize
def split_comma_list(comma_list):
    """
    Take a comma deliminated string and split it into a list of strings
    """
    return comma_list.split(',')


@memoize
def get_conversion_plus_io(backend_model, tier):
    """
    from a carrier_tier, return the primary tier (of `in`, `out`) and
    corresponding decision variable (`carrier_con` and `carrier_prod`, respectively)
    """
    if 'out' in tier:
        return 'out', backend_model.carrier_prod
    elif 'in' in tier:
        return 'in', backend_model.carrier_con


def get_var(backend_model, var, dims=None, sparse=False):
    """
    Return output for variable `var` as a pandas.Series (1d),
    pandas.Dataframe (2d), or xarray.DataArray (3d and higher).

    Parameters
    ----------
    var : variable name as string, e.g. 'resource'
    dims : list, optional
        indices as strings, e.g. ('loc_techs', 'timesteps');
        if not given, they are auto-detected
    sparse : bool, optional; default = False
        If extracting Pyomo Param data, the output sparse array includes inputs
        the user left as NaN replaced with the default value for that Param.
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

    if sparse:
        result = pd.DataFrame.from_dict(var_container.extract_values_sparse(), orient='index')
    else:
        result = pd.DataFrame.from_dict(var_container.extract_values(), orient='index')

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


@memoize
def loc_tech_is_in(backend_model, loc_tech, model_set):
    """
    Check if set exists and if loc_tech is in the set

    Parameters
    ----------
    loc_tech : string
    model_set : string
    """

    if hasattr(backend_model, model_set) and loc_tech in getattr(backend_model, model_set):
        return True
    else:
        return False
