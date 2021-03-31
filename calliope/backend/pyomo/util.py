"""
Copyright (C) since 2013 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

"""

import logging
import re

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
import xarray as xr
import pyomo.core as po

from calliope.core.util.tools import memoize
from calliope import exceptions

logger = logging.getLogger(__name__)


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
        logger.debug(
            "get_param: var {} and dims {} leading to default lookup".format(var, dims)
        )
        return backend_model.__calliope_defaults[var]
    except KeyError:  # try removing timestep
        try:
            if len(dims) > 2:
                return getattr(backend_model, var)[dims[:-1]]
            else:
                return getattr(backend_model, var)[dims[0]]
        except KeyError:  # Static default value
            logger.debug(
                "get_param: var {} and dims {} leading to default lookup".format(
                    var, dims
                )
            )
            return backend_model.__calliope_defaults[var]


def get_previous_timestep(timesteps, timestep):
    """Get the timestamp for the timestep previous to the input timestep"""
    return timesteps[timesteps.ord(timestep) - 1]


@memoize
def get_timestep_weight(backend_model):
    """
    Get the total number of years this model considers, by summing all
    timestep resolution with timestep weight (a weight/resolution of 1 = 1 hour)
    and divide it by number of hours in the year. Weight/resolution will almost
    always be 1 per step, unless time clustering/masking/resampling has taken place.
    """
    time_res = [po.value(i) for i in backend_model.timestep_resolution.values()]
    weights = [po.value(i) for i in backend_model.timestep_weights.values()]
    return sum(np.multiply(time_res, weights)) / 8760


@memoize
def get_conversion_plus_io(backend_model, tier):
    """
    from a carrier_tier, return the primary tier (of `in`, `out`) and
    corresponding decision variable (`carrier_con` and `carrier_prod`, respectively)
    """
    if "out" in tier:
        return "out", backend_model.carrier_prod
    elif "in" in tier:
        return "in", backend_model.carrier_con


def get_var(backend_model, var, dims=None, sparse=False, expr=False):
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
        raise exceptions.BackendError("Variable {} inexistent.".format(var))

    if not dims:
        if var + "_index" == var_container.index_set().name:
            dims = [i.name for i in var_container.index_set().subsets()]
        else:
            dims = [var_container.index_set().name]

    if sparse and not expr:
        result = pd.Series(var_container.extract_values_sparse())
    else:
        if expr:
            result = pd.Series(var_container._data).apply(
                lambda x: po.value(x) if not invalid(x) else np.nan
            )
        else:
            result = pd.Series(var_container.extract_values())
    if result.empty:
        raise exceptions.BackendError("Variable {} has no data.".format(var))

    result = result.rename_axis(index=dims)

    return xr.DataArray.from_series(result)


def loc_tech_is_in(backend_model, loc_tech, model_set):
    """
    Check if set exists and if loc_tech is in the set

    Parameters
    ----------
    loc_tech : string
    model_set : string
    """

    if hasattr(backend_model, model_set) and loc_tech in getattr(
        backend_model, model_set
    ):
        return True
    else:
        return False


def get_domain(var: xr.DataArray) -> str:
    def check_sign(var):
        if re.match("resource|node_coordinates|cost*", var.name):
            return ""
        else:
            return "NonNegative"

    if var.dtype.kind == "b":
        return "Boolean"
    elif is_numeric_dtype(var.dtype):
        return check_sign(var) + "Reals"
    else:
        return "Any"


def invalid(val) -> bool:
    if isinstance(val, po.base.param._ParamData):
        return val._value == po.base.param._NotValid or pd.isnull(po.value(val))
    else:
        return pd.isnull(val)


def datetime_to_string(
    backend_model: po.ConcreteModel, model_data: xr.Dataset
) -> xr.Dataset:
    """
    Convert from datetime to string xarray dataarrays, to reduce the memory
    footprint of converting datetimes from numpy.datetime64 -> pandas.Timestamp
    when creating the pyomo model object.

    Parameters
    ----------
    backend_model : the backend pyomo model object
    model_data : the Calliope xarray Dataset of model data
    """
    datetime_data = set()
    for attr in ["coords", "data_vars"]:
        for set_name, set_data in getattr(model_data, attr).items():
            if set_data.dtype.kind == "M":
                attrs = model_data[set_name].attrs
                model_data[set_name] = model_data[set_name].dt.strftime(
                    "%Y-%m-%d %H:%M"
                )
                model_data[set_name].attrs = attrs
                datetime_data.add((attr, set_name))
    backend_model.__calliope_datetime_data = datetime_data

    return model_data


def string_to_datetime(
    backend_model: po.ConcreteModel, model_data: xr.Dataset
) -> xr.Dataset:
    """
    Convert from string to datetime xarray dataarrays, reverting the process
    undertaken in
    datetime_to_string

    Parameters
    ----------
    backend_model : the backend pyomo model object
    model_data : the Calliope xarray Dataset of model data
    """
    for attr, set_name in backend_model.__calliope_datetime_data:
        if attr == "coords" and set_name in model_data:
            model_data.coords[set_name] = model_data[set_name].astype("datetime64[ns]")
        elif set_name in model_data:
            model_data[set_name] = (
                model_data[set_name].fillna(pd.NaT).astype("datetime64[ns]")
            )
    return model_data
