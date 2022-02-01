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
        return backend_model.__calliope_defaults.get(var, po.Param.NoValue)
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
            return backend_model.__calliope_defaults.get(var, po.Param.NoValue)


def get_previous_timestep(timesteps, timestep):
    """Get the timestamp for the timestep previous to the input timestep"""
    return timesteps[timesteps.ord(timestep) - 1]


@memoize
def get_loc_tech_carriers(backend_model, loc_carrier):
    """
    For a given loc_carrier concatenation, get lists of the relevant
    loc_tech_carriers which produce energy (loc_tech_carriers_prod), consume
    energy (loc_tech_carriers_con) and export energy (loc_tech_carriers_export)
    """
    lookup = backend_model.__calliope_model_data["data"]["lookup_loc_carriers"]
    loc_tech_carriers = split_comma_list(lookup[loc_carrier])

    loc_tech_carriers_prod = [
        i for i in loc_tech_carriers if i in backend_model.loc_tech_carriers_prod
    ]
    loc_tech_carriers_con = [
        i for i in loc_tech_carriers if i in backend_model.loc_tech_carriers_con
    ]

    if hasattr(backend_model, "loc_tech_carriers_export"):
        loc_tech_carriers_export = [
            i for i in loc_tech_carriers if i in backend_model.loc_tech_carriers_export
        ]
    else:
        loc_tech_carriers_export = []

    return (loc_tech_carriers_prod, loc_tech_carriers_con, loc_tech_carriers_export)


@memoize
def get_loc_tech(loc_tech_carrier):
    """
    Split the string of a loc_tech_carrier (e.g. `region1::ccgt::power`) to get
    just the loc_tech (e.g. `region1::ccgt`)
    """
    return loc_tech_carrier.rsplit("::", 1)[0]


@memoize
def get_timestep_weight(backend_model):
    """
    Get the total number of years this model considers, by summing all
    timestep resolution with timestep weight (a weight/resolution of 1 = 1 hour)
    and divide it by number of hours in the year. Weight/resolution will almost
    always be 1 per step, unless time clustering/masking/resampling has taken place.
    """
    model_data_dict = backend_model.__calliope_model_data
    time_res = list(model_data_dict["data"]["timestep_resolution"].values())
    weights = list(model_data_dict["data"]["timestep_weights"].values())
    return sum(np.multiply(time_res, weights)) / 8760


@memoize
def split_comma_list(comma_list):
    """
    Take a comma deliminated string and split it into a list of strings
    """
    return comma_list.split(",")


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
    expr : bool, optional
        If True, treat var as a pyomo expression, which requires calculating
        the result of the expression before translating into nd data structure
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
        if invalid(var_container.default()):
            result = pd.Series(var_container._data).apply(
                lambda x: po.value(x) if not invalid(x) else np.nan
            )
        else:
            result = pd.Series(var_container.extract_values_sparse())
    else:
        if expr:
            result = pd.Series(var_container._data).apply(po.value)
        else:
            result = pd.Series(var_container.extract_values())
    if result.empty:
        raise exceptions.BackendError("Variable {} has no data.".format(var))

    result = result.rename_axis(index=dims)

    return xr.DataArray.from_series(result)


@memoize
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


def get_domain(var: xr.DataArray, default) -> str:
    """
    Get the Pyomo 'domain' of an array of input data. This is required when
    initialising a pyomo Parameter. An initial attempt will be made to infer the array's
    domain, based on its dtype. If that fails, the result will be "Any".

    Args:
        var (xr.DataArray): Calliope model parameter.
        default ([type]): default value of the parameter (e.g. from config/defaults.yaml).

    Returns:
        str: Domain name recognised by Pyomo.
    """

    def check_sign(var):
        if re.match("resource|loc_coordinates|cost*", var.name):
            return ""
        elif re.match("group_carrier_con*", var.name):
            return "NonPositive"
        else:
            return "NonNegative"

    if var.dtype.kind == "b":
        return "Boolean"
    elif is_numeric_dtype(var.dtype) and isinstance(default, (int, float)):
        return check_sign(var) + "Reals"
    else:
        return "Any"


def invalid(val) -> bool:
    """
    Check whether an optimisation parameter is initialised and/or set to None.

    Args:
        val: Pyomo Parameter object or any other type.

    Returns:
        bool: True if the parameter is not valid for use in setting a constraint/objective
    """
    if isinstance(val, po.base.param._ParamData):
        return (
            val._value == po.Param.NoValue
            or val._value is None
            or po.value(val) is None
        )
    elif val == po.Param.NoValue:
        return True
    else:
        return pd.isnull(val)


def apply_equals(val) -> bool:
    """
    Check if a constraint should enforce a variable to be an exact value, rather than
    allowing a range. E.g. If a user sets `energy_cap_equals`, it is applied in
    preference of `energy_cap_min` and `energy_cap_max`.

    Args:
        val: Pyomo parameter value for the `..._equals` parameter in question.

    Raises:
        ValueError: Cannot set a variable to equal infinity; an exception is raised if this is attempted.

    Returns:
        bool: If True, setting the variable to `val` is the correct course of action.
    """
    if invalid(val) or (isinstance(po.value(val), bool) and po.value(val) is False):
        return False
    elif np.isinf(po.value(val)):
        raise ValueError(f"Cannot use inf for parameter {val.name}")
    else:
        return True


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
