# Copyright (C) since 2013 Calliope contributors listed in AUTHORS.
# Licensed under the Apache 2.0 License (see LICENSE file).

"""
io.py
~~~~~

Functions to read and save model results.

"""

import os
from typing import Union

# We import netCDF4 before xarray to mitigate a numpy warning:
# https://github.com/pydata/xarray/issues/7259
import netCDF4  # noqa: F401
import numpy as np
import pandas as pd
import xarray as xr

from calliope import exceptions
from calliope.attrdict import AttrDict


def read_netcdf(path):
    """Read model_data from NetCDF file"""
    with xr.open_dataset(path) as model_data:
        model_data.load()

    _deserialise(model_data.attrs)
    for var in model_data.data_vars.values():
        _deserialise(var.attrs)

    # Convert empty strings back to np.NaN
    # TODO: revert when this issue is solved: https://github.com/pydata/xarray/issues/1647
    # which it might be once this is merged: https://github.com/pydata/xarray/pull/7869
    for var_name, var_array in model_data.data_vars.items():
        if var_array.dtype.kind in ["U", "O"]:
            model_data[var_name] = var_array.where(lambda x: x != "")

    # FIXME some checks for consistency
    # use check_dataset from the checks module
    # also check the old checking from 0.5.x

    return model_data


def _pop_serialised_list(
    attrs: dict, serialised_items: Union[str, list, np.ndarray]
) -> Union[list, np.ndarray]:
    serialised_ = attrs.pop(serialised_items, [])
    if not isinstance(serialised_, (list, np.ndarray)):
        return [serialised_]
    else:
        return serialised_


def _serialise(attrs: dict) -> None:
    """Convert troublesome datatypes to nicer ones in xarray attribute dictionaries.

    This will tackle dictionaries (to string), booleans (to int), None (to string), and sets (to list).

    Args:
        attrs (dict):
            Attribute dictionary from an xarray Dataset/DataArray.
            Changes will be made in-place, so be sure to supply a copy of your dictionary if you want access to its original state.
    """
    # Convert dicts attrs to yaml strings
    dict_attrs = [k for k, v in attrs.items() if isinstance(v, dict)]
    attrs["serialised_dicts"] = dict_attrs
    for attr in dict_attrs:
        attrs[attr] = AttrDict(attrs[attr]).to_yaml()

    # Convert boolean attrs to ints
    bool_attrs = [k for k, v in attrs.items() if isinstance(v, bool)]
    attrs["serialised_bools"] = bool_attrs
    for attr in bool_attrs:
        attrs[attr] = int(attrs[attr])

    # Convert None attrs to 'None'
    none_attrs = [k for k, v in attrs.items() if v is None]
    attrs["serialised_nones"] = none_attrs
    for attr in none_attrs:
        attrs[attr] = "None"

    # Convert set attrs to lists
    set_attrs = [k for k, v in attrs.items() if isinstance(v, set)]
    for attr in set_attrs:
        attrs[attr] = list(attrs[attr])

    list_attrs = [k for k, v in attrs.items() if isinstance(v, list)]
    for attr in list_attrs:
        if any(not isinstance(i, str) for i in attrs[attr]):
            raise TypeError(
                f"Cannot serialise a sequence of values stored in a model attribute unless all values are strings, found: {attrs[attr]}"
            )
    else:
        attrs["serialised_sets"] = set_attrs


def _deserialise(attrs: dict) -> None:
    """Convert troublesome datatypes in xarray attribute dictionaries from their stored data type to the data types expected by Calliope.

    This will tackle dictionaries (from string), booleans (from int), None (form string), and sets (from list).

    Args:
        attrs (dict):
            Attribute dictionary from an xarray Dataset/DataArray.
            Changes will be made in-place, so be sure to supply a copy of your dictionary if you want access to its original state.
    """
    for attr in _pop_serialised_list(attrs, "serialised_dicts"):
        attrs[attr] = AttrDict.from_yaml_string(attrs[attr])
    for attr in _pop_serialised_list(attrs, "serialised_bools"):
        attrs[attr] = bool(attrs[attr])
    for attr in _pop_serialised_list(attrs, "serialised_nones"):
        attrs[attr] = None
    for attr in _pop_serialised_list(attrs, "serialised_sets"):
        attrs[attr] = set(attrs[attr])


def save_netcdf(model_data, path, model=None):
    original_model_data_attrs = model_data.attrs
    model_data_attrs = original_model_data_attrs.copy()

    if model is not None and hasattr(model, "_model_def_dict"):
        # Attach initial model definition to _model_data
        model_data_attrs["_model_def_dict"] = model._model_def_dict.to_yaml()
        if hasattr(model, "_debug_data"):
            model_data_attrs["_debug_data"] = model._debug_data.to_yaml()

    _serialise(model_data_attrs)
    for var in model_data.data_vars.values():
        _serialise(var.attrs)

    encoding = {
        k: {"zlib": False, "_FillValue": None}
        if v.dtype.kind in ["U", "O"]
        else {"zlib": True, "complevel": 4}
        for k, v in model_data.data_vars.items()
    }

    try:
        model_data.attrs = model_data_attrs
        model_data.to_netcdf(path, format="netCDF4", encoding=encoding)
        model_data.close()  # Force-close NetCDF file after writing
    finally:  # Revert model_data.attrs back
        model_data.attrs = original_model_data_attrs
        for var in model_data.data_vars.values():
            _deserialise(var.attrs)


def save_csv(model_data, path, dropna=True):
    """
    If termination condition was not optimal, filters inputs only, and
    warns that results will not be saved.

    """
    os.makedirs(path, exist_ok=False)

    # a MILP model which optimises to within the MIP gap, but does not fully
    # converge on the LP relaxation, may return as 'feasible', not 'optimal'
    if "termination_condition" not in model_data.attrs or model_data.attrs[
        "termination_condition"
    ] in ["optimal", "feasible"]:
        data_vars = model_data.data_vars
    else:
        data_vars = model_data.filter_by_attrs(is_result=0).data_vars
        exceptions.warn(
            "Model termination condition was not optimal, saving inputs only."
        )

    for var_name, var in data_vars.items():
        in_out = "results" if var.attrs["is_result"] else "inputs"
        out_path = os.path.join(path, "{}_{}.csv".format(in_out, var_name))
        if not var.shape:
            series = pd.Series(var.item())
            keep_index = False
        else:
            series = var.to_series()
            keep_index = True
        if dropna:
            series = series.dropna()
        series.to_csv(out_path, header=True, index=keep_index)
