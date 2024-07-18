# Copyright (C) since 2013 Calliope contributors listed in AUTHORS.
# Licensed under the Apache 2.0 License (see LICENSE file).
"""Functions to read and save model results."""

import importlib.resources
from copy import deepcopy
from pathlib import Path

# We import netCDF4 before xarray to mitigate a numpy warning:
# https://github.com/pydata/xarray/issues/7259
import netCDF4  # noqa: F401
import numpy as np
import pandas as pd
import xarray as xr

from calliope import exceptions
from calliope.attrdict import AttrDict
from calliope.util.tools import listify

CONFIG_DIR = importlib.resources.files("calliope") / "config"


def read_netcdf(path):
    """Read model_data from NetCDF file."""
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

    return model_data


def _pop_serialised_list(
    attrs: dict, serialised_items: str | list
) -> list | np.ndarray:
    """Pop a list of serialised attributes from the attribute dictionary."""
    serialised_ = attrs.pop(serialised_items, [])
    return listify(serialised_)


def _serialise(attrs: dict) -> None:
    """Convert troublesome datatypes to nicer ones in xarray attribute dictionaries.

    This will tackle dictionaries (to string), booleans (to int), None (to string), and sets (to list).

    We also make note of any single element lists as they will be stored as simple strings in netcdf format,
    but we want them to be returned as lists on loading the data from file.

    Args:
        attrs (dict):
            Attribute dictionary from an xarray Dataset/DataArray.
            Changes will be made in-place, so be sure to supply a copy of your dictionary if you want access to its original state.
    """
    # Keep track of single element lists, to listify them again when loading from file.

    attrs["serialised_single_element_list"] = [
        k for k, v in attrs.items() if isinstance(v, list) and len(v) == 1
    ]

    # Convert dict attrs to yaml strings
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
        # Also keep track of single element sets
        attrs["serialised_single_element_list"].extend(
            [k for k in set_attrs if len(attrs[k]) == 1]
        )


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
    for attr in _pop_serialised_list(attrs, "serialised_single_element_list"):
        attrs[attr] = listify(attrs[attr])
    for attr in _pop_serialised_list(attrs, "serialised_sets"):
        attrs[attr] = set(attrs[attr])


def save_netcdf(model_data, path, **kwargs):
    """Save the model to a netCDF file."""
    original_model_data_attrs = deepcopy(model_data.attrs)
    for key, value in kwargs.items():
        model_data.attrs[key] = value

    _serialise(model_data.attrs)
    for var in model_data.data_vars.values():
        _serialise(var.attrs)

    encoding = {
        k: (
            {"zlib": False, "_FillValue": None}
            if v.dtype.kind in ["U", "O"]
            else {"zlib": True, "complevel": 4}
        )
        for k, v in model_data.data_vars.items()
    }

    try:
        model_data.to_netcdf(path, format="netCDF4", encoding=encoding)
        model_data.close()  # Force-close NetCDF file after writing
    finally:  # Revert model_data.attrs back
        model_data.attrs = original_model_data_attrs
        for var in model_data.data_vars.values():
            _deserialise(var.attrs)


def save_csv(
    model_data: xr.Dataset,
    path: str | Path,
    dropna: bool = True,
    allow_overwrite: bool = False,
):
    """Save results to CSV.

    One file per dataset array will be generated, with the filename matching the array name.

    If termination condition was not optimal, filters inputs only, and warns that results will not be saved.

    Args:
        model_data (xr.Dataset): Calliope model data.
        path (str | Path): Directory to which the CSV files will be saved
        dropna (bool, optional):
            If True, drop all NaN values in the data series before saving to file.
            Defaults to True.
        allow_overwrite (bool, optional):
            If True, allow the option to overwrite the directory contents if it already exists.
            This will overwrite CSV files one at a time, so if the dataset has different arrays to the previous saved models, you will get a mix of old and new files.
            Defaults to False.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=allow_overwrite)

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
        out_path = path / f"{in_out}_{var_name}.csv"
        if not var.shape:
            series = pd.Series(var.item())
            keep_index = False
        else:
            series = var.to_series()
            keep_index = True
        if dropna:
            series = series.dropna()
        series.to_csv(out_path, header=True, index=keep_index)


def load_config(filename: str):
    """Load model configuration from a file."""
    with importlib.resources.as_file(CONFIG_DIR / filename) as f:
        loaded = AttrDict.from_yaml(f)
    return loaded
