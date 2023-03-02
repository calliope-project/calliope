"""
Copyright (C) since 2013 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

io.py
~~~~~

Functions to read and save model results.

"""

import os

import xarray as xr
import numpy as np

from calliope._version import __version__
from calliope import exceptions
from calliope.core.attrdict import AttrDict


def read_netcdf(path):
    """Read model_data from NetCDF file"""
    with xr.open_dataset(path) as model_data:
        model_data.load()

    calliope_version = model_data.attrs.get("calliope_version", False)
    if calliope_version:
        if not str(calliope_version) in __version__:
            exceptions.warn(
                "This model data was created with Calliope version {}, "
                "but you are running {}. Proceed with caution!".format(
                    calliope_version, __version__
                )
            )
    for attr in _pop_serialised_list(model_data.attrs, "serialised_dicts"):
        model_data.attrs[attr] = AttrDict.from_yaml_string(model_data.attrs[attr])
    for attr in _pop_serialised_list(model_data.attrs, "serialised_bools"):
        model_data.attrs[attr] = bool(model_data.attrs[attr])
    for attr in _pop_serialised_list(model_data.attrs, "serialised_nones"):
        model_data.attrs[attr] = None

    # FIXME some checks for consistency
    # use check_dataset from the checks module
    # also check the old checking from 0.5.x

    return model_data


def _pop_serialised_list(attribute_dict, serialised_items):
    serialised_ = attribute_dict.pop(serialised_items, [])
    if not isinstance(serialised_, (list, np.ndarray)):
        return [serialised_]
    else:
        return serialised_


def save_netcdf(model_data, path, model=None):
    original_model_data_attrs = model_data.attrs
    model_data_attrs = model_data.attrs.copy()

    if model is not None and hasattr(model, "_model_run"):
        # Attach _model_run and _debug_data to _model_data
        model_run_to_save = model._model_run.copy()
        for k in ["timeseries_data", "timesteps"]:
            model_run_to_save.pop(k, None)
        model_data_attrs["_model_run"] = model_run_to_save.to_yaml()
        if hasattr(model, "_debug_data"):
            model_data_attrs["_debug_data"] = model._debug_data.to_yaml()

    # Convert dicts attrs to yaml strings
    dict_attrs = [k for k, v in model_data_attrs.items() if isinstance(v, dict)]
    model_data_attrs["serialised_dicts"] = dict_attrs
    for k in dict_attrs:
        model_data_attrs[k] = AttrDict(model_data_attrs[k]).to_yaml()

    # Convert boolean attrs to ints
    bool_attrs = [k for k, v in model_data_attrs.items() if isinstance(v, bool)]
    model_data_attrs["serialised_bools"] = bool_attrs
    for k in bool_attrs:
        model_data_attrs[k] = int(model_data_attrs[k])

    # Convert None attrs to 'None'
    none_attrs = [k for k, v in model_data_attrs.items() if v is None]
    model_data_attrs["serialised_nones"] = none_attrs
    for k in none_attrs:
        model_data_attrs[k] = "None"

    encoding = {
        k: {"zlib": False}
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

    for var in data_vars:
        in_out = "results" if model_data[var].attrs["is_result"] else "inputs"
        out_path = os.path.join(path, "{}_{}.csv".format(in_out, var))
        series = model_data[var].to_series()
        if dropna:
            series = series.dropna()
        series.to_csv(out_path, header=True)


def save_lp(model, path):
    if not model.run_config["backend"] == "pyomo":
        raise IOError("Only the pyomo backend can save to LP.")
    if not hasattr(model, "_backend_model"):
        model.run(build_only=True)
    model._backend_model.write(
        path, format="lp", io_options={"symbolic_solver_labels": True}
    )
