from typing import Optional
import os
import sys
import ast

import pytest
import xarray as xr

from calliope.backend import backends
import calliope
from calliope import AttrDict


constraint_sets = {
    k: [ast.literal_eval(i) for i in v]
    for k, v in AttrDict.from_yaml(
        os.path.join(os.path.dirname(__file__), "constraint_sets.yaml")
    )
    .as_dict_flat()
    .items()
}

defaults = AttrDict.from_yaml(
    os.path.join(os.path.dirname(calliope.__file__), "config", "defaults.yaml")
)

python36_or_higher = pytest.mark.skipif(
    sys.version_info < (3, 6), reason="Requires ordered dicts from Python >= 3.6"
)


def build_test_model(
    override_dict=None,
    scenario=None,
    model_file="model.yaml",
    timeseries_dataframes=None,
):
    return calliope.Model(
        os.path.join(os.path.dirname(__file__), "test_model", model_file),
        override_dict=override_dict,
        scenario=scenario,
        timeseries_dataframes=timeseries_dataframes,
    )


def check_error_or_warning(error_warning, test_string_or_strings):
    if hasattr(error_warning, "list"):
        output = ",".join(
            str(error_warning.list[i]) for i in range(len(error_warning.list))
        )
    elif hasattr(error_warning, "value"):
        output = str(error_warning.value)
    elif isinstance(error_warning, (list, set)):
        output = ",".join(error_warning)

    if isinstance(test_string_or_strings, list):
        result = all(test_string in output for test_string in test_string_or_strings)
    else:
        result = test_string_or_strings in output

    return result


def check_variable_exists(
    expr_or_constr: Optional[xr.DataArray], variable: str, idx: Optional[dict] = None
):
    """
    Search for existence of a decision variable in a Pyomo constraint.

    Parameters
    ----------
    backend_interface : solver interface library
    constraint : str, name of constraint which could exist in the backend
    variable : str, string to search in the list of variables to check if existing
    """
    if expr_or_constr is None:
        return False

    try:
        var_exists = expr_or_constr.body.astype(str).str.find(variable) > -1
    except (AttributeError, KeyError):
        var_exists = expr_or_constr.astype(str).str.find(variable) > -1

    if idx is not None:
        var_exists = var_exists.loc[idx]

    return var_exists.any()
