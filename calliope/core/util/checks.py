"""
Copyright (C) since 2013 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

run_checks.py
~~~~~~~~~~~~~

Checks for model consistency and possible errors when preparing run in the backend.

"""
import ast

import numpy as np
import xarray as xr
from calliope.core.attrdict import AttrDict
from calliope.backend.subsets import imask_where, combine_imasks


def _equation_parser(model_data, eq_dict, check_name):
    """
    Take a dictionary of the form
    {lhs: [param1, combination_method1, param2, ...],
     rhs: [param3, combination_method2, param4, ...],
     operator: operator1}
    where `param...` is a model_data data variable,
    `combination_method...` is a numpy method to combine arrays (e.g. `multiply`, `add`), and
    `operator...` is a mathematical operator to compare arrays (e.g. `<=`, `==`, `lt`, `ge`).

    The following example:
    {lhs: [energy_cap_max, multiply, storage_cap_max, divide, energy_eff],
     rhs: [energy_cap_per_storage_cap_min, add, parasitic_eff, multiply, energy_cap_max],
     operator: "le"}
    will be parsed as:
    ```
     np.divide(np.multiply(model_data.energy_cap_max, model_data.storage_cap_max), model_data.energy_eff)
     <=
     np.multiply(np.add(model_data.energy_cap_per_storage_cap_min, model_data.parasitic_eff), model_data.energy_cap_max)
    ```
    Contrary to standard mathematical operations, multiplications, divisions, additions, and subtractions are handled in the order they appear.
    However, nested lists can be included, which will be handled recursively, e.g.:
    `[foo, multiply, bar, divide, baz]` will be parsed as:
    `np.divide(np.multiply(foo, bar), baz))`

    while `[foo, multiply, [bar, divide, baz]]` will be parsed as:
    `np.multiply(model_data.foo, np.divide(bar, baz))`

    The parser returns a single boolean or a boolean array, where True refers to value(s) meeting the specified conditions.
    """

    return combine_imasks(
        _parse_vars(model_data, eq_dict["lhs"], check_name),
        _parse_vars(model_data, eq_dict["rhs"], check_name),
        eq_dict.operator,
    )


def _parse_vars(model_data, var_list, check_name):
    operators = []
    parsed_vars = []
    for var in var_list:
        if isinstance(var, list):
            parsed_vars.append(_parse_vars(var))
        elif isinstance(var, (int, float)):
            parsed_vars.append(var)
        elif hasattr(np, var):
            operators.append(getattr(np, var))
        elif var in model_data.data_vars.keys():
            parsed_vars.append(model_data[var])
        elif isinstance(var, str) and (
            var.startswith("run.") or var.startswith("model.")
        ):
            group, key = var.split(".", 1)
            config = AttrDict.from_yaml_string(model_data.attrs[f"{group}_config"])
            parsed_vars.append(config.get_key(key, 1))
        else:
            raise ValueError(
                f"Unable to parse variable {var} in tabular data check {check_name}"
            )

    assert len(parsed_vars) - 1 == len(
        operators
    ), f"Too many numpy operators compared to variables to check in tabular data check {check_name}"
    var = parsed_vars[0]
    for i in range(len(parsed_vars) - 1):
        var = operators[i](var, parsed_vars[i + 1])

    return var


def check_tabular_data(model_data, checklist_path):
    """
    Check for issues in tabular data based on parsing YAML configurations.
    YAML configurations include requirements values that can exist in specfic N-dimensional arrays,
    as well as requirements between arrays (where e.g. the values in one array must always be smaller than the values in another).

    Parameters
    ----------
    model_data: xarray Dataset
    checklist_path: str
        path to YAML checklist which includes the arrays to compare
        as well as the error/warning to raise on any element in the comparison returning True when checked.

    Returns
    -------
    warnings : list
        possible problems that do not prevent the model run
        from continuing
    errors : list
        serious issues that should raise a ModelError

    """
    checklist = AttrDict.from_yaml(checklist_path)

    logs = {"warning": [], "error": []}

    for check_name, check_config in checklist.items():
        imask = imask_where(model_data, check_name, check_config.fail_where)
        if isinstance(imask, xr.DataArray) and imask.any() or imask is True:
            if "fail_if_any" in check_config.keys():
                imask *= _equation_parser(
                    model_data, check_config["fail_if_any"], check_name
                )
            if isinstance(imask, xr.DataArray) and imask.any() or imask is True:
                for exctype, exclist in logs.items():
                    if exctype in check_config.keys():
                        exclist.append(check_config[exctype])

    return logs["warning"], logs["error"]
