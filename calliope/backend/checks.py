"""
Copyright (C) since 2013 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

run_checks.py
~~~~~~~~~~~~~

Checks for model consistency and possible errors when preparing run in the backend.

"""
import numpy as np
import xarray as xr
from calliope.core.attrdict import AttrDict
from calliope.backend.subsets import imask_where, _combine_imasks


def _equation_parser(model_data, eq_dict):
    def _parse_vars(var_list):
        operators = []
        parsed_vars = []
        for var in var_list:
            if isinstance(var, list):
                parsed_vars.append(_parse_vars(var))
            elif hasattr(np, var):
                operators.append(getattr(np, var))
            elif var in model_data.data_vars.keys():
                parsed_vars.append(model_data[var])
            elif isinstance(var, (int, float)) or (
                isinstance(var, str) and var.isnumeric()
            ):
                parsed_vars.append(var)
        assert len(parsed_vars) - 1 == len(operators)
        var = parsed_vars[0]
        for i in range(len(parsed_vars) - 1):
            var = operators[i](var, parsed_vars[i + 1])

        return var

    return _combine_imasks(
        _parse_vars(eq_dict["lhs"]), _parse_vars(eq_dict["rhs"]), eq_dict.operator
    )


def check_operate_params(model_data, checklist_path):
    """
    if model mode = `operate`, check for clashes in capacity constraints.
    In this mode, all capacity constraints are set to parameters in the backend,
    so can easily lead to model infeasibility if not checked.

    Returns
    -------
    comments : AttrDict
        debug output
    warnings : list
        possible problems that do not prevent the model run
        from continuing
    errors : list
        serious issues that should raise a ModelError

    """
    checklist = AttrDict.from_yaml(checklist_path)

    logs = {"warning": [], "error": []}

    for check_name, check_config in checklist.items():
        imask = imask_where(model_data, check_name, check_config.where)
        if isinstance(imask, xr.DataArray) and imask.any() or imask is True:
            if "assert" in check_config.keys():
                imask *= ~_equation_parser(model_data, check_config["assert"])
            if isinstance(imask, xr.DataArray) and imask.any() or imask is True:
                for exctype, exclist in logs.items():
                    if exctype in check_config.keys():
                        exclist.append(check_config[exctype])

    return logs["warning"], logs["error"]
