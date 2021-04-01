"""
Copyright (C) since 2013 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

run_checks.py
~~~~~~~~~~~~~

Checks for model consistency and possible errors when preparing run in the backend.

"""
import re
import ast

import numpy as np
import xarray as xr
from calliope.core.attrdict import AttrDict
from calliope.backend.subsets import imask_where, _combine_imasks, _imask_foreach


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


def check_operate_params(model_data, checklist_path, run_config=None):
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
                if "must_exist" in check_config.keys():
                    _create_var(check_config["must_exist"], model_data, run_config)

    return logs["warning"], logs["error"]


def _create_var(var_to_create, model_data, run_config):
    def __is_run_config(search_string):
        return re.search("^run\\.(.*)\\=(.*)$", search_string)

    def __set_in_dict(run_config, key_list, value):
        for key in key_list[:-1]:
            run_config = run_config.setdefault(key, {})
        run_config[key_list[-1]] = value

    def __model_data_var(search_string):
        return re.search("^([\\w\\-]*)\\[(.*)\\]\\=(.*)$", search_string)

    if __is_run_config(var_to_create) is not None:
        config_keys, config_val = __is_run_config(var_to_create).groups()
        config_key_list = config_keys.split(".")
        if not isinstance(config_key_list, list):
            config_key_list = [config_key_list]
        __set_in_dict(run_config, config_key_list, ast.literal_eval(config_val))

    elif __model_data_var(var_to_create) is not None:
        var_name, dims, var_val = __model_data_var(var_to_create).groups()

        imask = _imask_foreach(model_data, dims.replace(" ", "").split(","))
        new_data = imask.where(imask) * ast.literal_eval(var_val)
        if var_name in model_data.data_vars.keys():
            model_data.update({var_name: model_data[var_name].combine_first(new_data)})
        else:
            dim_order = [i for i in model_data.dims]
            model_data[var_name] = new_data.transpose(*[i for i in dim_order if i in new_data.dims])
            model_data[var_name].attrs = {"parameters": 1, "is_result": 0}
    else:
        raise ValueError(
            f"Malformed variable / run config option to create: {var_to_create}"
        )
