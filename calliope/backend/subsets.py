"""
Copyright (C) since 2013 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

subsets.py
~~~~~~~~~~

Create subsets over which the model objects are valid.

"""

import operator
import functools
import re
import ast

import xarray as xr
import pandas as pd
from calliope.core.attrdict import AttrDict
from calliope.core.util.dataset import reorganise_xarray_dimensions


def create_valid_subset(model_data, name, config):
    """
    Returns the subset for which a given constraint, variable or
    expression is valid, based on the given configuration. See `config/subsets.yaml` for
    the configuration definitions.

    Parameters
    ----------

    model_data : xarray.Dataset (calliope.Model._model_data)
    name : str
        Name of the constraint, variable or expression
    config : dict
        Configuration for the constraint, variable or expression

    Returns
    -------
    valid_subset : pandas.MultiIndex

    """

    # Start with a mask that is True where the tech exists at a node (across all timesteps and for a each carrier and cost, where appropriate)
    imask = _imask_foreach(model_data, config.foreach)
    if imask is False:  # i.e. not all of 'foreach' are in model_data
        return None
    # Add "where" info as imasks
    where_array = config.get_key("where", default=[])
    if where_array:
        imask = _imask_where(model_data, name, where_array, imask, "and_")

    # Add imask based on subsets
    imask = _subset_imask(name, config, imask)

    # Only build and return imask if there are some non-zero elements
    if isinstance(imask, xr.DataArray) and imask.sum() != 0:
        # Squeeze out any unwanted dimensions
        if len(imask.dims) > len(config.foreach):
            imask = imask.sum([i for i in imask.dims if i not in config.foreach]) > 0
        # We have a problem if we have too few dimensions at this point...
        if len(imask.dims) < len(config.foreach):
            raise ValueError(f"Missing dimension(s) in imask for set {name}")

        valid_subset = _get_valid_subset(
            reorganise_xarray_dimensions(imask).astype(bool)
        )

        return valid_subset

    else:
        return None


def _param_exists(model_data, param):
    # mask by NaN and INF/-INF values = False, otherwise True
    with pd.option_context("mode.use_inf_as_na", True):
        if isinstance(model_data.get(param), xr.DataArray):
            _da = model_data.get(param)
            return _da.where(pd.notnull(_da)).notnull()
        else:
            return False


def _val_is(model_data, param, val):
    if param.startswith(("model.", "run.")):
        group = param.split(".")[0]
        config = AttrDict.from_yaml_string(model_data.attrs[f"{group}_config"])
        # TODO: update to str.removeprefix() in Python 3.9+
        imask = config.get_key(param[len(f"{group}.") :], None) == ast.literal_eval(val)
    elif param in model_data.data_vars.keys():
        imask = model_data[param] == ast.literal_eval(val)
    else:
        return False

    return imask


def _get_valid_subset(imask):
    if len(imask.dims) == 1:
        return imask[imask].coords.to_index()
    else:
        mask_stacked = imask.stack(dim_0=imask.dims)
        return mask_stacked[mask_stacked].coords.to_index()


def _subset_imask(set_name, set_config, imask):
    # For some masks, we take a subset of a given dimension (e.g. only "out" in 'carrier_tier')
    for subset_name, subset in set_config.get("subset", {}).items():
        # Keep the axis if it is expected for this constraint/variable
        # Set those not in the set to False
        if not isinstance(subset, (list, set, tuple)):
            raise TypeError(
                f"set `{set_name}` must subset over an iterable, instead got non-iterable `{subset}` for subset `{subset_name}`"
            )
        if subset_name in set_config.foreach:
            imask.loc[{subset_name: ~imask[subset_name].isin(subset)}] = False
        # Otherwise squeeze out this dimension after slicing it
        else:
            imask = (
                imask.loc[{subset_name: imask[subset_name].isin(subset)}].sum(
                    subset_name
                )
                > 0
            )
    return imask


def _inheritance(model_data, tech_group):
    # Only for base tech inheritance
    return model_data.inheritance.str.endswith(tech_group)


VALID_HELPER_FUNCTIONS = {
    "inheritance": _inheritance,
}


def _imask_where(
    model_data, set_name, where_array, initial_imask=None, initial_operator=None
):
    """
    Example mask: [cost_purchase, and, [param(energy_cap_max), or, not inheritance(supply_plus)]]
    i.e. a list of "param(...)", "inheritance(...)" and operators.
    Sublists will be handled recursively.
    "not" before param/inheritance will invert the mask
    """
    imasks = []
    operators = []

    def __is_function(imask_string):
        return re.search(r"(\w+)\((\w+)\)", imask_string)

    def __is_equals_statement(imask_string):
        return re.search(r"([\w\.]+)\=([\'\w\.\:\,]+)", imask_string)

    for i in where_array:
        if isinstance(i, list):
            imasks.append(_imask_where(model_data, set_name, i))
        elif i in ["or", "and"]:
            operators.append(i)
        else:
            # If it's not one of the operators, it is either a function, a val=foo, or
            # a val on its own (indicating the value should just exist)

            _not = False
            if i.startswith("not "):
                _not = True
                i = i.replace("not ", "")
            if __is_function(i) is not None:
                func, val = __is_function(i).groups()
                imask = VALID_HELPER_FUNCTIONS[func](model_data, val)
            elif __is_equals_statement(i) is not None:
                param, val = __is_equals_statement(i).groups()
                imask = _val_is(model_data, param, val)
            elif i in model_data.data_vars.keys():
                imask = _param_exists(model_data, i)
            else:
                imask = False  # TODO: this should differntiate between a valid parameter not being in model_data and an e.g. incorrectly spelled parameter
            # Separately check whether the condition should be inverted
            if _not is True:
                imask = ~imask
            imasks.append(imask)
    if len(imasks) - 1 != len(operators):
        raise ValueError(
            f"'where' array for set `{set_name}` must be a list of statements comma separated by {{and, or}} operators."
        )
    # Run through and combine all imasks using defined operators
    imask = imasks[0]
    for i in range(len(imasks) - 1):
        imask = _combine_imasks(imask, imasks[i + 1], operators[i])

    if initial_imask is not None and initial_operator is not None:
        imask = _combine_imasks(imask, initial_imask, initial_operator)

    return imask


def _combine_imasks(curr_imask, new_imask, _operator):
    if _operator in ["or", "or_"]:
        _operator = "or_"
    elif _operator in ["and", "and_"]:
        _operator = "and_"
    else:
        raise ValueError(f"Operator `{_operator}` not recognised")
    imask = functools.reduce(getattr(operator, _operator), [curr_imask, new_imask])
    return imask


def _imask_foreach(model_data, foreach):
    if not all(i in model_data.dims for i in foreach):
        # ignore constraints/variables if the set doesn't even exist (e.g. datesteps)
        return False
    # Start with (carrier, node, tech) and go from there
    initial_imask = model_data.carrier.notnull() * model_data.node_tech.notnull()
    # Squeeze out any of (carrier, node, tech) not in foreach
    reduced_imask = (
        initial_imask.sum([i for i in initial_imask.dims if i not in foreach]) > 0
    )
    # Add other dimensions (costs, timesteps, etc.)
    imask = functools.reduce(
        operator.and_,
        [
            reduced_imask,
            *[model_data[i].notnull() for i in foreach if i not in reduced_imask.dims],
        ],
    )

    return imask
