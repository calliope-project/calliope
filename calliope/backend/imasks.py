"""
Copyright (C) since 2013 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

masks.py
~~~~~~~~

Functionality to generate an xarray.Dataset of boolean masks for all
constraints and decision variables, based on the model_data xarray.Dataset.

"""

import operator
import functools
import re
import ast

import xarray as xr
import pandas as pd
from calliope.core.attrdict import AttrDict
from calliope.core.util.dataset import reorganise_xarray_dimensions

# [storage_cap_min, or, inheritance(storage), and, export=True, and, run.mode=plan]


def build_imasks(model_data, imask_config):
    """
    Returns a dict of dicts containing valid indices.
    """

    imasks = {}
    for imask_type in imask_config.keys():  # "variables", "expressions", "constraints"
        imasks[imask_type] = {}
        for set_name, set_config in imask_config[imask_type].items():
            imask = create_imask(model_data, set_name, set_config)

            if imask is not None:  # All-zero imasks are not even added to imasks
                imasks[imask_type][set_name] = imask

    return imasks


def create_imask(model_data, set_name, set_config):
    """
    Create boolean masks for constraints and decision variables
    from data stored in config/sets.yaml
    """

    # Start with a mask that is True where the tech exists at a node (across all timesteps and for a each carrier and cost, where appropriate)
    imask = imask_foreach(model_data, set_config.foreach)
    if imask is False:  # i.e. not all of 'foreach' are in model_data
        return None
    # Add "where" info as imasks
    where_array = set_config.get_key("where", default=[])
    if where_array:
        imask = imask_where(model_data, set_name, where_array, imask, "and_")

    # Add imask based on subsets
    imask = subset_imask(set_name, set_config, imask)

    # Only build and return imask if there are some non-zero elements
    if isinstance(imask, xr.DataArray) and imask.sum() != 0:
        # Squeeze out any unwanted dimensions
        if len(imask.dims) > len(set_config.foreach):
            imask = (
                imask.sum([i for i in imask.dims if i not in set_config.foreach]) > 0
            )
        # We have a problem if we have too few dimensions at this point...
        if len(imask.dims) < len(set_config.foreach):
            raise ValueError(f"Missing dimension(s) in imask for set {set_name}")

        imask = reorganise_xarray_dimensions(imask).astype(bool)

        return get_valid_index(imask)

    else:
        return None


def param_exists(model_data, param):
    # mask by NaN and INF/-INF values = False, otherwise True
    with pd.option_context("mode.use_inf_as_na", True):
        if isinstance(model_data.get(param), xr.DataArray):
            _da = model_data.get(param)
            return _da.where(pd.notnull(_da)).notnull()
        else:
            return False


def inheritance(model_data, tech_group):
    # Only for base tech inheritance
    return model_data.inheritance.str.endswith(tech_group)


def val_is(model_data, param, val):
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


def get_valid_index(imask):
    if len(imask.dims) == 1:
        return imask[imask].coords.to_index()
    else:
        mask_stacked = imask.stack(dim_0=imask.dims)
        return mask_stacked[mask_stacked].coords.to_index()


def subset_imask(set_name, set_config, imask):
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


def imask_where(
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

    def _func(imask_string):
        return re.search(r"(\w+)\((\w+)\)", imask_string)

    def _val_is(imask_string):
        return re.search(r"([\w\.]+)\=([\'\w\.\:\,]+)", imask_string)

    for i in where_array:
        if isinstance(i, list):
            imasks.append(imask_where(model_data, set_name, i))
        elif i in ["or", "and"]:
            operators.append(i)
        else:
            # If it's not one of the operators, it is either a function, a val=foo, or
            # a val on its own (indicating the value should just exist)

            _not = False
            if i.startswith("not "):
                _not = True
                i = i.replace("not ", "")
            if _func(i) is not None:
                func, val = _func(i).groups()
                imask = globals()[func](model_data, val)
            elif _val_is(i) is not None:
                param, val = _val_is(i).groups()
                imask = val_is(model_data, param, val)
            elif i in model_data.data_vars.keys():
                imask = param_exists(model_data, i)
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
        imask = combine_imasks(imask, imasks[i + 1], operators[i])

    if initial_imask is not None and initial_operator is not None:
        imask = combine_imasks(imask, initial_imask, initial_operator)

    return imask


def combine_imasks(curr_imask, new_imask, _operator):
    if "or" in _operator:
        _operator = "or_"
    elif "and" in _operator:
        _operator = "and_"
    else:
        raise ValueError(f"Operator `{_operator}` not recognised")
    imask = functools.reduce(getattr(operator, _operator), [curr_imask, new_imask])
    return imask


def imask_foreach(model_data, foreach):
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
