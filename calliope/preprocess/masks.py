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

import xarray as xr
import pandas as pd
from calliope.core.attrdict import AttrDict
from calliope.core.util.dataset import reorganise_xarray_dimensions


# [storage_cap_min, or, inheritance(storage), and, export=True, and, run.mode=plan]
def create_imask_ds(model_data, sets):
    """
    Create boolean masks for constraints and decision variables
    from data stored in config/sets.yaml
    """

    ds = xr.Dataset()
    run_config = AttrDict.from_yaml_string(model_data.attrs["run_config"])

    for set_group, sets in sets.items():
        for set_name, set_config in sets.items():
            # Start with a mask that is True where the tech exists at a node (across all timesteps and for a each carrier and cost, where appropriate)
            imask = imask_foreach(model_data, set_config.foreach)
            if imask is False:  # i.e. not all of 'foreach' are in model_data
                continue

            # Add "where" info as imasks
            where_array = set_config.get_key("where", default=[])
            if where_array:
                imask = imask_where(model_data, where_array, imask, "and_")

            # Add imask based on subsets
            imask = subset_imask(set_config, imask)

            # Add to dataset if not already there
            if isinstance(imask, xr.DataArray) and imask.sum() != 0:
                # Squeeze out any unwanted dimensions
                if len(imask.dims) > len(set_config.foreach):
                    imask = imask.sum([i for i in imask.dims if i not in set_config.foreach]) > 0
                # We have a problem if we have too few dimensions at this point...
                if len(imask.dims) < len(set_config.foreach):
                    raise ValueError(f'Missing dimension(s) in imask for set {set_name}')
                if k not in ds.data_vars.keys():
                    ds = ds.merge(imask.to_dataset(name=set_name))
                else:  # if already in dataset, it should look exactly the same
                    assert (ds[set_name] == imask).all().item()
                ds[set_name].attrs[set_group] = 1  # give info on whether the mask is for a constraint, variable, etc.
                if set_group == 'variables':
                    ds[set_name].attrs['domain'] = set_config.get_key('domain', default='Reals')  # added info for variables

    return reorganise_xarray_dimensions(ds)


def param_exists(model_data, param):
    # mask by NaN and INF/-INF values = False, otherwise True
    pd.options.mode.use_inf_as_na = True  # can only catch infs as nans using pandas
    if isinstance(model_data.get(param), xr.DataArray):
        _da = model_data.get(param)
        return _da.where(pd.notnull(_da)).notnull()
    else:
        return False
    pd.options.mode.use_inf_as_na = False


def inheritance(model_data, tech_group):
    # Only for base tech inheritance
    return model_data.inheritance.str.endswith(tech_group)


def val_is(model_data, param, val):
    print(param)
    if "run." in param:
        run_config = AttrDict.from_yaml_string(model_data.attrs["run_config"])
        imask = run_config[param.strip("run.")] == val
    else:
        imask = model_data[param] == val

    return imask


def subset_imask(set_config, imask):
    # For some masks, we take a subset of a given dimension (e.g. only "out" in 'carrier_tier')
    for set_name, subset in set_config.get("subsets", {}).items():
        # Keep the axis if it is expected for this constrain/variable
        if set_name in set_config.foreach:
            imask = imask.loc[{set_name: [i for i in subset if i in imask[set_name]]}]
        # Otherwise squeeze out this dimension after slicing it
        else:
            imask = imask.loc[{set_name: [i for i in subset if i in imask[set_name]]}].sum(set_name) > 0


def imask_where(model_data, where_array, initial_imask=None, initial_operator=None):
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
        return re.search(r"([\w\.]+)\=(\w+)", imask_string)

    for i in where_array:
        if isinstance(i, list):
            imasks.append(imask_where(model_data, i))
        elif i in ["or", "and"]:
            operators.append(i)
        else:
            # If it's not one of the operators, it is either a function, a val=foo, or
            # a val on its own (indicating the value should just exist)

            _not = False
            if i.startswith("not "):
                _not = True
                i = i.strip("not ")
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
        # TODO: better error message
        raise ValueError("Expression must be a list of statements separated by operators.")
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
