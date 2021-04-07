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
import numpy as np
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

    # Start with a mask that is True where the tech exists at a node
    # (across all timesteps and for a each carrier and cost, where appropriate)
    imask = _imask_foreach(model_data, config.foreach)
    if imask is False:  # i.e. not all of 'foreach' are in model_data
        return None
    # Add "where" info as imasks
    where_array = config.get_key("where", default=[])
    if where_array:
        imask = imask_where(model_data, name, where_array, imask, "and_")

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
    """ Assign a model data variable NaN and INF/-INF values to False, otherwise True """
    with pd.option_context("mode.use_inf_as_na", True):
        if isinstance(model_data.get(param), xr.DataArray):
            _da = model_data.get(param)
            return _da.where(pd.notnull(_da)).notnull()
        else:
            return False


def _val_is(model_data, param, val, _operator):
    """
    Return True if `param` exists as a model data variable or a configuration option and if it matches `val`.
    If param is a model data variable, a boolean array will be returned.
    `_operator` defines the comparison between `param` and `val`, e.g. `==`, `<=`, etc...
    """
    if val == "inf":
        val = np.inf
    else:
        val = ast.literal_eval(val)
    if param.startswith(("model.", "run.")):
        group = param.split(".")[0]
        config = AttrDict.from_yaml_string(model_data.attrs[f"{group}_config"])
        # TODO: update to str.removeprefix() in Python 3.9+
        lhs = config.get_key(param[len(f"{group}.") :], None)
    elif param in model_data.data_vars.keys():
        lhs = model_data[param]
    else:
        return False

    imask = combine_imasks(lhs, val, _operator)
    return imask


def _get_valid_subset(imask):
    """ From a boolean N-dimensional array to a pandas Index (1 dimension) / MultiIndex (>1 dimension) where index values are `True`"""
    if len(imask.dims) == 1:
        return imask[imask].coords.to_index()
    else:
        mask_stacked = imask.stack(dim_0=imask.dims)
        return mask_stacked[mask_stacked].coords.to_index()


def _subset_imask(set_name, set_config, imask):
    """ For some masks, we take a subset of a given dimension (e.g. only "out" in "carrier_tier") """
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
    """ Checks for base tech group inheritance (supply, supply_plus, etc.)"""
    return model_data.inheritance.str.endswith(tech_group)


VALID_HELPER_FUNCTIONS = {
    "inheritance": _inheritance,
}


def imask_where(
    model_data, set_name, where_array, initial_imask=None, initial_operator=None
):
    """
    Take a set of arrays (`where_array`) and overlay them to create a multidimensional
    boolean array (here known as `imask`) including all dimensions in the `where_array`
     arrays, where an indexed element is True if:

    1. there is some non-null / non-infinite data in all `where_array` elements
        (`where_array` element is a simple string referring to model_data variable, e.g. `energy_cap_max`)
    2. a configuration option is set to a specific value
        (`where_array` element is of the form `run.config_option=val` or `model.config_option=val`).
        If `val` is a string, it must be enclosed in `'`, e.g. `'plan'`.
    3. `techs` inherit from a specific base tech group
        (`where_array` element is of the form `inheritance(tech_group_name)`, where `tech_group_name` is e.g. `supply`)

    Each element in the `where_array` is combined to produce the final boolean array
    based on connecting operators `and`/`or`
    For example: `[energy_cap_max, and, run.mode='plan', or, inheritance(supply)]`

    Sublists will be handled recursively.

    `not` before any `where_array` element that isn't an operator will invert
    the result of that comparison (e.g. `not inheritance(supply)`).

    An initial boolean array `initial_imask` can be optionally included as a starting point,
    along with an initial operator to set how it is to be compared to the imask produced by the `where_array`.

    Parameters
    ----------
    model_data: xarray Dataset
    set_name: str
        Name of the subset being produced, to use when raising errors/warnings
    where_array: list of strings
        Elements to find and compare to produce boolean imask, must be of the form
        [element_to_compare, operator, element_to_compare, operator, ...].
        `element_to_compare` can itself be a list of strings, and will be handled recursively.
    initial_imask: xarray DataArray or None, default=None
        Boolean array to initialise the where_array
    initial_operator: str or None, default=None
        Operator to connect `initial_imask` with the result of processing `where_array`.
        Can be one of "and" or "or".

    Returns
    -------
    imask: xarray DataArray or boolean
        Boolean N-dimensional array where (multi)index values are True for the
        (multi)index elements to be included in the final subset.
        If there are no valid (multi)index elements, imask will return `False`.
    """
    imasks = []
    operators = []

    def __is_function(imask_string):
        return re.search(r"(\w+)\((\w+)\)", imask_string)

    def __is_statement(imask_string):
        return re.search(r"([\w\.]+)(\=|\<|\>|\<=|\>=)([\'\w\.\:\,]+)", imask_string)

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
            if __is_function(i) is not None:
                func, val = __is_function(i).groups()
                imask = VALID_HELPER_FUNCTIONS[func](model_data, val)
            elif __is_statement(i) is not None:
                param, statement_operator, val = __is_statement(i).groups()
                imask = _val_is(model_data, param, val, statement_operator)
            elif i in model_data.data_vars.keys():
                imask = _param_exists(model_data, i)
            else:
                imask = False  # TODO: this should differentiate between a valid parameter not being in model_data and an e.g. incorrectly spelled parameter
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


def _translate_operator(_operator):
    """ Go from 'human-readable' operators to the `operator` package methods """
    if _operator == "or":
        _operator = "or_"
    elif _operator == "and":
        _operator = "and_"
    elif _operator == "=":
        _operator = "eq"
    elif _operator == "<":
        _operator = "lt"
    elif _operator == ">":
        _operator = "gt"
    elif _operator == "<=":
        _operator = "le"
    elif _operator == ">=":
        _operator = "ge"
    elif not hasattr(operator, _operator):
        raise ValueError(f"Operator `{_operator}` not recognised")

    return getattr(operator, _operator)


def combine_imasks(curr_imask, new_imask, _operator):
    """ Combine two booleans / boolean arrays using a specified operator (e.g. `and`, `or`)"""
    imask = functools.reduce(_translate_operator(_operator), [curr_imask, new_imask])
    return imask


def _imask_foreach(model_data, foreach):
    """
    Create an initial array of booleans based on core Calliope dimensions.
    These include `techs`, `nodes`, `timesteps`, `carriers`, `carrier_tiers`.
    If `foreach` is [nodes, techs] then all instances of a technology existing at a node
    will result in a value of True in the output array (all others will be False).
    The number of elements given in `foreach` is equal to the number of dimenstions on
    the output array. If not all dimensions exist in the model, this function will simply
    return False.

    """
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
