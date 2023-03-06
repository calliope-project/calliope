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
from calliope.core.util.dataset import reorganise_xarray_dimensions
from calliope.backend.subset_parser import generate_where_string_parser
from calliope.exceptions import print_warnings_and_raise_errors


def _inheritance(model_data, **kwargs):
    def __inheritance(tech_group):
        # Only for base tech inheritance
        return model_data.inheritance.str.endswith(tech_group)

    return __inheritance


VALID_HELPER_FUNCTIONS = {
    "inheritance": _inheritance,
}


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
    where_string = config.get_key("where", default=[])
    parsing_errors = []
    if where_string:
        parsed_string = generate_where_string_parser().parse_string(
            where_string, parse_all=True
        )
        where_string_evaluated = parsed_string[0].eval(
            model_data=model_data,
            helper_func_dict=VALID_HELPER_FUNCTIONS,
            errors=parsing_errors,
            defaults=model_data.attrs["defaults"],
        )
        print_warnings_and_raise_errors(errors=parsing_errors)
        imask = imask & where_string_evaluated

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
