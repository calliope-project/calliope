"""
Copyright (C) 2013-2017 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

"""

from calliope import exceptions

import xarray as xr
import pandas as pd


def get_loc_techs(loc_techs, tech=None, loc=None):
    """
    Get a list of loc_techs associated with the given technology and/or location.
    If multiple of both loc and tech are given, the function will return any
    combination of members of loc and tech lists found in loc_techs.

    Parameters
    ----------
    loc_techs : list
        set of loc_techs to search for the relevant tech and/or loc
    tech : string or list of strings, default None
        technology/technologies to search for in the set of location:technology
    loc : string or list of strings, default None
        location(s) to search for in the set of location:technology

    Returns
    -------
    relevant_loc_techs : list of strings
    """

    # If both are strings, there is only one loc:tech possibility to look for
    if (isinstance(tech, str) and isinstance(loc, str)
        and ':'.join([loc, tech]) in loc_techs):
        relevant_loc_techs = [':'.join([loc, tech])]

    tech = [tech] if tech is not None and isinstance(tech, str) else tech
    loc = [loc] if loc is not None and isinstance(loc, str) else loc

    if tech and not loc:
        relevant_loc_techs = [i for i in loc_techs if i.split(':', 1)[1] in tech]
    elif loc and not tech:
        relevant_loc_techs = [i for i in loc_techs if i.split(':', 1)[0] in loc]
    elif loc and tech:
        loc_techs_set = set(
            (i.split(':')[0], i.split(':')[1]) for i in loc_techs
        )
        possible_loc_techs = set((l, t) for l in loc for t in tech)
        relevant_loc_techs = [
            ':'.join(i) for i in possible_loc_techs.intersection(loc_techs_set)
        ]
    else:
        relevant_loc_techs = [None]

    return relevant_loc_techs


def split_loc_techs(data_var):
    """
    Get a DataArray with locations and technologies split into seperate
    coordinates.

    Parameters
    ----------
    data_var : xarray DataArray
        Variable from Calliope model_data, to split loc_techs dimension

    Returns
    -------
    updated_data_var : xarray DataArray
    """

    # Find the loc_techs dimension
    loc_tech_dim = [i for i in data_var.dims if 'loc_techs' in i]
    non_loc_tech_dims = list(set(data_var.dims).difference(loc_tech_dim))
    if not loc_tech_dim:
        return data_var

    elif len(loc_tech_dim) > 1:
        e = exceptions.ModelError
        raise e("Cannot split loc_techs dimension "
                "for DataArray {}".format(data_var.name))
    else:
        loc_tech_dim = loc_tech_dim[0]
        data_var_series = data_var.to_series().unstack(non_loc_tech_dims)
        data_var_series.index = pd.MultiIndex.from_tuples(
           data_var_series.index.str.split(':', 1).tolist(),
           names=['locs', 'techs']
        )
        updated_data_var = xr.DataArray.from_series(
            data_var_series.stack(non_loc_tech_dims)
        )
    return updated_data_var
