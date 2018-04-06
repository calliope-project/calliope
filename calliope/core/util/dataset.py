"""
Copyright (C) 2013-2018 Calliope contributors listed in AUTHORS.
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
        and '::'.join([loc, tech]) in loc_techs):
        relevant_loc_techs = ['::'.join([loc, tech])]

    tech = [tech] if tech is not None and isinstance(tech, str) else tech
    loc = [loc] if loc is not None and isinstance(loc, str) else loc

    if tech and not loc:
        relevant_loc_techs = [i for i in loc_techs if i.split('::')[1] in tech]
    elif loc and not tech:
        relevant_loc_techs = [i for i in loc_techs if i.split('::')[0] in loc]
    elif loc and tech:
        loc_techs_set = set(tuple(i.split('::')) for i in loc_techs)
        possible_loc_techs = set((l, t) for l in loc for t in tech)
        relevant_loc_techs = [
            '::'.join(i) for i in possible_loc_techs.intersection(loc_techs_set)
        ]
    else:
        relevant_loc_techs = [None]

    return relevant_loc_techs


def split_loc_techs(data_var, as_='DataArray'):
    """
    Get a DataArray with locations technologies, and possibly carriers
    split into separate coordinates.

    Parameters
    ----------
    data_var : xarray DataArray
        Variable from Calliope model_data, to split loc_techs dimension
    as_ : string
        'DataArray' to return xarray DataArray or 'Series' to return pandas
        Series with dimensions as a MultiIndex

    Returns
    -------
    updated_data_var : xarray DataArray of pandas Series
    """

    # Separately find the loc_techs(_carriers) dimension and all other dimensions
    loc_tech_dim = [i for i in data_var.dims if 'loc_tech' in i]
    if not loc_tech_dim:
        loc_tech_dim = [i for i in data_var.dims if 'loc_carrier' in i]
    non_loc_tech_dims = list(set(data_var.dims).difference(loc_tech_dim))

    if not loc_tech_dim:
        if as_ == 'Series':
            return data_var.to_series()
        elif as_ == 'DataArray':
            return data_var
        else:
            raise ValueError('`as_` must be `DataArray` or `Series`, '
                             'but `{}` given'.format(as_))

    elif len(loc_tech_dim) > 1:
        e = exceptions.ModelError
        raise e("Cannot split loc_techs or loc_techs_carrier dimension "
                "for DataArray {}".format(data_var.name))

    loc_tech_dim = loc_tech_dim[0]
    # xr.Datarray -> pd.Series allows for string operations
    data_var_df = data_var.to_series().unstack(non_loc_tech_dims)
    index_list = data_var_df.index.str.split('::').tolist()

    # carrier_prod, carrier_con, and carrier_export will return an index_list
    # of size 3, all others will be an index list of size 2
    possible_names = ['loc', 'tech', 'carrier']
    names = [i + 's' for i in possible_names if i in loc_tech_dim]

    data_var_df.index = pd.MultiIndex.from_tuples(index_list, names=names)

    # If there were no other dimensions other than loc_techs(_carriers) then
    # nothing was unstacked on creating data_var_df, so nothing is stacked now
    if isinstance(data_var_df, pd.Series):
        data_var_series = data_var_df
    else:
        data_var_series = data_var_df.stack(non_loc_tech_dims)

    if as_ == "Series":
        return data_var_series

    elif as_ == "DataArray":
        updated_data_var = xr.DataArray.from_series(data_var_series)
        updated_data_var.attrs = data_var.attrs

        return updated_data_var

    else:
        raise ValueError('`as_` must be `DataArray` or `Series`, '
                         'but `{}` given'.format(as_))


def reorganise_dataset_dimensions(dataset):
    """
    Reorganise the Dataset dimensions to be alphabetical *except*
    `timesteps`, which must always come last in any DataArray's dimensions
    """

    new_dims = (
        sorted(list(set(dataset.dims.keys()) - set(['timesteps'])))
    ) + ['timesteps']

    updated_dataset = dataset.transpose(*new_dims).reindex(
        {k:dataset[k] for k in new_dims})

    return updated_dataset
