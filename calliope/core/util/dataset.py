"""
Copyright (C) since 2013 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

"""

from calliope import exceptions
import xarray as xr
import pandas as pd


def reorganise_xarray_dimensions(data):
    """
    Reorganise Dataset or DataArray dimensions to be alphabetical *except*
    `timesteps`, which must always come last in any DataArray's dimensions
    """

    if not (isinstance(data, xr.Dataset) or isinstance(data, xr.DataArray)):
        raise TypeError(
            "Must provide either xarray Dataset or DataArray to be reorganised"
        )

    steps = [i for i in ["datesteps", "timesteps"] if i in data.dims]

    if isinstance(data, xr.Dataset):
        new_dims = (sorted(list(set(data.dims.keys()) - set(steps)))) + steps
    elif isinstance(data, xr.DataArray):
        new_dims = (sorted(list(set(data.dims) - set(steps)))) + steps

    updated_data = data.transpose(*new_dims).reindex({k: data[k] for k in new_dims})

    return updated_data
