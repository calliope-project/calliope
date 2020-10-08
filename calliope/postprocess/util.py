"""
Copyright (C) since 2013 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

util.py
~~~~~~~

Analysis util functions.

"""


def subset_sum_squeeze(data, subset={}, sum_dims=None, squeeze=False):
    """
    Take an xarray DataArray and apply indexing, summing and squeezing,
    to prepare it for analysis.

    Parameters
    ----------
    data : xarray DataArray
        A Calliope model data variable, either input or output, which has been
        reformatted to deconcatenate loc_techs (or loc_tech_carriers/loc_carriers)
        using calliope.Model().get_formatted_array(original_data)
    subset : dict, default {}
        key:value pairs for indexing data. Uses xarray `loc[]` to index.
    sum_dims : str or list of strings, default None
        Names of dimensions over which to sum the data.
    squeeze : bool, str, or list of strings, default False
        If True, remove all dimensions of length 1
        If string, try to remove that dimension, if it is of length 1
        If list of strings, try to remove all listed dimensions, if they are length 1

    Returns
    -------
    data : xarray DataArray

    Examples
    --------
    (in) data = carrier_prod, dimensions = (locs: 2, techs: 5, carriers: 1, timesteps: 100)
    subset_sum_squeeze(
        data, subset={'techs': ['ccgt', 'csp']}, sum_dims='locs', squeeze=True
    )
    (out) data = carrier_prod, dimensions = (techs: 2, timesteps: 100)

    """
    if subset:  # first, subset the data
        allowed_subsets = {k: v for k, v in subset.items() if k in data.dims}
        allowed_items = {}

        for k, v in allowed_subsets.items():
            if isinstance(v, str):
                v = [v]
            allowed_items[k] = [i for i in v if i in data[k].values]

        data = data.loc[allowed_items]

    if sum_dims:  # second, sum along all necessary dimensions
        data = data.sum(sum_dims)

    if squeeze:  # finally, squeeze out single length dimensions
        if len(data.techs) == 1:
            dims_to_squeeze = [
                i for i in data.dims if len(data[i]) == 1 and i != "techs"
            ]
            data = data.squeeze(dims_to_squeeze)
        else:
            data = data.squeeze()

    return data
