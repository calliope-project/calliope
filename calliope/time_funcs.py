"""
Copyright (C) 2013-2016 Stefan Pfenninger.
Licensed under the Apache 2.0 License (see LICENSE file).

time_funcs.py
~~~~~~~~~~~~~

"""

import xarray as xr
from xarray.ufuncs import fabs

from .core import plugin_load
from . import data_tools as dt
from . import time_clustering


def normalize(data):
    """
    Return a copy of data, with the absolute taken and normalized to 0-1.

    The maximum across all regions and timesteps is used to normalize.

    """
    ds = data.copy(deep=True)  # Work off a copy
    for var in dt.get_datavars(data):
        y_var = '_y_def_{}'.format(var)
        for y in ds.coords[y_var].values:
            # Get max across all regions to normalize against
            norm_max = fabs(ds[var].loc[{y_var: y}]).max()
            for x in ds.coords['x'].values:
                df = ds[var].loc[{'x': x, y_var: y}]
                ds[var].loc[{'x': x, y_var: y}] = fabs(df) / norm_max
    return ds


def apply_clustering(data, timesteps=None, clustering_func=None, **args):
    """
    Modifies the passed model's data in-place

    Returns
    -------
    data : xarray.Dataset
        Original data
    new_data : xarray.Dataset
    new_data_scaled : xarray.Dataset

    """
    # Only apply clustering function on subset of masked timesteps
    if timesteps is not None:
        data_to_cluster = data.loc[{'t': timesteps}]
    else:
        data_to_cluster = data

    data_normalized = normalize(data_to_cluster)

    # Get function from `clustering_func` string
    func = plugin_load(clustering_func, builtin_module='time_clustering')

    result = func(data_normalized, **args)
    clusters = result[0]  # Ignore other stuff returned

    data_new = time_clustering.map_clusters_to_data(data_to_cluster, clusters, how='mean')
    # new_data_normalized = map_clusters_to_data(data_normalized, clusters, how='mean')

    if timesteps is not None:
        # Drop timesteps from old data
        data_excluded_from_clustering = data.drop(timesteps, dim='t')
        # Combine the clustered with the old unclustered data
        data_new = xr.concat([data_excluded_from_clustering, data_new], dim='t')
        # Ensure time dimension is ordered
        data_new = data_new.loc[{'t': data_new.t.to_pandas().index.sort_values()}]

    # Scale the new/combined data so that the mean for each (x, y, variable)
    # combination matches that from the original data
    data_new_scaled = data_new.copy(deep=True)
    for var in dt.get_datavars(data_to_cluster):
        scale_to_match_mean = (data[var].mean(dim='t') / data_new[var].mean(dim='t')).fillna(0)
        data_new_scaled[var] = data_new[var] * scale_to_match_mean

    return data_new_scaled
