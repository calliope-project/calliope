"""
Copyright (C) 2013-2016 Stefan Pfenninger.
Licensed under the Apache 2.0 License (see LICENSE file).

time_funcs.py
~~~~~~~~~~~~~

Functions to process time series data.

"""

import logging

import pandas as pd
import xarray as xr
from xarray.ufuncs import fabs

from . import utils
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


def _combine_datasets(data0, data1):
    # Combine the clustered with the old unclustered data
    data_new = xr.concat([data0, data1], dim='t')
    # Ensure time dimension is ordered
    data_new = data_new.loc[{'t': data_new.t.to_pandas().index.sort_values()}]
    return data_new


def apply_clustering(data, timesteps, clustering_func, **kwargs):
    """
    Apply the given clustering function to the given data.

    Parameters
    ----------
    data : xarray.Dataset
    timesteps : pandas.DatetimeIndex or list of timesteps or None
    clustering_func : str
        Name of clustering function.
    **args : dict
        Arguments passed to clustering_func.

    Returns
    -------
    data_new_scaled : xarray.Dataset

    """
    # Only apply clustering function on subset of masked timesteps
    if timesteps is not None:
        data_to_cluster = data.loc[{'t': timesteps}]
    else:
        data_to_cluster = data

    data_normalized = normalize(data_to_cluster)

    # Get function from `clustering_func` string
    func = utils.plugin_load(clustering_func, builtin_module='time_clustering')

    result = func(data_normalized, **kwargs)
    clusters = result[0]  # Ignore other stuff returned

    data_new = time_clustering.map_clusters_to_data(data_to_cluster, clusters, how='mean')
    # new_data_normalized = map_clusters_to_data(data_normalized, clusters, how='mean')

    if timesteps is not None:
        # Drop timesteps from old data
        data_new = _combine_datasets(data.drop(timesteps, dim='t'), data_new)

    # Scale the new/combined data so that the mean for each (x, y, variable)
    # combination matches that from the original data
    data_new_scaled = data_new.copy(deep=True)
    for var in dt.get_datavars(data_to_cluster):
        scale_to_match_mean = (data[var].mean(dim='t') / data_new[var].mean(dim='t')).fillna(0)
        data_new_scaled[var] = data_new[var] * scale_to_match_mean

    return data_new_scaled


_RESAMPLE_METHODS = {
    '_weights': 'mean',
    '_time_res': 'sum',
    'r': 'sum',
    'e_eff': 'mean',
}


def resample(data, timesteps, resolution):
    data_new = data.copy(deep=True)
    if timesteps is not None:
        data_new = data_new.loc[{'t': timesteps}]

    # First create a new resampled dataset of the correct size by
    # using first-resample, which should be a quick way to achieve this
    data_rs = data_new.resample(resolution, dim='t', how='first')

    for var in data_new.data_vars:
        if var in _RESAMPLE_METHODS:
            how = _RESAMPLE_METHODS[var]
            data_rs[var] = data_new[var].resample(resolution, dim='t', how=how)
        else:
            # If we don't know how to resample a var, we drop it
            logging.error('Dropping {} because it has no resampling method.'.format(var))
            data_rs = data_rs.drop(var)

    # Get rid of the filled-in NaN timestamps
    data_rs = data_rs.dropna(dim='t', how='all')

    if timesteps is not None:
        # Combine leftover parts of passed in data with new data
        data_rs = _combine_datasets(data.drop(timesteps, dim='t'), data_rs)

    return data_rs


def drop(data, timesteps, padding=None):
    """
    Drop timesteps from data, with optional padding
    around into the contiguous areas encompassed by the timesteps.

    """
    freq = dt.get_freq(data)  # FIXME freq should be an attr on data

    if padding:
        # Series of 1 where timesteps 'exist' and 0 where they don't
        s = (pd.Series(1, index=timesteps)
               .reindex(pd.date_range(timesteps[0], timesteps[-1], freq=freq))
               .fillna(0))

        # Blocks of contiguous 1's in the series
        blocks = (s != s.shift()).cumsum().drop(s[s==0].index)

        # Groups of contiguous areas
        groups = blocks.groupby(blocks).apply(lambda x: (x.index[0], x.index[-1]))

        # Reduce size of each block by `padding` on both sides
        padding = pd.Timedelta(padding)
        dt_indices = [pd.date_range(g[0] + padding, g[1] - padding, freq=freq)
                      for g in groups]

        # Concatenate the DatetimeIndexes by using dummy Series
        timesteps = pd.concat([pd.Series(0, index=i) for i in dt_indices]).index

    return data.drop(timesteps, dim='t')
