"""
Copyright (C) 2013-2017 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

time_funcs.py
~~~~~~~~~~~~~

Functions to process time series data.

"""

import logging

import pandas as pd
import xarray as xr
from xarray.ufuncs import fabs  # pylint: disable=no-name-in-module

from . import utils
from . import time_clustering


def normalized_copy(data):
    """
    Return a copy of data, with the absolute taken and normalized to 0-1.

    The maximum across all regions and timesteps is used to normalize.

    """
    ds = data.copy(deep=True)  # Work off a copy
    data_vars_in_t = [v for v in time_clustering._get_datavars(data)
                      if 't' in data[v].dims]
    for var in data_vars_in_t:
        for y in ds.coords['y'].values:
            # Get max across all regions to normalize against
            norm_max = fabs(ds[var].loc[{'y': y}]).max()
            for x in ds.coords['x'].values:
                df = ds[var].loc[{'x': x, 'y': y}]
                ds[var].loc[{'x': x, 'y': y}] = fabs(df) / norm_max
    return ds


def _copy_non_t_vars(data0, data1):
    """Copies non-t-indexed variables from data0 into data1, then
    returns data1"""
    non_t_vars = [v for v in data0.data_vars
                  if 't' not in data0[v].dims]
    # Manually copy over variables not in `t`. If we don't do this,
    # these vars get polluted with a superfluous `t` dimension
    for v in non_t_vars:
        data1[v] = data0[v]
    return data1


def _combine_datasets(data0, data1):
    """Concatenates data0 and data1 along the t dimension"""
    data_new = xr.concat([data0, data1], dim='t')
    # Ensure time dimension is ordered
    data_new = data_new.loc[{'t': data_new.t.to_pandas().index.sort_values()}]

    return data_new


def apply_clustering(data, timesteps, clustering_func, how, normalize=True, **kwargs):
    """
    Apply the given clustering function to the given data.

    Parameters
    ----------
    data : xarray.Dataset
    timesteps : pandas.DatetimeIndex or list of timesteps or None
    clustering_func : str
        Name of clustering function.
    how : str
        How to map clusters to data. 'mean' or 'closest'.
    normalize : bool, optional
        If True (default), data is normalized before clustering is applied,
        using :func:`~calliope.time_funcs.normalized_copy`.
    **kwargs : optional
        Arguments passed to clustering_func.

    Returns
    -------
    data_new_scaled : xarray.Dataset

    """
    # Only apply clustering function on subset of masked timesteps
    if timesteps is None:
        data_to_cluster = data
    else:
        data_to_cluster = data.loc[{'t': timesteps}]

    if normalize:
        data_normalized = normalized_copy(data_to_cluster)
    else:
        data_normalized = data_to_cluster

    # Get function from `clustering_func` string
    func = utils.plugin_load(clustering_func, builtin_module='time_clustering')

    result = func(data_normalized, **kwargs)
    clusters = result[0]  # Ignore other stuff returned

    data_new = time_clustering.map_clusters_to_data(data_to_cluster, clusters,
                                                    how=how)

    if timesteps is None:
        data_new = _copy_non_t_vars(data, data_new)
    else:
        # Drop timesteps from old data
        data_new = _copy_non_t_vars(data, data_new)
        data_new = _combine_datasets(data.drop(timesteps, dim='t'), data_new)
        data_new = _copy_non_t_vars(data, data_new)

    # Scale the new/combined data so that the mean for each (x, y, variable)
    # combination matches that from the original data
    data_new_scaled = data_new.copy(deep=True)
    data_vars_in_t = [v for v in time_clustering._get_datavars(data)
                      if 't' in data[v].dims]
    for var in data_vars_in_t:
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

    timestep_vars = [v for v in data_new.data_vars
                     if 't' in data_new[v].dims]

    # Resampling adds spurious `t` dimension to non-t vars, correct that
    for v in data_rs.data_vars:
        if v not in timestep_vars:
            data_rs[v] = data[v]

    for var in timestep_vars:
        if var in _RESAMPLE_METHODS:
            how = _RESAMPLE_METHODS[var]
            data_rs[var] = data_new[var].resample(resolution, dim='t', how=how)
        else:
            # If we don't know how to resample a var, we drop it
            logging.error('Dropping {} because it has no resampling method.'.format(var))
            data_rs = data_rs.drop(var)

    # Get rid of the filled-in NaN timestamps
    data_rs = data_rs.dropna(dim='t', how='all')
    data_rs.attrs['opmode_safe'] = True  # Resampling still permits operational mode

    if timesteps is not None:
        # Combine leftover parts of passed in data with new data
        data_rs = _copy_non_t_vars(data, data_rs)
        data_rs = _combine_datasets(data.drop(timesteps, dim='t'), data_rs)
        data_rs = _copy_non_t_vars(data, data_rs)
        # Having timesteps with different lengths does not permit operational mode
        data_rs.attrs['opmode_safe'] = False

    return data_rs


def drop(data, timesteps, padding=None):
    """
    Drop timesteps from data, with optional padding
    around into the contiguous areas encompassed by the timesteps.

    """
    if padding:
        ts_per_day = time_clustering._get_timesteps_per_day(data)
        freq = '{}H'.format(24 / ts_per_day)

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

    # 'Distribute weight' of the dropped timesteps onto the remaining ones
    dropped_weight = data._weights.loc[{'t': timesteps}].sum()

    data = data.drop(timesteps, dim='t')

    data['_weights'] = data['_weights'] + (dropped_weight / len(data['_weights']))

    return data
