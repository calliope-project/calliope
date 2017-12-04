"""
Copyright (C) 2013-2017 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

time_masks.py
~~~~~~~~~~~~~

Functions to pick timesteps from data given certain criteria.

"""

import pandas as pd

from . import time_funcs
from . import utils


def _get_array(data, var, tech, **kwargs):
    subset = {'techs':tech}
    if kwargs is not None:
        subset.update({k:v for k, v in kwargs.items()})
    unusable_dims = (set(subset.keys())
                        .difference(["techs", "locs"])
                        .difference(data[var].dims)
                    )
    if unusable_dims:
        raise exceptions.ModelError("attempting to mask time based on "
                                    "technology {}, but dimension(s) "
                                    "{} don't exist for parameter {}".format(
                                        tech, unusable_dims, var.name))
    arr = utils.split_loc_techs(data[var].copy()).loc[subset]
    arr = arr.mean(dim=[i for i in arr.dims if i is not 'timesteps']).to_pandas()
    return arr


def zero(data, tech, var='resource', **kwargs):
    """
    Returns timesteps where ``var`` for the technology ``tech``
    across the given list of ``locations`` is zero.

    If ``locations`` not given, uses all available locations.

    """
    s = _get_array(data, var, tech, **kwargs)

    return s[s == 0].index


def _concat_indices(indices):
    return pd.concat([i.to_series() for i in indices]).sort_index().index


def _get_minmax_timestamps(series, length, n, how='max', padding=None):
    # Get the max/min timestamps
    group = series.groupby(pd.TimeGrouper(length)).mean()
    timesteps = []
    for _ in range(n):
        if how == 'max':
            ts = group.idxmax()
        elif how == 'min':
            ts = group.idxmin()
        timesteps.append(ts)
        group = group.drop(ts)

    # Get range of timestamps including padding
    full_timesteps = []
    for ts in timesteps:
        ts_end = ts + pd.Timedelta(length)
        if padding is not None:
            ts -= pd.Timedelta(padding)
            ts_end += pd.Timedelta(padding)
        ts_range = pd.date_range(ts, ts_end, freq='1H')[:-1]
        full_timesteps.append(ts_range)

    ts_index = _concat_indices(full_timesteps)

    return ts_index


def extreme(data, tech, var='resource', how='max',
            length='1D', n=1, groupby_length=None,
            padding=None, **kwargs):
    """
    Returns timesteps for period of ``length`` where ``var`` for the technology
    ``tech`` across the given list of ``locations`` is either minmal
    or maximal.

    Parameters
    ----------
    data : xarray.Dataset
    tech : str
        Technology whose `var` to find extreme for.
    var : str, optional
        default 'resource'
    how : str, optional
        'max' (default) or 'min'.
    length : str, optional
        Defaults to '1D'.
    n : int, optional
        Number of periods of `length` to look for, default is 1.
    groupby_length : str, optional
        Group time series and return `n` periods of `length`
        for each group.
    padding : int, optional
        Pad beginning and end of the unmasked area by the number of
        timesteps given.
    normalize : bool, optional
        If True (default), data is normalized
        using :func:`~calliope.time_funcs.normalized_copy`.
    kwargs : dict, optional
        Dimensions of the selected var over which to index. Any remaining
        dimensions will be flattened by mean

    """
    arr = _get_array(data, var, tech, **kwargs)

    return _extreme(arr, how, length, n, groupby_length, padding)


def extreme_diff(data, tech0, tech1, var='resource', how='max',
                 length='1D', n=1, groupby_length=None,
                 padding=None, normalize=True, **kwargs):
    if normalize:
        data_n = time_funcs.normalized_copy(data)
    else:
        data_n = data
    arr0 = _get_array(data_n, var, tech0, **kwargs)
    arr1 = _get_array(data_n, var, tech1, **kwargs)
    arr = arr0 - arr1

    return _extreme(arr, how, length, n, groupby_length, padding)


def _extreme(arr, how='max',
             length='1D', n=1, groupby_length=None,
             padding=None):

    if groupby_length:
        groupby = pd.TimeGrouper(groupby_length)
        group_indices = []
        grouping = arr.groupby(groupby)
        for k in grouping.groups.keys():
            s = grouping.get_group(k)
            group_indices.append(_get_minmax_timestamps(s, length, n, how, padding))
        ts_index = _concat_indices(group_indices)
    else:
        ts_index = _get_minmax_timestamps(arr, length, n, how, padding)

    return ts_index


_WEEK_DAY_FUNCS = {
    'extreme': extreme,
    'extreme_diff': extreme_diff
}


def week(data, day_func, **day_func_kwargs):
    # Get extreme day time index
    func = _WEEK_DAY_FUNCS[day_func]
    day = func(data, **day_func_kwargs)

    # Using day of week, figure out how many days before and after to get
    # a complete week
    days_before = 6 - day[0].dayofweek
    days_after = 6 - days_before

    # Turn it into a week
    # FIXME: assumes 1H timestep length
    start_hour = day[0] - pd.Timedelta('{}D'.format(days_before))
    end_hour = day[-1] + pd.Timedelta('{}D'.format(days_after))
    before = pd.date_range(start_hour, day[0], freq='1H')[:-1]
    after = pd.date_range(day[-1], end_hour, freq='1H')[1:]
    result_week = before.append(day).append(after)

    return result_week
