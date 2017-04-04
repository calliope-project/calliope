"""
Copyright (C) 2013-2017 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

time_masks.py
~~~~~~~~~~~~~

Functions to pick timesteps from data given certain criteria.

"""

import pandas as pd

from . import time_funcs


def _get_array(data, var, tech, locations):
    arr = data[var]
    arr = arr.loc[{'y': tech}]
    if locations is not None:
        arr = arr.loc[{'x': locations}]
    return arr


def zero(data, tech, var='r', locations=None):
    """
    Returns timesteps where ``var`` for the technology ``tech``
    across the given list of ``locations`` is zero.

    If ``locations`` not given, uses all available locations.

    """
    arr = _get_array(data, var, tech, locations)
    s = arr.mean(dim='x').to_pandas()  # Get a t-indexed Series

    return s[s == 0].index


def _concat_indices(indices):
    return pd.concat([i.to_series() for i in indices]).sort_index().index


def _get_minmax_timestaps(series, length, n, how='max', padding=None):
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


def extreme(data, tech, var='r', how='max',
            length='1D', n=1, groupby_length=None,
            locations=None, padding=None):
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
        default 'r'
    how : str, optional
        'max' (default) or 'min'.
    length : str, optional
        Defaults to '1D'.
    n : int, optional
        Number of periods of `length` to look for, default is 1.
    groupby_length : str, optional
        Group time series and return `n` periods of `length`
        for each group.
    locations : list, optional
        List of locations to use, if None, uses all available locations.
    padding : int, optional
        Pad beginning and end of the unmasked area by the number of
        timesteps given.
    normalize : bool, optional
        If True (default), data is normalized
        using :func:`~calliope.time_funcs.normalized_copy`.

    """
    arr = _get_array(data, var, tech, locations)

    return _extreme(arr, how, length, n, groupby_length, padding)


def extreme_diff(data, tech0, tech1, var='r', how='max',
                 length='1D', n=1, groupby_length=None,
                 locations=None, padding=None, normalize=True):
    if normalize:
        data_n = time_funcs.normalized_copy(data)
    else:
        data_n = data
    arr0 = _get_array(data_n, var, tech0, locations)
    arr1 = _get_array(data_n, var, tech1, locations)
    arr = arr0 - arr1

    return _extreme(arr, how, length, n, groupby_length, padding)


def _extreme(arr, how='max',
             length='1D', n=1, groupby_length=None,
             padding=None):

    full_series = arr.mean(dim='x').to_pandas()  # Get a t-indexed Series

    if groupby_length:
        groupby = pd.TimeGrouper(groupby_length)
        group_indices = []
        grouping = full_series.groupby(groupby)
        for k in grouping.groups.keys():
            s = grouping.get_group(k)
            group_indices.append(_get_minmax_timestaps(s, length, n, how, padding))
        ts_index = _concat_indices(group_indices)
    else:
        ts_index = _get_minmax_timestaps(full_series, length, n, how, padding)

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
