"""
Copyright (C) since 2013 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

masks.py
~~~~~~~~

Functions to pick timesteps from data given certain criteria.

"""

import pandas as pd

from calliope.time import funcs
from calliope.core.util.dataset import split_loc_techs
from calliope import exceptions


def _get_array(data, var, tech, **kwargs):
    subset = {"techs": tech}
    if kwargs is not None:
        subset.update({k: v for k, v in kwargs.items()})

    unusable_dims = (
        set(subset.keys()).difference(["techs", "locs"]).difference(data[var].dims)
    )
    if unusable_dims:
        raise exceptions.ModelError(
            "Attempting to mask time based on  technology {}, "
            "but dimension(s) {} do not exist for parameter {}".format(
                tech, unusable_dims, var.name
            )
        )

    arr = split_loc_techs(data[var].copy()).loc[subset]
    arr = arr.mean(dim=[i for i in arr.dims if i != "timesteps"]).to_pandas()
    return arr


def zero(data, tech, var="resource", **kwargs):
    """
    Returns timesteps where ``var`` for the technology ``tech`` is zero.

    kwargs are additional dimensions to subset on, for example,
    ``locs=['location1', 'location2]``

    """
    s = _get_array(data, var, tech, **kwargs)

    return s[s == 0].index


def _concat_indices(indices):
    return pd.concat([i.to_series() for i in indices]).sort_index().index


def _get_minmax_timestamps(series, length, n, how="max", padding=None):
    # Get the max/min timestamps
    group = series.groupby(pd.Grouper(freq=length)).mean()
    timesteps = []
    for _ in range(n):
        if how == "max":
            ts = group.idxmax()
        elif how == "min":
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
        ts_range = series[ts:ts_end].index[:-1]
        full_timesteps.append(ts_range)

    ts_index = _concat_indices(full_timesteps)

    return ts_index


def extreme(
    data,
    tech,
    var="resource",
    how="max",
    length="1D",
    n=1,
    groupby_length=None,
    padding=None,
    normalize=True,
    **kwargs,
):
    """
    Returns timesteps for period of ``length`` where ``var`` for the technology
    ``tech`` across the given list of ``locations`` is either minimal
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
    padding : str, optional
        Either Pandas frequency (e.g. '1D') or 'calendar_week'.
        If Pandas frequency, symmetric padding is undertaken, either side of `length`
        If 'calendar_week', padding is fit to the calendar week in which the
        extreme day(s) are found.
    normalize : bool, optional
        If True (default), data is normalized
        using :func:`~calliope.time.funcs.normalized_copy`.
    kwargs : dict, optional
        Dimensions of the selected var over which to index. Any remaining
        dimensions will be flattened by mean

    """
    if normalize:
        # Only normalise the desired var as rest of data may contain
        # non-numeric variables!
        data_n = funcs.normalized_copy(data[var].to_dataset(name=var))
    else:
        data_n = data
    arr = _get_array(data_n, var, tech, **kwargs)
    return _extreme_with_padding(arr, how, length, n, groupby_length, padding)


def extreme_diff(
    data,
    tech0,
    tech1,
    var="resource",
    how="max",
    length="1D",
    n=1,
    groupby_length=None,
    padding=None,
    normalize=True,
    **kwargs,
):
    """
    Returns timesteps for period of ``length`` where the diffence in extreme
    value for ``var`` between technologies ``tech0`` and ``tech1`` is either a
    minimum or a maximum.

    Parameters
    ----------
    data : xarray.Dataset
    tech0 : str
        First technology for which we find the extreme of `var`
    tech1 : str
        Second technology for which we find the extreme of `var`
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
    padding : str, optional
        Either Pandas frequency (e.g. '1D') or 'calendar_week'.
        If Pandas frequency, symmetric padding is undertaken, either side of `length`
        If 'calendar_week', padding is fit to the calendar week in which the
        extreme day(s) are found.
    normalize : bool, optional
        If True (default), data is normalized
        using :func:`~calliope.time.funcs.normalized_copy`.
    kwargs : dict, optional
        Dimensions of the selected var over which to index. Any remaining
        dimensions will be flattened by mean

    """
    if normalize:
        # Only normalise the desired var as rest of data may contain
        # non-numeric variables!
        data_n = funcs.normalized_copy(data[var].to_dataset(name=var))
    else:
        data_n = data
    arr0 = _get_array(data_n, var, tech0, **kwargs)
    arr1 = _get_array(data_n, var, tech1, **kwargs)
    arr = arr0 - arr1

    return _extreme_with_padding(arr, how, length, n, groupby_length, padding)


def _extreme(arr, how="max", length="1D", n=1, groupby_length=None, padding=None):

    if groupby_length:
        groupby = pd.Grouper(freq=groupby_length)
        group_indices = []
        grouping = arr.groupby(groupby)
        for k in grouping.groups.keys():
            s = grouping.get_group(k)
            group_indices.append(_get_minmax_timestamps(s, length, n, how, padding))
        ts_index = _concat_indices(group_indices)
    else:
        ts_index = _get_minmax_timestamps(arr, length, n, how, padding)

    return ts_index


def _extreme_with_padding(arr, how, length, n, groupby_length, padding):
    if padding == "calendar_week":
        if n != 1 or length != "1D":
            raise ValueError(
                "calendar_week padding only supports n=1 and length=1D for now."
            )
        result = _extreme(arr, how, length, n, groupby_length, padding=None)
        # get week padding for each day in result
        days = list(result.groupby(result.dayofyear).values())
        weeks = pd.DatetimeIndex(days[0])
        for d in days:
            weeks = weeks.union(_calendar_week_padding(d, arr))
        # concatenate the weeks into one index and drop possible duplicates
        return pd.DatetimeIndex(weeks).drop_duplicates()
    else:
        return _extreme(arr, how, length, n, groupby_length, padding)


def _calendar_week_padding(day, arr):
    """
    Given a day, returns the whole calendar week which contains that day

    """
    days = len(day.day.unique())
    if not days == 1:
        raise ValueError(
            "Only a single day at a time may be used for calendar_week padding, "
            "but {} days were passed.".format(days)
        )

    # Using day of week, figure out how many days before and after to get
    # a complete week
    days_before = 6 - day[0].dayofweek
    days_after = 6 - days_before

    # Turn it into a week
    start_time = day[0] - pd.Timedelta("{}D".format(days_before))
    end_time = day[-1] + pd.Timedelta("{}D".format(days_after))
    before = arr[start_time : day[0]].index[:-1]
    after = arr[day[-1] : end_time].index[1:]
    result_week = before.append(day).append(after)

    return result_week
