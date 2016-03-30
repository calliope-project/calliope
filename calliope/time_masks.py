"""
Copyright (C) 2013-2016 Stefan Pfenninger.
Licensed under the Apache 2.0 License (see LICENSE file).

time_masks.py
~~~~~~~~~~~~~

Defines time masks to be used with TimeSummarizer (see time_tools.py).

A mask function returns a pandas Series with the same index as the input
data that is 0 where the original resolution should be retained (unmasked
areas) and 1 where it should be reduced (masked areas).

The name of a mask must always be 'mask'.

"""

import pandas as pd


def mask_relative_difference(data, target1, target2, days=5,
                             padding_days=1, per_zone=False):
    """
    Masks everywhere except the ``days`` number of days where the
    resource of the two technologies ``target1`` and ``target2`` is most
    different.

    Parameters
    ----------
    data : Calliope model data
    target1 : str
    target2 : str
    days : int, default 5
    padding_days : int, default 1
    per_zone: bool, default False

    """
    data_length = len(data._dt)

    def _get_most_different(data, target1, target2,
                            days, padding_days, target_zone=None):
        # Get length of a day in timesteps
        day_len = int(24 / data.time_res_data)
        targets = (target1, target2)
        relative_series = []
        for t in targets:
            if not target_zone:
                t_series = data.r[t].sum(1)
            else:
                t_series = data.r[t].loc[:, target_zone]
            if t_series.max() < 0:  # Assume this is a demand series and <0
                relative = t_series / t_series.min()
            else:
                relative = t_series / t_series.max()
            relative_series.append(relative)

        diff = (relative_series[0] - relative_series[1]).abs()

        most_different = diff.sort_values(ascending=False, inplace=False)
        datetimes = data._dt.loc[most_different.index]
        # `result` are the datetimes for the found days (indexed by timestep index)
        result = datetimes.dt.date.drop_duplicates().head(days)

        periods = []
        for index, r in result.iteritems():
            # `selected` is the index of the first timestamp in the chosen day
            selected = data._dt[data._dt.dt.date == r].index[0]
            start = selected - padding_days * day_len
            end = selected + (padding_days + 1) * day_len
            periods.append((start, end))
        return periods

    if not per_zone:
        periods = _get_most_different(data, target1, target2, days,
                                      padding_days)
    else:
        periods = []
        for zone in data._x:
            p = _get_most_different(data, target1, target2, days,
                                    padding_days, target_zone=zone)
            periods.extend(p)

    # Mask where everything is 1 (reduce resolution) by default
    mask = pd.Series(1, index=list(range(data_length)))
    # For the desired time periods, change the mask to native resolution (0)
    for p in periods:
        mask.loc[p[0]:p[1] - 1] = 0
    mask.name = 'mask'
    return mask


def mask_zero(data, tech, var='r', locations=None):
    """
    Mask where ``var`` for the technology ``tech``
    across the given list of ``locations`` is zero.

    ``var`` defaults to ``r``.

    If ``locations`` not given, uses all available locations.

    """
    df = data[var][tech]
    if locations:
        df = df.loc[:, locations]
    # Retain original resolution everywhere by default (0)
    mask = pd.Series(0, index=list(range(len(df))))
    # Sum over all locations to find those times where the data==0 everywhere
    mask[df.sum(1) <= 0] = 1
    mask.name = 'mask'
    return mask


def mask_extreme(data, tech, var='r', how='max',
                 length=24, locations=None, padding=None):
    """
    Mask everywhere except the ``length`` where ``var`` for the technology
    ``tech`` across the given list of ``locations`` is either minmal
    or maximal.

    Parameters
    ----------
    data : Calliope model data
    tech : str
        Technology whose `var` to find extreme for.
    var : str, default 'r'
    how : str, default 'max'
        'max' or 'min'
    length : int, default 24
        Timesteps to mask.
    locations : list, default None
        List of locations to use, if None, uses all available locations.
    padding : int, default None
        Pad beginning and end of the unmasked area by the number of
        timesteps given.

    """
    df = data[var][tech]
    if locations:
        df = df.loc[:, locations]
    df = df.sum(axis=1)
    totals = []
    for i in range(0, len(df), length):
        totals.append(df[i:(i + 1) * length].sum(axis=0))
    totals = pd.Series(totals)
    if how == 'max':
        total_i = totals.argmax()
    elif how == 'min':
        total_i = totals.argmin()
    mask = pd.Series(1, index=list(range(len(df))))
    ifrom = total_i * length
    ito = (total_i + 1) * length - 1
    if padding:
        ifrom = ifrom - padding
        ito = ito + padding
        # Ensure ifrom and ito remain within bounds
        if ifrom < 0:
            ifrom = 0
        if ito > len(df):
            ito = len(df)
    mask.loc[ifrom:ito] = 0
    mask.name = 'mask'
    return mask


def mask_extreme_week(data, tech, var='r', how='sum', what='min'):
    """
    Mask everywhere except the week containing the day where ``var`` of
    ``tech`` is minimal (what='min') or maximal (what='max') across the
    sum (how='sum') of or the mode (most) of locations (how='mode').

    Parameters
    ----------
    data : Calliope model data
    tech : str
        Technology whose `var` to find extreme week for.
    var : str, default 'r'
    how : str, default 'sum'
        'sum' or 'mode'
    what : str, default 'min'
        'min' or 'max'

    """
    df = data[var][tech]
    # Get length of a day in timesteps
    day_len = int(24 / data.time_res_data)
    # Get day-wise sums
    dff_index = list(range(0, len(df), day_len))
    dff = df.rolling(window=day_len).sum().reindex(dff_index)
    # If what is 'min', this will get the 'idxmin' attribute (a method),
    # similar for 'max', else most likely raise an error!
    idx_extr = lambda x: getattr(x, 'idx{}'.format(what))
    if how == 'mode':
        # Find day where var/tech is min/max across the most locations
        selected = idx_extr(dff)().mode()[0]
    elif how == 'sum':
        # Find day where var/tech is min/max across all locations
        selected = idx_extr(dff.sum(axis=1))()
    d = data._dt.at[selected]
    # Determine the range for the calendar week
    # (7 days) to keep at full resolution
    week_start = selected - day_len * d.dayofweek
    week_end = selected + day_len * (7 - d.dayofweek)
    # Mask where everything is 1 (reduce resolution) by default
    mask = pd.Series(1, index=list(range(len(df))))
    # For the desired week, change the mask to native resolution (0)
    mask.loc[week_start:week_end - 1] = 0
    mask.name = 'mask'
    return mask
