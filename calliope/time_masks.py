"""
Copyright (C) 2013-2016 Stefan Pfenninger.
Licensed under the Apache 2.0 License (see LICENSE file).

time_masks.py
~~~~~~~~~~~~~

Functions to pick timesteps from data given certain criteria.

"""

import pandas as pd
from . import data_tools as dt


def _get_array(data, var, tech, locations):
    arr = data[var]
    y_coord = dt.get_y_coord(arr)
    arr = arr.loc[{y_coord: tech}]
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


def extreme(data, tech, var='r', how='max',
            length='1D', locations=None, padding=None):
    """
    Returns timesteps for period of ``length`` where ``var`` for the technology
    ``tech`` across the given list of ``locations`` is either minmal
    or maximal.

    Parameters
    ----------
    data : Calliope model data
    tech : str
        Technology whose `var` to find extreme for.
    var : str, optional
        default 'r'
    how : str, optional
        'max' (default) or 'min'.
    length : int, optional
        Timesteps to mask, default is 24.
    locations : list, optional
        List of locations to use, if None, uses all available locations.
    padding : int, optional
        Pad beginning and end of the unmasked area by the number of
        timesteps given.

    """
    arr = _get_array(data, var, tech, locations)
    s = arr.mean(dim='x').to_pandas()  # Get a t-indexed Series

    # Get the max or min timestamp
    group = s.groupby(pd.TimeGrouper(length)).mean()
    if how == 'max':
        ts = group.idxmax()
    elif how == 'min':
        ts = group.idxmin()

    # Get range of timestamps including padding
    ts_end = ts + pd.Timedelta(length)
    if padding is not None:
        ts -= pd.Timedelta(padding)
        ts_end += pd.Timedelta(padding)
    ts_range = pd.date_range(ts, ts_end, freq='1H')[:-1]

    return ts_range
