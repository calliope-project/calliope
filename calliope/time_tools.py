"""
Copyright (C) 2013-2015 Stefan Pfenninger.
Licensed under the Apache 2.0 License (see LICENSE file).

time_tools.py
~~~~~~~~~~~~~

Provides the TimeSummarizer class to dynamically adjust time resolution.

Also provides the masks_to_resolution_series function. A resolution series
is a pandas Series with the same index as the input data, where each
time step is one of:

* a positive integer (signifying how many timesteps to summarize),
* -1 (marks timesteps that will be summarized if following a positive integer,
  or dropped otherwise),
* or 0 (no resolution adjustment to this timestep).

The name of a resolution series must always be 'resolution_series'.

"""

import logging

import numpy as np
import pandas as pd

from . import utils
from . import exceptions


class TimeSummarizer(object):
    """
    Provides methods to reduce time resolution for the given model data.

    """
    def __init__(self):
        super(TimeSummarizer, self).__init__()
        # Format: {'data item': 'method'}
        self.known_data_types = {'_dt': 'NOOP',
                                 'a': self._reduce_r_to_a,
                                 'r': self._reduce_sum,
                                 'r_eff': self._reduce_average,
                                 'e_eff': self._reduce_average}
        self.source_parameters = {'a': 'r'}

    def _apply_method(self, data, method, s, param, src_param=None,
                      subkey=None):
        if method == 'NOOP':
            # NOOP method means we don't do anything, so return straight away
            return
        i = data._dt.index[s.start]
        if not src_param:
            src_param = param
        if subkey:
            data[param][subkey].loc[i, :] = method(data[src_param][subkey][s])
        else:
            data[param].at[i] = method(data[src_param][s])

    def _reduce_resolution(self, data, resolution, t_range):
        """
        Helper function called by dynamic_timestepper.

        Parameters
        ----------
        data : AttrDict
            data container
        resolution : int
            new time step resolution (in hours)
        t_range : (int, int)
            (absolute) time indices over which to apply resolution reduction

        """
        # Initialize some common data
        resolution = resolution / data.time_res_data
        assert resolution.is_integer()
        resolution = int(resolution)
        self.resolution = resolution
        # Set up time range slice
        s = slice(*t_range)
        # Go through each item in data and apply the appropriate method to it
        for k in list(data.keys()):
            if k in list(self.known_data_types.keys()):
                how = self.known_data_types[k]
                if k in self.source_parameters:
                    src = self.source_parameters[k]
                else:
                    src = None
                if isinstance(data[k], utils.AttrDict):
                    for kk in list(data[k].keys()):
                        self._apply_method(data, how, s, param=k,
                                           src_param=src, subkey=kk)
                else:
                    self._apply_method(data, how, s, param=k,
                                       src_param=src)
            else:  # k is not in known_data_types.keys()
                msg = 'Encountered unknown data type, skipping: {}'.format(k)
                logging.debug(msg)

    def dynamic_timestepper(self, data, series):
        """``series`` must be a series with the same length as the given data.

        For each timestep, the ``series`` can either be 0 (no
        adjustment), >0 (marks the first timestep to be compressed, with
        the integer value giving the number of timesteps to compress) or
        -1 (marks timesteps that will be dropped or compressed).

        For example, ``series`` could contain::

            [0, 0, 0, 3, -1, -1, 0, 0, 2, -1, 2, -1]

        This would result in data with the following new timestep resolution::

            [1, 1, 1, 3, 1, 1, 2, 2]

        Warning: modifies the passed data object in-place.
        Returns None on success.

        """
        assert len(series) == len(data['_dt']), \
            'Mask and data must have same length.'
        df = pd.DataFrame(series, index=series.index)
        df.columns = ['summarize']  # rename the single column
        resolution = data.time_res_static
        df['time_res'] = resolution
        df['to_keep'] = 0
        # Set to_keep to 1 where the series is 0, i.e. where timesteps
        # should not be touched
        df.loc[df[df.summarize == 0].index, 'to_keep'] = 1
        # Get all time steps that need summarizing
        entry_points = df.summarize[df.summarize > 0]
        # 1. Summarize by calling _reduce_resolution for the desired ranges
        for i, v in entry_points.items():
            ifrom = i
            ito = i + int(v / data.time_res_static)
            resolution = v
            self._reduce_resolution(data, resolution, t_range=(ifrom, ito))
            df.at[ifrom, 'to_keep'] = 1  # Mark the timestep to keep
            df.at[ifrom, 'time_res'] = resolution
        # 2. Replace all data with its subset where to_keep is 1
        # Create boolean mask with the right length -- but ignore index by
        # turning it into a list (since data is indexed by timestep id)
        boolean_mask = (df.to_keep == 1).tolist()
        for k in list(data.keys()):
            if k in list(self.known_data_types.keys()):
                if isinstance(data[k], utils.AttrDict):
                    for kk in list(data[k].keys()):
                        data[k][kk] = data[k][kk][boolean_mask]
                else:
                    data[k] = data[k][boolean_mask]
            # NB unknown data types are checked for and logged earlier, inside
            # _reduce_resolution()
        data.time_res_series = df.time_res[boolean_mask]
        # `df` is zero-indeed, so we have to re-set the actual timesteps as
        # index for time_res_series
        data.time_res_series.index = data._dt.index
        # Update data.time_res_static if the resolution reduction was uniform
        if len(data.time_res_series.unique()) == 1:
            data.time_res_static = resolution
        data.time_res_native = 0  # Unset native time_res flag

    def reduce_resolution(self, data, resolution):
        """
        Reduces the resolution of the entire ``data`` to ``resolution``,
        which must be an integer and greater than the initial resolution.

        Modifies the passed data object in-place and returns None.

        """
        series = resolution_series_uniform(data, resolution)
        self.dynamic_timestepper(data, series)

    def _reduce_average(self, df):
        return df.mean(0)

    def _reduce_sum(self, df):
        return df.sum(0)

    def _reduce_r_to_a(self, df):
        """
        Based on resource data given in ``df``, generate availability data

        """
        if (df.sum(axis=0) == np.inf).all():
            # If the df is all infinite, availability is 1
            result = 1
        elif df.max().max() == 0 and df.min().min() == 0:
            # And if it's all zero, availability is 0
            result = 0
        else:
            result = df.sum() / (df.max() * len(df))
            result = result.fillna(0)  # Fill NaNs from division by zero
        return result


def masks_to_resolution_series(masks, how='or', resolution=None,
                               drop_with_padding=None):
    """
    Converts a list of overlapping masks into a series of time step
    resolutions.

    Parameters
    ----------
    how : str, default 'or'
        ``or`` or ``and``.
    resolution : int, default None
        If given, will break up masked areas into timesteps of at most
        the given length (with possibly a over timestep at a lower
        length at the end of the masked area).
        If None, all contingent masked areas become single timesteps.
    drop_with_padding : int, default None
        If given, all masked areas are dropped, except for a padding area
        around the retained high-resolution areas. The padding is determined
        by ``drop_with_padding * resolution``, so if drop_with padding is
        1 and resolution is 24, a single 24-hour timestep each will be kept
        before and after each high-resolution area. If set and resolution
        is None an error is raised, unless set to 0, which will drop all
        non-masked areas without padding.

    """
    # Validate options
    if drop_with_padding and resolution is None and drop_with_padding != 0:
        e = exceptions.ModelError
        raise e('If drop_with_padding is given and > 0, '
                'resolution cannot be None.')

    if not isinstance(masks, list) or isinstance(masks, tuple):
        masks = [masks]
    # combine all masks into one
    df = pd.DataFrame({i: x for i, x in enumerate(masks)})

    if how == 'and':
        mask = df.sum(axis=1)
    elif how == 'or':
        # joiner: only return 1 if all items in the row are 1, else return 0
        joiner = lambda row: 1 if sum(row) == len(row) else 0
        mask = df.apply(joiner, axis=1)

    istart = 0
    end = False
    while not end:
        ifrom = mask[istart:].argmax()  # Find first 1
        ito = mask[ifrom:].argmin()  # Find first 0 after series of 1s
        if ifrom == ito:  # Reached the end!
            ito = len(mask)
            end = True
            # If `summarize` is zero at the very last entry
            # (`ito - `), we break out of the
            # loop to prevent it from adding a spurious summarization
            if mask[ito - 1] == 0:
                break
        step_resolution = ito - ifrom
        # Start by dropping the entire masked area, and if drop_with_padding
        # is 0, that's all we do (drop it all)
        mask[ifrom:ito] = -1
        if drop_with_padding and drop_with_padding > 0:
            # Drop masked areas with padding, but if drop_with_padding == 0,
            # don't do anything since we dimply drop it all
            # Need to add padding before and after unmasked areas
            if ifrom > 0:  # Don't add padding at beginning of the series
                pad_after_end = ifrom + drop_with_padding * resolution
                for i in range(ifrom, pad_after_end, resolution):
                    mask[i] = resolution
            if ito != len(mask):  # Don't add padding at the end of the series
                pad_before_start = ito - drop_with_padding * resolution
                for i in range(pad_before_start, ito, resolution):
                    mask[i] = resolution
        elif drop_with_padding is None:
            # If dropping is None, summarize into one step for now
            mask[ifrom] = step_resolution
        # Correct edge case where only one timestep would be "summarized"
        if mask[ifrom] >= 1 and step_resolution == 1:
            mask[ifrom] = 0
        istart = ito

    # If we didn't drop_with_padding, but want a resolution reduction
    # in masked areas, we go through the masked areas again
    # and split them into smaller chunks
    # FIXME assumes that data is in 1-hourly resolution?
    if drop_with_padding is None and resolution:
        for index, value in mask[mask > resolution].iteritems():
            end_index = index + value
            summary_index = list(range(index, end_index, resolution))
            for i in summary_index:
                if i + resolution < end_index:
                    mask[i] = resolution
                else:  # Make sure the last timestep isn't too long
                    mask[i] = end_index - i

    mask.name = 'resolution_series'
    return mask


def resolution_series_uniform(data, resolution):
    """
    Resolution series to reduce resolution uniformly.

    Parameters
    ----------
    data : Calliope model data
    resolution : int or float
        Resolution (in hours) to downsample all data to.

    """
    res_length = resolution / data.time_res_static
    df = data.r[list(data.r.keys())[0]]  # Grab length of data from any table
    summarize = pd.Series(-1, index=list(range(len(df))))
    # Set to 0 (keep timestep) for the given resolution
    for index, item in summarize.items():
        if index % res_length == 0:
            summarize.at[index] = resolution
    summarize.name = 'resolution_series'
    return summarize


def resolution_series_to_mask(resolution_series):
    """
    Turns a resolution series into a mask.

    """
    mask = resolution_series
    mask[mask != 0] = 1
    mask.name = 'mask'
    return mask
