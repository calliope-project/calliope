"""
Copyright (C) 2013 Stefan Pfenninger.
Licensed under the Apache 2.0 License (see LICENSE file).

time_tools.py
~~~~~~~~~~~~~

Provides the TimeSummarizer class to dynamically adjust time resolution.

"""

from __future__ import print_function
from __future__ import division

import numpy as np
import pandas as pd

from . import time_masks
from . import utils


class TimeSummarizer(object):
    """
    Provides methods to reduce time resolution for the given model data.

    """
    def __init__(self):
        super(TimeSummarizer, self).__init__()
        # Format: {'data item': 'method'}
        self.known_data_types = {'_t': self._reduce_cut,
                                 '_dt': self._reduce_cut,
                                 'a': self._reduce_r_to_a,
                                 'r': self._reduce_sum,
                                 'r_eff': self._reduce_average,
                                 'e_eff': self._reduce_average}

    def _reduce_resolution(self, data, resolution, t_range):
        """Helper function called by dynamic_timestepper."""
        # Initialize some common data
        resolution = resolution / data.time_res_data
        assert resolution.is_integer()
        resolution = int(resolution)
        self.resolution = resolution
        # Set up time range slice
        s = slice(*t_range)
        data_len = len(data['_t'][s])
        start_idx = data['_t'][s.start]
        # TODO all of these are not necessary any more now that only using
        # dynamic_timestepper, could vastly simplify all of this
        self.new_index = range(start_idx, start_idx + data_len, resolution)
        self.rolling_new_index = range(start_idx + resolution - 1,
                                       start_idx + data_len, resolution)
        # Go through each item in data and apply the appropriate method to it
        for k in data.keys():
            if k in self.known_data_types.keys():
                how = self.known_data_types[k]
                if isinstance(data[k], utils.AttrDict):
                    for kk in data[k].keys():
                        self._apply_method(data, how, s, key=k, subkey=kk)
                else:
                    self._apply_method(data, how, s, key=k)

    def _apply_method(self, data, method, s, key, subkey=None):
        if subkey:
            df = method(data[key][subkey][s])
        else:
            df = method(data[key][s])
        # Choose different iterator method for series and dataframe
        if isinstance(df, pd.Series):
            iterator = df.iteritems
        else:
            iterator = df.iterrows
        # Now iterate over each item in the reduced-resolution df,
        # updating the dataset item by item
        if subkey:
            for i, r in iterator():
                data[key][subkey].loc[i] = r
        else:
            for i, r in iterator():
                data[key].loc[i] = r

    def dynamic_timestepper(self, data, mask):
        """``mask`` must be a series with the same index as the given data.

        For each timestep, the ``mask`` can either be 0 (no
        adjustment), >0 (marks the first timestep to be compressed, with
        the integer value giving the number of timesteps to compress) or
        -1 (marks timesteps following a >0 timestep that will be compressed).

        For example, ``mask`` could contain::

            [0, 0, 0, 3, -1, -1, 0, 0, 2, -1, 2, -1]

        This would result in data with the following new timestep resolution::

            [1, 1, 1, 3, 1, 1, 2, 2]

        Warning: modifies the passed data object in-place.
        Returns None on success.

        """
        # Prepare the availability data based on resource
        for k in data.r.keys():
            data.a[k] = data.r[k].copy()
        # Set up the mask
        df = pd.DataFrame(mask, index=mask.index)
        df.columns = ['summarize']  # rename the single column
        df['time_res'] = data.time_res_static
        df['to_keep'] = True
        # Get all time steps that need summarizing
        entry_points = df.summarize[df.summarize > 0]
        for i, v in entry_points.iteritems():
            ifrom = i
            ito = i + int(v / data.time_res_static)
            resolution = v
            self._reduce_resolution(data, resolution, t_range=[ifrom, ito])
            # Mark the rows that need to be killed with False
            df.to_keep[ifrom+1:ito] = False
            df.time_res.iloc[ifrom] = resolution
        for k in data.keys():
            # # Special case for `_t`, which is the only known_data_type which is always 0-indexed
            # if k == '_t' or k == '_dt':
            #     # To get around non-matching index, we simply turn the boolean mask df into a list
            #     data[k] = data[k][(df.summarize < 2).tolist()]
            if k in self.known_data_types.keys():
                if isinstance(data[k], utils.AttrDict):
                    for kk in data[k].keys():
                        data[k][kk] = data[k][kk][df.to_keep]
                else:
                    data[k] = data[k][df.to_keep]
        data.time_res_series = df.time_res[df.to_keep]
        # NB data.time_res_static is not adjusted here, but this should be ok

    def reduce_resolution(self, data, resolution):
        """
        Reduces the resolution of the entire ``data`` to ``resolution``,
        which must be an integer and greater than the initial resolution.

        Modifies the passed data object in-place and returns None.

        """
        mask = time_masks.resolution_series_uniform(data, resolution)
        self.dynamic_timestepper(data, mask)
        data.time_res_static = resolution

    def _infinity_test(self, df):
        return (df.sum(axis=0) == np.inf).all()
        # The above approach wouldn't work in a df with mixed inf and non-inf
        # values, but this shouldn't happy in practice anyway
        # -- if a df contains inf it should be all inf!

    def _reduce_average(self, df):
        """Calculates rolling mean, then reindexes with ``self.new_index``."""
        if not self._infinity_test(df):
            df = pd.stats.moments.rolling_mean(df, self.resolution)
        df = df.reindex(self.rolling_new_index)
        df.index = self.new_index
        return df

    def _reduce_sum(self, df):
        """Calculates rolling sum, then reindexes with ``self.new_index``."""
        if not self._infinity_test(df):
            df = pd.stats.moments.rolling_sum(df, self.resolution)
        df = df.reindex(self.rolling_new_index)
        df.index = self.new_index
        return df

    def _reduce_cut(self, df):
        """Removes rows not in ``self.new_index``."""
        if len(self.new_index) == 1:
            df = pd.Series(df.iloc[0], index=self.new_index)
        else:
            df = df.reindex(self.new_index)
        return df

    def _reduce_r_to_a(self, df):
        """Using resource data as a basis, generates availability data"""
        # TODO NB this only works with the 'new' way of using only
        # dynamic_timestepper where we can assume that every call to a
        # _reduce method results in only one row being returned
        # and self.new_index being of length 1 only
        df = df.sum() / (df.max() * len(df))
        df = df.fillna(0)
        df = pd.DataFrame(df).T  # transpose the series to a frame
        # Next step is superfluous but kept for consistency
        df.index = self.new_index
        return df
