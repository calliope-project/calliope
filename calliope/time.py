from __future__ import print_function
from __future__ import division

import numpy as np
import pandas as pd

from . import utils


class TimeSummarizer(object):
    """Provides methods to reduce time resolution for a given set of model
    data. On initialization, it defines ``self.methods`` and
    ``self.known_data_types`` which may be inspected and overriden before use.

    """
    def __init__(self):
        super(TimeSummarizer, self).__init__()
        self.methods = {'weighted_average': self._reduce_weighted_average,
                        'average': self._reduce_average,
                        'sum': self._reduce_sum,
                        'cut': self._reduce_cut}
        # Format: {'data item': ('method', 'argument')}
        self.known_data_types = {'_t': ('cut'),
                                 '_dt': ('cut'),
                                 'r': ('sum'),
                                 'r_eff': ('average'),
                                 'e_eff': ('average')}

    def reduce_resolution(self, data, resolution):
        """
        Reduces the resolution of the entire ``data`` to ``resolution``,
        which must be an integer and greater than 1.

        The data must have a resolution of 1 to begin with.

        Warning: modifies the passed data object in-place.
        Returns None on success.

        """
        self._reduce_resolution(data, resolution)

    def _reduce_resolution(self, data, resolution, t_range=None):
        """Helper function called by both reduce_resolution and
        dynamic_timestepper."""
        # Initialize some common data
        self.resolution = resolution
        # Set up time range slice, if given
        if not t_range:
            s = slice(None)
            data_len = len(data['_t'])
            start_idx = data['_t'][0]
        else:
            s = slice(*t_range)
            data_len = len(data['_t'][s])
            start_idx = data['_t'][s.start]
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
        # If not t_range (implies working on entire time series), also add
        # time_res to dataset (if t_range set, this happens inside
        # dynamic_timestepper)
        if not t_range:
            data['time_res_series'] = pd.Series(resolution, index=data['_t'])

    def _apply_method(self, data, how, s, key, subkey=None):
        if len(how) == 2:
            method = self.methods[how[0]]
            if subkey:
                df = method(data[key][subkey][s], data[how[1]][subkey])
            else:
                df = method(data[key][s], data[how[1]])
        else:
            method = self.methods[how]
            if subkey:
                df = method(data[key][subkey][s])
            else:
                df = method(data[key][s])
        # If no slice, this implies we are working on whole time series,
        # so we replace the existing time series completely
        # to get around indexing problems
        if s == slice(None):
            if subkey:
                data[key][subkey] = df
            else:
                data[key] = df
        else:
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
        """``mask`` must be a df with the same index as the other dfs in
        data, and a ``summarize`` column that gives information on how to
        adjust timesteps.

        For each timestep, the ``summarize`` can either be 0 (no
        adjustment), >0 (marks the first timestep to be compressed, with
        the integer value giving the number of timesteps to compress) or
        -1 (marks timesteps following a >0 timestep that will be compressed).

        For example, the summarize column could contain::

            [0, 0, 0, 3, -1, -1, 0, 0, 2, -1, 2, -1]

        This would result in data with the following new timestep resolution::

            [1, 1, 1, 3, 1, 1, 2, 2]

        Warning: modifies the passed data object in-place.
        Returns None on success.

        """
        # Set up the mask
        df = mask
        df['time_res'] = 1
        df['to_keep'] = True
        # Get all time steps that need summarizing
        entry_points = df.summarize[df.summarize > 0]
        for i, v in entry_points.iteritems():
            ifrom = i
            ito = i + v
            resolution = v
            self._reduce_resolution(data, resolution, t_range=[ifrom, ito])
            # Mark the rows that need to be killed with False
            df.to_keep[ifrom+1:ito] = False
            df.time_res.iloc[ifrom] = resolution
        for k in data.keys():
            # Special case for `_t`, which is the only known_data_type which is always 0-indexed
            if k == '_t' or k == '_dt':
                # To get around non-matching index, we simply turn the boolean mask df into a list
                data[k] = data[k][(df.summarize < 2).tolist()]
            elif k in self.known_data_types.keys():
                if isinstance(data[k], utils.AttrDict):
                    for kk in data[k].keys():
                        data[k][kk] = data[k][kk][df.to_keep]
                else:
                    data[k] = data[k][df.to_keep]
        df = df[df.summarize < 2]
        data['time_res_series'] = df['time_res']

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
