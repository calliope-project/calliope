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

    def reduce_resolution(self, data, resolution, t_range=None):
        """
        Warning: modifies the passed data object in-place. Does not
        return anything on success.

        """
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
                        self.apply_method(data, how, s, key=k, subkey=kk)
                else:
                    self.apply_method(data, how, s, key=k)
        # If not t_range (implies working on entire time series), also add
        # time_res to dataset (if t_range set, this happens inside
        # dynamic_timestepper)
        if not t_range:
            data['time_res_series'] = pd.Series(resolution, index=data['_t'])

    def apply_method(self, data, how, s, key, subkey=None):
        if len(how) == 2:
            method = self.methods[how[0]]
            if subkey:
                df = method(data[key][subkey][s], data[how[1]][subkey])
            else:
                df = method(data[key][s], data[how[1]])
            # If no slice, this implies we are working on whole time series,
            # so we replace the existing time series completely
            # to get around indexing problems
            if s == slice(None):
                if subkey:
                    data[key][subkey] = df
                else:
                    data[key] = df
            else:
                if subkey:
                    data[key][subkey][s] = df
                else:
                    data[key[s]] = df
        else:
            method = self.methods[how]
            if subkey:
                df = method(data[key][subkey][s])
            else:
                df = method(data[key][s])
            if s == slice(None):
                if subkey:
                    data[key][subkey] = df
                else:
                    data[key] = df
            else:
                if subkey:
                    data[key][subkey][s] = df
                else:
                    data[key][s] = df

    def dynamic_timestepper(self, data, mask):
        """`mask` must be a df with the same index as the other dfs in
        data, and a 'summarize' column of 0s and 1s, such that contiguous
        groups of 1s are compressed into a single time step

        Warning: modifies the passed data object in-place. Does not
        return anything on success.

        """
        # Set up the mask
        df = mask
        df['time_res'] = 1
        # Apply the variable time step algorithm
        istart = 0
        end = False
        while not end:
            ifrom = istart + df.summarize[istart:].argmax()
            ito = ifrom + df.summarize[ifrom:].argmin()
            if ifrom == ito:  # Reached the end!
                # TODO this works if the final timesteps are part of a summary step
                # but need to verify if it also works if final timesteps are NOT
                # going to be folded into a summary step!
                ito = len(df.summarize)
                end = True
            resolution = ito - ifrom
            # Reduce time_res of all relevant series with an appropriate method
            self.reduce_resolution(data, resolution, t_range=[ifrom, ito])
            df.summarize[ifrom+1:ito] = 2
            df.time_res.iloc[ifrom] = len(df.summarize[ifrom:ito])
            istart = ito
        for k in data.keys():
            # Special case for `_t`, which is the only known_data_type which is always 0-indexed
            # To get around non-matching index, we simply turn the boolean mask df into a list
            if k == '_t':
                data[k] = data[k][(df.summarize < 2).tolist()]
            elif k in self.known_data_types.keys():
                data[k] = data[k][df.summarize < 2]
        df = df[df.summarize < 2]
        data['time_res_series'] = df['time_res']

    def _reduce_weighted_average(self, target, weight):
        """Custom weighted average. NB: Currently non-functional and
        not used.

        """
        df = target.reindex(self.new_index)
        for i in range(len(df)):
            weighted = 0
            for j in range(self.resolution):
                weighted += (weight.iloc[i*self.resolution+j, :]
                             * target.iloc[i*self.resolution+j, :])
            weighted = weighted / weight.iloc[i*self.resolution:i*self.resolution+self.resolution, :].sum()
            weighted[weighted.isnull()] = 0
            df.iloc[i, :] = weighted
        df = df.fillna(np.inf)
        target = df
        return target

    def _reduce_average(self, df):
        """Calculates rolling mean, then reindexes with ``self.new_index``.

        """
        df = pd.stats.moments.rolling_mean(df, self.resolution)
        df = df.reindex(self.rolling_new_index)
        df.index = self.new_index
        # Bad hack: if we have np.inf in input data, it gets replaced
        # by np.nan in the rolling stats functions. The following addresses
        # this, but would also replace np.nan caused by other reasons with
        # np.inf, which may lead to strange behavior!
        # (applies to _reduce_sum too)
        df = df.fillna(np.inf)
        return df

    def _reduce_sum(self, df):
        """Calculates rolling sum, then reindexes with ``self.new_index``.

        """
        df = pd.stats.moments.rolling_sum(df, self.resolution)
        df = df.reindex(self.rolling_new_index)
        df.index = self.new_index
        df = df.fillna(np.inf)
        return df

    def _reduce_cut(self, df):
        """Removes rows not in ``self.new_index``.

        """
        if len(self.new_index) == 1:
            df = pd.Series(df.iloc[0], index=self.new_index)
        else:
            df = df.reindex(self.new_index)
        return df
