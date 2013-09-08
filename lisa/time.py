from __future__ import print_function
from __future__ import division

import pandas as pd


class TimeSummarizer(object):
    """docstring for TimeSummarizer"""
    def __init__(self):
        super(TimeSummarizer, self).__init__()
        self.methods = {'weighted_average': self._reduce_weighted_average,
                        'average': self._reduce_average,
                        'sum': self._reduce_sum,
                        'cut': self._reduce_cut}
        # Format: {'data item': ('method', 'argument')}
        self.known_data_types = {'_t': ('cut'),
                                 '_dt': ('cut'),
                                 'dni': ('sum'),
                                 'n_sf': ('weighted_average', 'dni'),
                                 'n_el': ('average'),
                                 'D': ('sum')}

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
            self.new_index = range(0, data_len, resolution)
            self.rolling_new_index = range(resolution - 1, data_len,
                                           resolution)
        else:
            s = slice(*t_range)
            data_len = len(data['_t'][s])
            self.new_index = range(s.start, s.start + data_len, resolution)
            self.rolling_new_index = range(s.start + resolution - 1,
                                           s.start + data_len, resolution)
        # Go through each item in data and apply the appropriate method to it
        for k in data.keys():
            if k in self.known_data_types.keys():
                how = self.known_data_types[k]
                if len(how) == 2:
                    method = self.methods[how[0]]
                    data[k][s] = method(data[k][s], data[how[1]])
                else:
                    method = self.methods[how]
                    data[k][s] = method(data[k][s])

    def mask_where_zero_dni(self, data):
        """Return a mask to summarize where DNI across all sites is zero"""
        df = data['dni'].copy(deep=True)
        # Summing over all DNIs to find those times where DNI==0 everywhere
        df = pd.DataFrame({'data': df.sum(1)})
        df['summarize'] = 0
        df['summarize'][df['data'] <= 0] = 1
        return df

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
            df.time_res[ifrom] = len(df.summarize[ifrom:ito])
            istart = ito
        for k in data.keys():
            if k in self.known_data_types.keys():
                data[k] = data[k][df.summarize < 2]
        df = df[df.summarize < 2]
        data['time_res_series'] = df['time_res']

    def _reduce_weighted_average(self, target, weight):
        """Custom weighted average"""
        df = target.reindex(self.new_index)
        for i in range(len(df)):
            weighted = 0
            for j in range(self.resolution):
                weighted += (weight.iloc[i*self.resolution+j, :]
                             * target.iloc[i*self.resolution+j, :])
            weighted = weighted / weight.iloc[i*self.resolution:i*self.resolution+self.resolution, :].sum()
            weighted[weighted.isnull()] = 0
            df.iloc[i*self.resolution, :] = weighted
        target = df
        return target

    def _reduce_average(self, df):
        """Rolling mean"""
        df = pd.stats.moments.rolling_mean(df, self.resolution)
        df = df.reindex(self.rolling_new_index)
        df.index = self.new_index
        return df

    def _reduce_sum(self, df):
        """Rolling sum"""
        df = pd.stats.moments.rolling_sum(df, self.resolution)
        df = df.reindex(self.rolling_new_index)
        df.index = self.new_index
        return df

    def _reduce_cut(self, df):
        """Only keep the first timestep"""
        df = df.reindex(self.new_index)
        return df
