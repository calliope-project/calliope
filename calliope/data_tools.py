"""
Copyright (C) 2013-2016 Stefan Pfenninger.
Licensed under the Apache 2.0 License (see LICENSE file).

data_tools.py
~~~~~~~~~~~~~

"""


def get_timeres(model, verify=False):
    """Returns resolution of data in hours. Needs a properly
    formatted ``set_t.csv`` file to work.

    If ``verify=True``, verifies that the entire file is at the same
    resolution. ``model.get_timeres(verify=True)`` can be called
    after Model initialization to verify this.

    """
    datetime_index = model._sets['t']
    seconds = (datetime_index[0] - datetime_index[1]).total_seconds()
    if verify:
        for i in range(len(datetime_index) - 1):
            assert ((datetime_index[i] - datetime_index[i+1]).total_seconds()
                    == seconds)
    hours = abs(seconds) / 3600
    return hours


##
# Functions based on xarray data
##


def get_y_coord(array):
    if 'y' in array.coords:
        y = 'y'
    else:
        try:  # assumes a single y_ coord in array
            y = [k for k in array.coords if 'y_' in k][0]
        except IndexError:  # empty list
            y = None
    return y


def get_datavars(data):
    return [var for var in data.data_vars if not var.startswith('_')]


def get_timesteps_per_day(data):
    timesteps_per_day = data.attrs['time_res'] * 24
    if isinstance(timesteps_per_day, float):
        assert timesteps_per_day.is_integer(), 'Timesteps/day must be integer.'
        timesteps_per_day = int(timesteps_per_day)
    return timesteps_per_day


def get_freq(data):
    ts_per_day = get_timesteps_per_day(data)
    return ('{}H'.format(24 / ts_per_day))
