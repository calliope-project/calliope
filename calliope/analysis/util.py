"""
Copyright (C) 2013-2018 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

plotting.py
~~~~~~~~~~~

Functionality to plot model data.

"""

def subset_sum_squeeze(data, subset={}, sum_dims=None, squeeze=False):
    if subset:  # first, subset the data
        allowed_subsets = {k: v for k, v in subset.items() if k in data.dims}
        data = data.loc[allowed_subsets]

    if sum_dims:  # second, sum along all necessary dimensions
        data = data.sum(sum_dims)

    if squeeze and len(data.techs) > 1:  # finally, squeeze out single length dimensions
        data = data.squeeze()

    return data


def hex_to_rgba(hex_color, opacity):
    _NUMERALS = '0123456789abcdefABCDEF'
    _HEXDEC = {v: int(v, 16) for v in (x + y for x in _NUMERALS for y in _NUMERALS)}
    hex_color = hex_color.lstrip('#')
    rgb = [_HEXDEC[hex_color[0:2]], _HEXDEC[hex_color[2:4]], _HEXDEC[hex_color[4:6]]]
    return 'rgba({1}, {2}, {3}, {0})'.format(opacity, *rgb)
