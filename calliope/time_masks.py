"""
Copyright (C) 2013 Stefan Pfenninger.
Licensed under the Apache 2.0 License (see LICENSE file).

time_masks.py
~~~~~~~~~~~~~

Defines time masks to be used with TimeSummarizer in time.py.

A mask function returns a series with the same index as the input data
that is 0 where the original resolution should be retained and 1 where
it should be reduced.

"""


from __future__ import print_function
from __future__ import division

import pandas as pd


def masks_to_resolution_series(masks, how='or', min_resolution=None):
    """
    Converts a list of overlapping masks into a series of time step
    resolutions.

    ``how`` can be ``or`` (default) or ``and``.

    If ``min_resolution`` given, will break up masked areas into
    timesteps of at least the given resolution (with possibly a
    leftover timestep at a higher resolution at the end of the
    masked area).

    """
    if not isinstance(masks, list) or isinstance(masks, tuple):
        masks = [masks]
    # combine all masks into one
    df = pd.DataFrame({i: x for i, x in enumerate(masks)})
    if how == 'or':
        combined_mask = df.sum(axis=1)
    elif how == 'and':
        # joiner: only return 1 if all items in the row are 1, else return 0
        joiner = lambda row: 1 if sum(row) == len(row) else 0
        combined_mask = df.apply(joiner, axis=1)
    istart = 0
    end = False
    while not end:
        ifrom = combined_mask[istart:].argmax()
        ito = combined_mask[ifrom:].argmin()
        if ifrom == ito:  # Reached the end!
            ito = len(combined_mask)
            end = True
            # If `summarize` is zero at the very last entry
            # (`ito - `), we break out of the
            # loop to prevent it from adding a spurious summarization
            if combined_mask[ito - 1] == 0:
                break
        resolution = ito - ifrom
        combined_mask[ifrom] = resolution
        combined_mask[ifrom + 1:ito] = -1
        # Correct edge case where only one timestep would be "summarized"
        if combined_mask[ifrom] >= 1 and resolution == 1:
            combined_mask[ifrom] = 0
        istart = ito
    return combined_mask


def resolution_series_uniform(data, resolution):
    """
    Resolution series to reduce resolution uniformly.

    """
    res_length = resolution / data.time_res_static
    df = data.r[data.r.keys()[0]]  # Grab length of data from any table
    summarize = pd.Series(-1, index=range(len(df)))
    # Set to 0 (keep timestep) for the given resolution
    for index, item in summarize.iteritems():
        if index % res_length == 0:
            summarize.at[index] = resolution
    return summarize


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
    summarize = pd.Series(0, index=range(len(df)))
    # Summing over all DNIs to find those times where DNI==0 everywhere
    summarize[df.sum(1) <= 0] = 1
    return summarize


def mask_extreme(data, tech, var='r', how='max',
                 length=24, locations=None, padding=None):
    """
    Mask everywhere except the ``length`` where ``var`` for the technology
    ``tech`` across the given list of ``locations`` is either minmal or maximal.

    ``var`` defaults to ``r``.

    ``how`` can be either ``max`` (default) or ``min``.

    If ``locations`` not given, uses all available locations.

    If ``padding`` given, pads beginning and end of the unmasked area.

    """
    df = data[var][tech]
    if locations:
        df = df.loc[:, locations]
    df = df.sum(axis=1)
    totals = []
    for i in range(0, len(df), length):
        totals.append(df[i:(i+1)*length].sum(axis=0))
    totals = pd.Series(totals)
    if how == 'max':
        total_i = totals.argmax()
    elif how == 'min':
        total_i = totals.argmin()
    summarize = pd.Series(1, index=range(len(df)))
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
    summarize.loc[ifrom:ito] = 0
    return summarize
