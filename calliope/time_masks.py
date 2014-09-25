"""
Copyright (C) 2013 Stefan Pfenninger.
Licensed under the Apache 2.0 License (see LICENSE file).

time_masks.py
~~~~~~~~~~~~~

Defines time masks and resolution series to be used with
TimeSummarizer in time.py.

A mask function returns a series with the same index as the input data
that is 0 where the original resolution should be retained and 1 where
it should be reduced.

A resolution series function returns a a series with the same index
as the input data, where each time step is either a positive integer
(signifying how many timesteps to summarize), -1 (following a positive
integer and signifying the timesteps that are summarized), or 0 (no
adjustment to this timestep).

The name of a returned series must always be either 'mask' or
'resolution_series'.

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
    combined_mask.name = 'resolution_series'
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
    summarize.name = 'resolution_series'
    return summarize


def resolution_series_min_week(data, tech, var='r', resolution=24,
                               how='sum'):
    """
    Resolution series to keep the week where containing the day where
    ``var`` of ``tech`` is minimal across the sum of or the mode (most)
    of locations, depending on ``how``, and reduce everything else to
    the given resolution (in hours, daily by default).

    ``how`` can be 'sum' or 'mode'

    """
    df = data[var][tech]
    # Get length of a day in timesteps
    day_len = int(24 / data.time_res_data)
    # Get day-wise sums
    dff = pd.rolling_sum(df, window=day_len).reindex(range(0, len(df), day_len))
    if how == 'mode':
        # Get timestep where var/tech is minimal in the largest
        # number of locations
        selected = int(dff[dff > 0].idxmin().mode()[0])
    elif how == 'sum':
        selected = dff[dff > 0].sum(axis=1).idxmin()
    d = data._dt.at[selected]
    # Determine the range for the calendar week
    # (7 days) to keep at full resolution
    week_start = selected - day_len * d.dayofweek
    week_end = selected + day_len * (7 - d.dayofweek)
    # Mask where everything is -1 (summarize) by default
    mask = pd.DataFrame({'mask': -1}, index=range(len(df)))
    # Mark timesteps for summarization at given resolution
    summary_timestep_len = int(resolution / data.time_res_data)
    summary_index = range(0, len(df), summary_timestep_len)
    mask.loc[summary_index, 'mask'] = resolution
    # For the desired week, change the mask to native resolution (0)
    mask.loc[week_start:week_end - 1, 'mask'] = 0
    # Return only a series (why did I create a df anyway?)
    series = mask['mask']
    series.name = 'resolution_series'
    return series


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
    summarize.name = 'mask'
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
    summarize.name = 'mask'
    return summarize
