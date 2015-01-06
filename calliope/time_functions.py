"""
Copyright (C) 2013-2015 Stefan Pfenninger.
Licensed under the Apache 2.0 License (see LICENSE file).

time_functions.py
~~~~~~~~~~~~~~~~~

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

import pandas as pd


def masks_to_resolution_series(masks, how='or', max_timesteps=None):
    """
    Converts a list of overlapping masks into a series of time step
    resolutions.

    ``how`` can be ``or`` (default) or ``and``.

    If ``max_timesteps`` given, will break up masked areas into
    timesteps of at at most the given length (with possibly a
    leftover timestep at a lower length at the end of the
    masked area).

    """
    if not isinstance(masks, list) or isinstance(masks, tuple):
        masks = [masks]
    # combine all masks into one
    df = pd.DataFrame({i: x for i, x in enumerate(masks)})
    if how == 'or':
        mask = df.sum(axis=1)
    elif how == 'and':
        # joiner: only return 1 if all items in the row are 1, else return 0
        joiner = lambda row: 1 if sum(row) == len(row) else 0
        mask = df.apply(joiner, axis=1)
    istart = 0
    end = False
    while not end:
        ifrom = mask[istart:].argmax()
        ito = mask[ifrom:].argmin()
        if ifrom == ito:  # Reached the end!
            ito = len(mask)
            end = True
            # If `summarize` is zero at the very last entry
            # (`ito - `), we break out of the
            # loop to prevent it from adding a spurious summarization
            if mask[ito - 1] == 0:
                break
        resolution = ito - ifrom
        mask[ifrom] = resolution
        mask[ifrom + 1:ito] = -1
        # Correct edge case where only one timestep would be "summarized"
        if mask[ifrom] >= 1 and resolution == 1:
            mask[ifrom] = 0
        istart = ito
    # Apply max_timesteps
    if max_timesteps:
        for index, value in mask[mask > max_timesteps].iteritems():
            end_index = index + value
            summary_index = list(range(index, end_index, max_timesteps))
            for i in summary_index:
                if i + max_timesteps < end_index:
                    mask[i] = max_timesteps
                else:  # Make sure the last timestep isn't too long
                    mask[i] = end_index - i
    mask.name = 'resolution_series'
    return mask


def resolution_series_to_mask(resolution_series):
    """
    Turns a resolution series into a mask.

    """
    mask = resolution_series
    mask[mask != 0] = 1
    mask.name = 'mask'
    return mask


def resolution_series_uniform(data, resolution):
    """
    Resolution series to reduce resolution uniformly.

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


def resolution_series_extreme_week(data, tech, var='r', resolution=24,
                                   how='sum', what='min'):
    """
    Resolution series to keep the week containing the day where
    ``var`` of ``tech`` is minimal (what='min') or maximal (what='max')
    across the sum (how='sum') of or the mode (most) of locations,
    (how='mode'), and reduce everything else to the given resolution
    (in hours, daily by default).

    Parameters
    ----------
    data : Calliope model data
    tech : str
        technology whose `var` to find extreme week for
    var : str, default 'r'
    resolution : int, default 24
        resolution the non-extreme week is reduced to
    how : str, default 'sum'
        'sum' or 'mode'
    what : str, default 'min'
        'min' or 'max'

    """
    df = data[var][tech]
    # Get length of a day in timesteps
    day_len = int(24 / data.time_res_data)
    # Get day-wise sums
    dff_index = list(range(0, len(df), day_len))
    dff = pd.rolling_sum(df, window=day_len).reindex(dff_index)
    # If what is 'min', this will get the 'idxmin' attribute (a method),
    # similar for 'max', else most likely raise an error!
    idx_extr = lambda x: getattr(x, 'idx{}'.format(what))
    if how == 'mode':
        # Find day where var/tech is min/max across the most locations
        selected = idx_extr(dff)().mode()[0]
    elif how == 'sum':
        # Find day where var/tech is min/max across all locations
        selected = idx_extr(dff.sum(axis=1))()
    d = data._dt.at[selected]
    # Determine the range for the calendar week
    # (7 days) to keep at full resolution
    week_start = selected - day_len * d.dayofweek
    week_end = selected + day_len * (7 - d.dayofweek)
    # Mask where everything is -1 (summarize) by default
    mask = pd.DataFrame({'mask': -1}, index=list(range(len(df))))
    # Mark timesteps for summarization at given resolution
    summary_timestep_len = int(resolution / data.time_res_data)
    summary_index = list(range(0, len(df), summary_timestep_len))
    mask.loc[summary_index, 'mask'] = resolution
    # For the desired week, change the mask to native resolution (0)
    mask.loc[week_start:week_end - 1, 'mask'] = 0
    # Return only a series (why did I create a df anyway?)
    series = mask['mask']
    series.name = 'resolution_series'
    return series


def mask_zero(data, tech, var='r', locations=None):  # FIXME
    """
    Mask where ``var`` for the technology ``tech``
    across the given list of ``locations`` is zero.

    ``var`` defaults to ``r``.

    If ``locations`` not given, uses all available locations.

    """
    df = data[var][tech]
    if locations:
        df = df.loc[:, locations]
    summarize = pd.Series(0, index=list(range(len(df))))
    # Summing over all DNIs to find those times where DNI==0 everywhere
    summarize[df.sum(1) <= 0] = 1
    summarize.name = 'mask'
    return summarize


def mask_extreme(data, tech, var='r', how='max',
                 length=24, locations=None, padding=None):  # FIXME
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
        totals.append(df[i:(i + 1) * length].sum(axis=0))
    totals = pd.Series(totals)
    if how == 'max':
        total_i = totals.argmax()
    elif how == 'min':
        total_i = totals.argmin()
    summarize = pd.Series(1, index=list(range(len(df))))
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
