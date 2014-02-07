"""
Copyright (C) 2013 Stefan Pfenninger.
Licensed under the Apache 2.0 License (see LICENSE file).

time_masks.py
~~~~~~~~~~~~~

Defines time masks to be used with TimeSummarizer in time.py.

"""


from __future__ import print_function
from __future__ import division

import pandas as pd


def mask_where_zero(data, tech, var='r', locations=None):
    """Return a mask to summarize where ``var`` for the technology ``tech``
    across the given list of ``locations`` is zero.

    ``var`` defaults to ``r``.

    If ``locations`` not given, uses all available locations.

    """
    df = data[var][tech].copy(deep=True)
    if locations:
        df = df.loc[:, locations]
    # Summing over all DNIs to find those times where DNI==0 everywhere
    df = pd.DataFrame({'data': df.sum(1)})
    df['summarize'] = 0
    df['summarize'][df['data'] <= 0] = 1
    # Apply the variable time step algorithm
    istart = 0
    end = False
    while not end:
        ifrom = df.summarize[istart:].argmax()
        ito = df.summarize[ifrom:].argmin()
        if ifrom == ito:  # Reached the end!
            ito = len(df.summarize)
            end = True
            # If `summarize` is zero at the very last entry
            # (`ito - `), we break out of the
            # loop to prevent it from adding a spurious summarization
            if df.summarize[ito - 1] == 0:
                break
        resolution = ito - ifrom
        df.summarize[ifrom] = resolution
        df.summarize[ifrom+1:ito] = -1
        # Correct edge case where only one timestep would be "summarized"
        if df.summarize[ifrom] == 1 and resolution == 1:
            df.summarize[ifrom] = 0
        istart = ito
    return df
