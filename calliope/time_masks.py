from __future__ import print_function
from __future__ import division

import pandas as pd


def mask_where_zero(data, tech, var='r', nodes=None):
    """Return a mask to summarize where ``var`` for the technology ``tech``
    across the given list of ``nodes`` is zero.

    ``var`` defaults to ``r``.

    If ``nodes`` not given, uses all available nodes.

    """
    df = data[var][tech].copy(deep=True)
    if nodes:
        df = df.loc[:, nodes]
    # Summing over all DNIs to find those times where DNI==0 everywhere
    df = pd.DataFrame({'data': df.sum(1)})
    df['summarize'] = 0
    df['summarize'][df['data'] <= 0] = 1
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
        df.summarize[ifrom] = resolution
        df.summarize[ifrom+1:ito] = -1
        istart = ito
    return df
