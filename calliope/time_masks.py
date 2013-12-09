from __future__ import print_function
from __future__ import division

import pandas as pd


def simple_mask_where_zero(data, tech, var='r', nodes=None):
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
    return df
