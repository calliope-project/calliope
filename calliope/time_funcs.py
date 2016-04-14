"""
Copyright (C) 2013-2016 Stefan Pfenninger.
Licensed under the Apache 2.0 License (see LICENSE file).

time_funcs.py
~~~~~~~~~~~~~

A time function must take ``data`` and ``timesteps`` and optionally any
additional keyword arguments. It modifies the timesteps in the data and
returns the timestep weights for the modified timesteps.

"""

import pandas as pd

from . import time_tools


def _kmeans(data, target, timesteps=None, days=5):
    """
    Adjusts

    Parameters
    ----------
    data : Calliope model data
    target : series
        Series to use for clustering.
    timesteps : list or array, optional
        Optional subset of timesteps to use for clustering.
    days : int, optional
        Number of clusters (days) to return, default = 5.

    Returns
    -------
    weights :

    """
    import scipy.cluster.vq as vq

    timesteps_per_day = data.time_res_static * 24

    if timesteps is None:
        timesteps = data._dt.index  # All timesteps

    # Get a (m,d) matrix where m is number of days in the matrix
    # and d is the number of timesteps in a day
    d = target.loc[timesteps, :].sum(1)
    d = d.reshape(len(d) / timesteps_per_day, timesteps_per_day)
    # Run k-means algorithm for the desired number of days (clusters)
    centroids, distortion = vq.kmeans(d, days)

    # Determine the cluster membership of each day
    day_clusters = vq.vq(d, centroids)[0]

    # Create mapping of timesteps to clusters
    ts_clusters = pd.Series(day_clusters, index=timesteps[::timesteps_per_day])

    # Get date of each cluster
    clusters = sorted(ts_clusters.unique())
    dates = [data._dt[ts_clusters[ts_clusters == cluster].index].iat[0]
             for cluster in clusters]
    dates = pd.Series(dates, index=clusters)

    # Return a 'timestep to chosen day' map
    clusters = ts_clusters.map(lambda x: dates[x])

    # Modify the timesteps and get timestep weights
    weights = time_tools._apply_day_summarizer(data, clusters)
    return weights


def kmeans_typicaldays_singlevariable(data, variable, timesteps=None, days=5):
    df = data.get_key(variable)
    return _kmeans(data, df, timesteps, days)
