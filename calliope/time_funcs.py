"""
Copyright (C) 2013-2016 Stefan Pfenninger.
Licensed under the Apache 2.0 License (see LICENSE file).

time_funcs.py
~~~~~~~~~~~~~

A time function must take ``data`` and ``timesteps`` and optionally any
additional keyword arguments. It modifies the timesteps in the data and
returns the timestep weights for the modified timesteps.

"""

import numpy as np
import pandas as pd
import xarray as xr
from xarray.ufuncs import fabs

import scipy.cluster.vq as vq
from scipy.cluster import hierarchy

from .core import plugin_load
from . import data_tools as dt


def normalize(data):
    """
    Return a copy of data, with the absolute taken and normalized to 0-1.

    The maximum across all regions and timesteps is used to normalize.

    """
    ds = data.copy(deep=True)  # Work off a copy
    for var in dt.get_datavars(data):
        y_var = '_y_def_{}'.format(var)
        for y in ds.coords[y_var].values:
            # Get max across all regions to normalize against
            norm_max = fabs(ds[var].loc[{y_var: y}]).max()
            for x in ds.coords['x'].values:
                df = ds[var].loc[{'x': x, y_var: y}]
                ds[var].loc[{'x': x, y_var: y}] = fabs(df) / norm_max
    return ds


def reshape_for_clustering(data, tech=None, invert=False):
    y = dt.get_y_coord(data.r)
    timesteps_per_day = dt.get_timesteps_per_day(data)
    days = int(len(data.t) / timesteps_per_day)
    regions = int(len(data['x']))
    techs = int(len(data[y]))

    if tech is not None:
        X = data.r.loc[{y: tech}].values.reshape((days, timesteps_per_day * regions))
    else:
        X = data.r.transpose('t', 'x', y).values.reshape(days, timesteps_per_day * regions * techs)

    return np.nan_to_num(X)  # replace any NaN with 0


def reshape_clustered(clustered, data):
    y = dt.get_y_coord(data.r)
    timesteps_per_day = dt.get_timesteps_per_day(data)
    regions = int(len(data['x']))
    techs = int(len(data[y]))
    days = clustered.shape[0]

    arr = xr.DataArray(clustered.reshape(techs, days * timesteps_per_day, regions),
                       dims=(y, 't', 'x'))
    arr['x'] = data['x']
    arr[y] = data[y]

    return arr


def get_mean_from_clusters(data, clusters, timesteps_per_day):

    cluster_map = clusters.groupby(clusters).groups

    hour_of_day = pd.Series([pd.Timestamp(i).hour
                             for i in data.coords['t'].values])
    # hour_of_day = pd.Series([i - (i // timesteps_per_day) * timesteps_per_day
    #                          for i in data.coords['t'].values])

    ds = {}
    for var in dt.get_datavars(data):
        data_arrays = []
        array = data[var]
        y_coord = dt.get_y_coord(array)

        t_coords = ['{}-{}'.format(cid, t)
                    for cid in cluster_map
                    for t in range(timesteps_per_day)]

        # Check if this variable has any data
        if len(data.coords[y_coord].values) == 0:
            e = np.empty((0, len(t_coords), len(data.coords['x'].values)))
            ds[var] = xr.DataArray(e, dims=[y_coord, 't', 'x'],
                                   coords={'x': data.coords['x'].values,
                                           't': t_coords})
        else:
            for cluster_id, cluster_members in cluster_map.items():
                y_arrays = []
                for y in data.coords[y_coord].values:
                    d = (array.loc[{'t': cluster_members, y_coord: y}]
                              .to_pandas().groupby(lambda x: x.hour).mean()
                              .to_xarray().to_array(dim='x').T
                         )
                    d.coords[y_coord] = y
                    d = d.rename({'index': 't'})
                    d.coords['t'] = [i for i in t_coords
                                     if i.startswith('{}-'.format(cluster_id))]
                    y_arrays.append(d)
                data_arrays.append(xr.concat(y_arrays, dim=y_coord))
                # data_arrays[-1].coords['cluster'] = cluster_id

            ds[var] = xr.concat(data_arrays, dim='t')
    ds = xr.Dataset(ds)
    return ds


def map_clusters_to_data(data, clusters, how, **kwargs):
    """
    Returns a copy of data that has been clustered.

    how can be mean, centroid, closest_to_centroid

    """

    # Get all timesteps, not just the first per day
    ts_per_day = dt.get_timesteps_per_day(data)
    idx = clusters.index
    new_idx = pd.concat([pd.Series(1, pd.date_range(i, i + pd.Timedelta('1D'),
                                                    freq='1H')[:-1])
                         for i in idx]).index
    c = (clusters.reindex(new_idx)
                 .fillna(method='ffill').astype(int))

    # == PICK DATA ==
    # TODO: additional options for picking data
    # or pick cluster centroid
    # or pick the real day closest to cluster centroid
    # if using centroid, need kwargs: centroids=None, variable=None, tech=None,
    if how == 'mean':
        new_data = get_mean_from_clusters(data, c, ts_per_day)

        # Add timestep names by taking the median timestamp from daily clusters...
        # (a random way of doing it, but we want some label to apply)
        timestamps = clusters.groupby(clusters).apply(lambda x: x.index[int(len(x.index) / 2)])
        new_t_coord = pd.concat([pd.Series(pd.date_range(ts, ts + pd.Timedelta('1D'), freq='1H')[:-1])
                                for ts in timestamps], ignore_index=True)

        new_data.coords['t'] = new_t_coord.as_matrix()

    # == DETERMINE WEIGHTS ==
    value_counts = c.value_counts()
    weights = c.map(lambda x: value_counts[x])

    weights = weights / ts_per_day

    new_data['_weights'] = xr.DataArray(weights, dims=['t'])
    new_data['_time_res'] = xr.DataArray(np.ones(len(new_data['t'])) * (24 / ts_per_day),
                                         coords={'t': new_data['t']})

    return new_data


def apply_clustering(data, timesteps=None, clustering_func=None, **args):
    """
    Modifies the passed model's data in-place

    Returns
    -------
    data : xarray.Dataset
        Original data
    new_data : xarray.Dataset
    new_data_scaled : xarray.Dataset

    """
    timesteps = data.t[timesteps].values

    # Only apply clustering function on subset of masked timesteps
    if timesteps is not None:
        data_to_cluster = data.loc[{'t': timesteps}]
    else:
        data_to_cluster = data

    data_normalized = normalize(data_to_cluster)

    # Get function from `clustering_func` string
    func = plugin_load(clustering_func, builtin_module='time_funcs')

    result = func(data_normalized, **args)
    clusters = result[0]  # Ignore other stuff returned

    data_new = map_clusters_to_data(data_to_cluster, clusters, how='mean')
    # new_data_normalized = map_clusters_to_data(data_normalized, clusters, how='mean')

    if timesteps is not None:
        # Drop timesteps from old data
        data_excluded_from_clustering = data.drop(timesteps, dim='t')
        # Combine the clustered with the old unclustered data
        data_new = xr.concat([data_excluded_from_clustering, data_new], dim='t')

    # Scale the new/combined data so that the mean for each (x, y, variable)
    # combination matches that from the original data
    data_new_scaled = data_new.copy(deep=True)
    for var in dt.get_datavars(data_to_cluster):
        scale_to_match_mean = (data[var].mean(dim='t') / data_new[var].mean(dim='t')).fillna(0)
        data_new_scaled[var] = data_new[var] * scale_to_match_mean

    return data_new_scaled

def get_clusters_kmeans(data, tech=None, timesteps=None, k=5):
    """

    Returns
    -------
    clusters : dataframe
        Indexed by timesteps and with locations as columns, giving cluster
        membership for first timestep of each day.
    centroids

    """
    timesteps_per_day = dt.get_timesteps_per_day(data)

    if timesteps is not None:
        data = data.loc[{'t': timesteps}]
    else:
        timesteps = data.t.values

    X = reshape_for_clustering(data, tech)

    centroids, distortion = vq.kmeans(X, k)

    # Determine the cluster membership of each day
    day_clusters = vq.vq(X, centroids)[0]

    # Create mapping of timesteps to clusters
    clusters = pd.Series(day_clusters, index=timesteps[::timesteps_per_day])

    # Reshape centroids
    centroids = reshape_clustered(centroids, data)

    return clusters, centroids


def get_clusters_hierarchical(data, tech=None, max_d=None, k=None):
    """
    Parameters
    ----------
    max_d : float or int, optional
        Max distance for returning clusters.
    k : int, optional
        Number of desired clusters.

    Returns
    -------
    clusters
    X
    Z

    """
    data_n = normalize(data)
    X = reshape_for_clustering(data_n, tech)

    # Generate the linkage matrix
    Z = hierarchy.linkage(X, 'ward')

    if max_d:
        # Get clusters based on maximum distance
        clusters = hierarchy.fcluster(Z, max_d, criterion='distance')
    elif k:
        # Get clusters based on number of desired clusters
        clusters = hierarchy.fcluster(Z, k, criterion='maxclust')
    else:
        clusters = None

    # Make sure clusters are a pd.Series with a datetime index
    if clusters is not None:
        timesteps_per_day = dt.get_timesteps_per_day(data)
        timesteps = data.coords['t'].values  # All timesteps

        clusters = pd.Series(clusters, index=timesteps[::timesteps_per_day])

    return (clusters, X, Z)
