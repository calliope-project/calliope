"""
Copyright (C) 2013-2017 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

time_clustering.py
~~~~~~~~~~~~~~~~~~

Functions to cluster data along the time dimension.

"""

import numpy as np
import pandas as pd
import xarray as xr

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches
    from matplotlib.colors import rgb2hex
    import seaborn as sns
except ImportError:
    pass  # This is logged in analysis.py

import scipy.cluster.vq as vq
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist


def _get_y_coord(array):
    if 'y' in array.coords:
        y = 'y'
    else:
        try:  # assumes a single y_ coord in array
            y = [k for k in array.coords if 'y_' in k][0]
        except IndexError:  # empty list
            y = None
    return y


def _get_datavars(data):
    return [var for var in data.data_vars if not var.startswith('_')]


def _get_timesteps_per_day(data):
    timesteps_per_day = data.attrs['time_res'] * 24
    if isinstance(timesteps_per_day, float):
        assert timesteps_per_day.is_integer(), 'Timesteps/day must be integer.'
        timesteps_per_day = int(timesteps_per_day)
    return timesteps_per_day


def reshape_for_clustering(data, tech=None):
    y_coord = 'y_def_r'
    y_values = list(data.attrs['_sets'][y_coord])
    timesteps_per_day = _get_timesteps_per_day(data)
    days = int(len(data.t) / timesteps_per_day)
    regions = len(data['x'])
    techs = len(y_values)

    if tech is None:
        X = (data.r.loc[{'y': y_values}]
                   .transpose('t', 'x', 'y')
                   .values.reshape(days, timesteps_per_day * regions * techs))
    else:
        X = (data.r.loc[{'y': tech}]
                   .values
                   .reshape((days, timesteps_per_day * regions)))

    return np.nan_to_num(X)  # replace any NaN with 0


def reshape_clustered(clustered, data, tech=None):
    y_coord = 'y_def_r'
    y_values = list(data.attrs['_sets'][y_coord])
    timesteps_per_day = _get_timesteps_per_day(data)
    days = clustered.shape[0]
    regions = len(data['x'])
    techs = len(y_values)

    if tech is None:
        reshaped = clustered.reshape(techs, days * timesteps_per_day, regions)
    else:
        reshaped = clustered.reshape(1, days * timesteps_per_day, regions)

    arr = xr.DataArray(reshaped, dims=('y', 't', 'x'))

    arr['x'] = data['x']
    if tech is None:
        arr['y'] = y_values
    else:
        arr['y'] = tech

    return arr


def get_mean_from_clusters(data, clusters, timesteps_per_day):

    cluster_map = clusters.groupby(clusters).groups

    hour_of_day = pd.Series([pd.Timestamp(i).hour
                             for i in data.coords['t'].values])
    # hour_of_day = pd.Series([i - (i // timesteps_per_day) * timesteps_per_day
    #                          for i in data.coords['t'].values])

    ds = {}
    data_vars_in_t = [v for v in _get_datavars(data)
                      if 't' in data[v].dims]
    for var in data_vars_in_t:
        data_arrays = []
        array = data[var]
        y_coord = _get_y_coord(array)

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


def find_nearest_vector_index(array, value):
    return np.array([np.linalg.norm(x + y + z + u)
                     for (x, y, z, u) in array - value]).argmin()


def get_closest_days_from_clusters(data, mean_data, clusters):
    subset_y = list(data.attrs['_sets']['y_def_r'])
    dtindex = data['t'].to_index()
    ts_per_day = _get_timesteps_per_day(data)
    days = int(len(data['t']) / ts_per_day)
    n_x = len(data['x'])
    n_y = len(subset_y)

    chosen_days = {}

    for cluster in sorted(clusters.unique()):

        subset_t = [t for t in mean_data.t.values if t.startswith('{}-'.format(cluster))]

        target = mean_data['r'].loc[dict(t=subset_t, y=subset_y)].values

        lookup_array = data['r'].loc[dict(y=subset_y)].values
        lookup_array = lookup_array.reshape((n_y, days, ts_per_day, n_x)).transpose(1, 0, 2, 3)

        chosen_days[cluster] = find_nearest_vector_index(lookup_array, target)

    days_list = sorted(list(set(chosen_days.values())))
    new_t_coord = _hourly_from_daily_index(dtindex[::ts_per_day][days_list])

    chosen_day_timestamps = {k: dtindex[::ts_per_day][v]
                             for k, v in chosen_days.items()}

    new_data = data.loc[dict(t=new_t_coord)]

    return new_data, chosen_day_timestamps


def _hourly_from_daily_index(idx):
    dtrange = lambda i: pd.date_range(i, i + pd.Timedelta('1D'),
                                      freq='1H')[:-1]
    new_idx = pd.concat([pd.Series(1, dtrange(i))
                         for i in idx]).index
    return new_idx


def map_clusters_to_data(data, clusters, how):
    """
    Returns a copy of data that has been clustered.

    Parameters
    ----------
    how : str
        How to select data from clusters.
        Can be mean (centroid) or closest.

    """
    # FIXME hardcoded time intervals ('1H', '1D')

    # Get all timesteps, not just the first per day
    ts_per_day = _get_timesteps_per_day(data)
    idx = clusters.index
    new_idx = _hourly_from_daily_index(idx)
    clusters_timeseries = (clusters.reindex(new_idx)
                           .fillna(method='ffill').astype(int))

    new_data = get_mean_from_clusters(data, clusters_timeseries, ts_per_day)

    if how == 'mean':
        # Add timestep names by taking the median timestamp from daily clusters...
        # (a random way of doing it, but we want some label to apply)
        timestamps = clusters.groupby(clusters).apply(lambda x: x.index[int(len(x.index) / 2)])
        new_t_coord = pd.concat([pd.Series(pd.date_range(ts, ts + pd.Timedelta('1D'), freq='1H')[:-1])
                                for ts in timestamps], ignore_index=True)
        new_data.coords['t'] = new_t_coord.as_matrix()

        # Generate weights
        # weight of each timestep = number of timesteps in this timestep's cluster
        # divided by timesteps per day (since we're grouping days together and
        # a cluster consisting of 1 day = 24 hours should have weight of 1)
        value_counts = clusters_timeseries.value_counts() / ts_per_day
        # And turn the index into dates (days)
        value_counts = pd.DataFrame({
            'dates': timestamps,
            'counts': value_counts}).set_index('dates')['counts']

    elif how == 'closest':
        new_data, chosen_ts = get_closest_days_from_clusters(data, new_data, clusters)
        # Deal with the case where more than one cluster has the same closest day
        # An easy way is to rename the original clusters with the chosen days
        # So at this point, clusterdays_timeseries maps all timesteps to the day
        # of year of the cluster the timestep belongs to
        clusterdays_timeseries = clusters_timeseries.map(lambda x: chosen_ts[x])
        value_counts = clusterdays_timeseries.value_counts() / ts_per_day

    weights = (value_counts.reindex(_hourly_from_daily_index(value_counts.index))
                           .fillna(method='ffill'))
    new_data['_weights'] = xr.DataArray(weights, dims=['t'])
    new_data['_time_res'] = xr.DataArray(np.ones(len(new_data['t'])) * (24 / ts_per_day),
                                         coords={'t': new_data['t']})

    return new_data


def get_clusters_kmeans(data, tech=None, timesteps=None, k=5):
    """
    Parameters
    ----------
    data : xarray.Dataset
        Should be normalized

    Returns
    -------
    clusters : dataframe
        Indexed by timesteps and with locations as columns, giving cluster
        membership for first timestep of each day.
    centroids

    """
    timesteps_per_day = _get_timesteps_per_day(data)

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
    centroids = reshape_clustered(centroids, data, tech)

    return clusters, centroids


def get_clusters_hierarchical(data, tech=None, max_d=None, k=None):
    """
    Parameters
    ----------
    data : xarray.Dataset
        Should be normalized
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
    X = reshape_for_clustering(data, tech)

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
        timesteps_per_day = _get_timesteps_per_day(data)
        timesteps = data.coords['t'].values  # All timesteps

        clusters = pd.Series(clusters, index=timesteps[::timesteps_per_day])

    return (clusters, X, Z)


def fancy_dendrogram(*args, **kwargs):
    """
    Code adapted from:
    https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/

    """
    max_d = kwargs.pop('max_d', None)
    if max_d and 'color_threshold' not in kwargs:
        kwargs['color_threshold'] = max_d
    annotate_above = kwargs.pop('annotate_above', 0)

    ddata = hierarchy.dendrogram(*args, **kwargs)

    if not kwargs.get('no_plot', False):
        plt.xlabel('Sample index or (cluster size)')
        plt.ylabel('Distance')
        for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            if y > annotate_above:
                plt.plot(x, y, 'o', c=c)
                plt.annotate("%.3g" % y, (x, y), xytext=(0, -5),
                             textcoords='offset points',
                             va='top', ha='center', fontsize=10)
        if max_d:
            plt.axhline(y=max_d, c='grey', lw=1.0, alpha=0.5)
    return ddata


def draw_dendrogram(Z, max_d):
    fig = plt.figure(figsize=(12, 6))
    ax = plt.subplot(111)

    # link_color_pal = sns.color_palette("hls", 8)
    link_color_pal = sns.color_palette("Set2", 10)

    hierarchy.set_link_color_palette([rgb2hex(i) for i in link_color_pal])

    color_threshold = max_d

    fancy_dendrogram(
        Z,
        ax=ax,
        color_threshold=color_threshold,
        truncate_mode='lastp',
        p=20,
        leaf_rotation=90,
        leaf_font_size=10,
        show_contracted=True,
        annotate_above=5,  # useful in small plots so annotations don't overlap
        max_d=color_threshold,
    )

    sns.despine()

    # Modify the contracted markers
    for child in ax.get_children():
        if isinstance(child, matplotlib.patches.Ellipse):
            child.set_zorder(1000)
            child.set_alpha(0.3)

    return fig


def cophenetic_corr(X, Z):
    """
    Get the Cophenetic Correlation Coefficient of a clustering with help
    of the cophenet() function. This (very very briefly) compares (correlates)
    the actual pairwise distances of all your samples to those implied by the
    hierarchical clustering. The closer the value is to 1, the better
    the clustering preserves the original distances.

    Source:
    https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/

    """
    c, coph_dists = hierarchy.cophenet(Z, pdist(X))
    return c, coph_dists
