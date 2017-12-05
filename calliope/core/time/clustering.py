"""
Copyright (C) 2013-2017 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

clustering.py
~~~~~~~~~~~~~

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

from calliope import exceptions
from calliope.core.util.loc_tech import get_loc_techs


def _get_datavars(data):
    return [var for var in data.data_vars if not var.startswith('_')]


def _get_timesteps_per_day(data):
    timesteps_per_day = data.attrs['timestep_resolution'] * 24
    if isinstance(timesteps_per_day, float):
        assert timesteps_per_day.is_integer(), 'Timesteps/day must be integer.'
        timesteps_per_day = int(timesteps_per_day)
    return timesteps_per_day


def _stack_data(data):
    """
    Stack all non-time dimensions of an xarray DataArray
    """
    non_time_dims = list(set(data.dims).difference(['time']))
    if len(non_time_dims) >= 1:
        stacked_var = data.stack(
            stacked=[i for i in data.dims if i is not 'time']
        )
    else:
        e = exceptions.ModelError
        raise e("Cannot conduct time clustering with variable {} as it has no "
        "non-time dimensions.".format(data.name))
    return stacked_var


def reshape_for_clustering(data, loc_techs=None, variables=None):
    """
    Create an array of timeseries values, where each day has a row of all
    hourly timeseries data from all relevant variables

    Parameters
    ----------
    data : xarray Dataset
        Dataset with all non-time dependent variables removed
    loc_techs : string or list-like, default = None
        If clustering over a subset of loc_techs, they are listed here
    variables : string or list-like, default = None
        If clustering over a subset of timeseries variables, they are listed here

    Returns
    -------
    X : numpy 2darray
        Array with all rows as days and all other data in columns, with NaNs
        converted to zeros
    """
    # to create an array with days as rows, we need to get timesteps per day
    timesteps_per_day = _get_timesteps_per_day(data)
    days = int(len(data.timesteps) / timesteps_per_day)

    reshaped_data = np.array([[] for i in range(days)])
    # if 'variables' is given then we will loop over that, otherwise we loop over
    # all timeseries variables
    relevent_vars = variables if variables else data.data_vars

    # loop over variables to get our arrays
    for var in relevent_vars:
        temp_data = data[var].copy()
        # if there is a loc_tech subset, index over that
        if loc_techs:
            loc_tech_dim = [i for i in data[var].dims if 'loc_techs' in i][0]
            relevent_loc_techs = list(
                set(temp_data[loc_tech_dim]).intersection(loc_techs)
            )
            temp_data = temp_data.loc[{loc_tech_dim: relevent_loc_techs}]

        # stack all non-time dimensions to get one row of data for each timestep
        stacked_var = _stack_data(temp_data)
        # reshape the array to split days and timesteps within days, now we have
        # one row of data per day
        reshaped_data = np.concatenate((reshaped_data,
            stacked_var.values.reshape(days,
                timesteps_per_day * len(stacked_var.stacked)
            )), axis=1
        )
    # put all the columns of data together, keeping rows as days. Also convert
    # all nans to zeros
    X = np.nan_to_num(reshaped_data)

    return X


def reshape_clustered(clustered, data, loc_techs=None, variables=None):
    """
    Repopulate a Dataset from an array that has each day as a row and all other
    variable information (timesteps in a day, loc_techs, etc.) as columns.

    The array here is expected to be the same shape as the array output from
    `reshape_for_clustering`

    Parameters
    ----------
    clustered : numpy 2darray
        array of days, clustered as per given clustering method
    data : xarray Dataset
        Dataset with all non-time dependent variables removed
    loc_techs : string or list-like, default = None
        If clustering over a subset of loc_techs, they are listed here
    variables : string or list-like, default = None
        If clustering over a subset of timeseries variables, they are listed here

    Returns
    -------
    X : xarray Dataset
        Dataset with DataArray variables associated with each timeseries
        variable of interest.
    """

    timesteps_per_day = _get_timesteps_per_day(data)
    clusters = clustered.shape[0]

    # if 'variables' is given then we will loop over that, otherwise we loop over
    # all timeseries variables
    relevent_vars = variables if variables else data.data_vars

    reshaped_data = dict()

    # Initialise the location of the last column of data for a given variable
    previous_last_column = 0
    # loop over variables to get our arrays
    for var in relevent_vars:
        temp_data = data[var].copy()
        # if there is a loc_tech subset, index over that
        if loc_techs:
            loc_tech_dim = [i for i in data[var].dims if 'loc_techs' in i][0]
            relevent_loc_techs = list(
                set(temp_data[loc_tech_dim]).intersection(loc_techs)
            )
            temp_data = temp_data.loc[{loc_tech_dim: relevent_loc_techs}]

        # list of strings of non-time dimensions for our variable
        non_time_dims = set(temp_data.dims).difference(['timesteps'])

        # length of each of those non-time dimensions
        non_time_dim_lengths = [len(temp_data[i]) for i in non_time_dims]

        # length of each dimension in final output
        reshaped_dims = ([clusters, timesteps_per_day] + non_time_dim_lengths)

        last_column = previous_last_column + (
            timesteps_per_day * np.prod(non_time_dim_lengths)
        )

        data_range = range(previous_last_column, last_column)

        # store information in dictionary, for later conversion to Dataset
        reshaped_data[var] = {
            'data':clustered[:, data_range].reshape(reshaped_dims),
            'dims':['clusters', 'timesteps'] + list(non_time_dims)
        }

        previous_last_column = last_column

    # output is a dataset, built from dictionaries
    reshaped_dataset = xr.Dataset.from_dict(reshaped_data)

    return reshaped_dataset


def get_mean_from_clusters(data, clusters, timesteps_per_day):
    """
    Clusters are days which are considered similar to each other. Here we find
    the mean value (in each tech dimension) for each cluster and return it

    Parameters
    ----------
    data : xarray Dataset
        Dataset with all non-time dependent variables removed
    clusters : pandas DataFrame
        index as days, columns as cluster group to which that day is associated
    timesteps_per_day : int
        Number of timesteps in each day

    Returns
    -------
    ds : xarray Dataset
        Dataset of timeseries DataArrays. Time dimension is only equal to the
        number of clusters * timesteps_per_day.
    """

    cluster_map = clusters.groupby(clusters).groups

    hour_of_day = pd.Series([pd.Timestamp(i).hour
                             for i in data.coords['timesteps'].values])

    ds = {}
    t_coords = ['{}-{}'.format(cid, t)
                for cid in cluster_map
                for t in range(timesteps_per_day)]
    for var in data.data_vars:
        loc_tech_dim = [i for i in data[var].dims if 'loc_techs' in i][0]
        data_arrays = []
        array = data[var].copy()
        for cluster_id, cluster_members in cluster_map.items():
            loc_tech_arrays = []
            var_techs = set(
                i.split(':', 1)[1] for i in data[var][loc_tech_dim].values
            )
            for tech in var_techs:
                relevent_loc_techs = (
                    get_loc_techs(data[loc_tech_dim].values, tech)
                )
                d = (array.loc[{'timesteps': cluster_members,
                                loc_tech_dim: relevent_loc_techs}]
                          .groupby('timesteps.hour').mean(dim='timesteps')
                          .rename({'hour': 'timesteps'})
                        )
                d.coords['timesteps'] = [i for i in t_coords
                                    if i.startswith('{}-'.format(cluster_id))]
                loc_tech_arrays.append(d)
            data_arrays.append(xr.concat(loc_tech_arrays, dim=loc_tech_dim))

            ds[var] = xr.concat(data_arrays, dim='timesteps')
    ds = xr.Dataset(ds)
    return ds


def find_nearest_vector_index(array, value):
    return np.array([np.linalg.norm(sum(y)) for y in array - value]).argmin()


def get_closest_days_from_clusters(data, mean_data, clusters):

    dtindex = data['timesteps'].to_index()
    ts_per_day = _get_timesteps_per_day(data)
    days = int(len(data['timesteps']) / ts_per_day)

    chosen_days = {}

    for cluster in sorted(clusters.unique()):

        subset_t = [t for t in mean_data.timesteps.values
                    if t.startswith('{}-'.format(cluster))]

        target = reshape_for_clustering(mean_data.loc[dict(timesteps=subset_t)])

        lookup_array = reshape_for_clustering(data)

        chosen_days[cluster] = find_nearest_vector_index(lookup_array, target)

    days_list = sorted(list(set(chosen_days.values())))
    new_t_coord = _hourly_from_daily_index(dtindex[::ts_per_day][days_list])

    chosen_day_timestamps = {k: dtindex[::ts_per_day][v]
                             for k, v in chosen_days.items()}

    new_data = data.loc[dict(timesteps=new_t_coord)]

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
    new_data.attrs = data.attrs

    if how == 'mean':
        # Add timestep names by taking the median timestamp from daily clusters...
        # (a random way of doing it, but we want some label to apply)
        timestamps = clusters.groupby(clusters).apply(
            lambda x: x.index[int(len(x.index) / 2)]
        )
        new_t_coord = pd.concat([
            pd.Series(
                pd.date_range(ts, ts + pd.Timedelta('1D'), freq='1H')[:-1]
            )
            for ts in timestamps], ignore_index=True
        )
        new_data.coords['time'] = new_t_coord.as_matrix()

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
    new_data['timestep_weights'] = xr.DataArray(weights, dims=['timesteps'])
    new_data['timestep_resolution'] = xr.DataArray(np.ones(len(new_data['timesteps']))
                                                * (24 / ts_per_day),
                                                dims=['timesteps'],
                                                coords={'timesteps': new_data['timesteps']})
    del new_data.attrs['timestep_resolution']
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
        data = data.loc[{'timesteps': timesteps}]
    else:
        timesteps = data.timesteps.values

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
        timesteps = data.coords['timesteps'].values  # All timesteps

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
