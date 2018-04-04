"""
Copyright (C) 2013-2018 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

clustering.py
~~~~~~~~~~~~~

Functions to cluster data along the time dimension.

"""
import numpy as np
import pandas as pd
import xarray as xr

from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist
from sklearn.metrics import mean_squared_error
from sklearn import cluster as sk_cluster

from calliope import exceptions


def _stack_data(data, dates, times):
    """
    Stack all non-time dimensions of an xarray DataArray
    """
    data_to_stack = data.assign_coords(
        timesteps=pd.MultiIndex.from_product([dates, times], names=['dates', 'times'])
    ).unstack('timesteps')
    non_date_dims = list(set(data_to_stack.dims).difference(['dates', 'times'])) + ['times']
    if len(non_date_dims) >= 2:
        stacked_var = data_to_stack.stack(stacked=non_date_dims)
    else:
        raise exceptions.ModelError(
            "Cannot conduct time clustering with variable {} as it has no "
            "non-time dimensions.".format(data.name)
        )
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
    timesteps = data.timesteps.to_index()
    if timesteps.dtype.kind == 'M': # 'M' = datetime format
        dates = np.unique(timesteps.date)
        times = np.unique(timesteps.time)
    else: # mean_data from get_closest_days_from_clusters is in the format cluster-timestep
        dates = np.unique([i.split('-')[0] for i in timesteps])
        times = np.unique([i.split('-')[1] for i in timesteps])

    reshaped_data = np.array([[] for i in dates])
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
                set(temp_data[loc_tech_dim].values).intersection(loc_techs)
            )
            temp_data = temp_data.loc[{loc_tech_dim: relevent_loc_techs}]

        # stack all non-time dimensions to get one row of data for each timestep
        stacked_var = _stack_data(temp_data, dates, times)
        # reshape the array to split days and timesteps within days, now we have
        # one row of data per day
        reshaped_data = np.concatenate([reshaped_data, stacked_var.values], axis=1)
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
    timesteps = data.timesteps.to_index()
    if timesteps.dtype.kind == 'M': # 'M' = datetime format
        dates = np.unique(timesteps.date)
        times = np.unique(timesteps.time)
    else: # mean_data from get_closest_days_from_clusters is in the format cluster-timestep
        dates = np.unique([i.split('-')[0] for i in timesteps])
        times = np.unique([i.split('-')[1] for i in timesteps])

    clusters = clustered.shape[0]

    # if 'variables' is given then we will loop over that, otherwise we loop over
    # all timeseries variables
    relevent_vars = variables if variables else data.data_vars

    reshaped_data = []

    # Initialise the location of the last column of data for a given variable
    previous_last_column = 0
    # loop over variables to get our arrays
    for var in relevent_vars:
        temp_data = data[var].copy()
        # if there is a loc_tech subset, index over that
        if loc_techs:
            loc_tech_dim = [i for i in data[var].dims if 'loc_techs' in i][0]
            relevent_loc_techs = list(
                set(temp_data[loc_tech_dim].values).intersection(loc_techs)
            )
            temp_data = temp_data.loc[{loc_tech_dim: relevent_loc_techs}]

        # Create an xarray DataArray with the correct dimensions
        stacked_var = _stack_data(temp_data, dates, times)[:clusters]

        # get last column index for this variable's data
        last_column = previous_last_column + len(stacked_var.stacked)

        # Update data to be the contents of 'clustered'
        stacked_var.loc[{}] = clustered[:, range(previous_last_column, last_column)]

        # rename and unstack
        reshaped_var = (
            stacked_var.assign_coords(dates=[i for i in range(clusters)])
                .rename({'dates': 'clusters'}).unstack('stacked')
        )

        # store information in dictionary, for later conversion to Dataset
        reshaped_data.append(reshaped_var.to_dataset(name=var))

        previous_last_column = last_column

    # output is a dataset, built from dictionaries
    reshaped_dataset = xr.merge(reshaped_data)

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

    ds = {}
    t_coords = [
        '{}-{}'.format(cid, t) for cid in cluster_map for t in range(timesteps_per_day)
    ]
    for var in data.data_vars:
        clustered_array = data[var][:, :len(t_coords)].copy()
        clustered_array['timesteps'] = t_coords
        for cluster_id, cluster_members in cluster_map.items():
            current_cluster = [
                i for i in t_coords if i.startswith('{}-'.format(cluster_id))
            ]
            clustered_array.loc[{'timesteps': current_cluster}] = (
                data[var].loc[{'timesteps': cluster_members}]
                .groupby('timesteps.hour').mean(dim='timesteps').values
            )
        ds[var] = clustered_array
    ds = xr.Dataset(ds)
    return ds


def find_nearest_vector_index(array, value, metric='rmse'):
    """
    compares the data for one cluster to every day in the timeseries, to find the
    day which most closely matches.

    Parameters
    ----------
    array : np.ndarray
        full timeseries array,
        shape = (num_dates, num_loc_techs * num_vars * num_timesteps_per_day)
        The shape is acheived by running a dataset through clustering.reshape_for_clustering
    value : np.ndarray
        one cluster of data,
        shape = (1, num_loc_techs * num_vars * num_timesteps_per_day)
        The shape is acheived by running the mean clustered dataset,
        subset over one cluster, through clustering.reshape_for_clustering
    metric : str, default = 'rmse'
        Error metric with which to compare the array and the values.
        If 'rmse', will compare the root mean square error between `value` and
        each date in `array`.
        If 'abs' will compare the absolute difference between `value` and
        each date in `array`.
    """
    if metric == 'rmse':
        error = np.array([mean_squared_error(i, value[0]) for i in array])
    elif metric == 'abs':
        error = np.array([np.linalg.norm(sum(y)) for y in array - value])
    else:
        raise ValueError(
            'Error metric can only be `rmse` or `abs`, {} given'.format(metric)
        )

    return error.argmin()


def get_closest_days_from_clusters(data, mean_data, clusters, timesteps_per_day):
    """
    Given a set of mean cluster timeseries profiles, find the day in the full
    timeseries that matches each cluster profile most closely.

    Parameters
    ----------
    data : xarray Dataset
        Input model data, with all non-timeseries data vars dropped
    mean_data : xarray Dataset
        Mean per cluster of input model data, resulting timeseries is of the form
        X-Y where X is the cluster number and Y is the timestep (both integers)
    clusters : pd.Series
        Cluster to which each date is aligned.
        Index = dates of the full timeseries, values = cluster number
    timesteps_per_day : int
        Number of timesteps in a day. We expect uniform timesteps between days,
        which is checked prior to reaching this function.

    Returns
        new_data : xarray Dataset
            data, indexed over only the timesteps within the days that most
            closely match the cluster means.
            Number of timesteps = timesteps_per_day * num_clusters. May be lower
            than this if many cluster means match with the same day
        chosen_day_timestepas : dict
            The day assigned to each cluster. key = cluster number, value = date.
    """
    dtindex = data['timesteps'].to_index()

    chosen_days = {}

    for cluster in sorted(clusters.unique()):

        subset_t = [t for t in mean_data.timesteps.values
                    if t.startswith('{}-'.format(cluster))]

        target = reshape_for_clustering(mean_data.loc[dict(timesteps=subset_t)])

        lookup_array = reshape_for_clustering(data)

        chosen_days[cluster] = find_nearest_vector_index(lookup_array, target)

    days_list = sorted(list(set(chosen_days.values())))
    new_t_coord = _hourly_from_daily_index(dtindex[::timesteps_per_day][days_list])

    chosen_day_timestamps = {k: dtindex[::timesteps_per_day][v]
                             for k, v in chosen_days.items()}

    new_data = data.loc[dict(timesteps=new_t_coord)]

    return new_data, chosen_day_timestamps


def _hourly_from_daily_index(idx):
    dtrange = lambda i: pd.date_range(
        i, i + pd.Timedelta('1D'), freq='1H'
    )[:-1]
    new_idx = pd.concat([pd.Series(1, dtrange(i))
                         for i in idx]).index
    return new_idx


def map_clusters_to_data(data, clusters, how, daily_timesteps):
    """
    Returns a copy of data that has been clustered.

    Parameters
    ----------
    how : str
        How to select data from clusters. Can be mean (centroid) or closest real
        day to the mean (by root mean square error).

    """
    # FIXME hardcoded time intervals ('1H', '1D')

    # Get all timesteps, not just the first per day
    timesteps_per_day = len(daily_timesteps)
    idx = clusters.index
    new_idx = _hourly_from_daily_index(idx)
    clusters_timeseries = (clusters.reindex(new_idx)
                           .fillna(method='ffill').astype(int))

    new_data = get_mean_from_clusters(data, clusters_timeseries, timesteps_per_day)
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
        new_data.coords['timesteps'] = new_t_coord.as_matrix()

        # Generate weights
        # weight of each timestep = number of timesteps in this timestep's cluster
        # divided by timesteps per day (since we're grouping days together and
        # a cluster consisting of 1 day = 24 hours should have weight of 1)
        value_counts = clusters_timeseries.value_counts() / timesteps_per_day
        # And turn the index into dates (days)
        value_counts = pd.DataFrame({
            'dates': timestamps,
            'counts': value_counts}).set_index('dates')['counts']

    elif how == 'closest':
        new_data, chosen_ts = get_closest_days_from_clusters(data, new_data, clusters, timesteps_per_day)
        # Deal with the case where more than one cluster has the same closest day
        # An easy way is to rename the original clusters with the chosen days
        # So at this point, clusterdays_timeseries maps all timesteps to the day
        # of year of the cluster the timestep belongs to
        clusterdays_timeseries = clusters_timeseries.map(lambda x: chosen_ts[x])
        value_counts = clusterdays_timeseries.value_counts() / timesteps_per_day
        timestamps = pd.DataFrame.from_dict(chosen_ts, orient='index')[0]

    _clusters = xr.DataArray(
        data=np.full(len(new_data.timesteps.values), np.nan),
        dims='timesteps',
        coords={'timesteps': new_data.timesteps.values}
    )
    for cluster in timestamps.index:
        _clusters.loc[_clusters.timesteps.to_index().date == timestamps[cluster].date()] = cluster
    new_data['clusters'] = _clusters.astype(int)
    weights = (value_counts.reindex(_hourly_from_daily_index(value_counts.index))
                           .fillna(method='ffill'))
    new_data['timestep_weights'] = xr.DataArray(weights, dims=['timesteps'])
    days = np.unique(new_data.timesteps.to_index().date)
    new_data['timestep_resolution'] = (
        xr.DataArray(np.repeat(daily_timesteps, len(days)),
                     dims=['timesteps'],
                     coords={'timesteps': new_data['timesteps']})
    )
    return new_data


def get_clusters_kmeans(
        data, timesteps_per_day,
        tech=None, timesteps=None, k=None, variables=None
    ):
    """
    Parameters
    ----------
    data : xarray.Dataset
        Should be normalized
    timesteps_per_day
    tech : list, optional
        list of strings referring to technologies by which clustering is undertaken.
        If none (default), all technologies within timeseries variables will be used.
    timesteps : list or str, optional
        Subset of the time domain within which to apply clustering.
    k : int, optional
        Number of clusters to create. If none (default), will use Hartigan's rule
        to infer a reasonable number of clusters.
    variables : list, optional
        data variables (e.g. `resource`, `energy_eff`) by whose values the data
        will be clustered. If none (default), all timeseries variables will be used.

    Returns
    -------
    clusters : dataframe
        Indexed by timesteps and with locations as columns, giving cluster
        membership for first timestep of each day.
    centroids

    """

    if timesteps is not None:
        data = data.loc[{'timesteps': timesteps}]
    else:
        timesteps = data.timesteps.values

    X = reshape_for_clustering(data, tech, variables)

    if not k:
        k = hartigan_n_clusters(X)
        exceptions.warn(
            'Used Hartigan\'s rule to determine '
            '{} is a good number of clusters.'.format(k)
        )

    clustered_data = sk_cluster.KMeans(k).fit(X)

    # Determine the cluster membership of each day
    day_clusters = clustered_data.labels_

    # Create mapping of timesteps to clusters
    clusters = pd.Series(day_clusters, index=timesteps[::timesteps_per_day])

    # Reshape centroids
    centroids = reshape_clustered(clustered_data.cluster_centers_, data, tech, variables)

    # Get inertia, for e.g. checking clustering with Hartigan's rule
    inertia = clustered_data.inertia_

    return clusters, centroids, inertia


def hartigan_n_clusters(X, threshold=10):
    """
    Try clustering using an sklearn.cluster method, for several cluster sizes.
    Using Hartigan's rule, we will return the number of clusters after which
    the benefit of clustering is low.
    """
    def _H_rule(inertia, inertia_plus_one, n_clusters, len_input):
        # see http://www.dcs.bbk.ac.uk/~mirkin/papers/00357_07-216RR_mirkin.pdf
        return ((inertia / inertia_plus_one) - 1) * (len_input - n_clusters - 1)

    len_input = len(X)
    n_clusters = 1
    HK = threshold + 1

    while n_clusters <= len_input and HK > threshold:

        kmeans = sk_cluster.KMeans(n_clusters=n_clusters).fit(X)
        kmeans_plus_one = sk_cluster.KMeans(n_clusters=n_clusters + 1).fit(X)

        inertia = kmeans.inertia_
        inertia_plus_one = kmeans_plus_one.inertia_

        HK = _H_rule(inertia, inertia_plus_one, n_clusters, len_input)

        n_clusters += 1

    if HK > threshold:  # i.e. we went to the limit where n_clusters = len_input
        exceptions.warn("Based on threshold, number of clusters = number of dates")
        return len_input
    else:
        return n_clusters - 1


# TODO get hierarchical clusters using scikitlearn too
# TODO change scipy for scikitlearn in Calliope requirements
def get_clusters_hierarchical(
        data, timesteps_per_day,
        tech=None, max_d=None, k=None, variables=None
        ):
    """
    Parameters
    ----------
    data : xarray.Dataset
        Should be normalized
    timesteps_per_day
    tech : list, optional
        list of strings referring to technologies by which clustering is undertaken.
        If none (default), all technologies within timeseries variables will be used.
    max_d : float or int, optional
        Max distance for returning clusters.
    k : int, optional
        Number of clusters to create. If none (default), will use Hartigan's rule
        to infer a reasonable number of clusters.
    variables : list, optional
        data variables (e.g. `resource`, `energy_eff`) by whose values the data
        will be clustered. If none (default), all timeseries variables will be used.

    Returns
    -------
    clusters
    X
    Z

    """
    X = reshape_for_clustering(data, tech, variables)

    # Generate the linkage matrix
    Z = hierarchy.linkage(X, 'ward')

    if max_d:
        # Get clusters based on maximum distance
        clusters = hierarchy.fcluster(Z, max_d, criterion='distance')
    elif k:
        # Get clusters based on number of desired clusters
        clusters = hierarchy.fcluster(Z, k, criterion='maxclust')
    else:
        k = hartigan_n_clusters(X)
        exceptions.warn(
            'Used Hartigan\'s rule to determine '
            '{} is a good number of clusters.'.format(k)
        )
        clusters = hierarchy.fcluster(Z, k, criterion='maxclust')

    # Make sure clusters are a pd.Series with a datetime index
    if clusters is not None:
        timesteps = data.coords['timesteps'].values  # All timesteps

        clusters = pd.Series(clusters, index=timesteps[::timesteps_per_day])

    return (clusters, X, Z)


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
