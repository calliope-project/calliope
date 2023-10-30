# Copyright (C) since 2013 Calliope contributors listed in AUTHORS.
# Licensed under the Apache 2.0 License (see LICENSE file).

"""
time.py
~~~~~~~

Functionality to add and process time varying parameters

"""
import logging

import numpy as np
import pandas as pd
import xarray as xr

from calliope import exceptions

LOGGER = logging.getLogger(__name__)


def add_time_dimension(
    data: xr.Dataset, timeseries_vars: set[str], timeseries_data: pd.DataFrame
):
    """
    Once all constraints and costs have been loaded into the model dataset, any
    timeseries data is loaded from file and substituted into the model dataset

    Parameters:
    -----------
    data : xarray Dataset
        A data structure which has already gone through `constraints_to_dataset`,
        `costs_to_dataset`, and `add_attributes`
    model_run : AttrDict
        Calliope model_run dictionary

    Returns:
    --------
    data : xarray Dataset
        A data structure with an additional time dimension to the input dataset,
        with all relevant `file=` and `df= `entries replaced with the correct data.

    """
    key_errors = []
    # Search through every constraint/cost for use of '='
    for variable in timeseries_vars:
        # 2) convert to a Pandas Series to do 'string contains' search
        data_series = data[variable].to_series().dropna()

        # 3) get Series of all uses of 'file=' or 'df=' for this variable (timeseries keys)
        try:
            tskeys = data_series[
                data_series.str.contains("file=") | data_series.str.contains("df=")
            ]
        except AttributeError:
            continue

        # 4) If no use of 'file=' or 'df=' then we can be on our way
        if tskeys.empty:
            continue

        # 5) remove all before '=' and split filename and node column
        tskeys = (
            tskeys.str.split("=")
            .str[1]
            .str.rsplit(":", n=1, expand=True)
            .reset_index()
            .rename(columns={0: "source", 1: "column"})
            .set_index(["source", "column"])
        )

        # 6) Get all timeseries data from dataframes stored in model_run
        try:
            var_timeseries_data = timeseries_data.loc[:, tskeys.index]
        except KeyError:
            key_errors.append(
                f"file:column combinations `{tskeys.index.values}` not found, but are"
                f" requested by parameter `{variable}`."
            )
            continue

        var_timeseries_data.columns = pd.MultiIndex.from_frame(tskeys)

        # 7) Add time dimension to the relevent DataArray and update the '='
        # dimensions with the time varying data (static data is just duplicated
        # at each timestep)

        data[variable] = (
            xr.DataArray.from_series(var_timeseries_data.unstack())
            .reindex(data[variable].coords)
            .fillna(data[variable])
        )
    if key_errors:
        exceptions.print_warnings_and_raise_errors(errors=key_errors)

    # Add timestep_resolution by looking at the time difference between timestep n
    # and timestep n + 1 for all timesteps
    # Last timestep has no n + 1, so will be NaT (not a time), we ffill this.
    # Time resolution is saved in hours (i.e. nanoseconds / 3600e6)
    timestep_resolution = (
        data.timesteps.diff("timesteps", label="lower")
        .reindex({"timesteps": data.timesteps})
        .rename("timestep_resolution")
    )

    if len(data.timesteps) == 1:
        exceptions.warn(
            "Only one timestep defined. Inferring timestep resolution to be 1 hour"
        )
        timestep_resolution = timestep_resolution.fillna(pd.Timedelta("1 hour"))
    else:
        timestep_resolution = timestep_resolution.ffill("timesteps")

    data["timestep_resolution"] = timestep_resolution / pd.Timedelta("1 hour")

    data["timestep_weights"] = xr.DataArray(
        np.ones(len(data.timesteps)), dims=["timesteps"]
    )

    return data


def update_dtypes(model_data):
    """
    Update dtypes to not be 'Object', if possible.
    Order of preference is: bool, int, float
    """
    # TODO: this should be redundant once typedconfig is in (params will have predefined dtypes)
    for var_name, var in model_data.data_vars.items():
        if var.dtype.kind == "O":
            no_nans = var.where(var != "nan", drop=True)
            model_data[var_name] = var.where(var != "nan")
            if no_nans.isin(["True", 0, 1, "False", "0", "1"]).all():
                # Turn to bool
                model_data[var_name] = var.isin(["True", 1, "1"])
            else:
                try:
                    model_data[var_name] = var.astype(np.int_, copy=False)
                except (ValueError, OverflowError):
                    try:
                        model_data[var_name] = var.astype(np.float_, copy=False)
                    except ValueError:
                        None
    return model_data


def resample(data: xr.Dataset, resolution: str):
    """
    Function to resample timeseries data from the input resolution (e.g. 1H), to
    the given resolution (e.g. 2H)

    Parameters
    ----------
    data : xarray.Dataset
        calliope model data, containing only timeseries data variables
    timesteps : str or list; optional
        If given, apply resampling to a subset of the timeseries data
    resolution : str
        time resolution of the output data, given in Pandas time frequency format.
        E.g. 1H = 1 hour, 1W = 1 week, 1M = 1 month, 1T = 1 minute. Multiples allowed.

    """
    resample_kwargs = {"indexer": {"timesteps": resolution}, "skipna": True}
    data_non_ts = data.drop_dims("timesteps")
    data_ts = data.drop(data_non_ts.data_vars)
    data_ts_resampled = data_ts.resample(**resample_kwargs).first(keep_attrs=True)

    for var_name, var_data in data_ts.data_vars.items():
        resampler = var_data.resample(**resample_kwargs)
        if var_name in [
            "timestep_resolution",
            "source_min",
            "sink_min",
            "source_max",
            "sink_max",
            "source_equals",
            "sink_equals",
        ]:
            method = "sum"
        elif var_data.dtype.kind in ["f", "i"]:
            method = "mean"
        else:
            method = "first"
        data_ts_resampled[var_name] = getattr(resampler, method)(keep_attrs=True)
        LOGGER.debug(
            f"Time Resampling | {var_name} | resampling function used: {method}"
        )

    data_new = xr.merge([data_non_ts, data_ts_resampled])

    # Resampling still permits operational mode
    data_new.attrs["allow_operate_mode"] = 1

    return data_new


def cluster(
    data: xr.Dataset,
    clustering_timeseries: pd.Series,
):
    """
    Apply the given clustering time series to the given data.

    Parameters
    ----------
    data : xarray.Dataset
    clustering_timeseries

    Returns
    -------
    data_new_scaled : xarray.Dataset

    """
    representative_days = pd.to_datetime(clustering_timeseries.dropna()).dt.date
    grouper = representative_days.to_frame("clusters").groupby("clusters")
    data_new = data.sel(
        timesteps=data.timesteps.dt.date.isin(representative_days.values)
    )
    date_series = data_new.timesteps.dt.date.to_series()
    data_new["timestep_cluster"] = xr.DataArray(
        grouper.ngroup().reindex(date_series).values, dims="timesteps"
    )
    data_new["timestep_weights"] = xr.DataArray(
        grouper.value_counts().reindex(date_series).values, dims="timesteps"
    )
    data_new.coords["datesteps"] = representative_days.index.rename("datesteps")
    data_new.coords["clusters"] = np.unique(data_new["timestep_cluster"].values)
    # Clustering no longer permits operational mode
    data_new.attrs["allow_operate_mode"] = 0
    _lookup_clusters(data_new, grouper)

    return data_new


def _lookup_clusters(dataset: xr.Dataset, grouper: pd.Series) -> xr.Dataset:
    """
    For any given timestep in a time clustered model, get:
    1. the first and last timestep of the cluster,
    2. the last timestep of the cluster corresponding to a date in the original timeseries
    """

    dataset["lookup_cluster_first_timestep"] = dataset.timesteps.isin(
        dataset.timesteps.groupby("timesteps.date").first()
    )
    dataset["lookup_cluster_last_timestep"] = dataset.timesteps.isin(
        dataset.timesteps.groupby("timesteps.date").last()
    )

    dataset["lookup_datestep_cluster"] = xr.DataArray(
        grouper.ngroup(), dims="datesteps"
    )

    last_timesteps = (
        dataset.timestep_cluster.to_series()
        .reset_index()
        .groupby("timestep_cluster")
        .last()
    )
    dataset["lookup_datestep_last_cluster_timestep"] = (
        dataset.lookup_datestep_cluster.to_series()
        .map(last_timesteps.timesteps)
        .to_xarray()
    )

    return dataset
