# Copyright (C) since 2013 Calliope contributors listed in AUTHORS.
# Licensed under the Apache 2.0 License (see LICENSE file).

"""
time.py
~~~~~~~

Functionality to add and process time varying parameters

"""
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr

from calliope import exceptions

LOGGER = logging.getLogger(__name__)


def add_inferred_time_params(model_data: xr.Dataset):
    # Add timestep_resolution by looking at the time difference between timestep n
    # and timestep n + 1 for all timesteps
    # Last timestep has no n + 1, so will be NaT (not a time), we ffill this.
    # Time resolution is saved in hours (i.e. nanoseconds / 3600e6)
    timestep_resolution = (
        model_data.timesteps.diff("timesteps", label="lower")
        .reindex({"timesteps": model_data.timesteps})
        .rename("timestep_resolution")
    )

    if len(model_data.timesteps) == 1:
        exceptions.warn(
            "Only one timestep defined. Inferring timestep resolution to be 1 hour"
        )
        timestep_resolution = timestep_resolution.fillna(pd.Timedelta("1 hour"))
    else:
        timestep_resolution = timestep_resolution.ffill("timesteps")

    model_data["timestep_resolution"] = timestep_resolution / pd.Timedelta("1 hour")

    model_data["timestep_weights"] = xr.DataArray(
        np.ones(len(model_data.timesteps)), dims=["timesteps"]
    )

    return model_data


def clean_data_source_timeseries(
    ds: xr.Dataset, init_config: dict, source_file: str
) -> xr.Dataset:
    time_subset: Optional[list[str]] = init_config.get("time_subset", None)
    time_format: str = init_config["time_format"]

    datetime_indices = [i for i in ds.dims if str(i).endswith("steps")]
    for index_name in datetime_indices:
        ds.coords[index_name] = _datetime_index(
            ds.coords[index_name].to_index(), time_format, source_file
        )
        if time_subset is not None:
            _check_time_subset(ds.coords[index_name].to_index(), time_subset)
            ds = ds.sel(**{index_name: slice(*time_subset)})

    return ds


def resample(data: xr.Dataset, resolution: str) -> xr.Dataset:
    """
    Function to resample timeseries data from the input resolution (e.g. 1H), to
    the given resolution (e.g. 2H)

    Args:
        data (xarray.Dataset): Calliope model data, containing only timeseries data variables.
        resolution (str):
            time resolution of the output data, given in Pandas time frequency format.
            E.g. 1H = 1 hour, 1W = 1 week, 1M = 1 month, 1T = 1 minute.
            Multiples allowed.

    Returns:
        xarray.Dataset:
            `data` resampled according to `resolution`.

    """
    resample_kwargs = {"indexer": {"timesteps": resolution}, "skipna": True}
    data_non_ts = data.drop_dims("timesteps")
    data_ts = data.drop_vars(data_non_ts.data_vars)
    data_ts_resampled = data_ts.resample(**resample_kwargs).first(keep_attrs=True)

    for var_name, var_data in data_ts.data_vars.items():
        resampler = var_data.resample(**resample_kwargs)
        if var_name in [
            "timestep_resolution",
            "source_use_min",
            "sink_use_min",
            "source_use_max",
            "sink_use_max",
            "source_use_equals",
            "sink_use_equals",
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


def cluster(data: xr.Dataset, clustering_file: str | Path, time_format: str):
    """
    Apply the given clustering time series to the given data.

    Args:
        data (xarray.Dataset): Calliope model data, containing only timeseries data variables.
        clustering_file (str | Path): Path to file containing rows of dates and the corresponding datestamp to which they are to be clustered.
        time_format (str): The format that dates in `clustering_file` have been defined (e.g., "%Y-%m-%d").

    Returns:
        xarray.Dataset:
            `data` clustered according to contents of `clustering_file`.

    """
    clustering_timeseries = pd.read_csv(clustering_file, index_col=0).squeeze()
    clustering_timeseries.index = _datetime_index(
        clustering_timeseries.index, time_format, Path(clustering_file).name
    )
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


def _datetime_index(index: pd.Index, format: str, source_name: str) -> pd.Index:
    try:
        return pd.to_datetime(index, format=format)
    except ValueError as e:
        raise exceptions.ModelError(
            f"Error in parsing dates in timeseries data from {source_name} using datetime format `{format}`. "
            f"Full error: {e}"
        )


def _check_time_subset(ts_index: pd.Index, time_subset: list[str]):
    try:
        time_subset_dt = pd.to_datetime(time_subset, format="ISO8601")
    except ValueError as e:
        raise exceptions.ModelError(
            "Timeseries subset must be in ISO format (anything up to the  "
            "detail of `%Y-%m-%d %H:%M:%S`.\n User time subset: {}\n "
            "Error caused: {}".format(time_subset, e)
        )

    df_start_time = ts_index[0]
    df_end_time = ts_index[-1]
    if (
        time_subset_dt[0].date() < df_start_time.date()
        or time_subset_dt[1].date() > df_end_time.date()
    ):
        raise exceptions.ModelError(
            f"subset time range {time_subset} is outside the input data time range "
            f"[{df_start_time}, {df_end_time}]"
        )


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
