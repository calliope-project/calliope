# Copyright (C) since 2013 Calliope contributors listed in AUTHORS.
# Licensed under the Apache 2.0 License (see LICENSE file).
"""Functionality to add and process time varying parameters."""

import logging
from typing import overload

import numpy as np
import pandas as pd
import xarray as xr

from calliope import exceptions
from calliope.schemas.math_schema import CalliopeBuildMath

LOGGER = logging.getLogger(__name__)


def add_inferred_time_params(model_data: xr.Dataset):
    """Add timestep_resolution.

    Done by ooking at the time difference between timestep n and timestep n + 1 for all timesteps.
    Last timestep has no n + 1, so will be NaT (not a time), we ffill this.
    Time resolution is saved in hours (i.e. nanoseconds / 3600e6)
    """
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


def timeseries_to_datetime(ds: xr.Dataset, time_format: str, id: str) -> xr.Dataset:
    """Find all dimensions ending in `steps` and try to cast to datetime type.

    Args:
        ds (xr.Dataset): Dataset possibly containing dimensions with `steps` suffix.
        time_format (str): Time format to use on casting timeseries strings to datetime.
        id (str): Identifier for `ds` to use in logging.

    Returns:
        xr.Dataset: Input `ds` with datetime timeseries coordinates.
    """
    datetime_dims = [i for i in ds.dims if str(i).endswith("steps")]
    for dim_name in datetime_dims:
        LOGGER.debug(
            f"{id} | Updating `{dim_name}` dimension index values to datetime format."
        )
        ds.coords[dim_name] = _datetime_index(
            ds.coords[dim_name].to_index(), time_format
        )
    return ds


def resample(
    data: xr.Dataset, math: CalliopeBuildMath, dim: str, resolution: str
) -> xr.Dataset:
    """Function to resample timeseries data.

    Transforms the input resolution (e.g. 1h), to the given resolution (e.g. 2h).

    Args:
        data (xarray.Dataset): Calliope model data, containing only timeseries data variables.
        math (CalliopeBuildMath): Calliope math object to access input resampling methods.
        dim (str):
            Name of the dimension to resample. This should be a time dimension, e.g.
            "timesteps" or "datesteps". It is assumed that the dimension contains datetime values
            and that the values are in ascending order.
        resolution (str):
            time resolution of the output data, given in Pandas time frequency format.
            E.g. 1h = 1 hour, 1W = 1 week, 1M = 1 month, 1T = 1 minute.
            Multiples allowed.

    Returns:
        xarray.Dataset:
            `data` resampled according to `resolution`.

    """
    resample_kwargs = {"indexer": {dim: resolution}, "skipna": True}
    data_non_ts = data.drop_dims(dim)
    data_ts = data.drop_vars(data_non_ts.data_vars)
    try:
        data_ts_resampled = data_ts.resample(**resample_kwargs).first(keep_attrs=True)
    except ValueError:
        raise exceptions.ModelError(
            f"Unknown `{dim}` resampling frequency: {resolution}"
        )

    for var_name, var_data in data_ts.data_vars.items():
        resampler = var_data.resample(**resample_kwargs)
        var_math = math.find(var_name, subset=["parameters", "lookups"])[1]
        resample_method = var_math["resample_method"]

        if resample_method == "sum":
            method_kwargs = {"min_count": 1}
        else:
            method_kwargs = {}

        data_ts_resampled[var_name] = getattr(resampler, resample_method)(
            keep_attrs=True, **method_kwargs
        )
        LOGGER.debug(
            f"Time Resampling | {var_name} | resampling function used: {resample_method}"
        )

    data_new = xr.merge([data_non_ts, data_ts_resampled])

    return data_new


def cluster(data: xr.Dataset, clustering_param: str, time_format: str):
    """Apply the given clustering time series to the given data.

    Args:
        data (xarray.Dataset): Calliope model data, containing only timeseries data variables.
        clustering_param (str): Name of array in data to use for clustering.
        time_format (str): The format that dates in `clustering_param` have been defined (e.g., "%Y-%m-%d" or "ISO8601").

    Returns:
        xarray.Dataset:
            `data` clustered according to contents of `clustering_file`.

    """
    clustering_timeseries = data[clustering_param].to_series()
    representative_days = _datetime_index(
        clustering_timeseries.dropna() + " 00:00:00", time_format
    ).dt.date
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
    _lookup_clusters(data_new, grouper)

    return data_new


@overload
def _datetime_index(index: pd.Index | np.ndarray, format: str) -> pd.DatetimeIndex: ...


@overload
def _datetime_index(index: pd.Series, format: str) -> pd.Series: ...


def _datetime_index(
    index: pd.Index | np.ndarray | pd.Series, format: str
) -> pd.Index | pd.Series:
    try:
        if format == "ISO8601":
            dt = pd.to_datetime(index, format=format)
        else:
            dt = pd.to_datetime(index, format=format, exact=False)
    except ValueError as e:
        raise exceptions.ModelError(
            f"Error in parsing dates in timeseries data from using datetime format `{format}`. "
            f"Full error: {e}"
        )
    else:
        return dt


def check_time_subset(ts_index: pd.Index, time_subset: list[str]):
    """Check if the user-configured time subset is in the correct format and it matches the range of the input data.

    We do not allow the time subset to have _no_ overlap with a timeseries index.
    This overlap is checked based on dates, not any higher resolution.

    Args:
        ts_index (pd.Index): Timeseries index to check for overlap.
        time_subset (list[str]): Time subset as a slicer of start and end timestamps.

    Raises:
        exceptions.ModelError: Time subset string format must conform to ISO8601.
        exceptions.ModelError: Cannot have time subsets that do not overlap the timeseries index.
    """
    if len(time_subset) != 2:
        raise exceptions.ModelError(
            f"Timeseries subset must be a list of two timestamps. Received: {time_subset}"
        )
    try:
        time_subset_dt = pd.to_datetime(time_subset, format="ISO8601")
    except ValueError as e:
        raise exceptions.ModelError(
            "Timeseries subset must be in ISO format (anything up to the  "
            f"detail of `%Y-%m-%d %H:%M:%S`).\n User time subset: {time_subset}\n "
            f"Error caused: {e}"
        )

    df_start_time = ts_index[0]
    df_end_time = ts_index[-1]
    if (
        time_subset_dt[1].date() < df_start_time.date()
        or time_subset_dt[0].date() > df_end_time.date()
    ):
        raise exceptions.ModelError(
            f"subset time range {time_subset} is outside the input data time range "
            f"[{df_start_time}, {df_end_time}]"
        )


def _check_missing_data(ds: xr.Dataset, dim_name: str):
    """Check if there are any parameters with timeseries data that doesn't cover the whole time period.

    We assume this is _not_ intended (e.g. loading in one dataset with a shorter time length than expected).

    Args:
        ds (xr.Dataset): Dataset with timeseries dimension `dim_name` present.
        dim_name (str): Name of the timeseries dimension.
    """
    datetime_ds = ds[[k for k, v in ds.data_vars.items() if dim_name in v.dims]]
    is_missing = (
        datetime_ds.notnull().any(dim_name) & ~datetime_ds.notnull().all(dim_name)
    ).to_dataframe()
    missing_data = is_missing[is_missing].stack()
    if not missing_data.empty:
        exceptions.warn(
            f"Possibly missing data on the {dim_name} dimension for:\n{missing_data}"
        )


def _lookup_clusters(dataset: xr.Dataset, grouper: pd.Series) -> xr.Dataset:
    """Clustering lookup functionality.

    For any given timestep in a time clustered model, get:
    1. the first and last timestep of the cluster,
    2. the last timestep of the cluster corresponding to a date in the original timeseries
    """
    dataset["cluster_first_timestep"] = dataset.timesteps.isin(
        dataset.timesteps.groupby("timesteps.date").first()
    )
    dataset["lookup_cluster_last_timestep"] = (
        dataset.timesteps.groupby("timesteps.date")
        .last()
        .rename({"date": "timesteps"})
        .reindex_like(dataset.timesteps)
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
