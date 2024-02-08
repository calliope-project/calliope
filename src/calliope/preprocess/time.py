# Copyright (C) since 2013 Calliope contributors listed in AUTHORS.
# Licensed under the Apache 2.0 License (see LICENSE file).

"""
time.py
~~~~~~~

Functionality to add and process time varying parameters

"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from calliope import exceptions
from calliope.util.schema import MODEL_SCHEMA, extract_from_schema

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


def subset_timeseries(ds: xr.Dataset, time_subset: list[str]) -> xr.Dataset:
    """Subset all timeseries dimensions according to an input slice of start and end times.

    Args:
        ds (xr.Dataset): Dataset containing timeseries data to subset.
        time_subset (list[str]): List of length 2, containing start and end times to subset on.

    Returns:
        xr.Dataset: Input `ds` with subset timeseries coordinates.
    """

    datetime_dims = [k for k, v in ds.coords.items() if v.dtype.kind == "M"]
    for dim_name in datetime_dims:
        _check_time_subset(ds.coords[dim_name].to_index(), time_subset)
        ds = ds.sel(**{dim_name: slice(*time_subset)})
        _check_missing_data(ds, dim_name)
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
    resampling_methods = extract_from_schema(MODEL_SCHEMA, "x-resample_method")

    for var_name, var_data in data_ts.data_vars.items():
        resampler = var_data.resample(**resample_kwargs)
        if var_name == "timestep_resolution":
            method = "sum"
        elif var_name in resampling_methods:
            method = resampling_methods.get(var_name, None)
        elif var_data.dtype.kind in ["f", "i"]:
            method = "mean"
        else:
            method = "first"

        if method == "sum":
            method_kwargs = {"min_count": 1}
        else:
            method_kwargs = {}

        data_ts_resampled[var_name] = getattr(resampler, method)(
            keep_attrs=True, **method_kwargs
        )
        LOGGER.debug(
            f"Time Resampling | {var_name} | resampling function used: {method}"
        )

    data_new = xr.merge([data_non_ts, data_ts_resampled])

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
        clustering_timeseries.index + " 00:00:00", time_format
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


def _datetime_index(index: pd.Index, format: str) -> pd.Index:
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


def _check_time_subset(ts_index: pd.Index, time_subset: list[str]):
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
    try:
        time_subset_dt = pd.to_datetime(time_subset, format="ISO8601")
    except ValueError as e:
        raise exceptions.ModelError(
            "Timeseries subset must be in ISO format (anything up to the  "
            "detail of `%Y-%m-%d %H:%M:%S`).\n User time subset: {}\n "
            "Error caused: {}".format(time_subset, e)
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
