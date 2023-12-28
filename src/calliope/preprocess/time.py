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


def add_time_dimension(
    model_data: xr.Dataset,
    init_config: dict,
    timeseries_dfs: Optional[dict[str, pd.DataFrame]],
):
    """
    Once all constraints and costs have been loaded into the model dataset, any
    timeseries data is loaded from file and substituted into the model dataset

    Parameters:
    -----------
    model_data : xarray Dataset
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
    timeseries_loader = TimeseriesLoader(init_config, timeseries_dfs)
    # Search through every constraint/cost for use of '='
    for var_name, var_data in model_data.data_vars.items():
        # 1) get Series of all uses of 'file=' or 'df=' for this variable (timeseries keys)
        if var_data.astype(str).str.contains("^file=|df=").any():
            var_series = var_data.to_series()
            tskeys = var_series[var_series.str.contains("^file=|df=").notnull()]
        else:
            continue

        # 2) If no use of 'file=' or 'df=' then we can be on our way
        if tskeys.empty:
            continue

        # 3) split data source ("df/file"), filename, and node column
        tskeys = (
            tskeys.dropna()
            .str.split("=|:", expand=True)
            .rename(columns={0: "source_type", 1: "source", 2: "column"})
        )
        if "column" not in tskeys.columns:
            tskeys = tskeys.assign(column=np.nan)
        if "nodes" in var_data.dims:
            node_info = (
                tskeys.index.get_level_values("nodes").unique()
                if isinstance(tskeys.index, pd.MultiIndex)
                else tskeys.index
            )
            tskeys["column"] = tskeys["column"].fillna(
                node_info.to_series().align(tskeys)[0]
            )

        # 4) Get all timeseries data from dataframes stored in model_run
        ts_df = tskeys.apply(
            timeseries_loader.load_timeseries, var_name=var_name, axis=1
        ).rename_axis(columns="timesteps")

        # 5) Add time dimension to the relevent DataArray and update the '='
        # dimensions with the time varying data (static data is just duplicated
        # at each timestep)

        model_data[var_name] = (
            ts_df.stack()
            .to_xarray()
            .reindex(var_data.coords)
            .fillna(var_data)
            .assign_attrs(var_data.attrs)
        )
    if any(
        var_data.astype(str).str.contains("^file=|df=").any()
        for var_data in model_data.data_vars.values()
    ):
        raise exceptions.ModelError(
            "Some lengths of input timeseries data are too short relative to your chosen time subset."
        )
    return model_data


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


def _datetime_index(index: pd.Index, format: str, source_name: str) -> pd.Index:
    try:
        return pd.to_datetime(index, format=format)
    except ValueError as e:
        raise exceptions.ModelError(
            f"Error in parsing dates in timeseries data from {source_name} using datetime format `{format}`. "
            f"Full error: {e}"
        )


class TimeseriesLoader:
    def __init__(
        self, init_config: dict, timeseries_dfs: Optional[dict[str, pd.DataFrame]]
    ):
        self.ts_cache: dict = {}
        self._timeseries_dfs = timeseries_dfs
        self._time_data_path: Path = init_config["time_data_path"]
        self._time_subset: Optional[list[str]] = init_config["time_subset"]
        self._time_format: str = init_config["time_format"]

        if self._timeseries_dfs is not None:
            if not isinstance(self._timeseries_dfs, dict) or not all(
                isinstance(v, (pd.Series, pd.DataFrame))
                for v in self._timeseries_dfs.values()
            ):
                raise exceptions.ModelError(
                    "`timeseries_dataframes` must be dict of pandas DataFrames."
                )

    def load_timeseries(self, df: pd.DataFrame, var_name: str) -> pd.Series:
        source = df.loc["source"]
        source_type = df.loc["source_type"]
        column = df.loc["column"]

        if source_type == "df" and self._timeseries_dfs is not None:
            try:
                df = self._timeseries_dfs[source]
            except KeyError:
                raise KeyError(
                    f"Attempted to load dataframe with undefined key: {source}"
                )
        elif source_type == "df" and self._timeseries_dfs is None:
            raise exceptions.ModelError(
                "Missing timeseries dataframes passed as an argument in calliope.Model(...)."
            )
        elif source_type == "file" and source in self.ts_cache:
            df = self.ts_cache[source]
        else:
            df = self._load_timeseries_from_file(source)
            self.ts_cache[source] = df

        clean_df = self._reformat_timeseries_df(df, source)

        if pd.isnull(column) and len(clean_df.columns) > 1:
            raise exceptions.ModelError(
                f"Timeseries data contains multiple columns but no column specified in reference from input parameter `{var_name}`"
            )
        elif pd.isnull(column) and len(clean_df.columns) == 1:
            series = clean_df.squeeze()
        else:
            series = clean_df[column]

        return series

    def _load_timeseries_from_file(self, ts_file_path: str):
        file_path = self._time_data_path / ts_file_path
        df = pd.read_csv(file_path, index_col=0)
        return df

    def _reformat_timeseries_df(self, df: pd.DataFrame, source: str) -> pd.DataFrame:
        try:
            df.apply(pd.to_numeric)
        except ValueError as e:
            raise exceptions.ModelError(
                f"Error in loading data from {source}. Ensure all entries are numeric. Full error: {e}"
            )
        df.index = _datetime_index(df.index, self._time_format, source)
        subset_df = self._subset_index(df)
        return subset_df

    def _subset_index(self, df: pd.DataFrame) -> pd.DataFrame:
        if self._time_subset is None:
            return df

        try:
            time_subset_dt = pd.to_datetime(self._time_subset, format="ISO8601")
        except ValueError as e:
            raise exceptions.ModelError(
                "Timeseries subset must be in ISO format (anything up to the  "
                "detail of `%Y-%m-%d %H:%M:%S`.\n User time subset: {}\n "
                "Error caused: {}".format(self._time_subset, e)
            )

        df_start_time = df.index[0]
        df_end_time = df.index[-1]
        if (
            time_subset_dt[0].date() < df_start_time.date()
            or time_subset_dt[1].date() > df_end_time.date()
        ):
            raise exceptions.ModelError(
                f"subset time range {self._time_subset} is outside the input data time range "
                f"[{df_start_time}, {df_end_time}]"
            )

        # We eventually subset using the input strings to capture entire days
        # E.g., ["2005-01-01", "2005-01-02"] will go to the last timestep on "2005-01-02"
        subset_df = df.loc[slice(*self._time_subset), :]
        if subset_df.empty:
            raise exceptions.ModelError(
                f"The time slice {time_subset_dt} creates an empty timeseries array."
            )
        return subset_df


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
    data_ts = data.drop_vars(data_non_ts.data_vars)
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


def cluster(data: xr.Dataset, clustering_file: str | Path, time_format: str):
    """
    Apply the given clustering time series to the given data.

    Parameters
    ----------
    data : xarray.Dataset
    clustering_file

    Returns
    -------
    data_new_scaled : xarray.Dataset

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
