"""
Copyright (C) since 2013 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

time.py
~~~~~~~

Functionality to add and process time varying parameters

"""
import xarray as xr
import numpy as np
import pandas as pd

from calliope import exceptions
from calliope.core.attrdict import AttrDict
from calliope.core.util.tools import plugin_load
from calliope.preprocess import checks
from calliope.core.util.dataset import reorganise_xarray_dimensions


def apply_time_clustering(model_data, model_run):
    """
    Take a Calliope model_data post time dimension addition, prior to any time
    clustering, and apply relevant time clustering/masking techniques.
    See doi: 10.1016/j.apenergy.2017.03.051 for applications.

    Techniques include:
    - Clustering timeseries into a selected number of 'representative' days.
        Days with similar profiles and daily magnitude are grouped together and
        represented by one 'representative' day with a greater weight per time
        step.
    - Masking timeseries, leading to variable timestep length
        Only certain parts of the input are shown at full resolution, with other
        periods being clustered together into a single timestep.
        E.g. Keep high resolution in the week with greatest wind power variability,
        smooth all other timesteps to 12H
    - Timestep resampling
        Used to reduce problem size by reducing resolution of all timeseries data.
        E.g. resample from 1H to 6H timesteps


    Parameters
    ----------
    model_data : xarray Dataset
        Preprocessed Calliope model_data, as produced using
        `calliope.preprocess.build_model_data`
        and found in model._model_data_original
    model_run : bool
        preprocessed model_run dictionary, as produced by
        Calliope.preprocess_model

    Returns
    -------
    data : xarray Dataset
        Dataset with optimisation parameters as variables, optimisation sets as
        coordinates, and other information in attributes. Time dimension has
        been updated as per user-defined clustering techniques (from model_run)

    """
    time_config = model_run.model["time"]

    data = model_data.copy(deep=True)

    ##
    # Process masking and get list of timesteps to keep at high res
    ##
    if "masks" in time_config:
        masks = {}
        # time.masks is a list of {'function': .., 'options': ..} dicts
        for entry in time_config.masks:
            entry = AttrDict(entry)
            mask_func = plugin_load(
                entry.function, builtin_module="calliope.time.masks"
            )
            mask_kwargs = entry.get_key("options", default=AttrDict()).as_dict()
            masks[entry.to_yaml()] = mask_func(data, **mask_kwargs)
        data.attrs["masks"] = masks
        # Concatenate the DatetimeIndexes by using dummy Series
        chosen_timesteps = pd.concat(
            [pd.Series(0, index=m) for m in masks.values()]
        ).index
        # timesteps: a list of timesteps NOT picked by masks
        timesteps = pd.Index(data.timesteps.values).difference(chosen_timesteps)
    else:
        timesteps = None

    ##
    # Process function, apply resolution adjustments
    ##
    if "function" in time_config:
        func = plugin_load(time_config.function, builtin_module="calliope.time.funcs")
        func_kwargs = time_config.get("function_options", AttrDict()).as_dict()
        if "file=" in func_kwargs.get("clustering_func", ""):
            func_kwargs.update({"model_run": model_run})
        data = func(data=data, timesteps=timesteps, **func_kwargs)

    return data


def add_time_dimension(data, model_run):
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
    for variable in model_run.timeseries_vars:
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
            .str.rsplit(":", 1, expand=True)
            .reset_index()
            .rename(columns={0: "source", 1: "column"})
            .set_index(["source", "column"])
        )

        # 6) Get all timeseries data from dataframes stored in model_run
        try:
            timeseries_data = model_run.timeseries_data.loc[:, tskeys.index]
        except KeyError:
            key_errors.append(
                f"file:column combinations `{tskeys.index.values}` not found, but are"
                f" requested by parameter `{variable}`."
            )
            continue

        timeseries_data.columns = pd.MultiIndex.from_frame(tskeys)

        # 7) Add time dimension to the relevent DataArray and update the '='
        # dimensions with the time varying data (static data is just duplicated
        # at each timestep)

        data[variable] = (
            xr.DataArray.from_series(timeseries_data.unstack())
            .reindex(data[variable].coords)
            .fillna(data[variable])
        )
    if key_errors:
        exceptions.print_warnings_and_raise_errors(errors=key_errors)

    # Add timestep_resolution by looking at the time difference between timestep n
    # and timestep n + 1 for all timesteps
    # Last timestep has no n + 1, so will be NaT (not a time), we ffill this.
    # Time resolution is saved in hours (i.e. nanoseconds / 3600e6)
    data["timestep_resolution"] = data.timesteps.diff(
        "timesteps", label="lower"
    ).reindex({"timesteps": data.timesteps}).ffill("timesteps").rename(
        "timestep_resolution"
    ) / pd.Timedelta(
        "1 hour"
    )
    if len(data.timesteps) == 1:
        exceptions.warn(
            "Only one timestep defined. Inferring timestep resolution to be 1 hour"
        )
        data["timestep_resolution"] = data["timestep_resolution"].fillna(1)

    data["timestep_weights"] = xr.DataArray(
        np.ones(len(data.timesteps)), dims=["timesteps"]
    )

    return data


def add_max_demand_timesteps(model_data):
    model_data["max_demand_timesteps"] = (
        (
            model_data.resource.where(model_data.resource < 0)
            * model_data.carrier.loc[
                {
                    "carrier_tiers": model_data.carrier_tiers.isin(
                        (["in", "in_2", "in_3"])
                    )
                }
            ].sum("carrier_tiers")
        )
        .sum(["nodes", "techs"])
        .idxmin("timesteps")
    )
    return model_data


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
                    model_data[var_name] = var.astype(np.int, copy=False)
                except (ValueError, OverflowError):
                    try:
                        model_data[var_name] = var.astype(np.float, copy=False)
                    except ValueError:
                        None
    return model_data
