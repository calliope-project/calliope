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
    data["timesteps"] = pd.to_datetime(data.timesteps.data)

    # Search through every constraint/cost for use of '='
    for variable in data.data_vars:
        # 1) If '=' in variable, it will give the variable a string data type
        if data[variable].dtype.kind != "U":
            continue

        # 2) convert to a Pandas Series to do 'string contains' search
        data_series = data[variable].to_series()

        # 3) get Series of all uses of 'file=' or 'df=' for this variable (timeseries keys)
        tskeys = data_series[
            data_series.str.contains("file=") | data_series.str.contains("df=")
        ]

        # 4) If no use of 'file=' or 'df=' then we can be on our way
        if tskeys.empty:
            continue

        # 5) remove all before '=' and split filename and location column
        tskeys = tskeys.str.split("=").str[1].str.rsplit(":", 1)
        if isinstance(tskeys.index, pd.MultiIndex):
            tskeys.index = tskeys.index.remove_unused_levels()

        # 6) Get all timeseries data from dataframes stored in model_run
        timeseries_data = []
        key_errors = []
        for loc_tech, (tskey, column) in tskeys.items():
            try:
                timeseries_data.append(
                    model_run.timeseries_data[tskey].loc[:, column].values
                )
            except KeyError:
                key_errors.append(
                    "column `{}` not found in dataframe `{}`, but was requested by "
                    "loc::tech `{}`.".format(column, tskey, loc_tech)
                )
        if key_errors:
            exceptions.print_warnings_and_raise_errors(errors=key_errors)

        timeseries_data_series = pd.DataFrame(
            index=tskeys.index, columns=data.timesteps.values, data=timeseries_data
        ).stack()
        timeseries_data_series.index.set_names("timesteps", level=-1, inplace=True)

        # 7) Add time dimension to the relevent DataArray and update the '='
        # dimensions with the time varying data (static data is just duplicated
        # at each timestep)
        timeseries_data_array = xr.broadcast(data[variable], data.timesteps)[0].copy()
        timeseries_data_array.loc[
            xr.DataArray.from_series(timeseries_data_series).coords
        ] = xr.DataArray.from_series(timeseries_data_series).values

        data[variable] = timeseries_data_array

    # Add timestep_resolution by looking at the time difference between timestep n
    # and timestep n + 1 for all timesteps
    time_delta = (data.timesteps.shift(timesteps=-1) - data.timesteps).to_series()

    # Last timestep has no n + 1, so will be NaT (not a time),
    # we duplicate the penultimate time_delta instead
    time_delta[-1] = time_delta[-2]
    time_delta.name = "timestep_resolution"
    # Time resolution is saved in hours (i.e. seconds / 3600)
    data["timestep_resolution"] = xr.DataArray.from_series(
        time_delta.dt.total_seconds() / 3600
    )

    data["timestep_weights"] = xr.DataArray(
        np.ones(len(data.timesteps)), dims=["timesteps"]
    )

    return data


def add_max_demand_timesteps(model_data):
    max_demand_timesteps = []

    # Get all loc_techs with a demand resource
    loc_techs_with_demand_resource = list(
        set(model_data.coords["loc_techs_finite_resource"].values).intersection(
            model_data.coords["loc_techs_demand"].values
        )
    )

    for carrier in list(model_data.carriers.data):
        # Filter demand loc_techs for this carrier
        loc_techs = [
            i
            for i in loc_techs_with_demand_resource
            if "{}::{}".format(i, carrier)
            in model_data.coords["loc_tech_carriers_con"].values
        ]

        carrier_demand = (
            model_data.resource.loc[dict(loc_techs_finite_resource=loc_techs)]
            .sum(dim="loc_techs_finite_resource")
            .copy()
        )

        # Only keep negative (=demand) values
        carrier_demand[carrier_demand.values > 0] = 0

        max_demand_timesteps.append(carrier_demand.to_series().idxmin())

    model_data["max_demand_timesteps"] = xr.DataArray(
        max_demand_timesteps, dims=["carriers"]
    )

    return model_data


def add_zero_carrier_ratio_sets(model_data):
    carrier_ratios = model_data.get("carrier_ratios", None)

    if carrier_ratios is None:
        return model_data

    zero_dims = (
        carrier_ratios.where(carrier_ratios == 0)
        .dropna("loc_tech_carriers_conversion_plus", how="all")
        .dropna("carrier_tiers", how="all")
    )

    if zero_dims.any().item() is False:
        return model_data

    zero_dims = zero_dims.stack(
        loc_tech_carrier_tiers_conversion_plus_zero_ratio=[
            "loc_tech_carriers_conversion_plus",
            "carrier_tiers",
        ]
    ).dropna("loc_tech_carrier_tiers_conversion_plus_zero_ratio", how="all")

    return model_data.assign_coords(
        loc_tech_carrier_tiers_conversion_plus_zero_ratio_constraint=[
            "::".join(i)
            for i in zero_dims.loc_tech_carrier_tiers_conversion_plus_zero_ratio.values
        ]
    )


def final_timedimension_processing(model_data):

    # Final checking of the data
    model_data, final_check_comments, warns, errors = checks.check_model_data(
        model_data
    )
    exceptions.print_warnings_and_raise_errors(warnings=warns, errors=errors)

    model_data = add_max_demand_timesteps(model_data)
    model_data = add_zero_carrier_ratio_sets(model_data)

    model_data = reorganise_xarray_dimensions(model_data)

    return model_data
