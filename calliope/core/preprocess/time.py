"""
Copyright (C) 2013-2017 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

time.py
~~~~~~~~~~~~~~~~~~

Functionality to add and process time varying parameters

"""

import xarray as xr
import numpy as np
import pandas as pd

from calliope import exceptions
from calliope.core.attrdict import AttrDict
from calliope.core.util.tools import plugin_load
from calliope._version import __version__
from calliope.core.preprocess import checks
from calliope.core.util.dataset import reorganise_dataset_dimensions


def apply_time_clustering(model_data_original, model_run):
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
    model_data_original : xarray Dataset
        Preprocessed Calliope model_data, as produced using
        `calliope.core.preprocess_data.build_model_data`
        and found in model._model_data_original
    model_run : bool
        preprocessed model_run dictionary, as produced by
        Calliope.core.preprocess_model

    Returns
    -------
    data : xarray Dataset
        Dataset with optimisation parameters as variables, optimisation sets as
        coordinates, and other information in attributes. Time dimension has
        been updated as per user-defined clustering techniques (from model_run)
    """

    # Carry y_ subset sets over to data for easier data analysis
    time_config = model_run.model.get('time', None)
    if not time_config:
        return reorganise_dataset_dimensions(model_data_original)  # Nothing more to do here
    else:
        data = model_data_original.copy(deep=True)

    # Add temporary 'timesteps per day' attribute
    timestep_resolution = data['timestep_resolution'].values[0]
    if not np.all(data['timestep_resolution'] == timestep_resolution):
        raise exceptions.ModelError('For clustering, timestep resolution must be uniform. ')
    timesteps_per_day = 24 / timestep_resolution
    if isinstance(timesteps_per_day, float):
        assert timesteps_per_day.is_integer(), 'Timesteps/day must be integer.'
        timesteps_per_day = int(timesteps_per_day)
    data.attrs['_timesteps_per_day'] = timesteps_per_day

    ##
    # Process masking and get list of timesteps to keep at high res
    ##
    if 'masks' in time_config:
        masks = {}
        # time.masks is a list of {'function': .., 'options': ..} dicts
        for entry in time_config.masks:
            entry = AttrDict(entry)
            mask_func = plugin_load(entry.function, builtin_module='calliope.core.time.masks')
            mask_kwargs = entry.get_key('options', default={})
            masks[entry.to_yaml()] = mask_func(data, **mask_kwargs)
        data.attrs['masks'] = masks
        # Concatenate the DatetimeIndexes by using dummy Series
        chosen_timesteps = pd.concat([pd.Series(0, index=m)
                                     for m in masks.values()]).index
        # timesteps: a list of timesteps NOT picked by masks
        timesteps = pd.Index(data.time.values).difference(chosen_timesteps)
    else:
        timesteps = None

    ##
    # Process function, apply resolution adjustments
    ##
    if 'function' in time_config:
        func = plugin_load(
            time_config.function, builtin_module='calliope.core.time.funcs')
        func_kwargs = time_config.get('function_options', {})
        data = func(data=data, timesteps=timesteps, **func_kwargs)

    # Final checking of the data
    final_check_comments, warnings, errors = checks.check_model_data(data)
    checks.print_warnings_and_raise_errors(warnings=warnings, errors=errors)

    # Temporary timesteps per day attribute is no longer needed
    try:
        del data.attrs['_timesteps_per_day']
    except KeyError:
        pass

    return reorganise_dataset_dimensions(data)


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
        with all relevant `file=` entries replaced with data from file.

    """
    data['timesteps'] = pd.to_datetime(data.timesteps)

    # Search through every constraint/cost for use of '='
    for variable in data.data_vars:
        # 1) If '=' in variable, it will give the variable a string data type
        if data[variable].dtype.kind != 'U':
            continue

        # 2) convert to a Pandas Series to do 'string contains' search
        data_series = data[variable].to_series()

        # 3) get a Series of all the uses of 'file=' for this variable
        filenames = data_series[data_series.str.contains('=')]

        # 4) If no use of 'file=' then we can be on our way
        if filenames.empty:
            continue

        # 5) remove all before '=' and split filename and location column
        filenames = filenames.str.split('=').str[1].str.rsplit(':', 1)

        # 6) Get all timeseries data from dataframes stored in model_run
        timeseries_data = [model_run.timeseries_data[file].loc[:, column].values
                           for (file, column) in filenames.values]

        timeseries_data_series = pd.DataFrame(index=filenames.index,
                                              columns=data.timesteps.values,
                                              data=timeseries_data).stack()
        timeseries_data_series.index.rename('timesteps', -1, inplace=True)

        # 7) Add time dimension to the relevent DataArray and update the '='
        # dimensions with the time varying data (static data is just duplicated
        # at each timestep)
        timeseries_data_array = xr.broadcast(data[variable], data.timesteps)[0].copy()

        timeseries_data_array.loc[
            xr.DataArray.from_series(timeseries_data_series).coords
        ] = xr.DataArray.from_series(timeseries_data_series).values

        # 8) assign correct dtype (might be string/object accidentally)
        if all(np.where(
            ((timeseries_data_array=='True') | (timeseries_data_array=='False'))
            )):
            # Turn to bool
            timeseries_data_array.loc[dict()] = timeseries_data_array == 'True'
            timeseries_data_array = timeseries_data_array.astype(np.bool, copy=False)
        else:
            try:
                timeseries_data_array = timeseries_data_array.astype(np.float, copy=False)
            except:
                None
        data[variable] = timeseries_data_array

    # Add time_resolution and timestep_weight variables
    # parsed_timesteps = pd.to_datetime(data.timesteps)
    seconds = abs(
        pd.to_datetime(data.timesteps.values[0]) -
        pd.to_datetime(data.timesteps.values[1])
    ).total_seconds()
    hours = seconds / 3600

    data['timestep_resolution'] = xr.DataArray(
            np.ones(len(data.timesteps)) * hours,
            dims=['timesteps']
    )

    data['timestep_weights'] = xr.DataArray(
            np.ones(len(data.timesteps)),
            dims=['timesteps']
    )

    return None
