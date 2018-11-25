"""
Copyright (C) 2013-2018 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

"""
import pandas as pd
import xarray as xr

##
# CHECKS TO IMPLEMENT IN PREPROCESSING
##
def checks(model_data):
    horizon = model_data.attrs['run.operation.horizon']
    window = model_data.attrs['run.operation.window']

    # window and horizon must be in 'H', 'D' or 'W' units

    # window must be smaller than or equal to the first horizon

    # total length of horizons must be less than timeseries length available


def apply_horizon(data, windowsteps, extents):
    window_forecasts = []
    for windowstep in windowsteps:
        window_forecast = []
        for _, d_from, d_to, resolution in extents:
            # Set extents to start from current windowstep
            d_from = windowstep + d_from
            d_to = windowstep + d_to
            if resolution == 'window':
                resampled = data.loc[dict(timesteps=slice(d_from, d_to))]
            else:
                resampled = data.loc[dict(timesteps=slice(d_from, d_to))].resample(timesteps=resolution).mean('timesteps')
            window_forecast.append(resampled)
        window_forecasts.append(xr.concat(window_forecast, dim='timesteps').drop('timesteps'))
    forecast_data = xr.concat(window_forecasts, dim='windowsteps')

    forecast_data['windowsteps'] = windowsteps
    forecast_data = forecast_data.rename({'timesteps': 'horizonsteps'})

    return forecast_data


def hours_from_datestring(string):
    num, freq = int(string[:-1]), string[-1:]
    freqs = {
        'H': 1,
        'D': 24,
        'W': 24 * 7,
    }
    if freq not in freqs.keys():
        raise ValueError('Frequency must be one of {}'.format(freqs.keys()))
    else:
        hours = num * freqs[freq]

    return hours


def generate_horizonstep_resolution(data):
    """
    Returns
    -------
    resolutions : pandas.Series
        horizonstep resolutions indexed
        by the starting timestep of the given horizonstep.
    horizon_extents : dict
        dict with horizon resolutions as keys and a list of [from, to]
        timesteps as values

    """
    # horizon: a {horizon_size: horizon_resolution} dict
    horizon = {
        k.split('.')[-1]: v for k, v in data.attrs.items()
        if 'run.operation.horizon' in k
    }

    # t_ -> timesteps, h_ -> horizonsteps
    t_resolutions = data.timestep_resolution.to_series()
    first_timestep = t_resolutions.index[0]

    window = data.attrs['run.operation.window']
    windowstep = pd.DateOffset(hours=hours_from_datestring(window))

    # TODO: Add check that window < first horizon

    # Initialise lists with window information
    h_resolutions = [t_resolutions.loc[:first_timestep + windowstep][:-1]]

    horizon_extents = [[
        window,
        first_timestep - first_timestep,
        h_resolutions[0].index[-1] - first_timestep,
        'window'
    ]]

    from_timestep = first_timestep + windowstep
    for horizon_size, horizon_resolution in horizon.items():
        offset = pd.DateOffset(hours=hours_from_datestring(horizon_size))
        resolution = '{}H'.format(hours_from_datestring(horizon_resolution))
        to_ts_pos = t_resolutions.index.get_loc(first_timestep + offset) - 1
        to_timestep = t_resolutions.index[to_ts_pos]
        this_horizon = t_resolutions.loc[from_timestep:to_timestep]
        this_res = this_horizon.resample(resolution).sum()
        h_resolutions.append(this_res)
        horizon_extents.append([
            horizon_size,
            from_timestep - first_timestep,
            to_timestep - first_timestep,
            resolution
        ])
        from_timestep = t_resolutions.loc[to_timestep:].index[1]

    resolutions = pd.concat(h_resolutions, axis=0).reset_index(drop=True)
    resolutions.index.name = 'horizonsteps'

    return resolutions, horizon_extents


def generate_forecasts(model_data):
    """
    Generates a xr.DataArray of forecasts, indexed by windowsteps
    (timestamps of the timestep where each window starts) and
    horizonsteps (integer-indexed) duration

    """
    window = model_data.attrs['run.operation.window']

    # Only proceed if we have uniform 1-hourly data resolution
    # FIXME: is this actually necessary?
    assert all(model_data.timestep_resolution == 1)

    resolutions, extents = generate_horizonstep_resolution(model_data)

    max_extent = hours_from_datestring(extents[-1][0])
    max_windowstep = model_data.timesteps.to_series()[-1] - pd.DateOffset(hours=max_extent)

    windowsteps = model_data.timestep_resolution.to_series().loc[:max_windowstep].resample(window).first().index
    windowsteps.name = 'windowsteps'

    timeseries_vars = [
        v for v in model_data.data_vars
        if 'timesteps' in model_data[v].dims
    ]
    forecasts = {}

    for var in timeseries_vars:
        forecasts[var + '_forecast'] = apply_horizon(
            model_data[var], windowsteps, extents
        )

    forecast_dataset = xr.Dataset(forecasts)
    forecast_dataset['horizonstep_resolution'] = resolutions
    for k in forecast_dataset.data_vars.keys():
        forecast_dataset[k].attrs['is_result'] = 0


    return forecast_dataset
