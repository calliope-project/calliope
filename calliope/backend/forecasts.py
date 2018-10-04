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
    forecasts = []
    for windowstep in windowsteps:
        step_data = []
        for _, d_from, d_to, resolution in extents:
            # Set extents to start from current windowstep
            d_from = windowstep + d_from
            d_to = windowstep + d_to
            resampled = data.loc[
                dict(timesteps=slice(d_from, d_to))
            ].to_pandas().T.resample(resolution).mean().reset_index(drop=True)
            step_data.append(resampled)
        forecasts.append(xr.DataArray(pd.concat(step_data, axis=0, ignore_index=True)))
    forecast_data = xr.concat(forecasts, dim='windowsteps')

    forecast_data['windowsteps'] = windowsteps
    forecast_data = forecast_data.rename({'dim_0': 'horizonsteps'})

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
    # horizon: a list of {horizon_size: horizon_resolution} dicts
    horizon = data.attrs['run.operation.horizon']
    horizon = [
        (list(h.keys())[0], list(h.values())[0])
        for h in horizon
    ]

    # t_ -> timesteps, h_ -> horizonsteps
    t_resolutions = data.timestep_resolution.to_series()

    h_resolutions = []
    horizon_extents = []

    first_timestep = t_resolutions.index[0]
    from_timestep = first_timestep
    for horizon_size, horizon_resolution in horizon:
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
    max_windowstep = model_data.timesteps.to_series()[0] + pd.DateOffset(hours=max_extent)

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

    return forecast_dataset
