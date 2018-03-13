"""
Copyright (C) 2013-2018 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

"""

from calliope.backend.pyomo.model import generate_model, solve_model, load_results, get_result_array
from calliope.core.util.tools import log_time
from calliope import exceptions

import numpy as np
import xarray as xr

def run(model_data, timings):
    if model_data.attrs['model.mode'] == 'operate':
        return run_iterative(model_data, timings)

    log_time(timings, 'run_start', comment='Pyomo backend: starting model run')

    backend_model = generate_model(model_data)

    log_time(
        timings, 'run_backend_model_generated', time_since_start=True,
        comment='Pyomo backend: model generated'
    )

    solver = model_data.attrs['run.solver']
    solver_io = model_data.attrs.get('run.solver_io', None)
    solver_options = model_data.attrs.get('run.solver_options', None)
    save_logs = model_data.attrs.get('run.save_logs', None)

    log_time(
        timings, 'run_solver_start',
        comment='Pyomo backend: sending model to solver'
    )

    results = solve_model(
        backend_model, solver=solver,
        solver_io=solver_io, solver_options=solver_options, save_logs=save_logs
    )

    log_time(
        timings, 'run_solver_exit', time_since_start=True,
        comment='Pyomo backend: solver finished running'
    )

    load_results(backend_model, results)

    log_time(
        timings, 'run_results_loaded',
        comment='Pyomo backend: loaded results'
    )

    results = get_result_array(backend_model)

    log_time(
        timings, 'run_solution_returned', time_since_start=True,
        comment='Pyomo backend: generated solution array'
    )

    return results, backend_model


def run_iterative(model_data, timings):
    """
    For use when mode = 'operate', to allow the model to be built, edited, and
    iteratively run within Pyomo
    """
    log_time(timings, 'run_start', comment='Pyomo backend: starting model run in operational mode')

    # Initialize our variables
    solver = model_data.attrs['run.solver']
    solver_io = model_data.attrs.get('run.solver_io', None)
    solver_options = model_data.attrs.get('run.solver_options', None)
    save_logs = model_data.attrs.get('run.save_logs', None)
    window = model_data.attrs['model.operation.window']
    horizon = model_data.attrs['model.operation.horizon']
    window_to_horizon = horizon - window

    # get the cumulative sum of timestep resolution, to find where we hit our window and horizon
    timestep_cumsum = model_data.timestep_resolution.cumsum('timesteps').to_pandas()
    # get the timesteps at which we start and end our windows
    window_ends = timestep_cumsum.where(
        (timestep_cumsum % window == 0) | (timestep_cumsum == timestep_cumsum[-1])
    )
    window_starts = timestep_cumsum.where(
        (~np.isnan(window_ends.shift(1))) | (timestep_cumsum == timestep_cumsum[0])
    ).dropna()

    window_ends = window_ends.dropna()
    horizon_ends = timestep_cumsum[timestep_cumsum.isin(window_ends.values + window_to_horizon)]

    if not any(window_starts):
        raise exceptions.ModelError(
            'Not enough timesteps or incorrect timestep resolution to run in '
            'operational mode with an optimisation window of {}'.format(window)
        )



    # We will only update timseries parameters
    timeseries_data_vars = [
        k for k, v in model_data.data_vars.items() if 'timesteps' in v.dims
    ]

    # Loop through each window, solve over the horizon length, and add result to result_array
    # we only go as far as the end of the last horizon, which may clip the last bit of data
    result_array = []

    for i in range(len(window_starts)):
        start_timestep = window_starts.index[i]

        # Build full model in first instance
        if i == 0:
            warmstart = False
            end_timestep = horizon_ends.index[i]
            window_model_data = model_data.loc[dict(timesteps=slice(start_timestep, end_timestep))]

            log_time(timings, 'model_gen_1', comment='Pyomo backend: generating initial model')
            backend_model = generate_model(window_model_data)

        # Build the full model in the last instance(s), where number of timesteps is less than the horizon length
        elif i > len(horizon_ends) - 1:
            warmstart = False
            end_timestep = window_ends.index[i]
            window_model_data = model_data.loc[dict(timesteps=slice(start_timestep, end_timestep))]

            log_time(timings, 'model_gen_{}'.format(i + 1),
                comment='Pyomo backend: iteration {}: generating new model for '
                        'end of timeseries, with horizon = {} timesteps'.format(i + 1, window_ends[i] - window_starts[i]))
            backend_model = generate_model(window_model_data)

        # Update relevent Pyomo Params in intermediate instances
        else:
            warmstart = True
            end_timestep = horizon_ends.index[i]
            window_model_data = model_data.loc[dict(timesteps=slice(start_timestep, end_timestep))]

            log_time(timings, 'model_gen_{}'.format(i + 1),
                     comment='Pyomo backend: iteration {}: updating model parameters'.format(i + 1))
            # Pyomo model sees the same timestamps each time, we just change the values associated with those timestamps
            for var in timeseries_data_vars:
                # New values
                var_series = window_model_data[var].to_series().dropna().replace('inf', np.inf)
                # Same timestamps
                var_series.index = backend_model.__calliope_model_data__['data'][var].keys()
                var_dict = var_series.to_dict()
                # Update pyomo Param with new dictionary
                getattr(backend_model, var).reconstruct(var_dict)

        log_time(timings, 'model_run_{}'.format(i + 1), time_since_start=True,
                 comment='Pyomo backend: iteration {}: sending model to solver'.format(i + 1))

        # After iteration 1, warmstart = True, which should speed up the process
        # Note: Warmstart isn't possible with GLPK (dealt with later on)
        results = solve_model(
            backend_model, solver=solver, solver_io=solver_io,
            solver_options=solver_options, save_logs=save_logs, warmstart=warmstart
        )

        log_time(
            timings, 'run_solver_exit_{}'.format(i + 1), time_since_start=True,
            comment='Pyomo backend: iteration {}: solver finished running'.format(i + 1)
        )
        # xarray dataset is built for each iteration
        load_results(backend_model, results)
        results = get_result_array(backend_model)

        # We give back the actual timesteps for this iteration and take a slice equal to the window length
        results['timesteps'] = window_model_data.timesteps.copy()

        # We always save the window data. Until the last window(s) this will crop
        # the window_to_horizon timesteps. In the last window(s), optimistion will
        # only be occurring over a window length anyway
        results = results.loc[dict(timesteps=slice(None, window_ends.index[i]))]
        result_array.append(results)

        # Set up initial storage for the next iteration
        if 'loc_techs_store' in model_data:
            backend_model.storage_initial.reconstruct(
                results.storage.loc[dict(timesteps=window_ends.index[i])]
                .to_series().dropna().replace('inf', np.inf).to_dict()
            )

        log_time(
            timings, 'run_solver_exit_{}'.format(i + 1), time_since_start=True,
            comment='Pyomo backend: iteration {}: generated solution array'.format(i + 1)
        )

    # concatenate results over the timestep dimension to get one xarray dataset of interest
    results = xr.concat(result_array, dim='timesteps')

    log_time(
        timings, 'run_solution_returned', time_since_start=True,
        comment='Pyomo backend: generated full solution array'
    )

    return results, backend_model
