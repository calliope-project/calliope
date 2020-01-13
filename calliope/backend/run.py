"""
Copyright (C) 2013-2019 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

"""
import logging

import numpy as np
import xarray as xr

from calliope.core.util.logging import log_time
from calliope import exceptions
from calliope.backend import checks
from calliope.backend.pyomo import model as run_pyomo
from calliope.backend.pyomo import interface as pyomo_interface

from calliope.core.util.observed_dict import UpdateObserverDict
from calliope.core.attrdict import AttrDict

logger = logging.getLogger(__name__)


def run(model_data, timings, build_only=False):
    """
    Parameters
    ----------
    model_data : xarray.Dataset
        Pre-processed dataset of Calliope model data.
    timings : dict
        Stores timings of various stages of model processing.
    build_only : bool, optional
        If True, the backend only constructs its in-memory representation
        of the problem rather than solving it. Used for debugging and
        testing.

    """

    BACKEND = {
        'pyomo': run_pyomo
    }

    INTERFACE = {
        'pyomo': pyomo_interface
    }

    run_config = AttrDict.from_yaml_string(model_data.attrs['run_config'])

    if run_config['mode'] == 'plan':
        results, backend = run_plan(
            model_data, timings,
            backend=BACKEND[run_config.backend], build_only=build_only
        )

    elif run_config['mode'] == 'operate':
        results, backend = run_operate(
            model_data, timings,
            backend=BACKEND[run_config.backend], build_only=build_only
        )

    return results, backend, INTERFACE[run_config.backend].BackendInterfaceMethods


def run_plan(model_data, timings, backend, build_only, backend_rerun=False):

    log_time(logger, timings, 'run_start', comment='Backend: starting model run')

    if not backend_rerun:
        backend_model = backend.generate_model(model_data)

        log_time(
            logger, timings, 'run_backend_model_generated', time_since_run_start=True,
            comment='Backend: model generated'
        )

    else:
        backend_model = backend_rerun

    run_config = backend_model.__calliope_run_config
    solver = run_config['solver']
    solver_io = run_config.get('solver_io', None)
    solver_options = run_config.get('solver_options', None)
    save_logs = run_config.get('save_logs', None)

    if build_only:
        results = xr.Dataset()

    else:
        log_time(
            logger, timings, 'run_solver_start',
            comment='Backend: sending model to solver'
        )

        results = backend.solve_model(
            backend_model, solver=solver,
            solver_io=solver_io, solver_options=solver_options, save_logs=save_logs
        )

        log_time(
            logger, timings, 'run_solver_exit', time_since_run_start=True,
            comment='Backend: solver finished running'
        )

        termination = backend.load_results(backend_model, results)

        log_time(
            logger, timings, 'run_results_loaded',
            comment='Backend: loaded results'
        )

        results = backend.get_result_array(backend_model, model_data)
        results.attrs['termination_condition'] = termination

        if results.attrs['termination_condition'] in ['optimal', 'feasible']:
            results.attrs['objective_function_value'] = backend_model.obj()

        log_time(
            logger, timings, 'run_solution_returned', time_since_run_start=True,
            comment='Backend: generated solution array'
        )

    return results, backend_model


def run_operate(model_data, timings, backend, build_only):
    """
    For use when mode is 'operate', to allow the model to be built, edited, and
    iteratively run within Pyomo.

    """
    log_time(logger, timings, 'run_start',
             comment='Backend: starting model run in operational mode')

    defaults = UpdateObserverDict(
        initial_yaml_string=model_data.attrs['defaults'], name='defaults', observer=model_data
    )
    run_config = UpdateObserverDict(
        initial_yaml_string=model_data.attrs['run_config'], name='run_config', observer=model_data
    )

    # New param defaults = old maximum param defaults (e.g. energy_cap gets default from energy_cap_max)
    operate_params = {
        k.replace('_max', ''): v for k, v in defaults.items() if k.endswith('_max')
    }
    operate_params['purchased'] = 0  # no _max to work from here, so we hardcode a default

    defaults.update(operate_params)

    # Capacity results (from plan mode) can be used as the input to operate mode
    if (any(model_data.filter_by_attrs(is_result=1).data_vars) and
            run_config.get('operation.use_cap_results', False)):
        # Anything with is_result = 1 will be ignored in the Pyomo model
        for varname, varvals in model_data.data_vars.items():
            if varname in operate_params.keys():
                varvals.attrs['is_result'] = 1
                varvals.attrs['operate_param'] = 1

    else:
        cap_max = xr.merge([
            v.rename(k.replace('_max', ''))
            for k, v in model_data.data_vars.items() if '_max' in k
        ])
        cap_equals = xr.merge([
            v.rename(k.replace('_equals', ''))
            for k, v in model_data.data_vars.items() if '_equals' in k
        ])
        caps = cap_max.update(cap_equals)
        for cap in caps.data_vars.values():
            cap.attrs['is_result'] = 1
            cap.attrs['operate_param'] = 1
        model_data.update(caps)

    comments, warnings, errors = checks.check_operate_params(model_data)
    exceptions.print_warnings_and_raise_errors(warnings=warnings, errors=errors)

    # Initialize our variables
    solver = run_config['solver']
    solver_io = run_config.get('solver_io', None)
    solver_options = run_config.get('solver_options', None)
    save_logs = run_config.get('save_logs', None)
    window = run_config['operation']['window']
    horizon = run_config['operation']['horizon']
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
        and v.attrs['is_result'] == 0
    ]

    # Loop through each window, solve over the horizon length, and add result to
    # result_array we only go as far as the end of the last horizon, which may
    # clip the last bit of data
    result_array = []
    # track whether each iteration finds an optimal solution or not
    terminations = []

    if build_only:
        iterations = [0]
    else:
        iterations = range(len(window_starts))

    for i in iterations:
        start_timestep = window_starts.index[i]

        # Build full model in first instance
        if i == 0:
            warmstart = False
            end_timestep = horizon_ends.index[i]
            timesteps = slice(start_timestep, end_timestep)
            window_model_data = model_data.loc[dict(timesteps=timesteps)]

            log_time(
                logger, timings, 'model_gen_1',
                comment='Backend: generating initial model'
            )

            backend_model = backend.generate_model(window_model_data)

        # Build the full model in the last instance(s),
        # where number of timesteps is less than the horizon length
        elif i > len(horizon_ends) - 1:
            warmstart = False
            end_timestep = window_ends.index[i]
            timesteps = slice(start_timestep, end_timestep)
            window_model_data = model_data.loc[dict(timesteps=timesteps)]

            log_time(
                logger, timings, 'model_gen_{}'.format(i + 1),
                comment=(
                    'Backend: iteration {}: generating new model for '
                    'end of timeseries, with horizon = {} timesteps'
                    .format(i + 1, window_ends[i] - window_starts[i])
                )
            )

            backend_model = backend.generate_model(window_model_data)

        # Update relevent Pyomo Params in intermediate instances
        else:
            warmstart = True
            end_timestep = horizon_ends.index[i]
            timesteps = slice(start_timestep, end_timestep)
            window_model_data = model_data.loc[dict(timesteps=timesteps)]

            log_time(
                logger, timings, 'model_gen_{}'.format(i + 1),
                comment='Backend: iteration {}: updating model parameters'.format(i + 1)
            )
            # Pyomo model sees the same timestamps each time, we just change the
            # values associated with those timestamps
            for var in timeseries_data_vars:
                # New values
                var_series = window_model_data[var].to_series().dropna().replace('inf', np.inf)
                # Same timestamps
                var_series.index = backend_model.__calliope_model_data['data'][var].keys()
                var_dict = var_series.to_dict()
                # Update pyomo Param with new dictionary

                getattr(backend_model, var).store_values(var_dict)

        if not build_only:
            log_time(
                logger, timings, 'model_run_{}'.format(i + 1), time_since_run_start=True,
                comment='Backend: iteration {}: sending model to solver'.format(i + 1)
            )
            # After iteration 1, warmstart = True, which should speed up the process
            # Note: Warmstart isn't possible with GLPK (dealt with later on)
            _results = backend.solve_model(
                backend_model, solver=solver, solver_io=solver_io,
                solver_options=solver_options, save_logs=save_logs, warmstart=warmstart,
            )

            log_time(
                logger, timings, 'run_solver_exit_{}'.format(i + 1), time_since_run_start=True,
                comment='Backend: iteration {}: solver finished running'.format(i + 1)
            )
            # xarray dataset is built for each iteration
            _termination = backend.load_results(backend_model, _results)
            terminations.append(_termination)

            _results = backend.get_result_array(backend_model, model_data)

            # We give back the actual timesteps for this iteration and take a slice
            # equal to the window length
            _results['timesteps'] = window_model_data.timesteps.copy()

            # We always save the window data. Until the last window(s) this will crop
            # the window_to_horizon timesteps. In the last window(s), optimistion will
            # only be occurring over a window length anyway
            _results = _results.loc[dict(timesteps=slice(None, window_ends.index[i]))]
            result_array.append(_results)

            # Set up initial storage for the next iteration
            if 'loc_techs_store' in model_data.dims.keys():
                storage_initial = _results.storage.loc[{'timesteps': window_ends.index[i]}].drop('timesteps')
                model_data['storage_initial'].loc[storage_initial.coords] = storage_initial.values
                backend_model.storage_initial.store_values(
                    storage_initial.to_series().dropna().to_dict()
                )

            # Set up total operated units for the next iteration
            if 'loc_techs_milp' in model_data.dims.keys():
                operated_units = _results.operating_units.sum('timesteps').astype(np.int)
                model_data['operated_units'].loc[{}] += operated_units.values
                backend_model.operated_units.store_values(
                    operated_units.to_series().dropna().to_dict()
                )

            log_time(
                logger, timings, 'run_solver_exit_{}'.format(i + 1), time_since_run_start=True,
                comment='Backend: iteration {}: generated solution array'.format(i + 1)
            )

    if build_only:
        results = xr.Dataset()
    else:
        # Concatenate results over the timestep dimension to get a single
        # xarray Dataset of interest
        results = xr.concat(result_array, dim='timesteps')
        if all(i == 'optimal' for i in terminations):
            results.attrs['termination_condition'] = 'optimal'
        elif all(i in ['optimal', 'feasible'] for i in terminations):
            results.attrs['termination_condition'] = 'feasible'
        else:
            results.attrs['termination_condition'] = ','.join(terminations)

        log_time(
            logger, timings, 'run_solution_returned', time_since_run_start=True,
            comment='Backend: generated full solution array'
        )

    return results, backend_model
