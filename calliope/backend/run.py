"""
Copyright (C) 2013-2018 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

"""
import ruamel.yaml

from calliope.core.util.logging import log_time
from calliope import exceptions
from calliope.backend import checks, forecasts

import numpy as np
import xarray as xr
import pandas as pd
import calliope.backend.pyomo.model as run_pyomo
import calliope.backend.pyomo.interface as pyomo_interface


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

    run_backend = model_data.attrs['run.backend']

    if model_data.attrs['run.mode'] == 'plan':
        results, backend = run_plan(
            model_data, timings,
            backend=BACKEND[run_backend], build_only=build_only
        )

    elif model_data.attrs['run.mode'] == 'operate':

        results, backend = run_operate(
            model_data, timings,
            backend=BACKEND[run_backend], build_only=build_only
        )

    return results, backend, INTERFACE[run_backend].BackendInterfaceMethods


def run_plan(model_data, timings, backend, build_only, backend_rerun=False):

    log_time(timings, 'run_start', comment='Backend: starting model run')

    if not backend_rerun:
        backend_model = backend.generate_model(model_data)

        log_time(
            timings, 'run_backend_model_generated', time_since_start=True,
            comment='Backend: model generated'
        )

    else:
        backend_model = backend_rerun

    solver = model_data.attrs['run.solver']
    solver_io = model_data.attrs.get('run.solver_io', None)
    solver_options = {
        k.split('.')[-1]: v
        for k, v in model_data.attrs.items() if '.solver_options.' in k
    }
    save_logs = model_data.attrs.get('run.save_logs', None)

    if build_only:
        results = xr.Dataset()

    else:
        log_time(
            timings, 'run_solver_start',
            comment='Backend: sending model to solver'
        )

        results = backend.solve_model(
            backend_model, solver=solver,
            solver_io=solver_io, solver_options=solver_options, save_logs=save_logs
        )

        log_time(
            timings, 'run_solver_exit', time_since_start=True,
            comment='Backend: solver finished running'
        )

        termination = backend.load_results(backend_model, results)

        log_time(
            timings, 'run_results_loaded',
            comment='Backend: loaded results'
        )

        results = backend.get_result_array(backend_model, model_data)
        results.attrs['termination_condition'] = termination

        log_time(
            timings, 'run_solution_returned', time_since_start=True,
            comment='Backend: generated solution array'
        )

    return results, backend_model


def run_operate(model_data, timings, backend, build_only):
    """
    For use when mode is 'operate', to allow the model to be built, edited, and
    iteratively run within Pyomo.

    """
    log_time(timings, 'run_start',
             comment='Backend: starting model run in operational mode')

    if any([i.endswith('_forecasts') for i in list(model_data.data_vars)]):
            # FIXME "temporary" and undocumented hack to allow user-supplied
            # operate_forecasts, with no sense checking whatsoever,
            # so to be used with caution and restraint
            assert 'horizonstep_resolution' in model_data.data_vars
    else:
        forecasts_data = forecasts.generate_forecasts(model_data)
        model_data.merge(forecasts_data, inplace=True)

    defaults = ruamel.yaml.load(model_data.attrs['defaults'], Loader=ruamel.yaml.Loader)
    operate_params = ['purchased'] + [
        i.replace('_max', '') for i in defaults if i[-4:] == '_max'
    ]

    # Capacity results (from plan mode) can be used as the input to operate mode
    if (any(model_data.filter_by_attrs(is_result=1).data_vars) and
            model_data.attrs.get('run.operation.use_cap_results', False)):
        # Anything with is_result = 1 will be ignored in the Pyomo model
        for varname, varvals in model_data.data_vars.items():
            if varname in operate_params:
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

    # Storage initial is carried over between iterations, so must be defined along with storage
    if ('loc_techs_store' in model_data.dims.keys() and
            'storage_initial' not in model_data.data_vars.keys()):
        model_data['storage_initial'] = (
            xr.DataArray([0 for loc_tech in model_data.loc_techs_store.values],
                         dims='loc_techs_store')
        )
        model_data['storage_initial'].attrs['is_result'] = 0
        exceptions.ModelWarning(
            'Initial stored energy not defined, set to zero for all '
            'loc::techs in loc_techs_store, for use in iterative optimisation'
        )
    # Operated units is carried over between iterations, so must be defined in a milp model
    if ('loc_techs_milp' in model_data.dims.keys() and
            'operated_units' not in model_data.data_vars.keys()):
        model_data['operated_units'] = (
            xr.DataArray([0 for loc_tech in model_data.loc_techs_milp.values],
                         dims='loc_techs_milp')
        )
        model_data['operated_units'].attrs['is_result'] = 1
        model_data['operated_units'].attrs['operate_param'] = 1
        exceptions.ModelWarning(
            'daily operated units not defined, set to zero for all '
            'loc::techs in loc_techs_milp, for use in iterative optimisation'
        )

    comments, warnings, errors = checks.check_operate_params(model_data)
    exceptions.print_warnings_and_raise_errors(warnings=warnings, errors=errors)

    # Initialize our variables
    solver = model_data.attrs['run.solver']
    solver_io = model_data.attrs.get('run.solver_io', None)
    solver_options = model_data.attrs.get('run.solver_options', None)
    save_logs = model_data.attrs.get('run.save_logs', None)
    window = model_data.attrs['run.operation.window']
    windowsteps = model_data.windowsteps.values

    # We will only update timseries parameters
    timeseries_data_vars = [
        k for k, v in model_data.data_vars.items() if 'timesteps' in v.dims and
        v.attrs['is_result'] == 0
    ]

    # Drop all timeseries vars, we only work with forecasts
    operate_model_data = model_data.drop(timeseries_data_vars)
    # Rename forecasts to their original names, so Pyomo understands them
    operate_model_data.rename(
        {var + '_forecast': var for var in timeseries_data_vars}, inplace=True
    )
    operate_model_data = operate_model_data.drop(['timesteps', 'timestep_resolution'])
    operate_model_data.rename(
        {'horizonstep_resolution': 'timestep_resolution', 'horizonsteps': 'timesteps'},
        inplace=True
    )

    # Loop through each window, solve over the horizon length, and add result to
    # result_array we only go as far as the end of the last horizon, which may
    # clip the last bit of data
    result_array = []
    # track whether each iteration finds an optimal solution or not
    terminations = []

    for windowstep in windowsteps:
        i = 0
        # window_model_data should look like a plan mode calliope model, with only timesteps
        # as a time dimension
        window_model_data = operate_model_data.loc[dict(windowsteps=windowstep)].drop('windowsteps')

        # Build full model in first instance
        if windowstep == windowsteps[0]:
            warmstart = False

            log_time(
                timings, 'model_gen_1',
                comment='Backend: generating initial model'
            )

            backend_model = backend.generate_model(window_model_data)

        # Update relevent Pyomo Params in intermediate instances
        else:
            warmstart = True

            log_time(
                timings, 'model_gen_{}'.format(i + 1),
                comment='Backend: iteration {}: updating model parameters'.format(i + 1)
            )
            # Pyomo model sees the same timestamps each time, we just change the
            # values associated with those timestamps
            for var in timeseries_data_vars:
                # New values
                var_series = window_model_data[var].to_series().dropna().replace('inf', np.inf)
                # Same timestamps
                var_series.index = backend_model.__calliope_model_data__['data'][var].keys()
                var_dict = var_series.to_dict()
                # Update pyomo Param with new dictionary
                for k, v in getattr(backend_model, var).items():
                    if k in var_dict:
                        v.set_value(var_dict[k])

        if build_only:
            break  # i.e. only the first windowstep is built if 'build_only'
        else:
            log_time(
                timings, 'model_run_{}'.format(i + 1), time_since_start=True,
                comment='Backend: iteration {}: sending model to solver'.format(i + 1)
            )
            # After iteration 1, warmstart = True, which should speed up the process
            # Note: Warmstart isn't possible with GLPK (dealt with later on)
            _results = backend.solve_model(
                backend_model, solver=solver, solver_io=solver_io,
                solver_options=solver_options, save_logs=save_logs, warmstart=warmstart,
            )

            log_time(
                timings, 'run_solver_exit_{}'.format(i + 1), time_since_start=True,
                comment='Backend: iteration {}: solver finished running'.format(i + 1)
            )
            # xarray dataset is built for each iteration
            _termination = backend.load_results(backend_model, _results)
            terminations.append(_termination)

            _results = backend.get_result_array(backend_model, model_data)

            # We give back the actual timesteps for this iteration and take a slice
            # equal to the window length
            window_end = pd.DateOffset(hours=forecasts.hours_from_datestring(window) - 1)
            window_slice = slice(pd.to_datetime(windowstep), pd.to_datetime(windowstep) + window_end)
            window_timesteps = model_data.timesteps.loc[{'timesteps': window_slice}]
            _results = _results.loc[{'timesteps': slice(0, len(window_timesteps) - 1)}]
            _results['timesteps'] = window_timesteps
            result_array.append(_results)

            # Set up initial storage for the next iteration
            if 'loc_techs_store' in model_data.dims.keys():
                storage_initial = _results.storage.loc[{'timesteps': _results.timesteps[-1]}].drop('timesteps')
                operate_model_data['storage_initial'].loc[storage_initial.coords] = storage_initial.values
                for k, v in backend_model.storage_initial.items():
                    v.set_value(storage_initial.to_series().dropna().to_dict()[k])

            # Set up total operated units for the next iteration
            if 'loc_techs_milp' in model_data.dims.keys():
                operated_units = _results.operating_units.sum('timesteps').astype(np.int)
                operate_model_data['operated_units'].loc[{}] += operated_units.values
                for k, v in backend_model.operated_units.items():
                    v.set_value(operated_units.to_series().dropna().to_dict()[k])

            log_time(
                timings, 'run_solver_exit_{}'.format(i + 1), time_since_start=True,
                comment='Backend: iteration {}: generated solution array'.format(i + 1)
            )

            i += 1

    if build_only:
        results = xr.Dataset()
    else:
        # Concatenate results over the timestep dimension to get a single
        # xarray Dataset of interest
        results = xr.concat(result_array, dim='timesteps')
        results.reindex(timesteps=model_data.timesteps)
        if all(i == 'optimal' for i in terminations):
            results.attrs['termination_condition'] = 'optimal'
        else:
            results.attrs['termination_condition'] = ','.join(terminations)

        log_time(
            timings, 'run_solution_returned', time_since_start=True,
            comment='Backend: generated full solution array'
        )

    return results, backend_model
