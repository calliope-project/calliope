"""
Copyright (C) 2013-2017 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

"""

from calliope.backend.pyomo.model import generate_model, solve_model, load_results
from calliope.core.util.tools import log_time


def run(model_data, timings):
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

    # load_results(backend_model, results)
    log_time(timings, 'run_results_loaded')

    # # FIXME extract results into a solution xarray

    # # FIXME don't return model, return solution
    log_time(timings, 'run_solution_returned', time_since_start=True)
    return backend_model
