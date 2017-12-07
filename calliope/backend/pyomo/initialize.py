"""
Copyright (C) 2013-2017 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

"""

from calliope.backend.pyomo.model import generate_model, solve_model, load_results


def run(model_data):
    backend_model = generate_model(model_data)

    solver = model_data.attrs['run.solver']
    solver_io = model_data.attrs.get('run.solver_io', None)
    solver_options = model_data.attrs.get('run.solver_options', None)
    save_logs = model_data.attrs.get('run.save_logs', None)

    results = solve_model(
        backend_model, solver=solver,
        solver_io=solver_io, solver_options=solver_options, save_logs=save_logs
    )

    # load_results(backend_model, results)

    # # FIXME extract results into a solution xarray

    # # FIXME don't return model, return solution
    # return backend_model
