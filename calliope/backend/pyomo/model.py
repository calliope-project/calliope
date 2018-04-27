"""
Copyright (C) 2013-2018 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

"""

import logging
import os
import ruamel.yaml
from contextlib import redirect_stdout, redirect_stderr

import numpy as np
import pandas as pd
import xarray as xr

import pyomo.core as po  # pylint: disable=import-error
from pyomo.opt import SolverFactory  # pylint: disable=import-error

# pyomo.environ is needed for pyomo solver plugins
import pyomo.environ  # pylint: disable=unused-import,import-error

# TempfileManager is required to set log directory
from pyutilib.services import TempfileManager  # pylint: disable=import-error

from calliope.backend.pyomo.util import get_var
from calliope.core.util.tools import load_function
from calliope.core.util.logging import LogWriter, logger
from calliope.core.util.dataset import reorganise_dataset_dimensions
from calliope import exceptions


def generate_model(model_data):
    """
    Generate a Pyomo model.

    """
    backend_model = po.ConcreteModel()
    mode = model_data.attrs['run.mode']  # 'plan' or 'operate'
    backend_model.mode = mode

    # Sets
    for coord in list(model_data.coords):
        set_data = list(model_data.coords[coord].data)
        # Ensure that time steps are pandas.Timestamp objects
        if isinstance(set_data[0], np.datetime64):
            set_data = pd.to_datetime(set_data)
        setattr(
            backend_model, coord,
            po.Set(initialize=set_data, ordered=True)
        )

    # "Parameters"
    model_data_dict = {
        'data': {
            k: v.to_series().dropna().replace('inf', np.inf).to_dict()
            for k, v in model_data.data_vars.items()
            if v.attrs['is_result'] == 0 or v.attrs.get('operate_param', 0) == 1
        },
        'dims': {
            k: v.dims
            for k, v in model_data.data_vars.items()
            if v.attrs['is_result'] == 0 or v.attrs.get('operate_param', 0) == 1
        },
        'sets': list(model_data.coords),
        'attrs': {k: v for k, v in model_data.attrs.items() if k is not 'defaults'}
    }
    # Dims in the dict's keys are ordered as in model_data, which is enforced
    # in model_data generation such that timesteps are always last and the
    # remainder of dims are in alphabetic order
    backend_model.__calliope_model_data__ = model_data_dict
    backend_model.__calliope_defaults__ = (
        ruamel.yaml.load(model_data.attrs['defaults'], Loader=ruamel.yaml.Loader)
    )

    for k, v in model_data_dict['data'].items():
        if k in backend_model.__calliope_defaults__.keys():
            setattr(
                backend_model, k,
                po.Param(*[getattr(backend_model, i)
                           for i in model_data_dict['dims'][k]],
                         initialize=v, mutable=True,
                         default=backend_model.__calliope_defaults__[k])
            )
        elif k == 'timestep_resolution' or k == 'timestep_weights':  # no default value to look up
            setattr(
                backend_model, k,
                po.Param(backend_model.timesteps, initialize=v, mutable=True)
            )
        elif mode == 'operate' and model_data[k].attrs.get('operate_param') == 1:
            setattr(
                backend_model, k,
                po.Param(getattr(backend_model, model_data_dict['dims'][k][0]),
                         initialize=v, mutable=True)
            )


    # Variables
    load_function(
        'calliope.backend.pyomo.variables.initialize_decision_variables'
    )(backend_model)

    # Constraints
    constraints_to_add = [
        'energy_balance.load_constraints',
        'dispatch.load_constraints',
        'network.load_constraints',
        'costs.load_constraints',
        'policy.load_constraints'
    ]

    if mode != 'operate':
        constraints_to_add.append('capacity.load_constraints')

    if hasattr(backend_model, 'loc_techs_conversion'):
        constraints_to_add.append('conversion.load_constraints')

    if hasattr(backend_model, 'loc_techs_conversion_plus'):
        constraints_to_add.append('conversion_plus.load_constraints')

    if hasattr(backend_model, 'loc_techs_milp') or hasattr(backend_model, 'loc_techs_purchase'):
        constraints_to_add.append('milp.load_constraints')

    # Export comes last as it can add to the cost expression, this could be
    # overwritten if it doesn't come last
    if hasattr(backend_model, 'loc_techs_export'):
        constraints_to_add.append('export.load_constraints')

    for c in constraints_to_add:
        load_function(
            'calliope.backend.pyomo.constraints.' + c
        )(backend_model)

    # FIXME: Optional constraints
    # optional_constraints = model_data.attrs['constraints']
    # if optional_constraints:
    #     for c in optional_constraints:
    #         self.add_constraint(load_function(c))

    # Objective function
    # FIXME re-enable loading custom objectives

    # fetch objective function by name, pass through objective options
    # if they are present
    objective_function = ('calliope.backend.pyomo.objective.' +
                          model_data.attrs['run.objective'])
    objective_args = dict([(k.split('.')[-1], v)
                          for k, v in model_data.attrs.items()
                          if (k.startswith('run.objective_options'))
                           ])
    load_function(objective_function)(backend_model, **objective_args)


    # delattr(backend_model, '__calliope_model_data__')

    return backend_model


def solve_model(backend_model, solver,
                solver_io=None, solver_options=None, save_logs=False,
                **solve_kwargs):
    """
    Solve a Pyomo model using the chosen solver and all necessary solver options

    Returns a Pyomo results object
    """
    opt = SolverFactory(solver, solver_io=solver_io)

    if solver_options:
        for k, v in solver_options.items():
            opt.options[k] = v

    if save_logs:
        solve_kwargs.update({
            'symbolic_solver_labels': True,
            'keepfiles': True
        })
        os.makedirs(save_logs, exist_ok=True)
        TempfileManager.tempdir = save_logs  # Sets log output dir
    if 'warmstart' in solve_kwargs.keys() and solver == 'glpk':
        exceptions.ModelWarning(
            'The chosen solver, GLPK, does not suport warmstart, which may '
            'impact performance.'
        )
        del solve_kwargs['warmstart']

    with redirect_stdout(LogWriter('solver', strip=True)):
        with redirect_stderr(LogWriter('error', strip=True)):
            results = opt.solve(backend_model, tee=True, **solve_kwargs)

    return results


def load_results(backend_model, results):
    """Load results into model instance for access via model variables."""
    not_optimal = (
        results['Solver'][0]['Termination condition'].key != 'optimal'
    )
    this_result = backend_model.solutions.load_from(results)

    if this_result is False or not_optimal:
        logger.critical('Problem status:')
        for l in str(results.Problem).split('\n'):
            logger.critical(l)
        logger.critical('Solver status:')
        for l in str(results.Solver).split('\n'):
            logger.critical(l)

        if not_optimal:
            message = 'Model solution was non-optimal.'
        else:
            message = 'Could not load results into model instance.'

        exceptions.BackendWarning(message)

    return results['Solver'][0]['Termination condition'].key


def get_result_array(backend_model, model_data):
    """
    From a Pyomo model object, extract decision variable data and return it as
    an xarray Dataset. Any rogue input parameters that are constructed inside
    the backend (instead of being passed by calliope.Model().inputs) are also
    added to calliope.Model()._model_data in-place.
    """
    all_variables = {
        i.name: get_var(backend_model, i.name) for i in backend_model.component_objects()
        if isinstance(i, po.base.var.IndexedVar)
    }

    # Get any parameters that did not appear in the user's model.inputs Dataset
    all_params = {
        i.name: get_var(backend_model, i.name)
        for i in backend_model.component_objects()
        if isinstance(i, po.base.param.IndexedParam) and
        i.name not in model_data.data_vars.keys()
    }

    results = reorganise_dataset_dimensions(xr.Dataset(all_variables))

    if all_params:
        additional_inputs = reorganise_dataset_dimensions(xr.Dataset(all_params))
        for var in additional_inputs.data_vars:
            additional_inputs[var].attrs['is_result'] = 0
        model_data.update(additional_inputs)

    return results
