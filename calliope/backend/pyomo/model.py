"""
Copyright (C) since 2013 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

"""

import logging
import os
from contextlib import redirect_stdout, redirect_stderr

import numpy as np
import pandas as pd
import xarray as xr

import pyomo.core as po  # pylint: disable=import-error
from pyomo.opt import SolverFactory  # pylint: disable=import-error

# pyomo.environ is needed for pyomo solver plugins
import pyomo.environ as pe  # pylint: disable=unused-import,import-error

# TempfileManager is required to set log directory
from pyomo.common.tempfiles import TempfileManager  # pylint: disable=import-error

from calliope.backend.pyomo.util import (
    get_var,
    get_domain,
    datetime_to_string,
    string_to_datetime,
)
from calliope.backend.pyomo import constraints
from calliope.core.util.tools import load_function
from calliope.core.util.logging import LogWriter
from calliope.core.util.dataset import reorganise_xarray_dimensions
from calliope import exceptions
from calliope.core.attrdict import AttrDict

logger = logging.getLogger(__name__)


def generate_model(model_data):
    """
    Generate a Pyomo model.

    """

    backend_model = po.ConcreteModel()
    model_data = datetime_to_string(backend_model, model_data)

    logger.info("Loading sets")
    # Sets
    for coord_name, coord_data in model_data.coords.items():
        if not coord_data.shape:
            continue
        set_data = list(coord_data.data)
        # Ensure that time steps are pandas.Timestamp objects
        if isinstance(set_data[0], np.datetime64):
            set_data = pd.to_datetime(set_data)
        setattr(backend_model, coord_name, po.Set(initialize=set_data, ordered=True))
    logger.info("Loading parameters")
    # "Parameters"
    model_data_dict = {
        "data": {
            k: v.to_series().dropna().replace("inf", np.inf).to_dict()
            for k, v in model_data.data_vars.items()
            if v.attrs["is_result"] == 0 or v.attrs.get("operate_param", 0) == 1
        },
        "dims": {
            k: v.dims
            for k, v in model_data.data_vars.items()
            if v.attrs["is_result"] == 0 or v.attrs.get("operate_param", 0) == 1
        },
        "sets": list(model_data.coords),
        "attrs": {k: v for k, v in model_data.attrs.items() if k != "defaults"},
    }

    # Dims in the dict's keys are ordered as in model_data, which is enforced
    # in model_data generation such that timesteps are always last and the
    # remainder of dims are in alphabetic order
    backend_model.__calliope_model_data = model_data_dict
    backend_model.__calliope_defaults = AttrDict.from_yaml_string(
        model_data.attrs["defaults"]
    )
    backend_model.__calliope_run_config = AttrDict.from_yaml_string(
        model_data.attrs["run_config"]
    )

    for k, v in model_data_dict["data"].items():
        _default = backend_model.__calliope_defaults.get(k, None)
        _kwargs = {
            "initialize": v,
            "mutable": True,
            "within": getattr(po, get_domain(model_data[k], default=_default)),
        }
        _kwargs["default"] = _default
        # In operate mode, e.g. energy_cap is a parameter, not a decision variable,
        # so add those in.
        if (
            backend_model.__calliope_run_config["mode"] == "operate"
            and model_data[k].attrs.get("operate_param") == 1
        ):
            dims = [getattr(backend_model, model_data_dict["dims"][k][0])]
        else:
            dims = [getattr(backend_model, i) for i in model_data_dict["dims"][k]]
        setattr(backend_model, k, po.Param(*dims, **_kwargs))

    for option_name, option_val in backend_model.__calliope_run_config[
        "objective_options"
    ].items():
        if option_name == "cost_class":
            objective_cost_class = {
                k: v for k, v in option_val.items() if k in backend_model.costs
            }

            backend_model.objective_cost_class = po.Param(
                backend_model.costs,
                initialize=objective_cost_class,
                mutable=True,
                within=po.Reals,
            )
        else:
            setattr(backend_model, "objective_" + option_name, option_val)

    # Variables
    load_function("calliope.backend.pyomo.variables.initialize_decision_variables")(
        backend_model
    )

    # Constraints
    constraints_to_add = [
        i.split(".py")[0]
        for i in os.listdir(constraints.__path__[0])
        if not i.startswith("_") and not i.startswith(".")
    ]

    # The list is sorted to ensure that some constraints are added after pyomo
    # expressions have been created in other constraint files.
    # Ordering is given by the number assigned to the variable ORDER within each
    # file (higher number = added later).
    try:
        constraints_to_add.sort(
            key=lambda x: load_function(
                "calliope.backend.pyomo.constraints." + x + ".ORDER"
            )
        )
    except AttributeError as e:
        raise AttributeError(
            "{}. This attribute must be set to an integer value based "
            "on the order in which the constraints in the file {}.py should be "
            "loaded relative to constraints in other constraint files. If order "
            "does not matter, set ORDER to a value of 10.".format(
                e.args[0], e.args[0].split(".")[-1].split("'")[0]
            )
        )

    logger.info(
        "constraints are loaded in the following order: {}".format(constraints_to_add)
    )

    for c in constraints_to_add:
        logger.info(f"creating {c} constraints")
        load_function("calliope.backend.pyomo.constraints." + c + ".load_constraints")(
            backend_model
        )

    # FIXME: Optional constraints
    # optional_constraints = model_data.attrs['constraints']
    # if optional_constraints:
    #     for c in optional_constraints:
    #         self.add_constraint(load_function(c))

    # Objective function
    # FIXME re-enable loading custom objectives

    # fetch objective function by name, pass through objective options
    # if they are present
    objective_function = (
        "calliope.backend.pyomo.objective."
        + backend_model.__calliope_run_config["objective"]
    )
    load_function(objective_function)(backend_model)

    model_data = string_to_datetime(backend_model, model_data)

    return backend_model


def solve_model(
    backend_model,
    solver,
    solver_io=None,
    solver_options=None,
    save_logs=False,
    opt=None,
    **solve_kwargs,
):
    """
    Solve a Pyomo model using the chosen solver and all necessary solver options

    Returns a Pyomo results object
    """
    if opt is None:
        opt = SolverFactory(solver, solver_io=solver_io)
        if "persistent" in solver:
            solve_kwargs.update({"save_results": False, "load_solutions": False})
            opt.set_instance(backend_model)

    if solver_options:
        for k, v in solver_options.items():
            opt.options[k] = v

    if save_logs:
        solve_kwargs.update({"symbolic_solver_labels": True, "keepfiles": True})
        os.makedirs(save_logs, exist_ok=True)
        TempfileManager.tempdir = save_logs  # Sets log output dir
    if "warmstart" in solve_kwargs.keys() and solver in ["glpk", "cbc"]:
        if solve_kwargs.pop("warmstart") is True:
            exceptions.warn(
                "The chosen solver, {}, does not suport warmstart, which may "
                "impact performance.".format(solver)
            )

    with redirect_stdout(LogWriter(logger, "debug", strip=True)):
        with redirect_stderr(LogWriter(logger, "error", strip=True)):
            # Ignore most of gurobipy's logging, as it's output is
            # already captured through STDOUT
            logging.getLogger("gurobipy").setLevel(logging.ERROR)
            if "persistent" in solver:
                results = opt.solve(tee=True, **solve_kwargs)
            else:
                results = opt.solve(backend_model, tee=True, **solve_kwargs)
    return results, opt


def load_results(backend_model, results, opt):
    """Load results into model instance for access via model variables."""
    termination = results.solver.termination_condition

    if termination == pe.TerminationCondition.optimal:
        try:
            opt.load_vars()
            this_result = True
        except AttributeError:
            this_result = backend_model.solutions.load_from(results)

    if termination != pe.TerminationCondition.optimal or this_result is False:
        logger.critical("Problem status:")
        for l in str(results.Problem).split("\n"):
            logger.critical(l)
        logger.critical("Solver status:")
        for l in str(results.Solver).split("\n"):
            logger.critical(l)

        if termination != pe.TerminationCondition.optimal:
            message = "Model solution was non-optimal."
        else:
            message = "Could not load results into model instance."

        exceptions.BackendWarning(message)

    return str(termination)


def get_result_array(backend_model, model_data):
    """
    From a Pyomo model object, extract decision variable data and return it as
    an xarray Dataset. Any rogue input parameters that are constructed inside
    the backend (instead of being passed by calliope.Model().inputs) are also
    added to calliope.Model()._model_data in-place.
    """
    all_variables = {
        i.name: get_var(backend_model, i.name)
        for i in backend_model.component_objects()
        if isinstance(i, po.base.Var)
    }
    all_variables.update(
        {
            i.name: get_var(backend_model, i.name, expr=True)
            for i in backend_model.component_objects()
            if isinstance(i, po.base.Expression)
        }
    )
    # Get any parameters that did not appear in the user's model.inputs Dataset
    all_params = {
        i.name: get_var(backend_model, i.name, expr=True)
        for i in backend_model.component_objects(ctype=po.base.param.IndexedParam)
        if i.name not in model_data.data_vars.keys() and "objective_" not in i.name
    }

    results = string_to_datetime(
        backend_model, reorganise_xarray_dimensions(xr.Dataset(all_variables))
    )

    if all_params:
        additional_inputs = reorganise_xarray_dimensions(xr.Dataset(all_params))
        for var in additional_inputs.data_vars:
            additional_inputs[var].attrs["is_result"] = 0
        model_data.update(additional_inputs)
        model_data = string_to_datetime(backend_model, model_data)

    return results
