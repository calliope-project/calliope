"""
Copyright (C) 2013-2019 Calliope contributors listed in AUTHORS.
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
import pyomo.environ  # pylint: disable=unused-import,import-error

# TempfileManager is required to set log directory
from pyutilib.services import TempfileManager  # pylint: disable=import-error

from calliope.backend.pyomo.util import get_var, get_domain, mask, within
from calliope.backend.pyomo import constraints
from calliope.core.util.tools import load_function
from calliope.core.util.logging import LogWriter
from calliope.core.util.dataset import reorganise_xarray_dimensions
from calliope import exceptions
from calliope.core.attrdict import AttrDict

logger = logging.getLogger(__name__)


def build_sets(model_data, backend_model):
    for coord_name, coord_vals in model_data.coords.items():
        setattr(
            backend_model,
            coord_name,
            po.Set(initialize=coord_vals.to_index(), ordered=True),
        )


def build_params(model_data, backend_model):
    # "Parameters"

    backend_model.__calliope_defaults = AttrDict.from_yaml_string(
        model_data.attrs["defaults"]
    )
    backend_model.__calliope_run_config = AttrDict.from_yaml_string(
        model_data.attrs["run_config"]
    )

    for k, v in model_data.data_vars.items():
        if v.attrs["is_result"] == 0 or v.attrs.get("operate_param", 0) == 1:
            with pd.option_context("mode.use_inf_as_na", True):
                _kwargs = {
                    "initialize": v.to_series().dropna().to_dict(),
                    "mutable": True,
                    "within": getattr(po, get_domain(v)),
                }
            if not pd.isnull(backend_model.__calliope_defaults.get(k, None)):
                _kwargs["default"] = backend_model.__calliope_defaults[k]
            # In operate mode, e.g. energy_cap is a parameter, not a decision variable,
            # so add those in.
            if (
                backend_model.__calliope_run_config["mode"] == "operate"
                and v.attrs.get("operate_param") == 1
            ):
                dims = [getattr(backend_model, v.dims[0])]
            else:
                dims = [getattr(backend_model, i) for i in v.dims]
            if k == "name":
                continue
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


def build_variables(backend_model, masks):
    for k, v in masks.filter_by_attrs(variables=1).data_vars.items():
        setattr(
            backend_model, k, po.Var(mask(v), domain=getattr(po, v.domain)),
        )
        if k == "unmet_demand":
            backend_model.bigM = backend_model.__calliope_run_config.get("bigM", 1e10)


def build_constraints(backend_model, k, v):
    print(k, v.nbytes)
    setattr(
        backend_model,
        f"{k}_constraint",
        po.Constraint(mask(v), rule=getattr(constraints, f"{k}_constraint_rule"),),
    )


def build_expressions(backend_model, masks):
    for k, v in masks.filter_by_attrs(expressions=1).data_vars.items():
        setattr(
            backend_model, k, po.Expression(mask(v), initialize=0.0),
        )


def build_objective(backend_model):
    objective_function = (
        "calliope.backend.pyomo.objective."
        + backend_model.__calliope_run_config["objective"]
    )
    load_function(objective_function)(backend_model)


def generate_model(model_data, masks):
    """
    Generate a Pyomo model.

    """
    backend_model = po.ConcreteModel()

    build_sets(model_data, backend_model)
    build_params(model_data, backend_model)
    build_variables(backend_model, masks)
    build_expressions(backend_model, masks)
    for k, v in masks.filter_by_attrs(constraints=1).data_vars.items():
        build_constraints(backend_model, k, v)
    build_objective(backend_model)
    # FIXME: Optional constraints
    # optional_constraints = model_data.attrs['constraints']
    # if optional_constraints:
    #     for c in optional_constraints:
    #         self.add_constraint(load_function(c))

    # Objective function
    # FIXME re-enable loading custom objectives

    # fetch objective function by name, pass through objective options
    # if they are present

    return backend_model


def solve_model(
    backend_model,
    solver,
    solver_io=None,
    solver_options=None,
    save_logs=False,
    **solve_kwargs,
):
    """
    Solve a Pyomo model using the chosen solver and all necessary solver options

    Returns a Pyomo results object
    """
    opt = SolverFactory(solver, solver_io=solver_io)

    if solver_options:
        for k, v in solver_options.items():
            opt.options[k] = v

    if save_logs:
        solve_kwargs.update({"symbolic_solver_labels": True, "keepfiles": True})
        os.makedirs(save_logs, exist_ok=True)
        TempfileManager.tempdir = save_logs  # Sets log output dir
    if "warmstart" in solve_kwargs.keys() and solver in ["glpk", "cbc"]:
        exceptions.warn(
            "The chosen solver, {}, does not suport warmstart, which may "
            "impact performance.".format(solver)
        )
        del solve_kwargs["warmstart"]

    with redirect_stdout(LogWriter(logger, "debug", strip=True)):
        with redirect_stderr(LogWriter(logger, "error", strip=True)):
            # Ignore most of gurobipy's logging, as it's output is
            # already captured through STDOUT
            logging.getLogger("gurobipy").setLevel(logging.ERROR)
            results = opt.solve(backend_model, tee=True, **solve_kwargs)
    return results


def load_results(backend_model, results):
    """Load results into model instance for access via model variables."""
    not_optimal = str(results["Solver"][0]["Termination condition"]) != "optimal"
    this_result = backend_model.solutions.load_from(results)

    if this_result is False or not_optimal:
        logger.critical("Problem status:")
        for l in str(results.Problem).split("\n"):
            logger.critical(l)
        logger.critical("Solver status:")
        for l in str(results.Solver).split("\n"):
            logger.critical(l)

        if not_optimal:
            message = "Model solution was non-optimal."
        else:
            message = "Could not load results into model instance."

        exceptions.BackendWarning(message)

    return str(results["Solver"][0]["Termination condition"])


def get_result_array(backend_model, model_data, masks):
    """
    From a Pyomo model object, extract decision variable data and return it as
    an xarray Dataset. Any rogue input parameters that are constructed inside
    the backend (instead of being passed by calliope.Model().inputs) are also
    added to calliope.Model()._model_data in-place.
    """
    all_variables = {
        i.name: get_var(backend_model, i.name, dims=masks[i.name].dims)
        for i in backend_model.component_objects()
        if isinstance(i, po.base.Var)
    }

    # Get any parameters that did not appear in the user's model.inputs Dataset
    all_params = {
        i.name: get_var(backend_model, i.name)
        for i in backend_model.component_objects()
        if isinstance(i, po.base.param.IndexedParam)
        and i.name not in model_data.data_vars.keys()
        and "objective_" not in i.name
    }

    results = reorganise_xarray_dimensions(xr.Dataset(all_variables))

    if all_params:
        additional_inputs = reorganise_xarray_dimensions(xr.Dataset(all_params))
        for var in additional_inputs.data_vars:
            additional_inputs[var].attrs["is_result"] = 0
        model_data.update(additional_inputs)

    return results
