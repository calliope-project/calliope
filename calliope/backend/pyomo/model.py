"""
Copyright (C) since 2013 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

"""

from calliope.backend.pyomo.constraints.capacity import get_capacity_bounds
import logging
import os
from contextlib import redirect_stdout, redirect_stderr

import pandas as pd
import xarray as xr

import pyomo.core as po
from pyomo.opt import SolverFactory

# pyomo.environ is needed for pyomo solver plugins
import pyomo.environ as pe  # pylint: disable=unused-import,import-error

# TempfileManager is required to set log directory
from pyomo.common.tempfiles import TempfileManager  # pylint: disable=import-error

from calliope.backend.pyomo.util import (
    get_var,
    get_domain,
    string_to_datetime,
    datetime_to_string,
)
from calliope.backend.subsets import create_valid_subset
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

    backend_model.__calliope_defaults = model_data.attrs["defaults"]
    backend_model.__calliope_run_config = model_data.attrs["run_config"]

    for k, v in model_data.data_vars.items():
        if v.attrs["is_result"] == 0 or (
            v.attrs.get("operate_param", 0) == 1
            and backend_model.__calliope_run_config["mode"] == "operate"
        ):
            with pd.option_context("mode.use_inf_as_na", True):
                _kwargs = {
                    "initialize": v.to_series().dropna().to_dict(),
                    "mutable": True,
                    "within": getattr(po, get_domain(v)),
                }
            if not pd.isnull(backend_model.__calliope_defaults.get(k, None)):
                _kwargs["default"] = backend_model.__calliope_defaults[k]
            dims = [getattr(backend_model, i) for i in v.dims]
            if hasattr(backend_model, k):
                logger.debug(
                    f"The parameter {k} is already an attribute of the Pyomo model."
                    "It will be prepended with `calliope_` for differentiatation."
                )
                k = f"calliope_{k}"
            setattr(backend_model, k, po.Param(*dims, **_kwargs))

    for option_name, option_val in backend_model.__calliope_run_config[
        "objective_options"
    ].items():
        if option_name == "cost_class":
            # TODO: shouldn't require filtering out unused costs (this should be caught by typedconfig?)
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
    backend_model.bigM = po.Param(
        initialize=backend_model.__calliope_run_config.get("bigM", 1e10),
        mutable=True,
        within=po.NonNegativeReals,
    )


def build_variables(backend_model, model_data, variable_definitions):
    for var_name, var_config in variable_definitions.items():
        subset = create_valid_subset(model_data, var_name, var_config)
        if subset is None:
            continue
        if "bounds" in var_config:
            kwargs = {"bounds": get_capacity_bounds(var_config.bounds)}
        else:
            kwargs = {}

        setattr(
            backend_model,
            var_name,
            po.Var(subset, domain=getattr(po, var_config.domain), **kwargs),
        )


def _load_rule_function(name):
    try:
        return getattr(constraints, name)
    except AttributeError:
        return None


def build_constraints(backend_model, model_data, constraint_definitions):
    for constraint_name, constraint_config in constraint_definitions.items():
        subset = create_valid_subset(model_data, constraint_name, constraint_config)
        if subset is None:
            continue
        setattr(
            backend_model,
            f"{constraint_name}_constraint",
            po.Constraint(
                subset,
                rule=_load_rule_function(f"{constraint_name}_constraint_rule"),
            ),
        )


def build_expressions(backend_model, model_data, expression_definitions):
    build_order_dict = {
        expr: config.get("build_order", 0)
        for expr, config in expression_definitions.items()
    }
    build_order = sorted(build_order_dict, key=build_order_dict.get)

    for expr_name in build_order:
        subset = create_valid_subset(
            model_data, expr_name, expression_definitions[expr_name]
        )
        if subset is None:
            continue
        expression_function = _load_rule_function(f"{expr_name}_expression_rule")
        if expression_function:
            kwargs = dict(rule=expression_function)
        else:
            kwargs = dict(initialize=0.0)
        setattr(backend_model, expr_name, po.Expression(subset, **kwargs))


def build_objective(backend_model):
    objective_function = (
        "calliope.backend.pyomo.objective."
        + backend_model.__calliope_run_config["objective"]
    )
    load_function(objective_function)(backend_model)


def generate_model(model_data):
    """
    Generate a Pyomo model.

    """
    backend_model = po.ConcreteModel()
    # remove pandas datetime from xarrays, to reduce memory usage on creating pyomo objects
    model_data = datetime_to_string(backend_model, model_data)

    subsets_config = model_data.attrs["subsets"]
    build_sets(model_data, backend_model)
    build_params(model_data, backend_model)
    build_variables(backend_model, model_data, subsets_config["variables"])
    build_expressions(backend_model, model_data, subsets_config["expressions"])
    build_constraints(backend_model, model_data, subsets_config["constraints"])
    build_objective(backend_model)
    # FIXME: Optional constraints
    # FIXME re-enable loading custom objectives

    # set datetime data back to datetime dtype
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
    subsets_config = model_data.attrs["subsets"]

    def _get_dim_order(foreach):
        return tuple([i for i in model_data.dims.keys() if i in foreach])

    all_variables = {
        i.name: get_var(
            backend_model,
            i.name,
            dims=_get_dim_order(subsets_config.variables[i.name].foreach),
        )
        for i in backend_model.component_objects(ctype=po.Var)
    }
    # Add in expressions, which are combinations of variables (e.g. costs)
    all_variables.update(
        {
            i.name: get_var(
                backend_model,
                i.name,
                dims=_get_dim_order(subsets_config.expressions[i.name].foreach),
                expr=True,
            )
            for i in backend_model.component_objects(ctype=po.Expression)
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
    results = string_to_datetime(backend_model, results)

    return results
