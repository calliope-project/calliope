from __future__ import annotations

from abc import ABC, abstractmethod
import re
from typing import (
    Any,
    SupportsFloat,
    Callable,
    Optional,
    Literal,
    TypedDict,
    TypeVar,
    Generic,
    Union,
)
import functools
import os
from contextlib import redirect_stdout, redirect_stderr
import logging

import xarray as xr
import pandas as pd
import pyomo.core as po
import pyomo.environ as pe
import pyomo.kernel as pmo
from pyomo.opt import SolverFactory

from pyomo.common.tempfiles import TempfileManager
import numpy as np
from pandas.api.types import is_numeric_dtype

from calliope.exceptions import BackendError, BackendWarning
from calliope.exceptions import warn as model_warn
from calliope.core.util.logging import LogWriter
from calliope.backend import parsing
from calliope.core.util.observed_dict import UpdateObserverDict


class ParsedComponents(TypedDict):
    variables: dict[str, parsing.ParsedVariable]
    constraints: dict[str, parsing.ParsedConstraint]
    objectives: dict[str, parsing.ParsedObjective]
    expressions: dict[str, parsing.ParsedExpression]


T = TypeVar("T")


class BackendModel(ABC, Generic[T]):
    logger = logging.getLogger(__name__)

    def __init__(
        self, parsed_components: ParsedComponents, defaults: dict, instance: T
    ):

        self.defaults = defaults.copy()
        self.parsed_components = parsed_components

        self.instance = instance
        self.dataset = xr.Dataset()

        self._datetime_data: set
        self._warnings: set = set()
        self._lookup_constraint_components: dict[str, pd.Series]
        self._lookup_param_or_var_set_names: dict[str, Optional[tuple]]

    @abstractmethod
    def add_parameter(
        self, parameter_name: str, parameter_values: xr.DataArray
    ) -> None:
        pass

    @abstractmethod
    def add_constraint(
        self,
        constraint_name: str,
        constraint_rule: Callable,
        constraint_subset: pd.Index,
    ) -> None:
        pass

    @abstractmethod
    def add_variable(
        self,
        variable_name: str,
        variable_subset: Optional[pd.Index],
        bounds_dict: dict[str, SupportsFloat],
        domain: str,
    ) -> None:
        pass

    @abstractmethod
    def add_objective(
        self,
        objective_name: str,
        objective_rule: Callable,
        domain: str,
        sense: Literal["minimise", "maximise"] = "minimise",
    ) -> None:
        pass

    @abstractmethod
    def get_parameter(
        self, parameter_name: str, index_lookup: Optional[tuple] = None
    ) -> Any:
        pass

    @abstractmethod
    def get_constraint(self, constraint_name: str) -> Optional[pd.DataFrame]:
        pass

    @abstractmethod
    def get_variable(
        self, variable_name: str, index_lookup: Optional[tuple] = None
    ) -> Any:
        pass

    @functools.cache
    def get_parameter_or_variable(
        self, obj_name: str, subset: Optional[pd.Index] = None
    ) -> Any:
        if obj_name in self.variables.keys():
            return self.get_variable(obj_name, subset)
        else:
            return self.get_parameter(obj_name, subset)

    @abstractmethod
    def solve(
        self,
        solver: str,
        solver_io: Optional[str] = None,
        solver_options: Optional[dict] = None,
        save_logs: Optional[str] = None,
        warmstart: bool = False,
        **solve_kwargs,
    ):
        pass

    def load_results(self):

        all_variables = {
            var.name: self.get_variable(var.name, as_array=True)
            for var in self.variables
        }

        results = self._string_to_datetime(xr.Dataset(all_variables))

        return results

    def _raise_error_on_preexistence(self, obj: str, obj_type: str):
        if obj in getattr(self, obj_type).keys():
            raise BackendError(
                f"Trying to add already existing {obj_type} {obj} to backend model."
            )


class PyomoBackendModel(BackendModel):
    def __init__(self, parsed_components: ParsedComponents, defaults: dict):
        BackendModel.__init__(
            self,
            parsed_components=parsed_components,
            defaults=defaults,
            instance=pmo.block(),
        )
        self.parameters = self.instance.parameters = pmo.parameter_dict()
        self.variables = self.instance.variables = pmo.variable_dict()
        self.expressions = self.instance.expressions = pmo.expression_dict()
        self.constraints = self.instance.constraints = pmo.constraint_dict()
        self.objectives = self.instance.objectives = pmo.objective_dict()

    def generate_backend_dataset(self, model_data: xr.Dataset, defaults: dict, run_config: UpdateObserverDict) -> None:
        dataset = xr.Dataset(model_data.coords)
        for param_name, param_data in model_data.data_vars.items():
            default_val = defaults.get(param_name, np.nan)
            dataset[param_name] = self.add_parameter(
                param_name, param_data, default_val
            )
        for param_name, default_val in defaults.items():
            if param_name in dataset:
                continue
            dataset[param_name] = self.add_parameter(
                param_name, xr.DataArray(default_val), use_inf_as_na=False
            )

        for option_name, option_val in run_config["objective_options"].items():
            if option_name == "cost_class":

                objective_cost_class = {
                    k: v for k, v in option_val.items() if k in model_data.costs
                }
                dataset["objective_cost_class"] = self.add_parameter(
                    "objective_cost_class",
                    xr.DataArray.from_series(
                        pd.Series(objective_cost_class).rename_axis(index="costs")
                    ),
                )
            else:
                dataset["objective_" + option_name] = self.add_parameter("objective_" + option_name, xr.DataArray(option_val))
        dataset["bigM"] = self.add_parameter("bigM", xr.DataArray(run_config.get("bigM", 1e10)))

        self.dataset = dataset

    def add_parameter(
        self,
        parameter_name: str,
        parameter_values: xr.DataArray,
        default: Any = np.nan,
        use_inf_as_na: bool = False,
    ) -> xr.DataArray:

        self._raise_error_on_preexistence(parameter_name, "parameters")

        parameter_da = self.apply_func(
            self._to_pyomo_param, parameter_values, default=default, use_inf_as_na=use_inf_as_na
        )
        if not parameter_values.shape and parameter_da.isnull().all():
            parameter_da = parameter_da.astype(float)

        if parameter_da.isnull().all():
            return parameter_da
        if not parameter_values.shape:
            parameter_dict = parameter_da.item()
        else:
            parameter_dict = pmo.parameter_dict(self.to_dict(parameter_da))

        self.instance.parameters[parameter_name] = parameter_dict
        return parameter_da

    def _add_constraint_or_expression(
        self,
        model_data: xr.Dataset,
        parsed_component: Union[parsing.ParsedConstraint, parsing.ParsedExpression],
        component_type_group: Literal["constraints", "expressions"],
    ) -> xr.DataArray:
        self._raise_error_on_preexistence(parsed_component.name, component_type_group)
        component_type = component_type_group.removesuffix("s")
        top_level_imask = parsed_component.evaluate_where(model_data, self.defaults, self)
        component_da = xr.DataArray().where(top_level_imask).astype(np.dtype("O"))

        if not top_level_imask.any():
            return xr.DataArray(None)

        for element in parsed_component.equations:

            imask = element.evaluate_where(model_data, self.defaults, self, top_level_imask)
            if not imask.any():
                continue

            expr = element.evaluate_expression(model_data, self)
            if component_da.where(imask).notnull().any():
                subset_overlap = (
                    component_da.where(imask).to_series().dropna().index
                )

                raise BackendError(
                    f"Trying to set two equations for the same index of {component_type}"
                    f"`{parsed_component.name}`:\n{subset_overlap}"
                )
            if component_type_group == "constraints":
                lhs, op, rhs = expr
                to_fill = self.apply_func(
                    self._to_pyomo_constraint,
                    imask,
                    xr.DataArray(lhs).squeeze(drop=True),
                    xr.DataArray(rhs).squeeze(drop=True),
                    op=op
                )
            elif component_type_group == "expressions":
                to_fill = self.apply_func(self._to_pyomo_expression, imask, expr.squeeze(drop=True))

            component_da = component_da.fillna(to_fill)
        return component_da.rename(parsed_component.name).assign_attrs(
            {component_type: 1}
        )

    def add_constraint(
        self,
        model_data: xr.Dataset,
        parsed_constraint: parsing.ParsedConstraint,
    ) -> Optional[xr.DataArray]:
        constraint_da = self._add_constraint_or_expression(
            model_data, parsed_constraint, "constraints"
        )
        if constraint_da.isnull().all():
            return None

        if constraint_da.shape == 0:
            self.instance.constraints[parsed_constraint.name] = constraint_da.item()
        else:
            self.instance.constraints[parsed_constraint.name] = pmo.constraint_dict(self.to_dict(constraint_da))
        return constraint_da

    def add_expression(
        self,
        model_data: xr.Dataset,
        parsed_expression: parsing.ParsedExpression,
    ) -> Optional[xr.DataArray]:
        expression_da = self._add_constraint_or_expression(
            model_data, parsed_expression, "expressions"
        )
        if expression_da.notnull().any():
            if expression_da.shape == 0:
                expression_dict = expression_da.item()
            else:
                expression_dict = pmo.expression_dict(self.to_dict(expression_da))

            self.instance.expressions[parsed_expression.name] = expression_dict

        return expression_da

    def apply_func(self, func: Callable, *args, **kwargs) -> xr.DataArray:
        return xr.apply_ufunc(
            func, *args, kwargs=kwargs, vectorize=True, keep_attrs=True, output_dtypes=[np.dtype("O")]
        )

    def to_dict(self, da: xr.DataArray) -> dict:
        da_stack = da.stack(all_dims=da.dims).dropna("all_dims")
        da_dict = da_stack.to_dict()
        return {da_dict["coords"]["all_dims"]["data"][i]: data for i, data in enumerate(da_dict["data"])}

    def add_variable(
        self,
        model_data: xr.Dataset,
        parsed_variable: parsing.ParsedVariable,
    ) -> None:

        #self._raise_error_on_preexistence(parsed_variable.name, "variables")

        imask = parsed_variable.evaluate_where(model_data, self.defaults, self)

        if imask is None:
            return None
        domain = parsed_variable._unparsed.get("domain", "real")
        domain_type = getattr(pmo, f"{domain.title()}Set")

        ub, lb = self._get_capacity_bounds(parsed_variable.bounds, imask)
        variable_da = self.apply_func(self._to_pyomo_variable, imask, ub, lb, domain_type=domain_type)

        self.instance.variables[parsed_variable.name] = pmo.variable_dict(self.to_dict(variable_da))

        return variable_da.rename(parsed_variable.name).assign_attrs({"variable": 1})

    def add_objective(
        self,
        model_data: xr.Dataset,
        parsed_objective: parsing.ParsedObjective,
    ) -> None:
        self._raise_error_on_preexistence(parsed_objective.name, "objectives")
        sense_dict = {"minimize": 1, "maximize": -1}
        n_valid_exprs = 0
        for equation in parsed_objective.equations:
            imask = equation.evaluate_where(model_data, self.defaults, self)
            if imask.any():
                expr = equation.evaluate_expression(model_data, self).item()
                n_valid_exprs += 1
        if n_valid_exprs > 1:
            raise BackendError(
                f"More than one {parsed_objective.name} objective is valid for this "
                "optimisation problem; only one is allowed."
            )

        objective = pmo.objective(expr, sense=sense_dict[parsed_objective.sense])
        self.instance.objectives[parsed_objective.name] = objective

        return (
            xr.DataArray(objective)
            .rename(parsed_objective.name)
            .assign_attrs({"objective": 1})
        )

    def get_parameter(
        self, parameter_name: str, subset: Optional[xr.DataArray] = None
    ) -> Optional[xr.DataArray]:
        parameter = self.dataset.get(parameter_name)
        if parameter is not None and subset is not None:
            parameter = parameter.where(subset)
        return parameter

    def get_parameter_array(self, parameter_name: str) -> Optional[xr.DataArray]:
        parameter = self.get_parameter(parameter_name)
        if isinstance(parameter, xr.DataArray):
            return self.apply_func(self._from_pyomo_obj, parameter)
        else:
            return parameter

    def get_constraint(
        self, constraint_name: str, eval_body: bool = False
    ) -> Optional[pd.DataFrame]:
        if constraint_name not in self.instance.constraints.keys():
            return None
        constraint_dict = dict()
        for idx, sub_constraint in self.instance.constraints[constraint_name].items():
            lb = sub_constraint.lb
            ub = sub_constraint.ub
            if eval_body:
                try:
                    body = sub_constraint.body()
                except ValueError:
                    body = sub_constraint.body.to_string()
            else:
                body = sub_constraint.body.to_string()
            constraint_dict[idx] = {"lb": lb, "body": body, "ub": ub}
        return pd.DataFrame(constraint_dict).T

    def get_variable(
        self, variable_name: str, subset: Optional[xr.DataArray] = None
    ) -> Optional[xr.DataArray]:
        variable = self.dataset.get(variable_name)
        if variable is not None and subset is not None:
            variable = variable.where(subset)
        return variable

    def get_variable_array(self, variable_name: str) -> Optional[xr.DataArray]:
        variable = self.get_variable(variable_name)
        if variable is not None:
            return self.apply_func(self._from_pyomo_obj, variable)

        return variable

    def solve(
        self,
        solver: str,
        solver_io: Optional[str] = None,
        solver_options: Optional[dict] = None,
        save_logs: Optional[str] = None,
        warmstart: bool = False,
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

        if save_logs is not None:
            solve_kwargs.update({"symbolic_solver_labels": True, "keepfiles": True})
            os.makedirs(save_logs, exist_ok=True)
            TempfileManager.tempdir = save_logs  # Sets log output dir
        if warmstart and solver in ["glpk", "cbc"]:
            model_warn(
                "The chosen solver, {}, does not suport warmstart, which may "
                "impact performance.".format(solver)
            )
            warmstart = False

        with redirect_stdout(LogWriter(self.logger, "debug", strip=True)):
            with redirect_stderr(LogWriter(self.logger, "error", strip=True)):
                # Ignore most of gurobipy's logging, as it's output is
                # already captured through STDOUT
                logging.getLogger("gurobipy").setLevel(logging.ERROR)
                results = opt.solve(self.instance, tee=True, **solve_kwargs)

        termination = results.solver[0].termination_condition

        if termination == pe.TerminationCondition.optimal:
            self.instance.load_solution(results.solution[0])

        else:
            self.logger.critical("Problem status:")
            for line in str(results.problem[0]).split("\n"):
                self.logger.critical(line)
            self.logger.critical("Solver status:")
            for line in str(results.solver[0]).split("\n"):
                self.logger.critical(line)

            BackendWarning("Model solution was non-optimal.")

        return str(termination)

    def _get_capacity_bounds(self, bounds: dict, imask: xr.DataArray):
        def __get_bound(bound):
            this_bound = bounds.get(bound, None)
            if isinstance(this_bound, str):
                return self.get_parameter(this_bound)
            else:
                return xr.DataArray(self._to_pyomo_param(this_bound))

        scale = __get_bound("scale")
        equals_ = __get_bound("equals")
        min_ = __get_bound("min")
        max_ = __get_bound("max")

        lb = equals_.fillna(min_)
        ub = equals_.fillna(max_)
        if scale.notnull().any():
            lb = lb * scale
            ub = ub * scale

        return ub.fillna(None), lb.fillna(None)

    @staticmethod
    def _to_pyomo_param(
        val: Any, default: Any = np.nan, use_inf_as_na: bool = True
    ) -> Union[type[ObjParameter], np.nan]:
        with pd.option_context("mode.use_inf_as_na", use_inf_as_na):
            if pd.isnull(val):
                if pd.isnull(default):
                    return np.nan
                else:
                    return ObjParameter(default)
            else:
                return ObjParameter(val)

    @staticmethod
    def _to_pyomo_constraint(mask, lhs, rhs, *, op):
        if not mask:
            return np.nan
        elif op == "==":
            return pmo.constraint(expr=lhs == rhs)
        elif op == "<=":
            return pmo.constraint(expr=lhs <= rhs)
        elif op == ">=":
            return pmo.constraint(expr=lhs >= rhs)

    @staticmethod
    def _to_pyomo_expression(mask, val: Any) -> Union[type[pmo.expression], float]:
        if not mask:
            return np.nan
        else:
            return pmo.expression(val)

    @staticmethod
    def _to_pyomo_variable(mask: Union[bool, np.bool_], ub: Any, lb: Any, *, domain_type: str) -> Union[type[pmo.variable], float]:
        if mask:
            return pmo.variable(ub=ub, lb=lb, domain_type=domain_type)
        else:
            return np.nan

    @staticmethod
    def _from_pyomo_obj(val: Any) -> Any:
        if pd.isnull(val):
            return np.nan
        else:
            return val.value


class ObjParameter(pmo.parameter):
    """A non-negative variable."""

    __slots__ = ()

    def __init__(self, value, **kwds):
        assert not pd.isnull(value)
        super(ObjParameter, self).__init__(value, **kwds)
        if "dtype" not in kwds:
            kwds["dtype"] = "O"

    @property
    def dtype(self):
        return "O"

    @dtype.setter
    def dtype(self, dtype):
        if dtype < 0:
            raise ValueError("lower bound must be non-negative")
        # calls the base class property setter
        pmo.parameter.dtype.fset(self, dtype)
