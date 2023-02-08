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
)
import functools
import os
from contextlib import redirect_stdout, redirect_stderr
import logging

import xarray as xr
import pandas as pd
import pyomo.core as po
import pyomo.environ as pe

from pyomo.common.tempfiles import TempfileManager
import numpy as np
from pandas.api.types import is_numeric_dtype

from calliope.exceptions import BackendError, BackendWarning
from calliope.exceptions import warn as model_warn
from calliope.core.util.logging import LogWriter
from calliope.backend import parsing


class ParsedComponents(TypedDict):
    variables: dict[str, parsing.ParsedVariable]
    constraints: dict[str, parsing.ParsedConstraint]
    objectives: dict[str, parsing.ParsedObjective]


T = TypeVar("T")


class BackendModel(ABC, Generic[T]):
    logger = logging.getLogger(__name__)

    def __init__(
        self, parsed_components: ParsedComponents, defaults: dict, instance: T
    ):

        self.defaults = defaults
        self.parsed_components = parsed_components

        self.instance = instance

        self._datetime_data: set
        self._warnings: set = set()
        self._lookup_constraint_components: dict[str, pd.Series]
        self._lookup_param_or_var_set_names: dict[str, Optional[tuple]]

        self.sets: set[str] = set()
        self.parameters: set[str] = set()
        self.variables: set[str] = set()
        self.constraints: set[str] = set()
        self.objectives: set[str] = set()

    @abstractmethod
    def add_set(self, set_name: str, set_values: xr.DataArray) -> None:
        pass

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
    def get_set(self, set_name: str):
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
        self, obj_name: str, index_lookup=Optional[tuple]
    ) -> Any:
        if obj_name in self.variables:
            return self.get_variable(obj_name, index_lookup)
        else:
            return self.get_parameter(obj_name, index_lookup)

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
        if obj in getattr(self, obj_type):
            raise BackendError(
                f"Trying to add already existing {obj_type} {obj} to backend model."
            )
        elif hasattr(self.instance, obj):
            found_obj = getattr(self.instance, obj)
            raise BackendError(
                f"Trying to add `{obj}` to backend model, "
                f"but a method of type {type(found_obj)} with that name already exists."
            )


class PyomoBackendModel(BackendModel):
    def __init__(self, parsed_components: ParsedComponents, defaults: dict):
        BackendModel.__init__(
            self,
            parsed_components=parsed_components,
            defaults=defaults,
            instance=pe.ConcreteModel(),
        )

    def add_set(self, set_name: str, set_values: xr.DataArray) -> None:
        self._raise_error_on_preexistence(set_name, po.Set)
        setattr(
            self.instance,
            set_name,
            po.Set(initialize=set_values.to_index(), ordered=True),
        )
        self.sets.add(set_name)

    def add_parameter(
        self, parameter_name: str, parameter_values: xr.DataArray, default: Any = None
    ) -> None:
        with pd.option_context("mode.use_inf_as_na", True):
            param_data_dict = parameter_values.to_series().dropna().to_dict()

        # We know this is an issue. We need to remove "name" as a Calliope input param
        if parameter_name == "name":
            return None
        else:
            self._raise_error_on_preexistence(parameter_name, po.Param)

        kwargs_ = {
            "initialize": param_data_dict,
            "mutable": True,
            "within": getattr(po, self._get_domain(parameter_values)),
        }
        if not pd.isnull(default):
            kwargs_["default"] = default
        dims = [self.get_set(str(set_name)) for set_name in parameter_values.dims]

        setattr(self.instance, parameter_name, po.Param(*dims, **kwargs_))

        self.parameters.add(parameter_name)
        self._lookup_param_or_var_set_names[parameter_name] = tuple(
            parameter_values.dims
        )
        # TODO: add objective options and bigM as model data variables
        """
        elif parameter is None:
            setattr(self.instance, parameter_name, po.Param(*dims, **kwargs_))
            self.instance = po.Param(
                self.get_set("costs"),
                initialize=objective_cost_class,
                mutable=True,
                within=po.Reals,
            )
        self.instance.bigM = po.Param(
            initialize=self.run_config.get("bigM", 1e10),
            mutable=True,
            within=po.NonNegativeReals,
        )
        """

    def add_constraint(
        self,
        constraint_name: str,
        constraint_rule: Callable,
        constraint_subset: pd.Index,
    ) -> None:
        self._raise_error_on_preexistence(constraint_name, po.Constraint)

        setattr(
            self.instance,
            constraint_name,
            po.Constraint(
                constraint_subset,
                rule=constraint_rule,
            ),
        )
        self.constraints.add(constraint_name)

    def add_variable(
        self,
        variable_name: str,
        variable_subset: Optional[pd.Index],
        bounds_dict: dict[str, SupportsFloat],
        domain: str,
    ) -> None:

        self._raise_error_on_preexistence(variable_name, po.Var)
        bounds = self._get_capacity_bounds(bounds_dict)

        setattr(
            self.instance,
            variable_name,
            po.Var(variable_subset, domain=getattr(po, domain), bounds=bounds),
        )
        if variable_subset is not None:
            dimensions = tuple(variable_subset.index.names)
        else:
            dimensions = None
        self._lookup_param_or_var_set_names[variable_name] = dimensions
        self.variables.add(variable_name)

    def add_objective(
        self,
        objective_name: str,
        objective_rule: Callable,
        domain: str,
        sense: Literal["minimise", "maximise"] = "minimise",
    ) -> None:
        self._raise_error_on_preexistence(objective_name, po.Objective)
        setattr(
            self.instance,
            objective_name,
            po.Objective(
                sense=getattr(po, sense),
                rule=objective_rule,
            ),
        )
        self.instance.obj.domain = getattr(po, domain)
        self.objectives.add(objective_name)

    def get_set(self, set_name: str) -> po.Set:
        return getattr(self.instance, set_name)

    def get_parameter(
        self, parameter_name: str, index_lookup: Optional[tuple] = None
    ) -> Any:
        """
        Get an input parameter held in a Pyomo object, or held in the defaults
        dictionary if that Pyomo object doesn't exist.

        Parameters
        ----------
        backend_model : Pyomo model instance
        var : str
        dims : single value or tuple

        """
        try:
            parameter = getattr(self.instance, parameter_name)
        except AttributeError:  # i.e. parameter doesn't exist at all
            self._warnings.add(
                f"Parameter {parameter_name} not added to backend model, leading to default lookup"
            )
            return self.defaults[parameter_name]
        else:
            if index_lookup is not None:
                return parameter[index_lookup]
            else:
                return parameter

    def get_parameter_array(
        self, parameter_name: str, sparse: bool = False
    ) -> xr.DataArray:
        parameter = self.get_parameter(parameter_name)
        if sparse:
            if self._invalid(parameter.default()):
                parameter_array = pd.Series(parameter._data).apply(
                    lambda x: po.value(x) if not self._invalid(x) else np.nan
                )
            else:
                parameter_array = pd.Series(parameter.extract_values_sparse())
        else:
            parameter_array = pd.Series(parameter.extract_values())

        return self._set_and_reorder_array_index(parameter_array.rename(parameter_name))

    def get_constraint(self, constraint_name: str) -> Optional[pd.DataFrame]:
        constraint_idx: pd.Series = self.parsed_components["constraints"][
            constraint_name
        ].index
        if constraint_idx is None or constraint_idx.empty:
            return None
        constraint_dict = dict()
        for idx, sub_constraint_name in constraint_idx.iterrows():
            constraint = getattr(self.instance, sub_constraint_name)[idx]
            constraint_dict[idx] = {
                "lower": po.value(constraint.lower),
                # TODO: return evaluated body value if requested (and optimisation is complete)
                "body": constraint.body.to_string(),
                "upper": po.value(constraint.upper),
            }
        return pd.DataFrame(constraint_dict)

    def get_variable(
        self, variable_name: str, index_lookup: Optional[tuple] = None
    ) -> Any:
        variable = getattr(self.instance, variable_name)
        if index_lookup is not None:
            return variable[index_lookup]

    def get_variable_array(self, variable_name: str) -> xr.DataArray:
        variable = self.get_variable(variable_name)
        variable_array = pd.Series(variable.extract_values())
        if variable_array.empty:
            raise BackendError(f"Variable {variable_array} has no data.")

        return self._set_and_reorder_array_index(variable_array.rename(variable_name))

    def _set_and_reorder_array_index(self, array: pd.Series) -> xr.DataArray:

        set_names = self._lookup_param_or_var_set_names[array.name]
        result_with_set_names = array.rename_axis(index=set_names)

        da = xr.DataArray.from_series(result_with_set_names)

        # Order of dimension set items is sorted by pd.Series above and may no longer match
        # the input calliope data set order. So we reorder the array dimensions here.
        return da.reindex(
            **{
                set_name: self.get_set(set_name)._ordered_values
                for set_name in set_names
            }
        )

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
        opt = po.SolverFactory(solver, solver_io=solver_io)

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

        termination = results.solver.termination_condition

        if termination == pe.TerminationCondition.optimal:
            self.instance.solutions.load_from(results)

        else:
            self.logger.critical("Problem status:")
            for line in str(results.Problem).split("\n"):
                self.logger.critical(line)
            self.logger.critical("Solver status:")
            for line in str(results.Solver).split("\n"):
                self.logger.critical(line)

            BackendWarning("Model solution was non-optimal.")

        return str(termination)

    @staticmethod
    def _get_domain(var: xr.DataArray) -> str:
        def check_sign(var):
            if re.match("resource|node_coordinates|cost*", var.name):
                return ""
            else:
                return "NonNegative"

        if var.dtype.kind == "b":
            return "Boolean"
        elif is_numeric_dtype(var.dtype):
            return check_sign(var) + "Reals"
        else:
            return "Any"

    @staticmethod
    def _invalid(val: Any) -> bool:
        if isinstance(val, po.base.param._ParamData):
            return val._value == po.Param.NoValue or po.value(val) is None
        elif val == po.Param.NoValue:
            return True
        else:
            return pd.isnull(val)

    def _get_capacity_bounds(self, bounds):
        def __get_bounds(_, *idx):
            def ___get_bound(bound):
                if bounds.get(bound) is not None:
                    return self.get_param(bounds.get(bound))[idx]
                else:
                    return None

            scale = ___get_bound("scale")
            equals_ = ___get_bound("equals")
            min_ = ___get_bound("min")
            max_ = ___get_bound("max")

            if not self._invalid(equals_):
                if not self._invalid(scale):
                    equals_ *= scale
                bound_tuple = (equals_, equals_)
            else:
                if self._invalid(min_):
                    min_ = None
                if self._invalid(max_):
                    max_ = None
                bound_tuple = (min_, max_)

            if not self._invalid(scale):
                bound_tuple = tuple(i * scale for i in bound_tuple)

            return bound_tuple

        return __get_bounds
