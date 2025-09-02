# Copyright (C) since 2013 Calliope contributors listed in AUTHORS.
# Licensed under the Apache 2.0 License (see LICENSE file).
"""Pyomo backend functionality."""

from __future__ import annotations

import logging
import re
from abc import ABC
from collections.abc import Iterable
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any, Literal, SupportsFloat, overload

import numpy as np
import pandas as pd
import pyomo.environ as pe  # type: ignore
import pyomo.kernel as pmo  # type: ignore
import xarray as xr
from pyomo.common.tempfiles import TempfileManager  # type: ignore
from pyomo.core import PyomoObject
from pyomo.core.kernel.piecewise_library.transforms import (
    PiecewiseLinearFunction,
    PiecewiseValidationError,
    piecewise_sos2,
)
from pyomo.opt import SolverFactory  # type: ignore
from pyomo.util.model_size import build_model_size_report  # type: ignore

from calliope.backend import backend_model, parsing
from calliope.backend.backend_model import ALL_COMPONENTS_T
from calliope.exceptions import BackendError, BackendWarning
from calliope.exceptions import warn as model_warn
from calliope.schemas import config_schema, math_schema
from calliope.util.logging import LogWriter

LOGGER = logging.getLogger(__name__)

COMPONENT_TRANSLATOR = {
    "parameter": "parameter",
    "variable": "variable",
    "global_expression": "expression",
    "constraint": "constraint",
    "piecewise_constraint": "block",
    "objective": "objective",
}


class PyomoBackendModel(backend_model.BackendModel):
    """Pyomo-specific backend functionality."""

    def __init__(
        self,
        inputs: xr.Dataset,
        math: math_schema.CalliopeBuildMath,
        build_config: config_schema.Build,
    ) -> None:
        """Pyomo solver interface class.

        Args:
            inputs (xr.Dataset): Calliope model data.
            math (AttrDict): Calliope math.
            build_config (config_schema.Build): Build configuration options.
        """
        super().__init__(inputs, math, build_config, pmo.block())

        self._instance.parameters = pmo.parameter_dict()
        self._instance.variables = pmo.variable_dict()
        self._instance.global_expressions = pmo.expression_dict()
        self._instance.constraints = pmo.constraint_dict()
        self._instance.piecewise_constraints = pmo.block_dict()
        self._instance.objectives = pmo.objective_dict()

        self._instance.dual = pmo.suffix(direction=pmo.suffix.IMPORT)
        self.shadow_prices = PyomoShadowPrices(self._instance.dual, self)

    def add_parameter(  # noqa: D102, override
        self, name: str, values: xr.DataArray, definition: math_schema.Parameter
    ) -> None:
        self._raise_error_on_preexistence(name, "parameters")
        if values.isnull().all():
            self.log(
                "parameters", name, "Empty component added; no data found in array."
            )
            values = xr.DataArray(np.nan, attrs=values.attrs)
        else:
            self._create_obj_list(name, "parameters")
            values = self._apply_func(
                self._to_pyomo_param, values.notnull(), 1, values, name=name
            )

        self._add_to_dataset(name, values, "parameters", definition.model_dump())

    def add_constraint(  # noqa: D102, override
        self, name: str, definition: math_schema.Constraint
    ) -> None:
        def _constraint_setter(
            element: parsing.ParsedBackendEquation, where: xr.DataArray, references: set
        ) -> xr.DataArray:
            expr = element.evaluate_expression(self, where=where, references=references)

            try:
                to_fill = self._apply_func(
                    self._to_pyomo_constraint, where, 1, expr, name=name
                )
            except BackendError as err:
                types = self._apply_func(lambda x: type(x).__name__, where, 1, expr)
                offending_items = types.to_series().dropna().str.startswith("bool")
                offending_idx = offending_items[offending_items].index.values
                err.args = (f"{err.args[0]}: {offending_idx}", *err.args[1:])
                raise err

            return to_fill

        self._add_component(name, definition, _constraint_setter, "constraints")

    def add_global_expression(  # noqa: D102, override
        self, name: str, definition: math_schema.GlobalExpression
    ) -> None:
        def _expression_setter(
            element: parsing.ParsedBackendEquation, where: xr.DataArray, references: set
        ) -> xr.DataArray:
            expr = element.evaluate_expression(self, where=where, references=references)
            expr = expr.squeeze(drop=True)

            to_fill = expr.where(where)

            return to_fill

        self._add_component(name, definition, _expression_setter, "global_expressions")

    def add_variable(  # noqa: D102, override
        self, name: str, definition: math_schema.Variable
    ) -> None:
        domain_dict = {"real": pmo.RealSet, "integer": pmo.IntegerSet}

        def _variable_setter(where, references):
            domain_type = domain_dict[definition.domain]
            bounds = definition.bounds
            lb = self._get_variable_bound(bounds["min"], name, references)
            ub = self._get_variable_bound(bounds["max"], name, references)
            var = self._apply_func(
                self._to_pyomo_variable,
                where,
                1,
                lb,
                ub,
                name=name,
                domain_type=domain_type,
            )
            var.attrs = {}
            return var

        self._add_component(name, definition, _variable_setter, "variables")

    def add_objective(  # noqa: D102, override
        self, name: str, definition: math_schema.Objective
    ) -> None:
        sense_dict = {"minimize": 1, "minimise": 1, "maximize": -1, "maximise": -1}

        sense = sense_dict[definition.sense]

        def _objective_setter(
            element: parsing.ParsedBackendEquation, where: xr.DataArray, references: set
        ) -> xr.DataArray:
            expr = element.evaluate_expression(self, references=references)
            objective = pmo.objective(expr.item(), sense=sense)
            if name == self.objective:
                text = "activated"
                objective.activate()
            else:
                text = "deactivated"
                objective.deactivate()
            self.log("objectives", name, f"Objective {text}.")

            self._instance.objectives[name].append(objective)
            return xr.DataArray(objective)

        self._add_component(name, definition, _objective_setter, "objectives")

    def set_objective(self, name: str) -> None:  # noqa: D102, override
        self.objectives[self.objective].item().deactivate()
        self.log("objectives", self.objective, "Objective deactivated.", level="info")

        self.objectives[name].item().activate()
        self.objective = name
        self.config = self.config.update({"objective": name})

        self.log("objectives", name, "Objective activated.", level="info")

    def get_parameter(  # noqa: D102, override
        self, name: str, as_backend_objs: bool = True
    ) -> xr.DataArray:
        parameter = self._get_component(name, "parameters")

        if as_backend_objs:
            return parameter

        param_as_vals = self._apply_func(
            self._from_pyomo_param, parameter.notnull(), 1, parameter
        )
        return param_as_vals.where(param_as_vals.notnull()).astype(float)

    @overload
    def get_constraint(  # noqa: D102, override
        self, name: str, as_backend_objs: Literal[True] = True, eval_body: bool = False
    ) -> xr.DataArray: ...

    @overload
    def get_constraint(  # noqa: D102, override
        self, name: str, as_backend_objs: Literal[False], eval_body: bool = False
    ) -> xr.Dataset: ...

    def get_constraint(  # noqa: D102, override
        self, name: str, as_backend_objs: bool = True, eval_body: bool = False
    ) -> xr.DataArray | xr.Dataset:
        constraint = self.constraints.get(name, None)
        if constraint is None:
            raise KeyError(f"Unknown constraint: {name}")
        if isinstance(constraint, xr.DataArray) and not as_backend_objs:
            lb, body, ub = self._apply_func(
                self._from_pyomo_constraint, constraint.notnull(), 3, constraint
            )
            body = self._from_pyomo_expr(body, eval_body)
            constraint = xr.Dataset(
                {"lb": lb, "body": body, "ub": ub}, attrs=constraint.attrs
            )
        return constraint

    def get_variable(  # noqa: D102, override
        self, name: str, as_backend_objs: bool = True
    ) -> xr.DataArray:
        variable = self._get_component(name, "variables")

        if as_backend_objs:
            return variable
        else:
            return self._apply_func(
                self._from_pyomo_param, variable.notnull(), 1, variable
            )

    def get_variable_bounds(self, name: str) -> xr.Dataset:  # noqa: D102, override
        variable = self.get_variable(name, as_backend_objs=True)
        lb, ub = self._apply_func(
            self._from_pyomo_variable_bounds, variable.notnull(), 2, variable
        )
        return xr.Dataset({"lb": lb, "ub": ub}, attrs=variable.attrs)

    def _get_expression(  # noqa: D102, override
        self,
        name: str,
        as_backend_objs,
        eval_body,
        component_type: Literal["global_expressions", "objectives"],
    ) -> xr.DataArray:
        expression = getattr(self, component_type).get(name, None)
        if expression is None:
            raise KeyError(f"Unknown {component_type.removesuffix('s')}: {name}")
        if not as_backend_objs:
            expression = self._from_pyomo_expr(expression, eval_body)

        return expression

    def _solve(  # noqa: D102, override
        self, solve_config: config_schema.Solve, warmstart: bool = False
    ) -> xr.Dataset:
        if solve_config.solver == "cbc" and self.shadow_prices.is_active:
            model_warn(
                "Switching off shadow price tracker as constraint duals cannot be accessed from the CBC solver"
            )
            self.shadow_prices.deactivate()
        opt = SolverFactory(solve_config.solver, solver_io=solve_config.solver_io)

        if solve_config.solver_options:
            for k, v in solve_config.solver_options.items():
                opt.options[k] = v

        solve_kwargs = {}
        if solve_config.save_logs is not None:
            solve_kwargs.update({"symbolic_solver_labels": True, "keepfiles": True})
            logdir = Path(solve_config.save_logs)
            logdir.mkdir(parents=True, exist_ok=True)
            TempfileManager.tempdir = logdir  # Sets log output dir

        if warmstart and solve_config.solver in ["glpk", "cbc"]:
            model_warn(
                f"The chosen solver, {solve_config.solver}, does not support warmstart, "
                "which may impact performance."
            )
            warmstart = False

        with redirect_stdout(LogWriter(self._solve_logger, "debug", strip=True)):  # type: ignore
            with redirect_stderr(LogWriter(self._solve_logger, "error", strip=True)):  # type: ignore
                # Ignore most of gurobipy's logging, as it's output is
                # already captured through STDOUT
                logging.getLogger("gurobipy").setLevel(logging.ERROR)
                results = opt.solve(self._instance, tee=True, **solve_kwargs)

        termination = results.solver[0].termination_condition

        if pe.TerminationCondition.to_solver_status(termination) == pe.SolverStatus.ok:
            self._instance.load_solution(results.solution[0])
            results = self.load_results()
        else:
            self._solve_logger.critical("Problem status:")
            for line in str(results.problem[0]).split("\n"):
                self._solve_logger.critical(line)
            self._solve_logger.critical("Solver status:")
            for line in str(results.solver[0]).split("\n"):
                self._solve_logger.critical(line)

            model_warn("Model solution was non-optimal.", _class=BackendWarning)
            results = xr.Dataset()

        results.attrs["termination_condition"] = str(termination)
        return results

    def verbose_strings(self) -> None:  # noqa: D102, override
        def __renamer(val, *idx):
            if pd.notna(val):
                val.calliope_coords = idx

        with self._datetime_as_string(self._dataset):
            for component_type in [
                "parameters",
                "variables",
                "constraints",
                "piecewise_constraints",
            ]:
                for da in self._dataset.filter_by_attrs(
                    coords_in_name=False, **{"obj_type": component_type}
                ).values():
                    self._apply_func(
                        __renamer, da.notnull(), 1, da, *[da.coords[i] for i in da.dims]
                    )
                    da.attrs["coords_in_name"] = True
        self._has_verbose_strings = True

    def to_lp(self, path: str | Path) -> None:  # noqa: D102, override
        self._instance.write(str(path), format="lp", symbolic_solver_labels=True)

    def _create_obj_list(self, key: str, component_type: ALL_COMPONENTS_T) -> None:
        """Attach an empty pyomo kernel list object to the pyomo model object.

        Args:
            key (str): Name of object
            component_type (str): Object type

        Raises:
            BackendError: Cannot overwrite object of same name and type.
        """
        component_dict = getattr(self._instance, component_type)
        if key in component_dict:
            raise BackendError(
                f"Trying to add already existing `{key}` to backend model {component_type}."
            )
        else:
            singular_component = component_type.removesuffix("s")
            component_dict[key] = getattr(
                pmo, f"{COMPONENT_TRANSLATOR[singular_component]}_list"
            )()

    def delete_component(  # noqa: D102, override
        self, key: str, component_type: ALL_COMPONENTS_T
    ) -> None:
        component_dict = getattr(self._instance, component_type)
        if key in component_dict:
            del component_dict[key]

        if key in self._dataset and self._dataset[key].obj_type == component_type:
            del self._dataset[key]

    def update_input(  # noqa: D102, override
        self, name: str, new_values: xr.DataArray | SupportsFloat
    ) -> None:
        orig, new, update = self._update_input(name, new_values, mutable=True)
        if update:
            self._apply_func(self._update_pyomo_param, new.notnull(), 1, orig, new)

    def update_variable_bounds(  # noqa: D102, override
        self,
        name: str,
        *,
        min: xr.DataArray | SupportsFloat | None = None,
        max: xr.DataArray | SupportsFloat | None = None,
    ) -> None:
        translator = {"min": "lb", "max": "ub"}
        variable_da = self.get_variable(name)

        for bound_name, new_bounds in {"min": min, "max": max}.items():
            if new_bounds is None:
                self.log(
                    "variables",
                    name,
                    f"{bound_name} bound not being updated as it has not been defined.",
                )
                continue

            existing_bound_param = self.math.variables[name].bounds[bound_name]
            if existing_bound_param in self.parameters:
                raise BackendError(
                    "Cannot update variable bounds that have been set by parameters. "
                    f"Use `update_input('{existing_bound_param}')` to update the {bound_name} bound of {name}."
                )

            bound_da = xr.DataArray(new_bounds)
            missing_dims_in_new_vals = set(variable_da.dims).difference(bound_da.dims)
            if missing_dims_in_new_vals:
                self.log(
                    "variables",
                    name,
                    f"New `{bound_name}` bounds will be broadcast along the {missing_dims_in_new_vals} dimension(s).",
                    "info",
                )
            self._apply_func(
                self._update_pyomo_variable,
                variable_da.notnull() & xr.DataArray(new_bounds).notnull(),
                1,
                variable_da,
                xr.DataArray(new_bounds),
                bound=translator[bound_name],
            )

    def fix_variable(  # noqa: D102, override
        self, name: str, where: xr.DataArray | None = None
    ) -> None:
        variable_da = self.get_variable(name)
        where_da = variable_da.notnull()
        if where is not None:
            where_da = where_da & where.fillna(False)
        self._apply_func(self._fix_pyomo_variable, where_da, 1, variable_da)

    def unfix_variable(  # noqa: D102, override
        self, name: str, where: xr.DataArray | None = None
    ) -> None:
        variable_da = self.get_variable(name)
        where_da = variable_da.notnull()
        if where is not None:
            where_da = where_da & where.fillna(False)
        self._apply_func(self._unfix_pyomo_variable, where_da, 1, variable_da)

    @property
    def has_integer_or_binary_variables(self) -> bool:  # noqa: D102, override
        model_report = build_model_size_report(self._instance)
        binaries = model_report["activated"]["binary_variables"]
        integers = model_report["activated"]["integer_variables"]
        number_of_binary_and_integer_vars = binaries + integers
        return number_of_binary_and_integer_vars > 0

    def _to_piecewise_constraint(  # noqa: D102, override
        self, x_var: Any, y_var: Any, *vals: float, name: str, n_breakpoints: int
    ) -> type[ObjPiecewiseConstraint]:
        y_vals = pd.Series(vals[n_breakpoints:]).dropna()
        x_vals = pd.Series(vals[:n_breakpoints]).dropna()
        try:
            var = ObjPiecewiseConstraint(
                breakpoints=x_vals,
                values=y_vals,
                input=x_var,
                output=y_var,
                require_bounded_input_variable=False,
            )
            self._instance.piecewise_constraints[name].append(var)
        except (PiecewiseValidationError, ValueError) as err:
            # We don't want to confuse the user with suggestions of pyomo options they can't access.
            err_message = err.args[0].split(" To avoid this error")[0]
            raise BackendError(err_message)
        return var

    def _to_pyomo_param(self, val: Any, *, name: str) -> type[ObjParameter] | float:
        """Utility function to generate a pyomo parameter for every element of an xarray DataArray.

        Output objects are of the type ObjParameter(pmo.parameter) since they need a
        "dtype" property to be handled by xarray.

        If not np.nan/None, output objects are also added to the backend model object in-place.


        Args:
            val (Any): Value to turn into a mutable pyomo parameter
            name (str): Name of parameter
            default (Any, optional): Default value if `val` is None/np.nan. Defaults to np.nan.

        Returns:
            type[ObjParameter] | float:
                If both `val` and `default` are np.nan/None, return np.nan.
                Otherwise return ObjParameter(val/default).
        """
        param = ObjParameter(val)
        self._instance.parameters[name].append(param)
        return param

    def _update_pyomo_param(self, orig: ObjParameter, new: Any) -> None:
        """Utility function to update pyomo parameter values in-place.

        Args:
            orig (ObjParameter): Pyomo parameter to update.
            new (Any): Value with which to update the parameter.
        """
        orig.value = new

    def _update_pyomo_variable(
        self, orig: ObjVariable, new: Any, *, bound: Literal["lb", "ub"]
    ) -> None:
        """Utility function to update pyomo variable bounds in-place.

        Args:
            orig (ObjVariable): Pyomo variable to update.
            new (Any): new value to set.
            bound (Literal["lb", "ub"]): upper / lower bound.
                lb (Any): Value with which to update the lower bound of the variable.
                ub (Any): Value with which to update the upper bound of the variable.
        """
        setattr(orig, bound, new)

    def _fix_pyomo_variable(self, orig: ObjVariable) -> None:
        """Utility function to fix a pyomo variable to its value in the optimisation model solution.

        Fixed variables will be considered as parameters in the subsequent solve.

        Args:
            orig (ObjVariable): Pyomo variable to fix.

        Raises:
            BackendError: Can only fix variables if they have values assigned to them from an optimal solution.
        """
        if orig.value is None:
            raise BackendError(
                "Cannot fix variable values without already having solved the model successfully."
            )
        else:
            orig.fix()

    def _unfix_pyomo_variable(self, orig: ObjVariable) -> None:
        """Utility function to unfix a pyomo variable so it can be considered a decision variable on the next solve.

        Args:
            orig (ObjVariable): Pyomo variable to unfix.

        """
        orig.unfix()

    def _to_pyomo_constraint(
        self, expr: Any, *, name: str
    ) -> type[ObjConstraint] | float:
        """Utility function to generate a pyomo constraint for every element of an xarray DataArray.

        If not np.nan/None, output objects are also added to the backend model object in-place.

        Args:
            mask (bool | np.bool_): If True, add constraint, otherwise return np.nan
            expr (Any): Equation expression.

        Kwargs:
            name (str): Name of constraint

        Returns:
            Union[type[ObjConstraint], float]:
                If mask is True, return np.nan.
                Otherwise return pmo_constraint(expr=lhs op rhs).
        """
        if isinstance(expr, np.bool_ | bool):
            raise BackendError(
                f"(constraints, {name}) | constraint array includes item(s) that resolves to a simple boolean. "
                "There must be a math component defined on at least one side of the equation"
            )
        constraint = ObjConstraint(expr=expr)
        self._instance.constraints[name].append(constraint)
        return constraint

    def _to_pyomo_variable(
        self,
        lb: Any,
        ub: Any,
        *,
        name: str,
        domain_type: Literal["RealSet", "IntegerSet"],
    ) -> type[ObjVariable] | float:
        """Utility function to generate a pyomo decision variable for every element of an xarray DataArray.

        If not np.nan/None, output objects are also added to the backend model object in-place.

        Args:
            mask (Union[bool, np.bool_]): If True, add variable, otherwise return np.nan.
            ub (Any): Upper bound to apply to the variable.
            lb (Any): Lower bound to apply to the variable.

        Kwargs:
            domain_type (Literal["RealSet", "IntegerSet"]):
                Domain over which variables are valid (real = continuous, integer = integer/binary)
            name (str): Name of variable.

        Returns:
            Union[type[ObjVariable], float]:
                If mask is True, return np.nan.
                Otherwise return pmo_variable(ub=ub, lb=lb, domain_type=domain_type).
        """
        var = ObjVariable(ub=ub, lb=lb, domain_type=domain_type)
        self._instance.variables[name].append(var)
        return var

    def _from_pyomo_expr(self, expr_da: xr.DataArray, eval_body: bool) -> xr.DataArray:
        """Evaluate an array of Pyomo expression objects.

        Args:
            expr_da (xr.DataArray): Array containing expression objects
            eval_body (bool):
                If True, attempt to evaluate objects as numeric values.
                Will return string values of the expression math if the optimisation problem hasn't been successfully solved.
                If False, will return string values of the expression math.

        Returns:
            xr.DataArray: Array of numeric or math string values.
        """
        if eval_body:
            try:
                expr = self._apply_func(
                    lambda expr: expr() if isinstance(expr, PyomoObject) else expr,
                    expr_da.notnull(),
                    1,
                    expr_da,
                )
            except ValueError:
                expr = expr_da.astype(str)
        else:
            expr = expr_da.astype(str)
        return expr.where(expr_da.notnull())

    @staticmethod
    def _from_pyomo_param(val: ObjParameter | ObjVariable | float) -> Any:
        """Evaluate value of Pyomo object.

        If the input object is a parameter, a numeric/string value will be given.
        If the input object is a global expression or variable, a numeric value will be given
        only if the backend model has been successfully optimised, otherwise evaluation will return None.

        Args:
            val (ObjParameter | ObjVariable | float):
                Item to be evaluated.

        Returns:
            Any: If the input is nullable, return np.nan, otherwise evaluate the "value" method of the object.
        """
        return val.value  # type: ignore

    @staticmethod
    def _from_pyomo_constraint(
        val: ObjConstraint,
    ) -> tuple[float, pmo.expression, float]:
        """Evaluate Pyomo constraint object.

        Args:
            val (ObjConstraint): constraint object to be evaluated
        Kwargs:
            eval_body (bool, optional):
                If True, attempt to evaluate the constraint object `body`, which will evaluate the
                linear expression contained in the constraint body and produce a numeric value.
                This will only succeed if the backend model has been successfully optimised,
                otherwise a string representation of the linear expression will be returned
                (same as eval_body=False). Defaults to False.

        Returns:
            pd.Series: Array of upper bound (ub), body, and lower bound (lb).
        """
        return val.lb, val.body, val.ub

    @staticmethod
    def _from_pyomo_variable_bounds(val: ObjVariable) -> tuple[float, float]:
        """Evaluate Pyomo decision variable object bounds.

        Args:
            val (ObjVariable): Variable object to be evaluated.

        Returns:
            pd.Series: Array of variable upper and lower bound.
        """
        return val.lb, val.ub


class CoordObj(ABC):
    """Generic class for name updates."""

    def __init__(self) -> None:
        """Class with methods to update the `name` property of inheriting classes."""
        self._calliope_coords: Iterable | None = None

    def _update_name(self, old_name: str) -> str:
        """Update string of a list containing a single number with a string of a list containing any arbitrary number of elements.

        Args:
            old_name (str): String representation of a list containing a single number

        Returns:
            str:
                If `self.calliope_coords` is None, returns `old_name`.
                Otherwise returns string representation of a list containing the contents of `self.calliope_coords`
        """
        if self._calliope_coords is None:
            return old_name

        if not self._calliope_coords:  # empty list = dimensionless component
            coord_list = ""
        else:
            coord_list = f"[{', '.join(str(i) for i in self._calliope_coords)}]"
        return re.sub(r"\[\d+\]", coord_list, old_name)

    @property
    def calliope_coords(self):
        """Get coordinates."""
        return self._calliope_coords

    @calliope_coords.setter
    def calliope_coords(self, val):
        """Set coordinates."""
        self._calliope_coords = val


class ObjParameter(pmo.parameter, CoordObj):
    """Pyomo parameter functionality."""

    def __init__(self, value, **kwds):
        """Instantiate a pyomo Parameter.

        A pyomo parameter (`a object for storing a mutable, numeric value that can be used to build a symbolic expression`)
        with added `dtype` property and a `name` property setter (via the `pmo.parameter.getname` method) which replaces a list position as a name with a list of strings.
        """
        assert not pd.isnull(value)
        pmo.parameter.__init__(self, value, **kwds)
        CoordObj.__init__(self)

    @property
    def dtype(self):
        """Get dtype."""
        return "O"

    def getname(self, *args, **kwargs):
        """Get name."""
        return self._update_name(pmo.parameter.getname(self, *args, **kwargs))


class ObjVariable(pmo.variable, CoordObj):
    """Pyomo variable functionality."""

    def __init__(self, **kwds):
        """Create a pyomo variable.

        Created with a `name` property setter (via the `pmo.variable.getname` method)
        which replaces a list position as a name with a list of strings.
        """
        pmo.variable.__init__(self, **kwds)
        CoordObj.__init__(self)

    def getname(self, *args, **kwargs):
        """Get variable name."""
        return self._update_name(pmo.variable.getname(self, *args, **kwargs))


class ObjConstraint(pmo.constraint, CoordObj):
    """Pyomo constraint functionality."""

    def __init__(self, **kwds):
        """Create a pyomo constraint.

        Created with a `name` property setter (via the `pmo.constraint.getname` method)
        which replaces a list position as a name with a list of strings.
        """
        pmo.constraint.__init__(self, **kwds)
        CoordObj.__init__(self)

    def getname(self, *args, **kwargs):
        """Get constraint name."""
        return self._update_name(pmo.constraint.getname(self, *args, **kwargs))


class ObjPiecewiseConstraint(piecewise_sos2, CoordObj):
    """Pyomo SOS2 piecewise constraint wrapper."""

    def __init__(self, **kwds):
        """Create a Pyomo SOS2 piecesise constraint.

        Created with a `name` property setter (via the `piecewise_sos2.getname` method)
        which replaces a list position as a name with a list of strings.
        """
        func = PiecewiseLinearFunction(
            breakpoints=kwds.pop("breakpoints"), values=kwds.pop("values")
        )
        piecewise_sos2.__init__(self, func, **kwds)
        CoordObj.__init__(self)

    def getname(self, *args, **kwargs):
        """Get piecewise constraint name."""
        return self._update_name(piecewise_sos2.getname(self, *args, **kwargs))


class PyomoShadowPrices(backend_model.ShadowPrices):
    """Pyomo shadow price functionality."""

    def __init__(self, dual_obj: pmo.suffix, backend_obj: PyomoBackendModel):
        """Create deactivated pyomo shadow price functions."""
        self._dual_obj = dual_obj
        self._backend_obj = backend_obj
        self.deactivate()

    def get(self, name: str) -> xr.DataArray:  # noqa: D102, override
        ...
        constraint = self._backend_obj.get_constraint(name, as_backend_objs=True)
        return self._backend_obj._apply_func(
            self._duals_from_pyomo_constraint,
            constraint.notnull(),
            1,
            constraint,
            dual_getter=self._dual_obj,
        )

    def activate(self):  # noqa: D102, override
        ...
        if self._backend_obj.has_integer_or_binary_variables:
            warning_text = "Shadow price tracking on a model with binary or integer variables is not possible. Proceeding without activating shadow price tracking."
            model_warn(warning_text, _class=BackendWarning)
        else:
            self._dual_obj.activate()

    def deactivate(self):  # noqa: D102, override
        ...
        self._dual_obj.deactivate()

    @property
    def is_active(self) -> bool:  # noqa: D102, override
        ...
        return self._dual_obj.active

    @property
    def available_constraints(self) -> Iterable:  # noqa: D102, override
        ...
        return self._backend_obj.constraints.data_vars

    @staticmethod
    def _duals_from_pyomo_constraint(
        val: pmo.constraint, *, dual_getter: pmo.suffix
    ) -> float:
        return dual_getter.get(val)
