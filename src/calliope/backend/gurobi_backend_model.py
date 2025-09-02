# Copyright (C) since 2013 Calliope contributors listed in AUTHORS.
# Licensed under the Apache 2.0 License (see LICENSE file).
"""Gurobi backend functionality."""

from __future__ import annotations

import importlib
import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal, SupportsFloat, overload

import numpy as np
import pandas as pd
import xarray as xr

from calliope.backend import backend_model, parsing
from calliope.backend.backend_model import ALL_COMPONENTS_T
from calliope.exceptions import BackendError, BackendWarning
from calliope.exceptions import warn as model_warn
from calliope.schemas import config_schema, math_schema

if importlib.util.find_spec("gurobipy") is not None:
    import gurobipy

LOGGER = logging.getLogger(__name__)


class GurobiBackendModel(backend_model.BackendModel):
    """gurobipy-specific backend functionality."""

    OBJECTIVE_SENSE_DICT: dict[str, int]
    if importlib.util.find_spec("gurobipy") is not None:
        OBJECTIVE_SENSE_DICT = {
            "minimize": gurobipy.GRB.MINIMIZE,
            "minimise": gurobipy.GRB.MINIMIZE,
            "maximize": gurobipy.GRB.MAXIMIZE,
            "maximise": gurobipy.GRB.MAXIMIZE,
        }
    else:
        # As of Gurobi v11, these are the correct integer values for the above constants
        OBJECTIVE_SENSE_DICT = {
            "minimize": 1,
            "minimise": 1,
            "maximize": -1,
            "maximise": -1,
        }

    def __init__(
        self,
        inputs: xr.Dataset,
        math: math_schema.CalliopeBuildMath,
        build_config: config_schema.Build,
    ) -> None:
        """Gurobi solver interface class.

        Args:
            inputs (xr.Dataset): Calliope model data.
            math (math_schema.CalliopeBuildMath): Calliope math.
            build_config (config_schema.Build): Build configuration options.
        """
        if importlib.util.find_spec("gurobipy") is None:
            raise ImportError(
                "Install the `gurobipy` package to build the optimisation problem with the Gurobi backend."
            )
        super().__init__(inputs, math, build_config, gurobipy.Model())
        self._instance: gurobipy.Model
        self.shadow_prices = GurobiShadowPrices(self)

    def add_parameter(  # noqa: D102, override
        self, name: str, values: xr.DataArray, definition: math_schema.Parameter
    ) -> None:
        self._raise_error_on_preexistence(name, "parameters")

        if values.isnull().all():
            self.log("parameters", name, "Component not added; no data found in array.")
            values = xr.DataArray(np.nan, attrs=values.attrs)

        self._add_to_dataset(name, values, "parameters", definition.model_dump())

    def add_constraint(  # noqa: D102, override
        self, name: str, definition: math_schema.Constraint
    ) -> None:
        def _constraint_setter(
            element: parsing.ParsedBackendEquation, where: xr.DataArray, references: set
        ) -> xr.DataArray:
            expr = element.evaluate_expression(self, where=where, references=references)
            to_fill = self._apply_func(self._instance.addConstr, where, 1, expr)

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
            self._clean_arrays(expr)
            return to_fill

        self._add_component(name, definition, _expression_setter, "global_expressions")

    def add_variable(  # noqa: D102, override
        self, name: str, definition: math_schema.Variable
    ) -> None:
        domain_dict = {"real": gurobipy.GRB.CONTINUOUS, "integer": gurobipy.GRB.INTEGER}

        def _variable_setter(where: xr.DataArray, references: set):
            domain_type = domain_dict[definition.domain]

            bounds = definition.bounds
            lb = self._get_variable_bound(bounds.min, name, references)
            ub = self._get_variable_bound(bounds.max, name, references)
            var = self._apply_func(
                self._instance.addVar, where, 1, lb, ub, vtype=domain_type
            )
            return var.fillna(value=np.nan)

        self._add_component(name, definition, _variable_setter, "variables")

    def add_objective(  # noqa: D102, override
        self, name: str, definition: math_schema.Objective
    ) -> None:
        sense = self.OBJECTIVE_SENSE_DICT[definition.sense]

        def _objective_setter(
            element: parsing.ParsedBackendEquation, where: xr.DataArray, references: set
        ) -> xr.DataArray:
            expr = element.evaluate_expression(self, references=references)

            if name == self.objective:
                self._instance.setObjective(expr.item(), sense=sense)
                self.objective = name
                self.log("objectives", name, "Objective activated.")

            return xr.DataArray(expr)

        self._add_component(name, definition, _objective_setter, "objectives")

    def set_objective(self, name: str) -> None:  # noqa: D102, override
        to_set = self.objectives[name]
        sense = self.OBJECTIVE_SENSE_DICT[to_set.attrs["sense"]]
        self._instance.setObjective(to_set.item(), sense=sense)
        self._instance.update()
        self.objective = name
        self.log("objectives", name, "Objective activated.", level="info")

    def get_parameter(  # noqa: D102, override
        self, name: str, as_backend_objs: bool = True
    ) -> xr.DataArray:
        parameter = self.parameters.get(name, None)
        if parameter is None:
            raise KeyError(f"Unknown parameter: {name}")

        return parameter.astype(float)

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
            raise BackendError("Cannot return a Gurobi constraint in string format")
        return constraint

    def get_variable(  # noqa: D102, override
        self, name: str, as_backend_objs: bool = True
    ) -> xr.DataArray:
        variable = self.variables.get(name, None)
        if variable is None:
            raise KeyError(f"Unknown variable: {name}")
        if as_backend_objs:
            return variable
        else:
            try:
                return self._apply_func(
                    self._from_gurobi_var, variable.notnull(), 1, variable
                )
            except AttributeError:
                return variable.astype(str).where(variable.notnull())

    def get_variable_bounds(self, name: str) -> xr.Dataset:  # noqa: D102, override
        variable = self.get_variable(name, as_backend_objs=True)

        lb, ub = self._apply_func(
            self._from_gurobi_variable_bounds, variable.notnull(), 2, variable
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
        if isinstance(expression, xr.DataArray) and not as_backend_objs:
            if not eval_body:
                return expression.astype(str).where(expression.notnull())
            else:
                try:
                    return self._apply_func(
                        self._from_gurobi_expr, expression.notnull(), 1, expression
                    )
                except (AttributeError, TypeError):
                    return expression.astype(str).where(expression.notnull())
        else:
            return expression

    def _solve(
        self, solve_config: config_schema.Solve, warmstart: bool = False
    ) -> xr.Dataset:
        self._instance.resetParams()

        if solve_config.solver_options is not None:
            for k, v in solve_config.solver_options.items():
                self._instance.setParam(k, v)

        if not warmstart:
            self._instance.setParam("LPWarmStart", 0)

        if solve_config.save_logs is not None:
            logdir = Path(solve_config.save_logs)
            self._instance.setParam("LogFile", (logdir / "gurobi.log").as_posix())

        self._instance.update()

        self._instance.optimize()

        termination = self._instance.status

        if termination == gurobipy.GRB.OPTIMAL:
            results = self.load_results()
        else:
            model_warn("Model solution was non-optimal.", _class=BackendWarning)
            results = xr.Dataset()

        termination = [
            i
            for i in dir(gurobipy.GRB.Status)
            if not i.startswith("_") and getattr(gurobipy.GRB.Status, i) == termination
        ][0].lower()
        results.attrs["termination_condition"] = str(termination)

        return results

    def verbose_strings(self) -> None:  # noqa: D102, override
        def __renamer(val, *idx, name: str, attr: str):
            if pd.notna(val):
                new_obj_name = f"{name}[{', '.join(idx)}]"
                setattr(val, attr, new_obj_name)

        self._instance.update()
        attribute_names = {
            "variables": "VarName",
            "constraints": "ConstrName",
            "piecewise_constraints": "GenConstrName",
        }
        with self._datetime_as_string(self._dataset):
            for da in self._dataset.filter_by_attrs(coords_in_name=False).values():
                if da.attrs["obj_type"] not in attribute_names.keys():
                    continue
                self._apply_func(
                    __renamer,
                    da.notnull(),
                    1,
                    da,
                    *[da.coords[i] for i in da.dims],
                    name=da.name,
                    attr=attribute_names[da.attrs["obj_type"]],
                )
                da.attrs["coords_in_name"] = True
        self._instance.update()

    def to_lp(self, path: str | Path) -> None:  # noqa: D102, override
        self._instance.update()

        if Path(path).suffix != ".lp":
            raise ValueError("File extension must be `.lp`")
        self._instance.write(str(path))

    def _create_obj_list(self, key: str, component_type: ALL_COMPONENTS_T) -> None:
        pass

    def delete_component(self, key: str, component_type: ALL_COMPONENTS_T) -> None:
        """Delete object from the backend model object linked to a component.

        Args:
            key (str): Name of object
            component_type (str): Object type
        """
        if key in self._dataset and self._dataset[key].obj_type == component_type:
            if component_type in ["variables", "constraints"]:
                self._apply_func(
                    self._del_gurobi_obj,
                    self._dataset[key].notnull(),
                    1,
                    self._dataset[key],
                )
            del self._dataset[key]
        self._instance.update()

    def update_input(  # noqa: D102, override
        self, name: str, new_values: xr.DataArray | SupportsFloat
    ) -> None:
        self._update_input(name, new_values, mutable=False)
        self._instance.update()

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
                self._update_gurobi_variable,
                variable_da.notnull() & xr.DataArray(new_bounds).notnull(),
                1,
                variable_da,
                xr.DataArray(new_bounds),
                bound=translator[bound_name],
            )
        self._instance.update()

    def fix_variable(  # noqa: D102, override
        self, name: str, where: xr.DataArray | None = None
    ) -> None:
        if self._instance.status != gurobipy.GRB.OPTIMAL:
            raise BackendError(
                "Cannot fix variable values without already having solved the model successfully."
            )

        variable_da = self.get_variable(name)
        if where is not None:
            variable_da = variable_da.where(where.fillna(0))

        self._apply_func(
            self._fix_gurobi_variable, variable_da.notnull(), 1, variable_da
        )

        self._instance.update()

    def unfix_variable(  # noqa: D102, override
        self, name: str, where: xr.DataArray | None = None
    ) -> None:
        raise BackendError(
            "Cannot unfix a variable using the Gurobi backend; "
            "you will need to rebuild your backend or update variable bounds to match the original bounds."
        )

    @property
    def has_integer_or_binary_variables(self) -> bool:  # noqa: D102, override
        self._instance.update()
        return any(
            var.vtype != gurobipy.GRB.CONTINUOUS for var in self._instance.getVars()
        )

    def _del_gurobi_obj(self, obj: Any) -> None:
        self._instance.remove(obj)

    def _to_piecewise_constraint(  # noqa: D102, override
        self,
        x_var: gurobipy.Var,
        y_var: gurobipy.Var,
        *vals: float,
        name: str,
        n_breakpoints: int,
    ) -> gurobipy.GenConstr:
        if not isinstance(x_var, gurobipy.Var) or not isinstance(y_var, gurobipy.Var):
            raise BackendError(
                "Gurobi backend can only build piecewise constraints using decision variables."
            )
        y_vals = list(pd.Series(vals[n_breakpoints:]).dropna().values)
        x_vals = list(pd.Series(vals[:n_breakpoints]).dropna().values)
        try:
            var = self._instance.addGenConstrPWL(
                xpts=x_vals, ypts=y_vals, xvar=x_var, yvar=y_var, name=name
            )
        except gurobipy.GurobiError as err:
            raise BackendError(err)
        return var

    def _update_gurobi_variable(
        self, orig: gurobipy.Var, new: Any, *, bound: Literal["lb", "ub"]
    ) -> None:
        """Utility function to update gurobi variable bounds in-place.

        Args:
            orig (gurobipy.Var): Gurobi variable to update.
            new (Any): Value with which to update the specified bound of the variable.
            bound (Literal[lb, ub]): Variable bound to update ("lb" = lower bound, "ub" = upper bound).
        """
        setattr(orig, bound, new)

    def _fix_gurobi_variable(self, orig: gurobipy.Var) -> None:
        """Utility function to fix a Gurobi variable to its value in the optimisation model solution.

        Fixed variables will be considered as parameters in the subsequent solve.

        Args:
            orig (gurobipy.Var): Gurobi variable to fix.

        Raises:
            BackendError: Can only fix variables if they have values assigned to them from an optimal solution.
        """
        self._update_gurobi_variable(orig, orig.x, bound="lb")  # type: ignore
        self._update_gurobi_variable(orig, orig.x, bound="ub")  # type: ignore

    @staticmethod
    def _from_gurobi_variable_bounds(val: gurobipy.Var) -> pd.Series:
        """Evaluate Gurobi decision variable object bounds.

        Args:
            val (gurobipy.Var): Variable object to be evaluated.

        Returns:
            pd.Series: Array of variable upper and lower bound.
        """
        return pd.Series(data=[val.lb, val.ub], index=["lb", "ub"])

    @staticmethod
    def _from_gurobi_var(val: gurobipy.Var) -> Any:
        """Evaluate Gurobi variable object.

        Args:
            val (gurobipy.LinExpr): expression object to be evaluated

        Returns:
            Any: If the input is nullable, return np.nan, otherwise a numeric value
            (eval_body=True and problem is optimised) or a string.
        """
        return val.x  # type: ignore

    @staticmethod
    def _from_gurobi_expr(
        val: gurobipy.LinExpr | gurobipy.Var | float,
    ) -> int | float | None:
        """Evaluate Gurobi object in an expression array.

        Args:
            val (gurobipy.LinExpr | gurobipy.Var | float): object to be evaluated; could be an expression, decision variable, or simple number stored in the global expression array.

        Returns:
            (int | float | None): the evaluated result.
        """
        if isinstance(val, gurobipy.LinExpr):
            return val.getValue()
        elif isinstance(val, gurobipy.Var):
            return val.x  # type: ignore
        elif isinstance(val, int | float):
            return val
        else:
            raise TypeError(
                f"Cannot convert Gurobi object of type {type(val)} to a numeric value."
            )


class GurobiShadowPrices(backend_model.ShadowPrices):
    """Gurobi shadow price functionality."""

    def __init__(self, backend_obj: GurobiBackendModel):
        """Create gurobipy shadow price functions."""
        self._backend_obj = backend_obj

    def get(self, name: str) -> xr.DataArray:  # noqa: D102, override
        constraint = self._backend_obj.get_constraint(name, as_backend_objs=True)
        return self._backend_obj._apply_func(
            self._duals_from_gurobi_constraint, constraint.notnull(), 1, constraint
        )

    def activate(self):  # noqa: D102, override
        pass

    def deactivate(self):  # noqa: D102, override
        pass

    @property
    def is_active(self) -> bool:  # noqa: D102, override
        return True

    @property
    def available_constraints(self) -> Iterable:  # noqa: D102, override
        return self._backend_obj.constraints.data_vars

    @staticmethod
    def _duals_from_gurobi_constraint(val: gurobipy.Constr) -> float:
        try:
            dual = val.Pi  # type: ignore
        except AttributeError:
            return np.nan
        else:
            return dual
