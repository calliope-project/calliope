# Copyright (C) since 2013 Calliope contributors listed in AUTHORS.
# Licensed under the Apache 2.0 License (see LICENSE file).
"""Highs backend functionality."""

from __future__ import annotations

import bisect
import importlib
import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal, SupportsFloat, overload

import numpy as np
import pandas as pd
import xarray as xr

from calliope.backend import backend_model
from calliope.backend.backend_model import ALL_COMPONENTS_T
from calliope.exceptions import BackendError, BackendWarning
from calliope.exceptions import warn as model_warn
from calliope.schemas import config_schema, math_schema

if importlib.util.find_spec("highspy") is not None:
    import highspy

LOGGER = logging.getLogger(__name__)


class HighsBackendModel(backend_model.BackendModel):
    """highspy-specific backend functionality."""

    if importlib.util.find_spec("highspy") is not None:
        OBJECTIVE_SENSE_DICT = {
            "minimize": highspy.ObjSense.kMinimize,
            "minimise": highspy.ObjSense.kMinimize,
            "maximize": highspy.ObjSense.kMaximize,
            "maximise": highspy.ObjSense.kMaximize,
        }
        VARIABLE_DOMAIN_DICT = {
            "real": highspy.HighsVarType.kContinuous,
            "integer": highspy.HighsVarType.kInteger,
        }
    else:
        OBJECTIVE_SENSE_DICT = {
            "minimize": 1,
            "minimise": 1,
            "maximize": -1,
            "maximise": -1,
        }
        VARIABLE_DOMAIN_DICT = {"real": "continuous", "integer": "integer"}

    def __init__(
        self,
        inputs: xr.Dataset,
        math: math_schema.CalliopeBuildMath,
        build_config: config_schema.Build,
    ) -> None:
        """Highs solver interface class.

        Args:
            inputs (xr.Dataset): Calliope model data.
            math (math_schema.CalliopeBuildMath): Calliope math.
            build_config (config_schema.Build): Build configuration options.
        """
        if importlib.util.find_spec("highspy") is None:
            raise ImportError(
                "Install the `highspy` package to build the optimisation problem with the Highs backend."
            )
        super().__init__(inputs, math, build_config, highspy.Highs())
        self._instance: highspy.Highs
        self.shadow_prices = HighsShadowPrices(self)

    def add_parameter(  # noqa: D102, override
        self, name: str, values: xr.DataArray, definition: math_schema.Parameter
    ) -> None:
        self._raise_error_on_preexistence(name, "parameters")

        if values.isnull().all():
            self.log("parameters", name, "Component not added; no data found in array.")
            values = xr.DataArray(np.nan, attrs=values.attrs)

        self._add_to_dataset(name, values, "parameters", definition.model_dump())

        if name not in self.math["parameters"]:
            self.math = self.math.update(
                {f"parameters.{name}": definition.model_dump()}
            )

    def _add_variable(  # noqa: D102, override
        self,
        name: str,
        where: xr.DataArray,
        references: set,
        domain_type: str,
        bounds: math_schema.Bounds,
    ) -> xr.DataArray:
        lb = self._get_variable_bound(bounds.min, name, references)
        ub = self._get_variable_bound(bounds.max, name, references)
        var = self._apply_func(
            self._instance.addVariable, where, 1, lb, ub, type=domain_type
        )

        return var.fillna(value=np.nan)

    def _add_global_expression(  # noqa: D102, override
        self, name: str, where: xr.DataArray, expression: xr.DataArray
    ) -> xr.DataArray:
        expression = expression.squeeze(drop=True)
        self._clean_arrays(expression)
        to_fill = expression.where(where)

        return to_fill

    def _add_constraint(  # noqa: D102, override
        self, name: str, where: xr.DataArray, expression: xr.DataArray
    ) -> xr.DataArray:
        to_fill = self._apply_func(self._instance.addConstr, where, 1, expression)

        return to_fill

    def _add_objective(  # noqa: D102, override
        self, name: str, where: xr.DataArray, expression: xr.DataArray, sense: int
    ) -> xr.DataArray:
        if name == self.objective:
            self._instance.setObjective(obj=expression.item(), sense=sense)
            self.objective = name
            self.log("objectives", name, "Objective activated.")
        return expression

    def set_objective(self, name: str) -> None:  # noqa: D102, override
        to_set = self.objectives[name]
        sense = self.OBJECTIVE_SENSE_DICT[self.math.objectives[name].sense]
        self._instance.setObjective(obj=to_set.item(), sense=sense)
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
            return self._apply_func(
                self._cons_to_str, constraint.notnull(), 1, constraint
            )
        return constraint

    def get_variable(  # noqa: D102, override
        self, name: str, as_backend_objs: bool = True
    ) -> xr.DataArray:
        variable = self.variables.get(name, None)
        if variable is None:
            raise KeyError(f"Unknown variable: {name}")
        if as_backend_objs or variable.isnull().all():
            return variable
        else:
            try:
                if not variable.dims:
                    da = xr.DataArray(self._instance.val(variable.item()))
                else:
                    df = pd.Series(
                        self._instance.vals(variable.to_series().dropna().to_dict())
                    ).reindex(variable.coords.to_index())
                    da = df.to_xarray()
                return da.rename(variable.name).assign_attrs(variable.attrs)
            except AttributeError:
                return self._apply_func(
                    self._expr_to_str, variable.notnull(), 1, variable
                )

    def get_variable_bounds(self, name: str) -> xr.Dataset:  # noqa: D102, override
        variable = self.get_variable(name, as_backend_objs=True)

        lb, ub = self._apply_func(
            self._from_highs_variable_bounds, variable.notnull(), 2, variable
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
            if not eval_body or not self._instance.getSolution().value_valid:
                return self._apply_func(
                    self._expr_to_str, expression.notnull(), 1, expression
                )
            else:
                return self._apply_func(
                    self._from_highs_expr,
                    expression.notnull(),
                    1,
                    expression,
                    col_values=self._instance.getSolution().col_value,
                )
        else:
            return expression

    def _solve(
        self, solve_config: config_schema.Solve, warmstart: bool = False
    ) -> xr.Dataset:
        self._instance.resetOptions()
        self._instance.clearSolver()
        if solve_config.solver_options is not None:
            for k, v in solve_config.solver_options.items():
                self._instance.setOptionValue(k, v)

        if warmstart:
            model_warn(
                "The chosen solver, highs, does not support warmstart, which may impact performance."
            )

        if solve_config.save_logs is not None:
            logdir = Path(solve_config.save_logs)
            self._instance.setOptionValue("log_file", (logdir / "highs.log").as_posix())

        self._instance.solve()
        termination = self._instance.getModelStatus()
        if termination == highspy.HighsModelStatus.kOptimal:
            results = self.load_results(
                solve_config.postprocessing_active, solve_config.zero_threshold
            )
        else:
            model_warn("Model solution was non-optimal.", _class=BackendWarning)
            results = xr.Dataset()

        termination = self._instance.modelStatusToString(termination).lower()
        results.attrs["termination_condition"] = str(termination)
        return results

    def verbose_strings(self) -> None:  # noqa: D102, override
        def __renamer(val, *idx, name: str):
            new_obj_name = f"{name}[{'__'.join(idx)}]"
            val.name = new_obj_name

        attribute_names = ["variables", "constraints", "piecewise_constraints"]
        with self._datetime_as_string(self._dataset):
            for da in self._dataset.filter_by_attrs(coords_in_name=False).values():
                if da.attrs["obj_type"] not in attribute_names:
                    continue
                self._apply_func(
                    __renamer,
                    da.notnull(),
                    1,
                    da,
                    *[da.coords[i].str.replace(" ", "_") for i in da.dims],
                    name=da.name,
                )
                da.attrs["coords_in_name"] = True

    def to_lp(self, path: str | Path) -> None:  # noqa: D102, override
        if Path(path).suffix != ".lp":
            raise ValueError("File extension must be `.lp`")
        self._instance.writeModel(str(path))

    def delete_component(self, key: str, component_type: ALL_COMPONENTS_T) -> None:
        """Delete object from the backend model object linked to a component.

        Args:
            key (str): Name of object
            component_type (str): Object type
        """
        if key not in self._dataset or self._dataset[key].obj_type != component_type:
            return
        if component_type in ("variables", "constraints"):
            deleted = self._collect_indices(self._dataset[key])
            del self._dataset[key]
            if deleted:
                sorted_deleted = sorted(deleted)
                arr = np.asarray(sorted_deleted, dtype=np.int32)
                if component_type == "variables":
                    self._instance.deleteCols(len(arr), arr)
                else:
                    self._instance.deleteRows(len(arr), arr)
                self._shift_stale_indices(sorted_deleted, component_type)
        else:
            del self._dataset[key]

    @staticmethod
    def _collect_indices(da: xr.DataArray) -> list[int]:
        """Gather the `.index` attribute of every non-null backend object in a DataArray."""
        indices: list[int] = []
        for val in da.values.ravel():
            if val is None or (isinstance(val, float) and np.isnan(val)):
                continue
            indices.append(int(val.index))
        return indices

    def _shift_stale_indices(
        self,
        sorted_deleted: list[int],
        component_type: Literal["variables", "constraints"],
    ) -> None:
        """Refresh `.index` on remaining backend objects after a batch delete.

        HiGHS shifts internal row/column indices on deleteRows/deleteCols, but
        the Python wrappers (`highs_var`, `highs_cons`, `highs_linear_expression`)
        hold copies of the old indices and must be corrected manually.
        """
        if not sorted_deleted:
            return

        if component_type == "variables":
            obj_targets = ("variables",)
            expr_targets = ("global_expressions", "objectives")
        else:
            obj_targets = ("constraints",)
            expr_targets = ()

        for da in self._dataset.data_vars.values():
            obj_type = da.attrs.get("obj_type")
            if obj_type in obj_targets:
                for val in da.values.ravel():
                    if val is None or (isinstance(val, float) and np.isnan(val)):
                        continue
                    val.index = val.index - bisect.bisect_left(
                        sorted_deleted, val.index
                    )
            elif obj_type in expr_targets:
                for val in da.values.ravel():
                    if not isinstance(val, highspy.highs.highs_linear_expression):
                        continue
                    val.idxs = [
                        i - bisect.bisect_left(sorted_deleted, i) for i in val.idxs
                    ]

    def update_input(  # noqa: D102, override
        self, name: str, new_values: xr.DataArray | SupportsFloat
    ) -> None:
        self._update_input(name, new_values, mutable=False)

    def update_variable_bounds(  # noqa: D102, override
        self,
        name: str,
        *,
        min: xr.DataArray | SupportsFloat | None = None,
        max: xr.DataArray | SupportsFloat | None = None,
    ) -> None:
        variable_da = self.get_variable(name)
        bound_das: dict = {}
        for bound_name, new_bounds in {"min": min, "max": max}.items():
            if new_bounds is None:
                self.log(
                    "variables",
                    name,
                    f"{bound_name} bound not being updated as it has not been defined.",
                )
                bound_das[bound_name] = xr.DataArray(np.nan)
                continue

            existing_bound_param = self.math.variables[name].bounds[bound_name]
            if existing_bound_param in self.parameters:
                raise BackendError(
                    "Cannot update variable bounds that have been set by parameters. "
                    f"Use `update_input('{existing_bound_param}')` to update the {bound_name} bound of {name}."
                )

            bound_das[bound_name] = xr.DataArray(new_bounds)
            missing_dims_in_new_vals = set(variable_da.dims).difference(
                bound_das[bound_name].dims
            )
            if missing_dims_in_new_vals:
                self.log(
                    "variables",
                    name,
                    f"New `{bound_name}` bounds will be broadcast along the {missing_dims_in_new_vals} dimension(s).",
                    "info",
                )

        self._apply_func(
            self._update_highs_variable,
            variable_da.notnull() & xr.DataArray(new_bounds).notnull(),
            1,
            variable_da,
            bound_das["min"],
            bound_das["max"],
        )

    def fix_variable(  # noqa: D102, override
        self, name: str, where: xr.DataArray | None = None
    ) -> None:
        if not self._instance.getSolution().value_valid:
            raise BackendError(
                "Cannot fix variable values without already having solved the model successfully."
            )

        variable_da = self.get_variable(name)
        if where is not None:
            variable_da = variable_da.where(where.fillna(0))

        self._apply_func(
            self._fix_highs_variable, variable_da.notnull(), 1, variable_da
        )

    def unfix_variable(  # noqa: D102, override
        self, name: str, where: xr.DataArray | None = None
    ) -> None:
        raise BackendError(
            "Cannot unfix a variable using the Highs backend; "
            "you will need to rebuild your backend or update variable bounds to match the original bounds."
        )

    @property
    def has_integer_or_binary_variables(self) -> bool:  # noqa: D102, override
        return any(
            self._instance.getColIntegrality(var.index)[1]
            == self.VARIABLE_DOMAIN_DICT["integer"]
            for var in self._instance.getVariables()
        )

    def _to_piecewise_constraint(  # noqa: D102, override
        self,
        x_var: highspy.highs.highs_var,
        y_var: highspy.highs.highs_var,
        *vals: float,
        name: str,
        n_breakpoints: int,
    ) -> None:
        raise NotImplementedError(
            "Piecewise constraints are not yet implemented for the Highs backend."
        )

    def _update_highs_variable(
        self, orig: highspy.highs.highs_var, lower_bound: float, upper_bound: float
    ) -> None:
        """Utility function to update highs variable bounds in-place.

        Args:
            orig (highspy.highs.highs_var): Highs variable to update.
            lower_bound (float): New variable lower bound.
            upper_bound (float): New variable upper bound.
        """
        orig_bounds = self._from_highs_variable_bounds(orig)
        lower_bound = orig_bounds.lb if pd.isna(lower_bound) else lower_bound
        upper_bound = orig_bounds.ub if pd.isna(upper_bound) else upper_bound
        self._instance.changeColBounds(orig.index, lower_bound, upper_bound)

    def _fix_highs_variable(self, orig: highspy.highs.highs_var) -> None:
        """Utility function to fix a Highs variable to its value in the optimisation model solution.

        Fixed variables will be considered as parameters in the subsequent solve.

        Args:
            orig (highspy.highs.highs_var): Highs variable to fix.

        Raises:
            BackendError: Can only fix variables if they have values assigned to them from an optimal solution.
        """
        bound = self._from_highs_var(orig)
        self._update_highs_variable(orig, bound, bound)  # type: ignore

    def _from_highs_variable_bounds(self, val: highspy.highs.highs_var) -> pd.Series:
        """Evaluate Highs decision variable object bounds.

        Args:
            val (highspy.highs.highs_var): Variable object to be evaluated.

        Returns:
            pd.Series: Array of variable upper and lower bound.
        """
        _, _, lb, ub, _ = self._instance.getCol(val.index)
        return pd.Series(data=[lb, ub], index=["lb", "ub"])

    def _from_highs_var(self, val: highspy.highs.highs_var) -> Any:
        """Evaluate Highs variable object.

        Args:
            val (highspy.LinExpr): expression object to be evaluated

        Returns:
            Any: If the input is nullable, return np.nan, otherwise a numeric value
            (eval_body=True and problem is optimised) or a string.
        """
        return self._instance.variableValue(val.index)

    def _from_highs_expr(
        self,
        val: highspy.LinExpr | highspy.highs.highs_var | float,
        *,
        col_values: list,
    ) -> int | float | None:
        """Evaluate Highs object in an expression array.

        Args:
            val (highspy.LinExpr | highspy.highs.highs_var | float): object to be evaluated; could be an expression, decision variable, or simple number stored in the global expression array.
            col_values (list): List of column values for variable evaluation.

        Returns:
            (int | float | None): the evaluated result.
        """
        if isinstance(val, highspy.highs.highs_linear_expression):
            return val.evaluate(col_values)
        elif isinstance(val, highspy.highs.highs_var):
            return self._instance.variableValue(val.index)
        elif isinstance(val, int | float):
            return val
        else:
            raise TypeError(
                f"Cannot convert highs object of type {type(val)} to a numeric value."
            )

    def _col_name(self, idx: int) -> str:
        """Return the variable name for a column index, or a generic fallback."""
        status, name = self._instance.getColName(int(idx))
        if status != highspy.HighsStatus.kOk or not name:
            return f"_v{int(idx)}"
        return name

    def _linexpr_to_str(self, expr: highspy.highs.highs_linear_expression) -> str:
        """Render a `highs_linear_expression` using variable names instead of column indices.

        The native `__str__` emits placeholders like `2.0_v0` regardless of whether
        column names have been set, which makes debug output and the
        `as_backend_objs=False` API surface useless for inspecting models.
        """
        parts: list[str] = []
        for idx, coef in zip(expr.idxs, expr.vals):
            name = self._col_name(idx)
            if coef == 1:
                parts.append(f"+ {name}")
            elif coef == -1:
                parts.append(f"- {name}")
            elif coef >= 0:
                parts.append(f"+ {coef:g}*{name}")
            else:
                parts.append(f"- {abs(coef):g}*{name}")
        if expr.constant:
            const = expr.constant
            if const >= 0:
                parts.append(f"+ {const:g}")
            else:
                parts.append(f"- {abs(const):g}")
        body = " ".join(parts).lstrip("+ ").strip()
        if not body:
            body = "0"
        return body

    def _expr_to_str(
        self,
        val: highspy.highs.highs_linear_expression | highspy.highs.highs_var | float,
    ) -> str:
        """Programmatic string repr for the values stored in expression/objective DataArrays."""
        if isinstance(val, highspy.highs.highs_linear_expression):
            return self._linexpr_to_str(val)
        if isinstance(val, highspy.highs.highs_var):
            return self._col_name(val.index)
        return str(val)

    def _cons_to_str(self, cons: highspy.highs.highs_cons) -> str:
        """Programmatic string repr for a `highs_cons`, using variable names and bounds."""
        expr = cons.expr()
        body = self._linexpr_to_str(expr)
        lb, ub = expr.bounds if expr.bounds is not None else (-np.inf, np.inf)
        if lb == ub:
            return f"{body} == {lb:g}"
        if np.isneginf(lb):
            return f"{body} <= {ub:g}"
        if np.isposinf(ub):
            return f"{body} >= {lb:g}"
        return f"{lb:g} <= {body} <= {ub:g}"


class HighsShadowPrices(backend_model.ShadowPrices):
    """Highs shadow price functionality."""

    def __init__(self, backend_obj: HighsBackendModel):
        """Create highspy shadow price functions."""
        self._backend_obj = backend_obj

    def get(self, name: str) -> xr.DataArray:  # noqa: D102, override
        constraint = self._backend_obj.get_constraint(name, as_backend_objs=True)
        return self._backend_obj._apply_func(
            self._duals_from_highs_constraint, constraint.notnull(), 1, constraint
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

    def _duals_from_highs_constraint(self, val: highspy.highs.highs_cons) -> float:
        try:
            dual = self._backend_obj._instance.constrDuals(val)  # type: ignore
        except AttributeError:
            return np.nan
        else:
            return dual
