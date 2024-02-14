# Copyright (C) since 2013 Calliope contributors listed in AUTHORS.
# Licensed under the Apache 2.0 License (see LICENSE file).

from __future__ import annotations

import logging
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from typing import Any, Iterator, Literal, Optional, SupportsFloat, TypeVar, Union

import gurobipy
import numpy as np
import pandas as pd
import xarray as xr

from calliope.backend import backend_model, parsing
from calliope.exceptions import BackendError, BackendWarning
from calliope.exceptions import warn as model_warn

T = TypeVar("T")
_COMPONENTS_T = Literal[
    "variables", "constraints", "objectives", "parameters", "global_expressions"
]

LOGGER = logging.getLogger(__name__)

COMPONENT_TRANSLATOR = {
    "parameter": "parameter",
    "variable": "variable",
    "global_expression": "expression",
    "constraint": "constraint",
    "objective": "objective",
}


class GurobiBackendModel(backend_model.BackendModel):
    def __init__(self, inputs: xr.Dataset, **kwargs) -> None:
        """Gurobi solver interface class.

        Args:
            inputs (xr.Dataset): Calliope model data.
        """
        super().__init__(inputs, gurobipy.Model(), **kwargs)
        self._instance: gurobipy.Model
        self.shadow_prices = GurobiShadowPrices(self)

        self._add_all_inputs_as_parameters()

    def add_parameter(
        self, parameter_name: str, parameter_values: xr.DataArray, default: Any = np.nan
    ) -> None:
        self._raise_error_on_preexistence(parameter_name, "parameters")

        parameter_da = parameter_values.fillna(default)
        if parameter_da.isnull().all():
            self.log(
                "parameters",
                parameter_name,
                "Component not added; no data found in array.",
            )
            parameter_da = parameter_da.astype(float)

        parameter_da.attrs["original_dtype"] = parameter_values.dtype
        self._add_to_dataset(parameter_name, parameter_da, "parameters", {})

    def add_constraint(
        self,
        name: str,
        constraint_dict: Optional[parsing.UnparsedConstraintDict] = None,
    ) -> None:
        def _constraint_setter(
            element: parsing.ParsedBackendEquation, where: xr.DataArray, references: set
        ) -> xr.DataArray:
            expr = element.evaluate_expression(self, where=where, references=references)
            func = np.frompyfunc(self._instance.addConstr, 1, 1)
            to_fill = func(expr, where=where.values)

            return to_fill

        self._add_component(name, constraint_dict, _constraint_setter, "constraints")

    def add_global_expression(
        self,
        name: str,
        expression_dict: Optional[parsing.UnparsedExpressionDict] = None,
    ) -> None:
        def _expression_setter(
            element: parsing.ParsedBackendEquation, where: xr.DataArray, references: set
        ) -> xr.DataArray:
            expr = element.evaluate_expression(self, where=where, references=references)
            expr = expr.squeeze(drop=True)

            to_fill = expr.where(where)
            self._clean_arrays(expr)
            return to_fill

        self._add_component(
            name, expression_dict, _expression_setter, "global_expressions"
        )

    def add_variable(
        self, name: str, variable_dict: Optional[parsing.UnparsedVariableDict] = None
    ) -> None:
        domain_dict = {"real": gurobipy.GRB.CONTINUOUS, "integer": gurobipy.GRB.INTEGER}
        if variable_dict is None:
            variable_dict = self.inputs.attrs["math"]["variables"][name]

        def _variable_setter(where: xr.DataArray, references: set):

            domain_type = domain_dict[variable_dict.get("domain", "real")]
            func = np.frompyfunc(
                partial(self._instance.addVar, vtype=domain_type), 2, 1
            )

            bounds = variable_dict["bounds"]
            lb = self._get_capacity_bound(bounds["min"], name, references)
            ub = self._get_capacity_bound(bounds["max"], name, references)
            var = func(
                lb.broadcast_like(where).values,
                ub.broadcast_like(where).values,
                where=where.values,
            )
            return where.copy(data=var).fillna(value=np.nan)

        self._add_component(name, variable_dict, _variable_setter, "variables")

    def add_objective(
        self, name: str, objective_dict: Optional[parsing.UnparsedObjectiveDict] = None
    ) -> None:
        min_ = gurobipy.GRB.MINIMIZE
        max_ = gurobipy.GRB.MAXIMIZE
        sense_dict = {
            "minimize": min_,
            "minimise": min_,
            "maximize": max_,
            "maximise": max_,
        }

        if objective_dict is None:
            objective_dict = self.inputs.attrs["math"]["objectives"][name]
        sense = sense_dict[objective_dict["sense"]]

        def _objective_setter(
            element: parsing.ParsedBackendEquation, where: xr.DataArray, references: set
        ) -> xr.DataArray:
            expr = element.evaluate_expression(self, references=references)

            if name == self.inputs.attrs["config"].build.objective:
                self._instance.setObjective(expr.item(), sense=sense)

                self.log("objectives", name, "Objective activated.")

            return xr.DataArray(expr)

        self._add_component(name, objective_dict, _objective_setter, "objectives")

    def get_parameter(self, name: str, as_backend_objs: bool = True) -> xr.DataArray:
        parameter = self.parameters.get(name, None)
        if parameter is None:
            raise KeyError(f"Unknown parameter: {name}")

        return parameter.astype(parameter.original_dtype)

    def get_constraint(
        self, name: str, as_backend_objs: bool = True, eval_body: bool = False
    ) -> Union[xr.DataArray, xr.Dataset]:
        constraint = self.constraints.get(name, None)
        if constraint is None:
            raise KeyError(f"Unknown constraint: {name}")
        if isinstance(constraint, xr.DataArray) and not as_backend_objs:
            raise BackendError("Cannot return a Gurobi constraint in string format")
        return constraint

    def get_variable(self, name: str, as_backend_objs: bool = True) -> xr.DataArray:
        variable = self.variables.get(name, None)
        if variable is None:
            raise KeyError(f"Unknown variable: {name}")
        if as_backend_objs:
            return variable
        else:
            try:
                func = np.frompyfunc(self._from_gurobi_var, 1, 1)
                return func(variable, where=variable.notnull().values).fillna(np.nan)
            except AttributeError:
                return variable.astype(str).where(variable.notnull())

    def get_variable_bounds(self, name: str) -> xr.Dataset:
        variable = self.get_variable(name, as_backend_objs=True)
        variable_attrs = self._apply_func(
            self._from_gurobi_variable_bounds,
            variable,
            output_core_dims=(["attributes"],),
        )
        variable_attrs.coords["attributes"] = ["lb", "ub"]
        return variable_attrs.to_dataset("attributes")

    def get_global_expression(
        self, name: str, as_backend_objs: bool = True, eval_body: bool = False
    ) -> xr.DataArray:
        global_expression = self.global_expressions.get(name, None)
        if global_expression is None:
            raise KeyError(f"Unknown global_expression: {name}")
        if isinstance(global_expression, xr.DataArray) and not as_backend_objs:
            if not eval_body:
                return global_expression.astype(str).where(global_expression.notnull())
            else:
                try:
                    func = np.frompyfunc(self._from_gurobi_expr, 1, 1)
                    return func(
                        global_expression, where=global_expression.notnull().values
                    ).fillna(np.nan)
                except AttributeError:
                    return global_expression.astype(str).where(
                        global_expression.notnull()
                    )
        else:
            return global_expression

    def _solve(
        self,
        solver: str,
        solver_io: Optional[str] = None,
        solver_options: Optional[dict] = None,
        save_logs: Optional[str] = None,
        warmstart: bool = False,
        **solve_config,
    ) -> xr.Dataset:
        self._instance.resetParams()

        if solver_options:
            for k, v in solver_options.items():
                self._instance.setParam(k, v)

        if not warmstart:
            self._instance.setParam("LPWarmStart", 0)

        if save_logs is not None:
            BackendWarning("No logs to save in Gurobi")

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

    def verbose_strings(self) -> None:
        def __renamer(val, *idx, name: str, attr: str):
            if pd.notnull(val):
                new_obj_name = f"{name}[{','.join(idx)}]"
                setattr(val, attr, new_obj_name)

        with self._datetime_as_string(self._dataset):
            for component_type, attr in {
                "variables": "VarName",
                "constraints": "ConstrName",
            }.items():
                for da in self._dataset.filter_by_attrs(
                    coords_in_name=False, **{"obj_type": component_type}
                ).values():
                    self._apply_func(
                        __renamer,
                        da,
                        *[da.coords[i] for i in da.dims],
                        name=da.name,
                        attr=attr,
                    )
                    da.attrs["coords_in_name"] = True

        self._instance.update()

    def to_lp(self, path: Union[str, Path]) -> None:
        self._instance.update()

        if Path(path).suffix != ".lp":
            raise ValueError("File extension must be `.lp`")
        self._instance.write(str(path))

    def _create_obj_list(self, key: str, component_type: _COMPONENTS_T) -> None:
        pass

    def delete_component(self, key: str, component_type: _COMPONENTS_T) -> None:
        """Delete object from the backend model object linked to a component.

        Args:
            key (str): Name of object
            component_type (str): Object type
        """
        if key in self._dataset and self._dataset[key].obj_type == component_type:
            if component_type in ["variables", "constraints"]:
                self._apply_func(self._del_gurobi_obj, self._dataset[key])
            del self._dataset[key]
        self._instance.update()

    def update_parameter(
        self, name: str, new_values: Union[xr.DataArray, SupportsFloat]
    ) -> None:
        new_values = xr.DataArray(new_values)
        parameter_da = self.get_parameter(name)
        missing_dims_in_new_vals = set(parameter_da.dims).difference(new_values.dims)

        if missing_dims_in_new_vals:
            self.log(
                "parameters",
                name,
                f"New values will be broadcast along the {missing_dims_in_new_vals} dimension(s)."
                "info",
            )
        new_parameter_da = new_values.broadcast_like(parameter_da).fillna(parameter_da)
        new_parameter_da.attrs = parameter_da.attrs

        self.delete_component(name, "parameters")
        self.add_parameter(
            name,
            new_parameter_da,
            default=self.inputs.attrs["defaults"].get(name, np.nan),
        )

        refs_to_update = self._find_all_references(parameter_da.attrs["references"])

        if refs_to_update:
            self.log(
                "parameters",
                name,
                f"The optimisation problem components {sorted(refs_to_update)} will be re-built.",
                "info",
            )
        self._rebuild_references(refs_to_update)
        self._instance.update()

    def update_variable_bounds(
        self,
        name: str,
        *,
        min: Optional[Union[xr.DataArray, SupportsFloat]] = None,
        max: Optional[Union[xr.DataArray, SupportsFloat]] = None,
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

            existing_bound_param = self.inputs.attrs["math"].get_key(
                f"variables.{name}.bounds.{bound_name}", None
            )
            if existing_bound_param in self.parameters:
                raise BackendError(
                    "Cannot update variable bounds that have been set by parameters. "
                    f"Use `update_parameter('{existing_bound_param}')` to update the {bound_name} bound of {name}."
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
                variable_da,
                xr.DataArray(new_bounds),
                bound=translator[bound_name],
            )
        self._instance.update()

    def fix_variable(self, name: str, where: Optional[xr.DataArray] = None) -> None:
        if self._instance.status != gurobipy.GRB.OPTIMAL:
            raise BackendError(
                "Cannot fix variable values without already having solved the model successfully."
            )

        variable_da = self.get_variable(name)
        if where is not None:
            variable_da = variable_da.where(where.fillna(0))
        self._apply_func(self._fix_gurobi_variable, variable_da)
        self._instance.update()

    def unfix_variable(self, name: str, where: Optional[xr.DataArray] = None) -> None:
        raise BackendError(
            "Cannot unfix a variable using the Gurobi backend; you will need to rebuild your backend or update variable bounds to match the original bounds."
        )

    def _del_gurobi_obj(self, obj: Any) -> None:
        if not pd.isnull(obj):
            self._instance.remove(obj)

    def _get_capacity_bound(
        self, bound: Any, name: str, references: set
    ) -> xr.DataArray:
        """
        Generate array for the upper/lower bound of a decision variable.
        Any NaN values will be replaced by None, which we will later interpret as there being no bound to apply.

        Args:
            bound (Any): The bound name (corresponding to an array in the model input data) or value.
            name (str): Name of decision variable.

        Returns:
            xr.DataArray: Where unbounded, the array entry will be None, otherwise a float value.
        """

        if isinstance(bound, str):
            self.log(
                "variables",
                name,
                f"Applying bound according to the {bound} parameter values.",
            )
            bound_array = self.get_parameter(bound)
            references.add(bound)
        else:
            bound_array = xr.DataArray(bound)

        return bound_array.fillna(None)

    def _update_gurobi_variable(
        self, orig: Optional[gurobipy.Var], new: Any, *, bound: Literal["lb", "ub"]
    ) -> None:
        """Utility function to update gurobi variable bounds in-place.

        Args:
            orig (gurobipy.Var): Gurobi variable to update.
        Keyword Args:
            lb (Any): Value with which to update the lower bound of the variable.
            ub (Any): Value with which to update the upper bound of the variable.
        """
        if pd.notnull(orig) and pd.notnull(new):  # type: ignore
            setattr(orig, bound, new)

    def _fix_gurobi_variable(self, orig: Optional[gurobipy.Var]) -> None:
        """Utility function to fix a Gurobi variable to its value in the optimisation model solution.
        Fixed variables will be considered as parameters in the subsequent solve.

        Args:
            orig (gurobipy.Var): Gurobi variable to fix.

        Raises:
            BackendError: Can only fix variables if they have values assigned to them from an optimal solution.
        """
        if pd.isnull(orig):  # type: ignore
            return None
        try:
            orig.x  # type: ignore
        except AttributeError:
            raise BackendError(
                "Cannot fix variable values without already having solved the model successfully."
            )
        else:
            self._update_gurobi_variable(orig, orig.x, bound="lb")  # type: ignore
            self._update_gurobi_variable(orig, orig.x, bound="ub")  # type: ignore

    @staticmethod
    def _from_gurobi_variable_bounds(val: Optional[gurobipy.Var]) -> pd.Series:
        """Evaluate Gurobi decision variable object bounds.

        Args:
            val (gurobipy.Var): Variable object to be evaluated.

        Returns:
            pd.Series: Array of variable upper and lower bound.
        """
        if pd.isnull(val):  # type: ignore
            vals = [np.nan, np.nan]
        else:
            vals = [val.lb, val.ub]  # type: ignore
        return pd.Series(data=vals, index=["lb", "ub"])

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
    def _from_gurobi_expr(val: gurobipy.LinExpr, *, eval_body: bool = False) -> Any:
        """Evaluate Gurobi expression object.

        Args:
            val (gurobipy.LinExpr): expression object to be evaluated
            eval_body (bool, optional):
                If True, attempt to evaluate the expression object, which will produce a numeric value.
                This will only succeed if the backend model has been successfully optimised,
                otherwise a string representation of the linear expression will be returned
                (same as eval_body=False). Defaults to False.

        Returns:
            Any: If the input is nullable, return np.nan, otherwise a numeric value
            (eval_body=True and problem is optimised) or a string.
        """
        return val.getValue()

    @contextmanager
    def _datetime_as_string(self, data: Union[xr.DataArray, xr.Dataset]) -> Iterator:
        """Context manager to temporarily convert np.dtype("datetime64[ns]") coordinates (e.g. timesteps) to strings with a resolution of minutes.

        Args:
            data (Union[xr.DataArray, xr.Dataset]): xarray object on whose coordinates the conversion will take place.
        """
        datetime_coords = set()
        for name_, vals_ in data.coords.items():
            if vals_.dtype.kind == "M":
                data.coords[name_] = data.coords[name_].dt.strftime("%Y-%m-%d %H:%M")
                datetime_coords.add(name_)
        try:
            yield
        finally:
            for name_ in datetime_coords:
                data.coords[name_] = xr.apply_ufunc(
                    pd.to_datetime, data.coords[name_], keep_attrs=True
                )


class GurobiShadowPrices(backend_model.ShadowPrices):
    def __init__(self, backend_obj: GurobiBackendModel):
        self._backend_obj = backend_obj
        self.deactivate()

    def get(self, name: str) -> xr.DataArray:
        constraint = self._backend_obj.get_constraint(name, as_backend_objs=True)
        return self._backend_obj._apply_func(
            self._duals_from_gurobi_constraint, constraint
        )

    def activate(self):
        pass

    def deactivate(self):
        pass

    @property
    def is_active(self) -> bool:
        return True

    @staticmethod
    def _duals_from_gurobi_constraint(val: Optional[gurobipy.Constr]) -> float:
        if pd.isnull(val):  # type: ignore
            return np.nan
        else:
            try:
                dual = val.Pi  # type: ignore
            except AttributeError:
                return np.nan
            else:
                return dual
