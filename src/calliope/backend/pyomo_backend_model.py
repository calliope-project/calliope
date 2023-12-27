# Copyright (C) since 2013 Calliope contributors listed in AUTHORS.
# Licensed under the Apache 2.0 License (see LICENSE file).

from __future__ import annotations

import logging
import os
import re
from abc import ABC
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from typing import (
    Any,
    Iterable,
    Iterator,
    Literal,
    Optional,
    SupportsFloat,
    TypeVar,
    Union,
)

import numpy as np
import pandas as pd
import pyomo.environ as pe  # type: ignore
import pyomo.kernel as pmo  # type: ignore
import xarray as xr
from pyomo.common.tempfiles import TempfileManager  # type: ignore
from pyomo.opt import SolverFactory  # type: ignore

from calliope.backend import backend_model, parsing
from calliope.exceptions import BackendError, BackendWarning
from calliope.exceptions import warn as model_warn
from calliope.util.logging import LogWriter

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


class PyomoBackendModel(backend_model.BackendModel):
    def __init__(self, inputs: xr.Dataset, **kwargs):
        super().__init__(inputs, pmo.block(), **kwargs)

        self._instance.parameters = pmo.parameter_dict()
        self._instance.variables = pmo.variable_dict()
        self._instance.global_expressions = pmo.expression_dict()
        self._instance.constraints = pmo.constraint_dict()
        self._instance.objectives = pmo.objective_dict()

        self._add_all_inputs_as_parameters()

    def add_parameter(
        self,
        parameter_name: str,
        parameter_values: xr.DataArray,
        default: Any = np.nan,
        use_inf_as_na: bool = False,
    ) -> None:
        self._raise_error_on_preexistence(parameter_name, "parameters")

        self._create_obj_list(parameter_name, "parameters")

        parameter_da = self._apply_func(
            self._to_pyomo_param,
            parameter_values,
            name=parameter_name,
            default=default,
            use_inf_as_na=use_inf_as_na,
        )
        if parameter_da.isnull().all():
            self.log(
                "parameters",
                parameter_name,
                "Component not added; no data found in array.",
            )
            self.delete_component(parameter_name, "parameters")
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

            to_fill = self._apply_func(
                self._to_pyomo_constraint, where, expr, name=name
            )
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

            to_fill = self._apply_func(
                self._to_pyomo_expression, where, expr, name=name
            )
            self._clean_arrays(expr)
            return to_fill

        self._add_component(
            name, expression_dict, _expression_setter, "global_expressions"
        )

    def add_variable(
        self, name: str, variable_dict: Optional[parsing.UnparsedVariableDict] = None
    ) -> None:
        domain_dict = {"real": pmo.RealSet, "integer": pmo.IntegerSet}

        if variable_dict is None:
            variable_dict = self.inputs.attrs["math"]["variables"][name]

        def _variable_setter(where):
            domain_type = domain_dict[variable_dict.get("domain", "real")]

            return self._apply_func(
                self._to_pyomo_variable,
                where,
                self._get_capacity_bound(variable_dict["bounds"]["max"], name=name),
                self._get_capacity_bound(variable_dict["bounds"]["min"], name=name),
                name=name,
                domain_type=domain_type,
            )

        self._add_component(name, variable_dict, _variable_setter, "variables")

    def add_objective(
        self, name: str, objective_dict: Optional[parsing.UnparsedObjectiveDict] = None
    ) -> None:
        sense_dict = {"minimize": 1, "minimise": 1, "maximize": -1, "maximise": -1}

        if objective_dict is None:
            objective_dict = self.inputs.attrs["math"]["objectives"][name]
        sense = sense_dict[objective_dict["sense"]]

        def _objective_setter(
            element: parsing.ParsedBackendEquation, where: xr.DataArray, references: set
        ) -> xr.DataArray:
            expr = element.evaluate_expression(self, references=references)
            objective = pmo.objective(expr.item(), sense=sense)
            if name == self.inputs.attrs["config"].build.objective:
                text = "activated"
                objective.activate()
            else:
                text = "deactivated"
                objective.deactivate()
            self.log("objectives", name, f"Objective {text}.")

            self._instance.objectives[name].append(objective)
            return xr.DataArray(objective)

        self._add_component(name, objective_dict, _objective_setter, "objectives")

    def get_parameter(self, name: str, as_backend_objs: bool = True) -> xr.DataArray:
        parameter = self.parameters.get(name, None)
        if parameter is None:
            raise KeyError(f"Unknown parameter: {name}")

        if as_backend_objs or not isinstance(parameter, xr.DataArray):
            return parameter

        param_as_vals = self._apply_func(self._from_pyomo_param, parameter)
        if parameter.original_dtype.kind == "M":  # i.e., np.datetime64
            self.log("parameters", name, "Converting Pyomo object to datetime dtype.")
            return xr.apply_ufunc(pd.to_datetime, param_as_vals)
        else:
            self.log(
                "parameters",
                name,
                f"Converting Pyomo object to {parameter.original_dtype} dtype.",
            )
            return param_as_vals.astype(parameter.original_dtype)

    def get_constraint(
        self, name: str, as_backend_objs: bool = True, eval_body: bool = False
    ) -> Union[xr.DataArray, xr.Dataset]:
        constraint = self.constraints.get(name, None)
        if constraint is None:
            raise KeyError(f"Unknown constraint: {name}")
        if isinstance(constraint, xr.DataArray) and not as_backend_objs:
            constraint_attrs = self._apply_func(
                self._from_pyomo_constraint,
                constraint,
                eval_body=eval_body,
                output_core_dims=(["attributes"],),
            )
            constraint_attrs.coords["attributes"] = ["lb", "body", "ub"]
            constraint = constraint_attrs.to_dataset("attributes")
        return constraint

    def get_variable(self, name: str, as_backend_objs: bool = True) -> xr.DataArray:
        variable = self.variables.get(name, None)
        if variable is None:
            raise KeyError(f"Unknown variable: {name}")
        if as_backend_objs:
            return variable
        else:
            return self._apply_func(self._from_pyomo_param, variable)

    def get_variable_bounds(self, name: str) -> xr.Dataset:
        variable = self.get_variable(name, as_backend_objs=True)
        variable_attrs = self._apply_func(
            self._from_pyomo_variable_bounds,
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
            return self._apply_func(
                self._from_pyomo_expr, global_expression, eval_body=eval_body
            )
        else:
            return global_expression

    def solve(
        self,
        solver: str,
        solver_io: Optional[str] = None,
        solver_options: Optional[dict] = None,
        save_logs: Optional[str] = None,
        warmstart: bool = False,
        **solve_kwargs,
    ):
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
                "The chosen solver, {}, does not support warmstart, which may "
                "impact performance.".format(solver)
            )
            warmstart = False

        with redirect_stdout(LogWriter(self._solve_logger, "debug", strip=True)):  # type: ignore
            with redirect_stderr(LogWriter(self._solve_logger, "error", strip=True)):  # type: ignore
                # Ignore most of gurobipy's logging, as it's output is
                # already captured through STDOUT
                logging.getLogger("gurobipy").setLevel(logging.ERROR)
                results = opt.solve(self._instance, tee=True, **solve_kwargs)

        termination = results.solver[0].termination_condition

        if termination == pe.TerminationCondition.optimal:
            self._instance.load_solution(results.solution[0])

        else:
            self._solve_logger.critical("Problem status:")
            for line in str(results.problem[0]).split("\n"):
                self._solve_logger.critical(line)
            self._solve_logger.critical("Solver status:")
            for line in str(results.solver[0]).split("\n"):
                self._solve_logger.critical(line)

            model_warn("Model solution was non-optimal.", _class=BackendWarning)

        return str(termination)

    def verbose_strings(self) -> None:
        def __renamer(val, *idx):
            if pd.notnull(val):
                val.calliope_coords = idx

        with self._datetime_as_string(self._dataset):
            for component_type in ["parameters", "variables", "constraints"]:
                for da in self._dataset.filter_by_attrs(
                    coords_in_name=False, **{"obj_type": component_type}
                ).values():
                    self._apply_func(__renamer, da, *[da.coords[i] for i in da.dims])
                    da.attrs["coords_in_name"] = True

    def to_lp(self, path: Union[str, Path]) -> None:
        self._instance.write(str(path), format="lp", symbolic_solver_labels=True)

    def _create_obj_list(self, key: str, component_type: _COMPONENTS_T) -> None:
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

    def delete_component(self, key: str, component_type: _COMPONENTS_T) -> None:
        """Delete a list object from the backend model object.

        Args:
            key (str): Name of object
            component_type (str): Object type
        """
        component_dict = getattr(self._instance, component_type)
        if key in component_dict:
            del component_dict[key]

        if key in self._dataset and self._dataset[key].obj_type == component_type:
            del self._dataset[key]

    def update_parameter(
        self, name: str, new_values: Union[xr.DataArray, SupportsFloat]
    ) -> None:
        new_values = xr.DataArray(new_values)
        parameter_da = self.get_parameter(name)
        missing_dims_in_new_vals = set(parameter_da.dims).difference(new_values.dims)
        missing_dims_in_orig_vals = set(new_values.dims).difference(parameter_da.dims)
        refs_to_update: set = set()

        if (
            (not parameter_da.shape and new_values.shape)
            or missing_dims_in_orig_vals
            or (parameter_da.isnull() & new_values.notnull()).any()
        ):
            refs_to_update = self._find_all_references(parameter_da.attrs["references"])
            if refs_to_update:
                self.log(
                    "parameters",
                    name,
                    "Defining values for a previously fully/partially undefined parameter. "
                    f"The optimisation problem components {sorted(refs_to_update)} will be re-built.",
                    "info",
                )
            self.delete_component(name, "parameters")
            self.add_parameter(
                name,
                new_values,
                default=self.inputs.attrs["defaults"].get(name, np.nan),
            )
            for ref in refs_to_update:
                self._rebuild_reference(ref)
            return None

        if missing_dims_in_new_vals:
            self.log(
                "parameters",
                name,
                f"New values will be broadcast along the {missing_dims_in_new_vals} dimension(s)."
                "info",
            )

        self._apply_func(self._update_pyomo_param, parameter_da, new_values)

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
                self._update_pyomo_variable,
                variable_da,
                xr.DataArray(new_bounds),
                bound=translator[bound_name],
            )

    def fix_variable(self, name: str, where: Optional[xr.DataArray] = None) -> None:
        variable_da = self.get_variable(name)
        if where is not None:
            variable_da = variable_da.where(where.fillna(0))
        self._apply_func(self._fix_pyomo_variable, variable_da)

    def unfix_variable(self, name: str, where: Optional[xr.DataArray] = None) -> None:
        variable_da = self.get_variable(name)
        if where is not None:
            variable_da = variable_da.where(where.fillna(0))
        self._apply_func(self._unfix_pyomo_variable, variable_da)

    def _get_capacity_bound(self, bound: Any, name: str) -> xr.DataArray:
        """
        Generate array for the upper/lower bound of a decision variable.
        Any NaN values will be replaced by None, which Pyomo will correctly interpret as there being no bound to apply.

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
        else:
            bound_array = xr.DataArray(bound)

        return bound_array.fillna(None)

    def _to_pyomo_param(
        self, val: Any, *, name: str, default: Any = np.nan, use_inf_as_na: bool = True
    ) -> Union[type[ObjParameter], float]:
        """
        Utility function to generate a pyomo parameter for every element of an
        xarray DataArray.
        Output objects are of the type ObjParameter(pmo.parameter) since they need a
        "dtype" property to be handled by xarray.

        If not np.nan/None, output objects are also added to the backend model object in-place.


        Args:
            val (Any): Value to turn into a mutable pyomo parameter
            name (str): Name of parameter
            default (Any, optional): Default value if `val` is None/np.nan. Defaults to np.nan.
            use_inf_as_na (bool, optional): If True, see np.inf as np.nan. Defaults to True.

        Returns:
            Union[type[ObjParameter], float]:
                If both `val` and `default` are np.nan/None, return np.nan.
                Otherwise return ObjParameter(val/default).
        """
        if use_inf_as_na:
            val = np.nan if val in [np.inf, -np.inf] else val
            default = np.nan if default in [np.inf, -np.inf] else default
        if pd.isnull(val):
            if pd.isnull(default):
                param = np.nan
            else:
                param = ObjParameter(default)
                self._instance.parameters[name].append(param)
        else:
            param = ObjParameter(val)
            self._instance.parameters[name].append(param)
        return param

    def _update_pyomo_param(self, orig: ObjParameter, new: Any) -> None:
        """Utility function to update pyomo parameter values in-place.

        Args:
            orig (ObjParameter): Pyomo parameter to update.
            new (Any): Value with which to update the parameter.
        """
        if pd.notnull(new):
            orig.value = new

    def _update_pyomo_variable(
        self, orig: ObjVariable, new: Any, *, bound: Literal["lb", "ub"]
    ) -> None:
        """Utility function to update pyomo variable bounds in-place.

        Args:
            orig (ObjVariable): Pyomo variable to update.
        Keyword Args:
            lb (Any): Value with which to update the lower bound of the variable.
            ub (Any): Value with which to update the upper bound of the variable.
        """
        if pd.notnull(orig) and pd.notnull(new):
            setattr(orig, bound, new)

    def _fix_pyomo_variable(self, orig: ObjVariable) -> None:
        """Utility function to fix a pyomo variable to its value in the optimisation model solution.
        Fixed variables will be considered as parameters in the subsequent solve.

        Args:
            orig (ObjVariable): Pyomo variable to fix.

        Raises:
            BackendError: Can only fix variables if they have values assigned to them from an optimal solution.
        """
        if pd.isnull(orig):
            return None
        elif orig.value is None:
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
        if pd.notnull(orig):
            orig.unfix()

    def _to_pyomo_constraint(
        self, mask: Union[bool, np.bool_], expr: Any, *, name: str
    ) -> Union[type[ObjConstraint], float]:
        """
        Utility function to generate a pyomo constraint for every element of an
        xarray DataArray.

        If not np.nan/None, output objects are also added to the backend model object in-place.

        Args:
            mask (Union[bool, np.bool_]): If True, add constraint, otherwise return np.nan
            expr (Any): Equation expression.

        Kwargs:
            name (str): Name of constraint

        Returns:
            Union[type[ObjConstraint], float]:
                If mask is True, return np.nan.
                Otherwise return pmo_constraint(expr=lhs op rhs).
        """

        if mask:
            constraint = ObjConstraint(expr=expr)
            self._instance.constraints[name].append(constraint)
            return constraint
        else:
            return np.nan

    def _to_pyomo_expression(
        self, mask: Union[bool, np.bool_], expr: Any, *, name: str
    ) -> Union[type[pmo.expression], float]:
        """
        Utility function to generate a pyomo expression for every element of an
        xarray DataArray.

        If not np.nan/None, output objects are also added to the backend model object in-place.


        Args:
            mask (Union[bool, np.bool_]): If True, add expression, otherwise return np.nan.
            expr (Any): Linear expression to add.
        Kwargs:
            name (str): Expression name.

        Returns:
            Union[type[pmo.expression], float]:
                If mask is True, return np.nan.
                Otherwise return pmo_expression(expr).
        """
        if mask:
            expr_obj = pmo.expression(expr)
            self._instance.global_expressions[name].append(expr_obj)
            return expr_obj
        else:
            return np.nan

    def _to_pyomo_variable(
        self,
        mask: Union[bool, np.bool_],
        ub: Any,
        lb: Any,
        *,
        name: str,
        domain_type: Literal["RealSet", "IntegerSet"],
    ) -> Union[type[ObjVariable], float]:
        """
        Utility function to generate a pyomo decision variable for every element of an
        xarray DataArray.

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
        if mask:
            var = ObjVariable(ub=ub, lb=lb, domain_type=domain_type)
            self._instance.variables[name].append(var)
            return var
        else:
            return np.nan

    @staticmethod
    def _from_pyomo_param(val: Union[ObjParameter, ObjVariable, float]) -> Any:
        """
        Evaluate value of Pyomo object.
        If the input object is a parameter, a numeric/string value will be given.
        If the input object is a global expression or variable, a numeric value will be given
        only if the backend model has been successfully optimised, otherwise evaluation will return None.

        Args:
            val (Union[ObjParameter, pmo.expression, ObjVariable, np.nan]):
                Item to be evaluated.

        Returns:
            Any: If the input is nullable, return np.nan, otherwise evaluate the "value" method of the object.
        """
        if pd.isnull(val):
            return np.nan
        else:
            return val.value  # type: ignore

    @staticmethod
    def _from_pyomo_constraint(
        val: ObjConstraint, *, eval_body: bool = False
    ) -> pd.Series:
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
        if pd.isnull(val):
            vals = [np.nan, np.nan, np.nan]
        else:
            if eval_body:
                try:
                    body = val.body()
                except ValueError:
                    body = val.body.to_string()
            else:
                body = val.body.to_string()
            vals = [val.lb, body, val.ub]
        return pd.Series(data=vals, index=["lb", "body", "ub"])

    @staticmethod
    def _from_pyomo_variable_bounds(val: ObjVariable) -> pd.Series:
        """Evaluate Pyomo decision variable object bounds.

        Args:
            val (ObjVariable): Variable object to be evaluated.

        Returns:
            pd.Series: Array of variable upper and lower bound.
        """
        if pd.isnull(val):
            vals = [np.nan, np.nan]
        else:
            vals = [val.lb, val.ub]
        return pd.Series(data=vals, index=["lb", "ub"])

    @staticmethod
    def _from_pyomo_expr(val: pmo.expression, *, eval_body: bool = False) -> Any:
        """Evaluate Pyomo expression object.

        Args:
            val (pmo.expression): expression object to be evaluated
            eval_body (bool, optional):
                If True, attempt to evaluate the expression object, which will produce a numeric value.
                This will only succeed if the backend model has been successfully optimised,
                otherwise a string representation of the linear expression will be returned
                (same as eval_body=False). Defaults to False.

        Returns:
            Any: If the input is nullable, return np.nan, otherwise a numeric value
            (eval_body=True and problem is optimised) or a string.
        """
        if pd.isnull(val):
            return np.nan
        else:
            if eval_body:
                expr = val()
                if expr is None:
                    return val.to_string()
                else:
                    return expr
            else:
                return val.to_string()

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


class CoordObj(ABC):
    """Class with methods to update the `name` property of inheriting classes"""

    def __init__(self) -> None:
        self._calliope_coords: Optional[Iterable] = None

    def _update_name(self, old_name: str) -> str:
        """
        Update string of a list containing a single number with a string of a list containing any arbitrary number of elements

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
        return self._calliope_coords

    @calliope_coords.setter
    def calliope_coords(self, val):
        self._calliope_coords = val


class ObjParameter(pmo.parameter, CoordObj):
    """
    A pyomo parameter (`a object for storing a mutable, numeric value that can be used to build a symbolic expression`)
    with added `dtype` property and a `name` property setter (via the `pmo.parameter.getname` method) which replaces a list position as a name with a list of strings.
    """

    def __init__(self, value, **kwds):
        assert not pd.isnull(value)
        pmo.parameter.__init__(self, value, **kwds)
        CoordObj.__init__(self)

    @property
    def dtype(self):
        return "O"

    def getname(self, *args, **kwargs):
        return self._update_name(pmo.parameter.getname(self, *args, **kwargs))


class ObjVariable(pmo.variable, CoordObj):
    """
    A pyomo variable with a `name` property setter (via the `pmo.variable.getname` method) which replaces a list position as a name with a list of strings.

    """

    def __init__(self, **kwds):
        pmo.variable.__init__(self, **kwds)
        CoordObj.__init__(self)

    def getname(self, *args, **kwargs):
        return self._update_name(pmo.variable.getname(self, *args, **kwargs))


class ObjConstraint(pmo.constraint, CoordObj):
    """
    A pyomo constraint with a `name` property setter (via the `pmo.constraint.getname` method) which replaces a list position as a name with a list of strings.

    """

    def __init__(self, **kwds):
        pmo.constraint.__init__(self, **kwds)
        CoordObj.__init__(self)

    def getname(self, *args, **kwargs):
        return self._update_name(pmo.constraint.getname(self, *args, **kwargs))
