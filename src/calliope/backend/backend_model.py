# Copyright (C) since 2013 Calliope contributors listed in AUTHORS.
# Licensed under the Apache 2.0 License (see LICENSE file).

from __future__ import annotations

import importlib
import logging
import typing
from abc import ABC, abstractmethod
from copy import deepcopy
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generic,
    Literal,
    Optional,
    SupportsFloat,
    TypeVar,
    Union,
)

import numpy as np
import xarray as xr

from calliope import exceptions
from calliope.attrdict import AttrDict
from calliope.backend import helper_functions, parsing
from calliope.util.schema import MATH_SCHEMA, update_then_validate_config, validate_dict

if TYPE_CHECKING:
    from calliope.backend.parsing import T as Tp

from calliope.exceptions import BackendError

T = TypeVar("T")
_COMPONENTS_T = Literal[
    "variables", "constraints", "objectives", "parameters", "global_expressions"
]

LOGGER = logging.getLogger(__name__)


class BackendModelGenerator(ABC):
    _VALID_COMPONENTS: tuple[_COMPONENTS_T, ...] = typing.get_args(_COMPONENTS_T)
    _COMPONENT_ATTR_METADATA = ["description", "unit"]

    def __init__(self, inputs: xr.Dataset, **kwargs):
        """Abstract base class to build a representation of the optimisation problem.

        Args:
            inputs (xr.Dataset): Calliope model data.
        """

        self._dataset = xr.Dataset()
        self.inputs = inputs.copy()
        self.inputs.attrs = deepcopy(inputs.attrs)
        self.inputs.attrs["config"]["build"] = update_then_validate_config(
            "build", self.inputs.attrs["config"], **kwargs
        )
        self._check_inputs()

        self._solve_logger = logging.getLogger(__name__ + ".<solve>")

    @abstractmethod
    def add_parameter(
        self,
        parameter_name: str,
        parameter_values: xr.DataArray,
        default: Any = np.nan,
        use_inf_as_na: bool = False,
    ) -> None:
        """
        Add input parameter to backend model in-place.
        If the backend interface allows for mutable parameter objects, they will be
        generated, otherwise a copy of the model input dataset will be used.
        In either case, NaN values are filled with the given parameter default value.

        Args:
            parameter_name (str): Name of parameter.
            parameter_values (xr.DataArray): Array of parameter values.
            default (Any, optional):
                Default value to fill NaN entries in parameter values array.
                Defaults to np.nan.
            use_inf_as_na (bool, optional):
                If True, will consider np.inf parameter value entries as np.nan and
                consequently try to fill those entries with the parameter default value.
                Defaults to False.
        """

    @abstractmethod
    def add_constraint(
        self, name: str, constraint_dict: parsing.UnparsedConstraintDict
    ) -> None:
        """
        Add constraint equation to backend model in-place.
        Resulting backend dataset entries will be constraint objects.

        Args:
            model_data (xr.Dataset):
                Calliope model data with which to create an array mask - only those
                dataset entries in the mask will be generated.
            name (str):
                Name of the constraint
            constraint_dict (parsing.UnparsedConstraintDict):
                Constraint configuration dictionary, ready to be parsed and then evaluated.
        """

    @abstractmethod
    def add_global_expression(
        self, name: str, expression_dict: parsing.UnparsedExpressionDict
    ) -> None:
        """
        Add global expression (arithmetic combination of parameters and/or decision variables)
        to backend model in-place.
        Resulting backend dataset entries will be linear expression objects.

        Args:
            model_data (xr.Dataset):
                Calliope model data with which to create an array mask - only those
                dataset entries in the mask will be generated.
            name (str):
                Name of the global expression
            expression_dict (parsing.UnparsedExpressionDict):
                Global expression configuration dictionary, ready to be parsed and then evaluated.
        """

    @abstractmethod
    def add_variable(
        self, name: str, variable_dict: parsing.UnparsedVariableDict
    ) -> None:
        """
        Add decision variable to backend model in-place.
        Resulting backend dataset entries will be decision variable objects.

        Args:
            model_data (xr.Dataset):
                Calliope model data with which to create an array mask - only those
                dataset entries in the mask will be generated.
            name (str):
                Name of the variable.
            variable_dict (parsing.UnparsedVariableDict):
                Variable configuration dictionary, ready to be parsed and then evaluated.
        """

    @abstractmethod
    def add_objective(
        self, name: str, objective_dict: parsing.UnparsedObjectiveDict
    ) -> None:
        """
        Add objective arithmetic to backend model in-place.
        Resulting backend dataset entry will be a single, unindexed objective object.

        Args:
            model_data (xr.Dataset):
                Calliope model data with which to create a constraint mask - only those
                dataset entries in the mask will be generated.
            name (str):
                Name of the objective.
            objective_dict (parsing.UnparsedObjectiveDict):
                Objective configuration dictionary, ready to be parsed and then evaluated.
        """

    def log(
        self,
        component_type: _COMPONENTS_T,
        component_name: str,
        message: str,
        level: Literal["info", "warning", "debug", "error", "critical"] = "debug",
    ):
        """Log to module-level logger with some prettification of the message

        Args:
            message (str): Message to log.
            level (Literal["info", "warning", "debug", "error", "critical"], optional): Log level. Defaults to "debug".
        """
        getattr(LOGGER, level)(
            f"Optimisation model | {component_type}:{component_name} | {message}"
        )

    def _check_inputs(self):
        data_checks = AttrDict.from_yaml(
            importlib.resources.files("calliope") / "config" / "model_data_checks.yaml"
        )
        errors = []
        warnings = []
        parser_ = parsing.where_parser.generate_where_string_parser()
        eval_kwargs = {
            "equation_name": "",
            "backend_interface": self,
            "input_data": self.inputs,
            "helper_functions": helper_functions._registry["where"],
            "apply_where": True,
        }
        for failure_check in data_checks["fail"]:
            parsed_ = parser_.parse_string(failure_check["where"], parse_all=True)
            failed = parsed_[0].eval("array", **eval_kwargs)
            if failed.any():
                errors.append(failure_check["message"])

        for warning_check in data_checks["warn"]:
            parsed_ = parser_.parse_string(warning_check["where"], parse_all=True)
            warned = parsed_[0].eval("array", **eval_kwargs)
            if warned.any():
                warnings.append(warning_check["message"])

        exceptions.print_warnings_and_raise_errors(warnings, errors)

    def _build(self) -> None:
        self._add_run_mode_custom_math()
        # The order of adding components matters!
        # 1. Variables, 2. Global Expressions, 3. Constraints, 4. Objectives
        for components in [
            "variables",
            "global_expressions",
            "constraints",
            "objectives",
        ]:
            component = components.removesuffix("s")
            for name in self.inputs.math[components]:
                getattr(self, f"add_{component}")(name)
            LOGGER.info(
                f"Optimisation Model: Generated optimisation problem {components}"
            )

    def _add_run_mode_custom_math(self) -> None:
        """If not given in the custom_math list, override model math with run mode math"""

        # FIXME: available modes should not be hardcoded here. They should come from a YAML schema.
        mode = self.inputs.attrs["config"].build.mode
        custom_math = self.inputs.attrs["applied_custom_math"]
        not_run_mode = {"plan", "operate", "spores"}.difference([mode])
        run_mode_mismatch = not_run_mode.intersection(custom_math)
        if run_mode_mismatch:
            exceptions.warn(
                f"Running in {mode} mode, but run mode(s) {run_mode_mismatch} custom "
                "math being loaded from file via the model configuration"
            )

        if mode != "plan" and mode not in custom_math:
            LOGGER.debug(f"Updating math formulation with {mode} mode custom math.")
            filepath = importlib.resources.files("calliope") / "math" / f"{mode}.yaml"
            self.inputs.math.union(AttrDict.from_yaml(filepath), allow_override=True)

        validate_dict(self.inputs.math, MATH_SCHEMA, "math")

    def _add_component(
        self,
        name: str,
        component_dict: Optional[Tp],
        component_setter: Callable,
        component_type: Literal[
            "variables", "global_expressions", "constraints", "objectives"
        ],
        break_early: bool = True,
    ) -> Optional[parsing.ParsedBackendComponent]:
        """Generalised function to add a optimisation problem component array to the model.

        Args:
            name:
                Name of the component.
                If not providing the `component_dict` directly, this name must be available in the input math provided on initialising the class.
            component_dict (Union[parsing.UnparsedConstraintDict, parsing.UnparsedExpressionDict, parsing.UnparsedObjectiveDict, parsing.UnparsedVariableDict]):
                Unparsed YAML dictionary configuration.
            component_setter (Callable):
                Function to combine evaluated xarray DataArrays into backend component objects.
                Will receive outputs of `evaluate_where` and (optionally, if `equations` is a component dictionary key) `evaluate_expression` as inputs.
            component_type (Literal["variables", "global_expressions", "constraints", "objectives"])

        Raises:
            BackendError:
                The sub-equations of the parsed component cannot generate component
                objects on duplicate index entries.
        """
        references: set[str] = set()

        if component_dict is None:
            component_dict = self.inputs.math[component_type][name]

        if break_early and not component_dict.get("active", True):
            self.log(
                component_type, name, "Component deactivated and therefore not built."
            )
            return None

        self._raise_error_on_preexistence(name, component_type)
        parsed_component = parsing.ParsedBackendComponent(
            component_type, name, component_dict
        )

        top_level_where = parsed_component.generate_top_level_where_array(
            self, align_to_foreach_sets=False, break_early=break_early
        )
        if break_early and not top_level_where.any():
            return parsed_component

        self._create_obj_list(name, component_type)

        equations = parsed_component.parse_equations(self.valid_component_names)
        if not equations:
            component_da = component_setter(
                parsed_component.drop_dims_not_in_foreach(top_level_where)
            )
        else:
            component_da = (
                xr.DataArray()
                .where(parsed_component.drop_dims_not_in_foreach(top_level_where))
                .astype(np.dtype("O"))
            )
        for element in equations:
            where = element.evaluate_where(self, initial_where=top_level_where)
            if break_early and not where.any():
                continue

            where = parsed_component.drop_dims_not_in_foreach(where)

            if component_da.where(where).notnull().any():
                if component_da.shape:
                    overlap = component_da.where(where).to_series().dropna().index
                    substring = (
                        f"trying to set two equations for the same index:\n{overlap}"
                    )
                else:
                    substring = "trying to set two equations for the same component."

                self.delete_component(name, component_type)
                raise BackendError(f"{element.name} | {substring}")

            to_fill = component_setter(element, where, references)
            component_da = component_da.fillna(to_fill)

        if break_early and component_da.isnull().all():
            self.delete_component(name, component_type)
            return parsed_component

        self._add_to_dataset(
            name, component_da, component_type, component_dict, references
        )

        return parsed_component

    @abstractmethod
    def delete_component(self, key: str, component_type: _COMPONENTS_T) -> None:
        """Delete a list object from the backend model object.

        Args:
            key (str): Name of object.
            component_type (str): Object type.
        """

    @abstractmethod
    def _create_obj_list(self, key: str, component_type: _COMPONENTS_T) -> None:
        """Attach an empty list object to the backend model object.
        This may be a backend-specific subclass of a standard list object.

        Args:
            key (str): Name of object.
            component_type (str): Object type.

        Raises:
            BackendError: Cannot overwrite object of same name and type.
        """

    def _add_all_inputs_as_parameters(self) -> None:
        """
        Add all parameters to backend dataset in-place.
        If model data does not include a parameter, their default values will be added here
        as unindexed backend dataset parameters.

        Args:
            model_data (xr.Dataset): Input model data.
            defaults (dict): Parameter defaults.
        """

        for param_name, param_data in self.inputs.filter_by_attrs(
            is_result=0
        ).data_vars.items():
            default_val = param_data.attrs.get("default", np.nan)
            self.add_parameter(param_name, param_data, default_val)
        for param_name, default_val in self.inputs.attrs["defaults"].items():
            if param_name in self.parameters.keys():
                continue
            self.log(
                "parameters", param_name, "Component not defined; using default value."
            )
            self.add_parameter(
                param_name, xr.DataArray(default_val), use_inf_as_na=False
            )
            self.parameters[param_name].attrs["is_result"] = 0
        LOGGER.info("Optimisation Model: Generated optimisation problem parameters")

    @staticmethod
    def _clean_arrays(*args) -> None:
        """
        Preemptively delete objects with large memory footprints that might otherwise
        stick around longer than necessary.
        """
        del args

    def _add_to_dataset(
        self,
        name: str,
        da: xr.DataArray,
        obj_type: _COMPONENTS_T,
        unparsed_dict: Union[parsing.UNPARSED_DICTS, dict],
        references: Optional[set] = None,
    ):
        """
        Add array of backend objects to backend dataset in-place.

        Args:
            name (str): Name of entry in dataset.
            da (xr.DataArray): Data to add.
            obj_type (str): Type of backend objects in the array.
            unparsed_dict (DT):
                Dictionary describing the object being added, from which descriptor attributes will be extracted and added to the array attributes.
            references (set):
                All other backend objects which are references in this backend object's linear expression(s).
                E.g. the constraint "flow_out / flow_out_eff <= flow_cap" references the variables ["flow_out", "flow_cap"]
                and the parameter ["flow_out_eff"].
                All referenced objects will have their "references" attribute updated with this object's name.
                Defaults to None.
        """

        add_attrs = {
            attr: unparsed_dict.get(attr)
            for attr in self._COMPONENT_ATTR_METADATA
            if attr in unparsed_dict.keys()
        }

        da.attrs.update(
            {
                "obj_type": obj_type,
                "references": set(),
                "coords_in_name": False,
                **add_attrs,  # type: ignore
            }
        )
        self._dataset[name] = da

        if references is not None:
            for reference in references:
                try:
                    self._dataset[reference].attrs["references"].add(name)
                except KeyError:
                    continue

    def _apply_func(
        self, func: Callable, *args, output_core_dims: tuple = ((),), **kwargs
    ) -> xr.DataArray:
        """
        Apply a function to every element of an arbitrary number of xarray DataArrays.

        Args:
            func (Callable):
                Un-vectorized function to call.
                Number of accepted args should equal len(args).
                Number of accepted kwargs should equal len(kwargs).
            args (xr.DataArray):
                xarray DataArrays which will be broadcast together and then iterated over
                to apply the function.
            output_core_dims (tuple):
                Additional dimensions which are expected to be passed back from `xr.apply_ufunc` after applying `func`.
                This is directly passed to `xr.apply_ufunc`; see their documentation for more details.
                Defaults to ((), )
            kwargs (dict[str, Any]):
                Additional keyword arguments to pass to `func`.

        Returns:
            xr.DataArray: Array with func applied to all elements.
        """
        return xr.apply_ufunc(
            func,
            *args,
            kwargs=kwargs,
            vectorize=True,
            keep_attrs=True,
            dask="parallelized",
            output_dtypes=[np.dtype("O")],
            output_core_dims=output_core_dims,
        )

    def _raise_error_on_preexistence(self, key: str, obj_type: _COMPONENTS_T):
        """
        We do not allow any overlap of backend object names since they all have to
        co-exist in the backend dataset.
        I.e., users cannot overwrite any backend component with another
        (of the same type or otherwise).

        Args:
            key (str): Backend object name
            obj_type (Literal["variables", "constraints", "objectives", "parameters", "expressions"]): Object type.

        Raises:
            BackendError:
                Raised if `key` already exists in the backend model
                (either with the same or different type as `obj_type`).
        """
        if key in self._dataset.keys():
            if key in getattr(self, obj_type):
                raise BackendError(
                    f"Trying to add already existing `{key}` to backend model {obj_type}."
                )
            else:
                other_obj_type = self._dataset[key].attrs["obj_type"].removesuffix("s")
                raise BackendError(
                    f"Trying to add already existing *{other_obj_type}* `{key}` "
                    f"as a backend model *{obj_type.removesuffix('s')}*."
                )

    @property
    def constraints(self):
        "Slice of backend dataset to show only built constraints"
        return self._dataset.filter_by_attrs(obj_type="constraints")

    @property
    def variables(self):
        "Slice of backend dataset to show only built variables"
        return self._dataset.filter_by_attrs(obj_type="variables")

    @property
    def parameters(self):
        "Slice of backend dataset to show only built parameters"
        return self._dataset.filter_by_attrs(obj_type="parameters")

    @property
    def global_expressions(self):
        "Slice of backend dataset to show only built global expressions"
        return self._dataset.filter_by_attrs(obj_type="global_expressions")

    @property
    def objectives(self):
        "Slice of backend dataset to show only built objectives"
        return self._dataset.filter_by_attrs(obj_type="objectives")

    @property
    def valid_component_names(self):
        def _filter(val):
            return val in ["variables", "parameters", "global_expressions"]

        in_data = set(self._dataset.filter_by_attrs(obj_type=_filter).data_vars.keys())
        in_math = set(
            name
            for component in ["variables", "global_expressions"]
            for name in self.inputs.math[component].keys()
        )
        return in_data.union(in_math)


class BackendModel(BackendModelGenerator, Generic[T]):
    def __init__(self, inputs: xr.Dataset, instance: T, **kwargs) -> None:
        """Abstract base class to build backend models that interface with solvers.

        Args:
            inputs (xr.Dataset): Calliope model data.
            instance (T): Interface model instance.
        """
        super().__init__(inputs, **kwargs)
        self._instance = instance

    @abstractmethod
    def get_parameter(
        self, parameter_name: str, as_backend_objs: bool = True
    ) -> xr.DataArray:
        """
        Extract parameter from backend dataset.

        Args:
            parameter_name (str): Name of parameter.
            TODO: hide this and create a method to edit parameter values
                  (to handle interfaces with non-mutable params)
            as_backend_objs (bool, optional):
                If True, will keep the array entries as backend interface objects,
                which can be updated to update the underlying model.
                Otherwise, parameter values are given directly, with default values in place of NaNs.
                Defaults to True.

        Returns:
            xr.DataArray: parameter array.
        """

    @abstractmethod
    def get_constraint(
        self,
        constraint_name: str,
        as_backend_objs: bool = True,
        eval_body: bool = False,
    ) -> Union[xr.DataArray, xr.Dataset]:
        """
        Get constraint data as either a table of details or as an array of backend interface objects.
        Can be used to inspect and debug built constraints.

        Args:
            constraint_name (str): Name of constraint, as given in YAML constraint key.
            TODO: hide this and create a method to edit constraints that handles differences in interface APIs.
            as_backend_objs (bool, optional):
                If True, will keep the array entries as backend interface objects,
                which can be updated to change the underlying model.
                Otherwise, constraint body, and lower and upper bounds are given in a table.
                Defaults to True.
            eval_body (bool, optional):
                If True and as_backend_objs is False, will attempt to evaluate the constraint body.
                If the model has been optimised, this attempt will produce a numeric value to see where the constraint sits between the lower or upper bound.
                If the model has not yet been optimised, this attempt will fall back on the same as
                if `eval_body` was set to False, i.e. a string representation of the linear expression in the constraint body.
                Defaults to False.

        Returns:
            Union[xr.DataArray, xr.Dataset]:
                If as_backend_objs is True, will return an xr.DataArray.
                Otherwise, a xr.Dataset will be given, indexed over the same dimensions as the xr.DataArray, with variables for the constraint body, and upper (`ub`) and lower (`lb`) bounds.
        """

    @abstractmethod
    def get_variable(
        self, variable_name: str, as_backend_objs: bool = True
    ) -> xr.DataArray:
        """Extract decision variable array from backend dataset

        Args:
            variable_name (str): Name of variable.
            TODO: hide this and create a method to edit variables that handles differences in interface APIs.
            as_backend_objs (bool, optional):
                If True, will keep the array entries as backend interface objects,
                which can be updated to update the underlying model.
                Otherwise, variable values are given directly.
                If the model has not been successfully optimised, variable values will all be None.
                Defaults to True.

        Returns:
            xr.DataArray: Decision variable array.
        """

    @abstractmethod
    def get_variable_bounds(self, name: str) -> xr.Dataset:
        """Extract decision variable upper and lower bound array from backend dataset

        Args:
            variable_name (str): Name of variable.

        Returns:
            xr.Dataset: Contains the arrays for upper ("ub", a.k.a. "max") and lower ("lb", a.k.a. "min") variable bounds.
        """

    @abstractmethod
    def get_global_expression(
        self, expression_name: str, as_backend_objs: bool = True, eval_body: bool = True
    ) -> xr.DataArray:
        """Extract global expression array from backend dataset

        Args:
            global_expression_name (str): Name of global expression
            TODO: hide this and create a method to edit expressions that handles differences in interface APIs.
            as_backend_objs (bool, optional):
                If True, will keep the array entries as backend interface objects,
                which can be updated to update the underlying model.
                Otherwise, global expression values are given directly.
                If the model has not been successfully optimised, expression values will all be provided as strings.
                Defaults to True.
            eval_body (bool, optional):
                If True and as_backend_objs is False, will attempt to evaluate the expression.
                If the model has been optimised, this attempt will produce a numeric value.
                If the model has not yet been optimised, this attempt will fall back on the same as
                if `eval_body` was set to False, i.e. a string representation of the linear expression.
                Defaults to True.

        Returns:
            xr.DataArray: global expression array.
        """

    @abstractmethod
    def update_parameter(
        self, name: str, new_values: Union[xr.DataArray, SupportsFloat]
    ) -> None:
        """Update parameter elements using an array of new values.
        If the parameter has not been previously defined, it will be added to the optimisation problem based on the new values given (with NaNs reverting to default values).
        If the new values have fewer dimensions than are on the parameter array, the new values will be broadcast across the missing dimensions before applying the update.

        Args:
            name (str): Parameter to update
            new_values (Union[xr.DataArray, SupportsFloat]): New values to apply. Any empty (NaN) elements in the array will be skipped.
        """

    @abstractmethod
    def update_variable_bounds(
        self,
        name: str,
        *,
        min: Optional[Union[xr.DataArray, SupportsFloat]] = None,
        max: Optional[Union[xr.DataArray, SupportsFloat]] = None,
    ) -> None:
        """
        Update the bounds on a decision variable.
        If the variable bounds are defined by parameters in the math formulation,
        the parameters themselves will be updated.

        Args:
            name (str): Variable to update.
            min (Union[xr.DataArray, SupportsFloat], optional):
                If provided, the Non-NaN values in the array will be used to defined new lower bounds in the decision variable.
                Defaults to None.
            max (Union[xr.DataArray, SupportsFloat], optional):
                If provided, the Non-NaN values in the array will be used to defined new upper bounds in the decision variable.
                Defaults to None.
        """

    @abstractmethod
    def fix_variable(self, name: str, where: Optional[xr.DataArray] = None) -> None:
        """
        Fix the variable value to the value quantified on the most recent call to `solve`.
        Fixed variables will be treated as parameters in the optimisation.

        Args:
            name (str): Variable to update.
            where (xr.DataArray, optional):
                If provided, only a subset of the coordinates in the variable will be fixed.
                Must be a boolean array or a float equivalent, where NaN is used instead of False.
                Defaults to None
        """

    @abstractmethod
    def unfix_variable(self, name: str, where: Optional[xr.DataArray] = None) -> None:
        """
        Unfix the variable so that it is treated as a decision variable in the next call to `solve`.

        Args:
            name (str): Variable to update
            where (xr.DataArray, optional):
                If provided, only a subset of the coordinates in the variable will be unfixed.
                Must be a boolean array or a float equivalent, where NaN is used instead of False.
                Defaults to None
        """

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
        """
        Optimise built model. If solution is optimal, interface objects
        (decision variables, global expressions, constraints, objective) can be successfully
        evaluated for their values at optimality.

        Args:
            solver (str): Name of solver to optimise with.
            solver_io (Optional[str], optional):
                If chosen solver has a python interface, set to "python" for potential
                performance gains, otherwise should be left as None. Defaults to None.
            solver_options (Optional[dict], optional):
                Solver options/parameters to pass directly to solver.
                See solver documentation for available parameters that can be influenced.
                Defaults to None.
            save_logs (Optional[str], optional):
                If given, solver logs and built LP file will be saved to this filepath.
                Defaults to None.
            warmstart (bool, optional):
                If True, and the chosen solver is capable of implementing it, an existing
                optimal solution will be used to warmstart the next solve run.
                Defaults to False.
        """

    def load_results(self) -> xr.Dataset:
        """
        Evaluate backend decision variables, global expressions, and parameters (if not in inputs)
        after a successful model run.

        Returns:
            xr.Dataset: Dataset of optimal solution results (all numeric data).
        """

        def _drop_attrs(da):
            da.attrs = {
                k: v for k, v in da.attrs.items() if k in self._COMPONENT_ATTR_METADATA
            }
            return da

        all_variables = {
            name_: _drop_attrs(self.get_variable(name_, as_backend_objs=False))
            for name_, var in self.variables.items()
            if var.notnull().any()
        }
        all_global_expressions = {
            name_: _drop_attrs(
                self.get_global_expression(name_, as_backend_objs=False, eval_body=True)
            )
            for name_, expr in self.global_expressions.items()
            if expr.notnull().any()
        }

        results = xr.Dataset({**all_variables, **all_global_expressions}).astype(float)

        return results

    @abstractmethod
    def verbose_strings(self) -> None:
        """
        Update optimisation model object string representations to include the index coordinates of the object.

        E.g., `variables(flow_out)[0]` will become `variables(flow_out)[power, region1, ccgt, 2005-01-01 00:00]`

        This takes approximately 10% of the peak memory required to initially build the optimisation problem, so should only be invoked if inspecting the model in detail (e.g., debugging)

        Only string representations of model parameters and variables will be updated since global expressions automatically show the string representation of their contents.
        """

    @abstractmethod
    def to_lp(self, path: Union[str, Path]) -> None:
        """Write the optimisation problem to file in the linear programming LP format.
        The LP file can be used for debugging and to submit to solvers directly.

        Args:
            path (Union[str, Path]): Path to which the LP file will be written.
        """

    def _find_all_references(self, initial_references: set) -> set:
        """
        Find all nested references to optimisation problem components from an initial set of references.

        Args:
            initial_references (set): names of optimisation problem components.

        Returns:
            set: `initial_references` + any names of optimisation problem components referenced from those initial references.
        """

        references = initial_references.copy()
        for reference in initial_references:
            new_refs = self._dataset[reference].attrs.get("references", {})
            references.update(self._find_all_references(new_refs))
        return references

    def _rebuild_reference(self, reference: str) -> None:
        """Delete and rebuild an optimisation problem component.

        Args:
            references (str): name of optimisation problem component.
        """
        obj_type = self._dataset[reference].attrs["obj_type"]
        self.delete_component(reference, obj_type)
        getattr(self, "add_" + obj_type.removesuffix("s"))(name=reference)
