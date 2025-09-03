# Copyright (C) since 2013 Calliope contributors listed in AUTHORS.
# Licensed under the Apache 2.0 License (see LICENSE file).
"""Methods to interface with the optimisation problem."""

from __future__ import annotations

import inspect
import logging
import time
import typing
from abc import ABC, ABCMeta, abstractmethod
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Literal,
    SupportsFloat,
    TypeVar,
    overload,
)

import numpy as np
import xarray as xr

from calliope import exceptions
from calliope.backend import helper_functions, parsing
from calliope.exceptions import BackendError
from calliope.exceptions import warn as model_warn
from calliope.preprocess.model_math import ORDERED_COMPONENTS_T
from calliope.schemas import config_schema, math_schema

if TYPE_CHECKING:
    from calliope.backend.parsing import T as Tp

T = TypeVar("T")
ALL_COMPONENTS_T = Literal["parameters", "lookups", ORDERED_COMPONENTS_T]


LOGGER = logging.getLogger(__name__)

VALIDATE_METHODS = [
    "add_constraint",
    "add_global_expression",
    "add_variable",
    "add_piecewise_constraint",
    "add_objective",
    "add_parameter",
    "add_lookup",
]
"""Methods whose input definitions will be validated when adding components."""


def validate_on_adding_component(func) -> Callable:
    """Decorator to validate the component definition when adding a new component as a dictionary.

    Args:
        func (Callable): The function with a definition to validate.
    """

    def validator(*args, **kwargs):
        # First, we have to turn all `args`` into `kwargs``
        # so that we can unambiguously select "definition" from the user input.
        sig = inspect.signature(func)
        bound_args = sig.bind_partial(*args, **kwargs)
        bound_args.apply_defaults()

        component_def = bound_args.arguments["definition"]

        validator = typing.get_type_hints(func)["definition"]

        if not isinstance(component_def, validator):
            bound_args.arguments["definition"] = validator.model_validate(component_def)
        return func(*bound_args.args, **bound_args.kwargs)

    return validator


class SelectiveWrappingMeta(ABCMeta):
    """Metaclass to selectively wrap methods in concrete classes inheriting from the backend ABC."""

    def __new__(mcs, name, bases, namespace):
        """Wrap methods in all new instances of the class."""
        cls = super().__new__(mcs, name, bases, namespace)

        for method_name in VALIDATE_METHODS:
            if method_name in namespace:
                original = getattr(cls, method_name)
                setattr(cls, method_name, validate_on_adding_component(original))
        return cls


class BackendModelGenerator(ABC, metaclass=SelectiveWrappingMeta):
    """Helper class for backends."""

    LID_COMPONENTS: tuple[ALL_COMPONENTS_T, ...] = typing.get_args(ALL_COMPONENTS_T)
    _COMPONENT_ATTR_METADATA = [
        "description",
        "unit",
        "default",
        "dtype",
        "domain",
        "title",
        "sense",
        "math_repr",
    ]

    objective: str
    """Optimisation problem objective name."""

    def __init__(
        self,
        inputs: xr.Dataset,
        math: math_schema.CalliopeBuildMath,
        build_config: config_schema.Build,
    ):
        """Abstract base class to build a representation of the optimisation problem.

        Args:
            inputs (xr.Dataset): Calliope model data.
            math (math_schema.CalliopeBuildMath): Calliope math.
            build_config (config_schema.Build): Build configuration options.
        """
        self._dataset = xr.Dataset()
        self.config = build_config
        self.math = math
        self._solve_logger = logging.getLogger(__name__ + ".<solve>")

        self.inputs = self._add_inputs(inputs)
        self.objective: str = self.config.objective

        self._check_inputs()

    def add_lookup(
        self, name: str, values: xr.DataArray, definition: math_schema.Lookup
    ) -> None:
        """Add input lookup array to backend model in-place.

        This directly passes a copy of the input lookup array to the backend.

        Args:
            name (str): Name of lookup.
            values (xr.DataArray): Array of lookup values.
            definition (math_schema.Lookup): Lookup math definition.
        """
        self._raise_error_on_preexistence(name, "lookups")

        if values.isnull().all():
            self.log("lookups", name, "Component not added; no data found in array.")
            values = values.astype(float)

        self._add_to_dataset(name, values, "lookups", definition.model_dump())

    @abstractmethod
    def add_parameter(
        self, name: str, values: xr.DataArray, definition: math_schema.Parameter
    ) -> None:
        """Add input parameter to backend model in-place.

        If the backend interface allows for mutable parameter objects, they will be
        generated, otherwise a copy of the model input dataset will be used.
        In either case, NaN values are filled with the given parameter default value.

        Args:
            name (str): Name of parameter.
            values (xr.DataArray): Array of parameter values.
            definition (math_schema.Parameter): Parameter math definition.
        """

    @abstractmethod
    def add_constraint(self, name: str, definition: math_schema.Constraint) -> None:
        """Add constraint equation to backend model in-place.

        Resulting backend dataset entries will be constraint objects.

        Args:
            name (str):
                Name of the constraint
            definition (math_schema.Constraint):
                Constraint configuration dictionary, ready to be parsed and then evaluated.
        """

    @abstractmethod
    def add_piecewise_constraint(
        self, name: str, definition: math_schema.PiecewiseConstraint
    ) -> None:
        """Add piecewise constraint equation to backend model in-place.

        Resulting backend dataset entries will be piecewise constraint objects.

        Args:
            name (str):
                Name of the piecewise constraint
            definition (math_schema.PiecewiseConstraint):
                Piecewise constraint configuration dictionary, ready to be parsed and then evaluated.
        """

    @abstractmethod
    def add_global_expression(
        self, name: str, definition: math_schema.GlobalExpression
    ) -> None:
        """Add global expression (arithmetic combination of parameters and/or decision variables) to backend model in-place.

        Resulting backend dataset entries will be linear expression objects.

        Args:
            name (str): name of the global expression
            definition (math_schema.GlobalExpression): Global expression configuration dictionary, ready to be parsed and then evaluated.
        """

    @abstractmethod
    def add_variable(self, name: str, definition: math_schema.Variable) -> None:
        """Add decision variable to backend model in-place.

        Resulting backend dataset entries will be decision variable objects.

        Args:
            name (str): name of the variable.
            definition (math_schema.Variable): Variable configuration dictionary.
        """

    @abstractmethod
    def add_objective(self, name: str, definition: math_schema.Objective) -> None:
        """Add objective arithmetic to backend model in-place.

        Resulting backend dataset entry will be a single, unindexed objective object.

        Args:
            name (str): name of the objective.
            definition (math_schema.Objective): Unparsed objective configuration dictionary.
        """

    @abstractmethod
    def set_objective(self, name: str) -> None:
        """Set a built objective to be the optimisation objective.

        Args:
            name (str): name of the objective.
        """

    def log(
        self,
        component_type: ALL_COMPONENTS_T,
        component_name: str,
        message: str,
        level: Literal["info", "warning", "debug", "error", "critical"] = "debug",
    ):
        """Log to module-level logger with some prettification of the message.

        Args:
            component_type (ALL_COMPONENTS_T): type of component.
            component_name (str): name of the component.
            message (str): message to log.
            level (Literal["info", "warning", "debug", "error", "critical"], optional): log level. Defaults to "debug".
        """
        getattr(LOGGER, level)(
            f"Optimisation model | {component_type}:{component_name} | {message}"
        )

    def _add_inputs(self, inputs: xr.Dataset):
        """Add default inputs to the model inputs dataset.

        Args:
            inputs (xr.Dataset): Model input data.

        Returns:
            xr.Dataset: Model input data with defaults added.
        """
        new_inputs = xr.Dataset()
        for obj_type in ["parameters", "lookups", "dimensions"]:
            for name, config in self.math[obj_type].root.items():
                attrs: dict = {"obj_type": obj_type, **config.model_dump()}
                default = config.default if obj_type == "lookups" else np.nan
                data = inputs.get(name, xr.DataArray(default))
                if obj_type == "dimensions" and name in inputs.dims:
                    new_inputs.coords[name] = data.assign_attrs(**attrs)
                elif obj_type != "dimensions":
                    new_inputs[name] = data.assign_attrs(**attrs)
                else:
                    continue

        return new_inputs

    def _check_inputs(self):
        data_checks = self.math.checks
        check_results = {"raise": [], "warn": []}
        parser_ = parsing.where_parser.generate_where_string_parser()
        eval_kwargs = {
            "backend_interface": self,
            "math": self.math,
            "input_data": self.inputs,
            "build_config": self.config,
            "helper_functions": helper_functions._registry["where"],
            "apply_where": True,
            "references": set(),
        }
        for name, check in data_checks.root.items():
            if check.active:
                parsed_ = parser_.parse_string(check.where, parse_all=True)
                evaluated = parsed_[0].eval("array", equation_name=name, **eval_kwargs)
                if (
                    evaluated.any()
                    and (evaluated & self.inputs.definition_matrix).any()
                ):
                    check_results[check.errors].append(check.message)

        exceptions.print_warnings_and_raise_errors(
            check_results["warn"], check_results["raise"]
        )

    def add_optimisation_components(self) -> None:
        """Parse math and inputs and set optimisation problem."""
        # The order of adding components matters!
        # 1. Variables, 2. Global Expressions, 3. Constraints, 4. Objectives
        self._load_inputs()
        for components in typing.get_args(ORDERED_COMPONENTS_T):
            component = components.removesuffix("s")
            ordered_items = sorted(
                self.math[components].root.items(),
                key=lambda item: getattr(item[1], "order", 0),
            )
            for name, definition in ordered_items:
                start = time.time()
                getattr(self, f"add_{component}")(name, definition)
                end = time.time() - start
                LOGGER.debug(
                    f"Optimisation Model | {components}:{name} | Built in {end:.4f}s"
                )
            LOGGER.info(f"Optimisation Model | {components} | Generated.")

    def _add_component(
        self,
        name: str,
        component_def: Tp,
        component_setter: Callable,
        component_type: ORDERED_COMPONENTS_T,
        break_early: bool = True,
    ) -> parsing.ParsedBackendComponent | None:
        """Generalised function to add a optimisation problem component array to the model.

        Args:
            name (str): name of the component.
            component_def (Tp): unparsed math component definition.
            component_setter (Callable): function to combine evaluated xarray DataArrays into backend component objects.
            component_type (Literal["variables", "global_expressions", "constraints", "piecewise_constraints", "objectives"]):
                type of the added component.
            break_early (bool, optional): break if the component is not active. Defaults to True.

        Raises:
            BackendError: The sub-equations of the parsed component cannot generate component
                objects on duplicate index entries.

        Returns:
            parsing.ParsedBackendComponent | None: parsed component. None if the break_early condition was met.
        """
        references: set[str] = set()
        default_empty = xr.DataArray(np.nan)
        if break_early and not component_def.active:
            self.log(
                component_type,
                name,
                "Component deactivated; only metadata will be stored if no other component with the same name is defined.",
            )
            if name in self._dataset.data_vars:
                return None
            else:
                component_da = default_empty
                parsed_component = None
        else:
            self._raise_error_on_preexistence(name, component_type)
            parsed_component = parsing.ParsedBackendComponent(
                component_type, name, component_def
            )

            top_level_where = parsed_component.generate_top_level_where_array(
                self,
                align_to_foreach_sets=False,
                break_early=break_early,
                references=references,
            )
            if break_early and not top_level_where.any():
                component_da = default_empty

            else:
                self._create_obj_list(name, component_type)
                equations = parsed_component.parse_equations(self.valid_component_names)

                if not equations:
                    component_da = component_setter(
                        parsed_component.drop_dims_not_in_foreach(top_level_where),
                        references,
                    )
                else:
                    component_da = (
                        xr.DataArray()
                        .where(
                            parsed_component.drop_dims_not_in_foreach(top_level_where)
                        )
                        .astype(np.dtype("O"))
                    )
                for element in equations:
                    where = element.evaluate_where(
                        self, initial_where=top_level_where, references=references
                    )
                    if break_early and not where.any():
                        continue

                    where = parsed_component.drop_dims_not_in_foreach(where)

                    if component_da.where(where).notnull().any():
                        if component_da.shape:
                            overlap = (
                                component_da.where(where).to_series().dropna().index
                            )
                            substring = f"trying to set two equations for the same index:\n{overlap}"
                        else:
                            substring = (
                                "trying to set two equations for the same component."
                            )

                        self.delete_component(name, component_type)
                        raise BackendError(f"{element.name} | {substring}")

                    to_fill = component_setter(element, where, references)
                    component_da = component_da.fillna(to_fill)

                if break_early and component_da.isnull().all():
                    self.delete_component(name, component_type)
                    # simplify the component since it's empty
                    component_da = default_empty

        self._add_to_dataset(
            name, component_da, component_type, component_def.model_dump(), references
        )
        if name not in self.math[component_type]:
            self.math = self.math.update(
                {f"{component_type}.{name}": component_def.model_dump()}
            )

        return parsed_component

    @abstractmethod
    def delete_component(self, key: str, component_type: ALL_COMPONENTS_T) -> None:
        """Delete a list object from the backend model object.

        Args:
            key (str): Name of object.
            component_type (str): Object type.
        """

    @abstractmethod
    def _create_obj_list(self, key: str, component_type: ALL_COMPONENTS_T) -> None:
        """Attach an empty list object to the backend model object.

        The attachment may be a backend-specific subclass of a standard list object.

        Args:
            key (str): Name of object.
            component_type (str): Object type.

        Raises:
            BackendError: Cannot overwrite object of same name and type.
        """

    def _load_inputs(self) -> None:
        """Add all parameters / lookups to backend dataset in-place.

        If model data does not include an entry, their default values will be added here
        in an unindexed form.

        Args:
            model_data (xr.Dataset): Input model data.
        """
        for name, data in self.inputs.data_vars.items():
            if data.obj_type == "parameters":
                self.add_parameter(name, data, self.math.parameters[name])
            elif data.obj_type == "lookups":
                self.add_lookup(name, data, self.math.lookups[name])

        LOGGER.info("Optimisation Model | parameters/lookups | Generated.")

    @staticmethod
    def _clean_arrays(*args) -> None:
        """Preemptively delete of objects with large memory footprints."""
        del args

    def _add_to_dataset(
        self,
        name: str,
        da: xr.DataArray,
        obj_type: ALL_COMPONENTS_T,
        attrs: dict,
        references: set | None = None,
    ):
        """Add array of backend objects to backend dataset in-place.

        Args:
            name (str): Name of entry in dataset.
            da (xr.DataArray): Data to add.
            obj_type (ALL_COMPONENTS_T): Type of backend objects in the array.
            attrs (Tp):
                Dictionary describing the object being added, from which descriptor
                attributes will be extracted and added to the array attributes.
            references (set | None, optional):
                All other backend objects which are references in this backend object's linear expression(s).
                E.g. the constraint "flow_out / flow_out_eff <= flow_cap" references the variables ["flow_out", "flow_cap"]
                and the parameter ["flow_out_eff"].
                All referenced objects will have their "references" attribute updated with this object's name.
                Defaults to None.
        """
        da.attrs = {
            "obj_type": obj_type,
            "references": set(),
            "coords_in_name": False,
            **{k: v for k, v in attrs.items() if k in self._COMPONENT_ATTR_METADATA},
        }
        self._dataset[name] = da
        if references is not None:
            self._update_references(name, references)

    def _update_references(self, name: str, references: set):
        """Update reference lists in dataset objects.

        Args:
            name (str): Name to update in reference lists.
            references (set): Names of dataset objects whose reference lists will be updated with `name`.
        """
        for reference in references:
            try:
                self._dataset[reference].attrs["references"].add(name)
            except KeyError:
                continue

    @overload
    def _apply_func(  # noqa: D102, override
        self, func: Callable, where: xr.DataArray, n_out: Literal[1], *args, **kwargs
    ) -> xr.DataArray: ...

    @overload
    def _apply_func(  # noqa: D102, override
        self,
        func: Callable,
        where: xr.DataArray,
        n_out: Literal[2, 3, 4, 5],
        *args,
        **kwargs,
    ) -> tuple[xr.DataArray, ...]: ...

    def _apply_func(
        self, func: Callable, where: xr.DataArray, n_out: int, *args, **kwargs
    ) -> xr.DataArray | tuple[xr.DataArray, ...]:
        """Apply a function to every element of an arbitrary number of xarray DataArrays.

        Args:
            func (Callable):
                Un-vectorized function to call.
                Number of accepted args should equal len(args).
                Number of accepted kwargs should equal len(kwargs).
            where (Optional[xr.DataArray]):
                If given, boolean array that will be used to:
                1. broadcast *args,
                2. mask the vectorised function call so that it is only applied on True elements.
            n_out (int): Number of expected output DataArrays.
            *args (xr.DataArray):
                xarray DataArrays which will have `func` applied on every element.
                If `where` is None, these arrays should already have been broadcast together.
            **kwargs (dict[str, Any]):
                Additional keyword arguments to pass to `func`.

        Returns:
            xr.DataArray | tuple[xr.DataArray, ...]: Array or tuple of arrays with `func` applied to all array elements.
        """
        if kwargs:
            func = partial(func, **kwargs)
        vectorised_func = np.frompyfunc(func, len(args), n_out)
        da = vectorised_func(
            *(arg.broadcast_like(where) for arg in args), where=where.values
        )
        if isinstance(da, xr.DataArray):
            da = da.fillna(np.nan)
        else:
            da = tuple(arr.fillna(np.nan) for arr in da)
        return da

    def _raise_error_on_preexistence(self, key: str, obj_type: ALL_COMPONENTS_T):
        """Detect if preexistent errors are present in the dataset.

        We do not allow any overlap of backend object names since they all have to
        co-exist in the backend dataset. I.e., users cannot overwrite any backend
        component with another (of the same type or otherwise).

        Args:
            key (str): Backend object name
            obj_type (ALL_COMPONENTS_T): Object type.

        Raises:
            BackendError: if `key` already exists in the backend model
                (either with the same or different type as `obj_type`).
        """
        if key in self._dataset.keys() and self.math.find(key)[1].active:
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
        """Slice of backend dataset to show only built constraints."""
        return self._dataset.filter_by_attrs(obj_type="constraints")

    @property
    def piecewise_constraints(self):
        """Slice of backend dataset to show only built piecewise constraints."""
        return self._dataset.filter_by_attrs(obj_type="piecewise_constraints")

    @property
    def variables(self):
        """Slice of backend dataset to show only built variables."""
        return self._dataset.filter_by_attrs(obj_type="variables")

    @property
    def parameters(self):
        """Slice of backend dataset to show only built parameters."""
        return self._dataset.filter_by_attrs(obj_type="parameters")

    @property
    def lookups(self):
        """Slice of backend dataset to show only built lookup arrays."""
        return self._dataset.filter_by_attrs(obj_type="lookups")

    @property
    def global_expressions(self):
        """Slice of backend dataset to show only built global expressions."""
        return self._dataset.filter_by_attrs(obj_type="global_expressions")

    @property
    def objectives(self):
        """Slice of backend dataset to show only built objectives."""
        return self._dataset.filter_by_attrs(obj_type="objectives")

    @property
    def valid_component_names(self) -> set:
        """Return a set of valid component names in the model data.

        Returns:
            set: set of valid names.
        """

        def _filter(val):
            return val in ["variables", "parameters", "global_expressions", "lookups"]

        in_data = set(self._dataset.filter_by_attrs(obj_type=_filter).data_vars.keys())
        in_math = set(
            name
            for component in ["variables", "global_expressions"]
            for name in self.math[component].root
        )
        return in_data.union(in_math)


class BackendModel(BackendModelGenerator, Generic[T]):
    """Calliope's backend model functionality."""

    def __init__(
        self,
        inputs: xr.Dataset,
        math: math_schema.CalliopeBuildMath,
        build_config: config_schema.Build,
        instance: T,
    ) -> None:
        """Abstract base class to build backend models that interface with solvers.

        Args:
            inputs (xr.Dataset): Calliope model data.
            math (AttrDict): Calliope math.
            build_config (config_schema.Build): Build configuration options.
            instance (T): Interface model instance.
        """
        super().__init__(inputs, math, build_config)
        self._instance = instance
        self.shadow_prices: ShadowPrices
        self._has_verbose_strings: bool = False

    def add_piecewise_constraint(  # noqa: D102, override
        self, name: str, definition: math_schema.PiecewiseConstraint
    ) -> None:
        if "breakpoints" in definition.foreach:
            raise BackendError(
                f"(piecewise_constraints, {name}) | `breakpoints` dimension should not be in `foreach`. "
                "Instead, index `x_values` and `y_values` parameters over `breakpoints`."
            )

        def _constraint_setter(where: xr.DataArray, references: set) -> xr.DataArray:
            expressions = []
            vals = []
            for axis in ["x", "y"]:
                dummy_expression_dict = {
                    "equations": [{"expression": definition[f"{axis}_expression"]}],
                    "foreach": definition.foreach,
                }
                parsed_component = parsing.ParsedBackendComponent(
                    "piecewise_constraints",
                    name,
                    math_schema.GlobalExpression.model_validate(dummy_expression_dict),
                )
                eq = parsed_component.parse_equations(self.valid_component_names)
                expression_da = eq[0].evaluate_expression(
                    self, where=where, references=references
                )
                val_name = definition[f"{axis}_values"]
                val_da = self.get_parameter(val_name)
                if "breakpoints" not in val_da.dims:
                    raise BackendError(
                        f"(piecewise_constraints, {name}) | "
                        f"`{axis}_values` must be indexed over the `breakpoints` dimension."
                    )
                references.add(val_name)
                expressions.append(expression_da)
                vals.extend([*val_da.to_dataset("breakpoints").data_vars.values()])

            try:
                return self._apply_func(
                    self._to_piecewise_constraint,
                    where,
                    1,
                    *expressions,
                    *vals,
                    name=name,
                    n_breakpoints=len(self.inputs.breakpoints),
                )
            except BackendError as err:
                raise BackendError(
                    f"(piecewise_constraints, {name}) | Errors in generating piecewise constraint: {err}"
                )

        self._add_component(
            name, definition, _constraint_setter, "piecewise_constraints"
        )

    def get_lookup(self, name: str) -> xr.DataArray:
        """Extract lookup from backend dataset.

        Args:
            name (str): Name of lookup.

        Returns:
            xr.DataArray: lookup array.
        """
        return self._get_component(name, "lookups")

    @abstractmethod
    def _to_piecewise_constraint(
        self, x_var: Any, y_var: Any, *vals: float, name: str, n_breakpoints: int
    ) -> Any:
        """Utility function to generate a pyomo piecewise constraint for every element of an xarray DataArray.

        The x-axis decision variable need not be bounded.
        This aligns piecewise constraint functionality with other possible backends (e.g., gurobipy).

        Args:
            x_var (Any): The x-axis decision variable to constrain.
            y_var (Any): The y-axis decision variable to constrain.
            *vals (xr.DataArray): The x-axis and y-axis decision variable values at each piecewise constraint breakpoint.
            name (str): The name of the piecewise constraint.
            n_breakpoints (int): number of breakpoints

        Returns:
            Any:
                Return piecewise_constraint object.
        """

    @abstractmethod
    def get_parameter(self, name: str, as_backend_objs: bool = True) -> xr.DataArray:
        """Extract parameter from backend dataset.

        Args:
            name (str): Name of parameter.
            as_backend_objs (bool, optional): TODO: hide this and create a method to edit parameter values (to handle interfaces with non-mutable params)
                If True, will keep the array entries as backend interface objects,
                which can be updated to update the underlying model.
                Otherwise, parameter values are given directly, with default values in place of NaNs.
                Defaults to True.

        Returns:
            xr.DataArray: parameter array.
        """

    @overload
    def get_constraint(  # noqa: D102, override
        self, name: str, as_backend_objs: Literal[True] = True, eval_body: bool = False
    ) -> xr.DataArray: ...

    @overload
    def get_constraint(  # noqa: D102, override
        self, name: str, as_backend_objs: Literal[False], eval_body: bool = False
    ) -> xr.Dataset: ...

    @abstractmethod
    def get_constraint(
        self, name: str, as_backend_objs: bool = True, eval_body: bool = False
    ) -> xr.DataArray | xr.Dataset:
        """Get constraint data from the backend for debugging.

        Dat can be returned as  a table of details or as an array of backend interface objects

        Args:
            name (str): Name of constraint, as given in YAML constraint key.
            as_backend_objs (bool, optional): TODO: hide this and create a method to edit constraints that handles differences in interface APIs
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
            xr.DataArray | xr.Dataset:
                If as_backend_objs is True, will return an xr.DataArray.
                Otherwise, a xr.Dataset will be given, indexed over the same dimensions as the xr.DataArray, with variables for the constraint body, and upper (`ub`) and lower (`lb`) bounds.
        """

    def get_piecewise_constraint(self, name: str) -> xr.DataArray:
        """Get piecewise constraint data as an array of backend interface objects.

        This method can be used to inspect and debug built piecewise constraints.

        Unlike other optimisation problem components, piecewise constraints can only be inspected as backend interface objects.
        This is because each element is a collection of variables, parameters, constraints, and expressions.

        Args:
            name (str): Name of piecewise constraint, as given in YAML piecewise constraint key.

        Returns:
            xr.DataArray: Piecewise constraint array.
        """
        return self._get_component(name, "piecewise_constraints")

    @abstractmethod
    def get_variable(self, name: str, as_backend_objs: bool = True) -> xr.DataArray:
        """Extract decision variable array from backend dataset.

        Args:
            name (str): Name of variable.
            as_backend_objs (bool, optional): TODO: hide this and create a method to edit variables that handles differences in interface APIs.
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
        """Extract decision variable upper and lower bound array from backend dataset.

        Args:
            name (str): Name of variable.

        Returns:
            xr.Dataset: Contains the arrays for upper ("ub", a.k.a. "max") and lower ("lb", a.k.a. "min") variable bounds.
        """

    def get_global_expression(
        self, name: str, as_backend_objs: bool = True, eval_body: bool = False
    ) -> xr.DataArray:
        """Extract global expression array from backend dataset.

        Args:
            name (str): Name of global expression.
            as_backend_objs (bool, optional): TODO: hide this and create a method to edit expressions that handles differences in interface APIs.
                If True, will keep the array entries as backend interface objects,
                which can be updated to update the underlying model.
                Otherwise, global expression values are given directly.
                If the model has not been successfully optimised, expression values will all be provided as strings.
                Defaults to True.
            eval_body (bool, optional):
                If True and `as_backend_objs is False`, will attempt to evaluate the expression.
                If the model has been optimised, this attempt will produce a numeric value.
                If the model has not yet been optimised, this attempt will fall back on the same as
                if `eval_body` was set to False, i.e. a string representation of the linear expression.
                Defaults to False.

        Returns:
            xr.DataArray: global expression array.
        """
        return self._get_expression(
            name, as_backend_objs, eval_body, "global_expressions"
        )

    def get_objective(
        self, name: str, as_backend_objs: bool = True, eval_body: bool = False
    ) -> xr.DataArray:
        """Extract objective from backend dataset.

        Args:
            name (str): Name of objective.
            as_backend_objs (bool, optional): TODO: hide this and create a method to edit expressions that handles differences in interface APIs.
                If True, will keep the array entries as backend interface objects,
                which can be updated to update the underlying model.
                Otherwise, objective values are given directly.
                If the model has not been successfully optimised, the objective will be provided as a string.
                Defaults to True.
            eval_body (bool, optional):
                If True and `as_backend_objs` is False, will attempt to evaluate the objective.
                If the model has been optimised, this attempt will produce a numeric value.
                If the model has not yet been optimised, this attempt will fall back on the same as
                if `eval_body` was set to False, i.e. a string representation of the linear expression.
                Defaults to False.

        Returns:
            xr.DataArray: objective array.
        """
        return self._get_expression(name, as_backend_objs, eval_body, "objectives")

    @abstractmethod
    def _get_expression(
        self,
        name: str,
        as_backend_objs: bool,
        eval_body: bool,
        component_type: Literal["global_expressions", "objectives"],
    ):
        """Extract an array of expressions from the backend dataset.

        Args:
            name (str): Name of expression array.
            as_backend_objs (bool): TODO: hide this and create a method to edit expressions that handles differences in interface APIs.
                If True, will keep the array entries as backend interface objects,
                which can be updated to update the underlying model.
                Otherwise, expression values are given directly.
                If the model has not been successfully optimised, the expression will be provided as a string.
            eval_body (bool):
                If True and `as_backend_objs` is False, will attempt to evaluate the expression.
                If the model has been optimised, this attempt will produce a numeric value.
                If the model has not yet been optimised, this attempt will fall back on the same as
                if `eval_body` was set to False, i.e. a string representation of the linear expression.
            component_type: (["global_expressions", "objectives"]):
                Type of expression to be accessed.

        Returns:
            xr.DataArray: expression array.
        """

    @abstractmethod
    def update_input(self, name: str, new_values: xr.DataArray | SupportsFloat) -> None:
        """Update input elements (parameters/lookups) using an array of new values.

        If the input has not been previously defined, it will be added to the
        optimisation problem based on the new values given (with NaNs reverting to
        default values).
        If the new values have fewer dimensions than are on the input array, the
        new values will be broadcast across the missing dimensions before applying the
        update.

        Args:
            name (str): Input array to update
            new_values (xr.DataArray | SupportsFloat): New values to apply. Any
                empty (NaN) elements in the array will be skipped.
        """

    def _update_input(
        self, name: str, new_values: xr.DataArray | SupportsFloat, mutable: bool
    ) -> tuple[xr.DataArray, xr.DataArray, bool]:
        """Update the input array with new values.

        Args:
            name (str): Name of the input array to update.
            new_values (xr.DataArray | SupportsFloat): New values to apply.
            mutable (bool): Whether the array being updated contains mutable objects or not.

        Returns:
            tuple[xr.DataArray, xr.DataArray, bool]:
                The original and new values and a flag for whether the original are mutable object that need updating with a backend-specific method.
        """
        new_values = xr.DataArray(new_values)
        obj_type, math = self.math.find(name, subset={"parameters", "lookups"})
        obj_type_singular = obj_type.removesuffix("s")
        dataset_da = getattr(self, f"get_{obj_type_singular}")(name)
        input_da = self.inputs[name]
        missing_dims_in_new_vals = set(dataset_da.dims).difference(new_values.dims)
        missing_dims_in_orig_vals = set(new_values.dims).difference(dataset_da.dims)

        if missing_dims_in_new_vals:
            self.log(
                obj_type,
                name,
                f"New values will be broadcast along the {missing_dims_in_new_vals} dimension(s)."
                "info",
            )
        new_input_da = new_values.broadcast_like(input_da).fillna(input_da)
        new_input_da.attrs = input_da.attrs
        self.inputs[name] = new_input_da

        if (
            (not dataset_da.shape and new_values.shape)
            or missing_dims_in_orig_vals
            or (dataset_da.isnull() & new_values.notnull()).any()
            or obj_type == "lookups"
            or not mutable
        ):
            self.delete_component(name, obj_type)
            getattr(self, f"add_{obj_type_singular}")(name, new_input_da, math)

            refs_to_update = self._find_all_references(dataset_da.attrs["references"])

            if refs_to_update:
                self.log(
                    obj_type,
                    name,
                    f"The optimisation problem components {sorted(refs_to_update)} will be re-built.",
                    "info",
                )
            self._rebuild_references(refs_to_update)

            if self._has_verbose_strings:
                self.verbose_strings()
            update_mutable = False
        else:
            update_mutable = True
        return dataset_da, new_values, update_mutable

    @abstractmethod
    def update_variable_bounds(
        self,
        name: str,
        *,
        min: xr.DataArray | SupportsFloat | None = None,
        max: xr.DataArray | SupportsFloat | None = None,
    ) -> None:
        """Update the bounds on a decision variable.

        If the variable bounds are defined by parameters in the math formulation,
        the parameters themselves will be updated.

        Args:
            name (str): Variable to update.
            min (xr.DataArray | SupportsFloat | None, optional):
                If provided, the Non-NaN values in the array will be used to defined new lower bounds in the decision variable.
                Defaults to None.
            max (xr.DataArray | SupportsFloat | None, optional):
                If provided, the Non-NaN values in the array will be used to defined new upper bounds in the decision variable.
                Defaults to None.
        """

    @abstractmethod
    def fix_variable(self, name: str, where: xr.DataArray | None = None) -> None:
        """Fix the variable value to the value quantified on the most recent call to `solve`.

        Fixed variables will be treated as parameters in the optimisation.

        Args:
            name (str): Variable to update.
            where (xr.DataArray | None, optional):
                If provided, only a subset of the coordinates in the variable will be fixed.
                Must be a boolean array or a float equivalent, where NaN is used instead of False.
                Defaults to None
        """

    @abstractmethod
    def unfix_variable(self, name: str, where: xr.DataArray | None = None) -> None:
        """Unfix the variable so that it is treated as a decision variable in the next call to `solve`.

        Args:
            name (str): Variable to update
            where (xr.DataArray | None, optional):
                If provided, only a subset of the coordinates in the variable will be unfixed.
                Must be a boolean array or a float equivalent, where NaN is used instead of False.
                Defaults to None
        """

    @abstractmethod
    def verbose_strings(self) -> None:
        """Update optimisation model object string representations to include the index coordinates of the object.

        E.g., `variables(flow_out)[0]` will become `variables(flow_out)[power, region1, ccgt, 2005-01-01 00:00]`

        This takes approximately 10% of the peak memory required to initially build the
        optimisation problem, so should only be invoked if inspecting the model in
        detail (e.g., debugging).

        Only string representations of model parameters and variables will be updated
        since global expressions automatically show the string representation of their
        contents.
        """

    @abstractmethod
    def to_lp(self, path: str | Path) -> None:
        """Write the optimisation problem to file in the linear programming LP format.

        The LP file can be used for debugging and to submit to solvers directly.

        Args:
            path (str | Path): Path to which the LP file will be written.
        """

    @property
    @abstractmethod
    def has_integer_or_binary_variables(self) -> bool:
        """Confirms if the built model has binary or integer decision variables.

        This can be used to understand how long the optimisation may take (MILP
        problems are harder to solve than LP ones), and to verify whether shadow prices
        can be tracked (they cannot be tracked in MILP problems).

        Returns:
            bool: True if the built model has binary or integer decision variables.
                False if all decision variables are continuous.
        """

    @abstractmethod
    def _solve(
        self, solve_config: config_schema.Solve, warmstart: bool = False
    ) -> xr.Dataset:
        """Optimise built model.

        If solution is optimal, interface objects (decision variables, global
        expressions, constraints, objective) can be successfully evaluated for their
        values at optimality.

        Args:
            solve_config: (config_schema.Solve): Calliope Solve configuration object.
            warmstart (bool, optional): If True, and the chosen solver is capable of implementing it, an existing
                optimal solution will be used to warmstart the next solve run.
                Defaults to False.

        Returns:
            xr.Dataset: Dataset of decision variable values if the solution was optimal/feasible,
                otherwise an empty dataset.
        """

    def load_results(self) -> xr.Dataset:
        """Load and evaluate model results after a successful run.

        Evaluates backend decision variables, global expressions, parameters (if not in
        inputs), and shadow_prices (if tracked).

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
        all_objectives = {
            name_: _drop_attrs(
                self.get_objective(name_, as_backend_objs=False, eval_body=True)
            )
            for name_, expr in self.objectives.items()
            if expr.notnull().any()
        }

        all_shadow_prices = {
            f"shadow_price_{constraint}": self.shadow_prices.get(constraint)
            for constraint in self.shadow_prices.tracked
        }

        results = xr.Dataset(
            {
                **all_variables,
                **all_global_expressions,
                **all_shadow_prices,
                **all_objectives,
            },
            attrs=self._dataset.attrs,
        ).astype(float)

        return results

    def _find_all_references(self, initial_references: set) -> set:
        """Find all nested references to optimisation problem components from an initial set of references.

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

    def _rebuild_references(self, references: set[str]) -> None:
        """Delete and rebuild optimisation problem components.

        Args:
            references (set[str]): names of optimisation problem components.
        """
        for component in typing.get_args(ORDERED_COMPONENTS_T):
            # Rebuild references in the order they are found in the backend dataset
            # which should correspond to the order they were added to the optimisation problem.
            refs = [k for k in getattr(self, component).data_vars if k in references]
            for ref in refs:
                self.delete_component(ref, component)
                def_ = self.math[component][ref]
                getattr(self, "add_" + component.removesuffix("s"))(ref, def_)

    def _get_component(self, name: str, component_group: str) -> xr.DataArray:
        component = getattr(self, component_group).get(name, None)
        if component is None:
            pretty_group_name = component_group.removesuffix("s").replace("_", " ")
            raise KeyError(f"Unknown {pretty_group_name}: {name}")
        return component

    def _get_variable_bound(
        self, bound: Any, name: str, references: set, fill_na: float | None = None
    ) -> xr.DataArray:
        """Generate array for either the upper or lower bound of a decision variable.

        Args:
            bound (Any): The bound name (corresponding to an array in the model input data) or value.
            name (str): Name of decision variable.
            references (set): set to store (inplace) the name given by `bound`, if `bound` is a reference to an input parameter.
            fill_na (Optional[float]):
                Fill bounds with this value, after trying to fill with the default value of the parameter,
                if `bound` is a reference to an input parameter.
                Defaults to None.

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
            fill_na = bound_array.attrs["default"]
            references.add(bound)
        else:
            bound_array = xr.DataArray(bound)
        filled_bound_array = bound_array.fillna(fill_na)
        filled_bound_array.attrs = {}
        return filled_bound_array

    @contextmanager
    def _datetime_as_string(self, data: xr.DataArray | xr.Dataset) -> Iterator:
        """Context manager to temporarily convert np.dtype("datetime64[ns]") coordinates (e.g. timesteps) to strings with a resolution of minutes.

        Args:
            data (Union[xr.DataArray, xr.Dataset]): xarray object on whose coordinates the conversion will take place.
        """
        datetime_coords = set()
        for name in data.coords:
            if self.math.dimensions[name].dtype in ["date", "datetime"]:
                data.coords[name] = data.coords[name].dt.strftime("%Y-%m-%d %H:%M")
                datetime_coords.add(name)
        try:
            yield
        finally:
            for name in datetime_coords:
                data.coords[name] = data.coords[name].astype("datetime64[ns]")


class ShadowPrices:
    """Object containing methods to interact with the backend object "shadow prices" tracker, which can be used to access duals for constraints.

    To keep memory overhead low. Shadow price tracking is deactivated by default.
    """

    _tracked: set = set()

    @abstractmethod
    def get(self, name) -> xr.DataArray:
        """Extract shadow prices (a.k.a. duals) from a constraint.

        Args:
            name (str): Name of constraint for which you're seeking duals.

        Returns:
            xr.DataArray: duals array.
        """

    @abstractmethod
    def activate(self):
        """Activate shadow price tracking."""

    @abstractmethod
    def deactivate(self):
        """Deactivate shadow price tracking."""

    @property
    @abstractmethod
    def is_active(self) -> bool:
        """Check whether shadow price tracking is active or not."""

    @property
    @abstractmethod
    def available_constraints(self) -> Iterable:
        """Iterable of constraints that are available to provide shadow prices on."""

    @property
    def tracked(self) -> set:
        """Constraints being tracked for automatic addition to the results dataset."""
        return self._tracked

    def track_constraints(self, constraints_to_track: list):
        """Track constraints if they are available in the built backend model.

        If there is at least one available constraint to track,
        `self.tracked` will be updated and shadow price tracking will be activated.

        Args:
            constraints_to_track (list): Constraint names to track
        """
        shadow_prices = set(constraints_to_track)
        invalid_constraints = shadow_prices.difference(self.available_constraints)
        valid_constraints = shadow_prices.intersection(self.available_constraints)
        if invalid_constraints:
            model_warn(
                f"Invalid constraints {invalid_constraints} in `config_schema.solve.shadow_prices`. "
                "Their shadow prices will not be tracked."
            )
        # Only actually activate shadow price tracking if at least one valid
        # constraint remains in the list after filtering out invalid ones
        if valid_constraints:
            self.activate()
        self._tracked = valid_constraints
