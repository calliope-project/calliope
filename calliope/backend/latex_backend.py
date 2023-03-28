from typing import Any, Callable, Optional, Literal, TypeVar, Generic, Union, Iterable

import xarray as xr
import numpy as np

from calliope.backend import backends
from calliope.backend import parsing, equation_parser
from calliope.exceptions import BackendError


class LatexBackendModel(backends.BackendModel):
    def __init__(self):
        """Abstract base class for interfaces to solvers.

        Args:
            instance (T): Interface model instance.
        """
        backends.BackendModel.__init__(self, instance=None)

    def add_parameter(
        self,
        parameter_name: str,
        parameter_values: xr.DataArray,
        default: Any = np.nan,
        use_inf_as_na: bool = False,
    ) -> None:
        self._add_to_dataset(parameter_name, parameter_values, "parameters")
        self.valid_arithmetic_components.add(parameter_name)

    def add_constraint(
        self,
        model_data: xr.Dataset,
        name: str,
        constraint_dict: parsing.UnparsedConstraintDict,
    ) -> None:
        self._add_constraint_or_expression(
            model_data,
            name,
            constraint_dict,
            lambda x: None,
            "constraints",
            equation_parser.generate_equation_parser,
        )

    def add_expression(
        self,
        model_data: xr.Dataset,
        name: str,
        expression_dict: parsing.UnparsedConstraintDict,
    ) -> None:
        self.valid_arithmetic_components.add(name)

        self._add_constraint_or_expression(
            model_data,
            name,
            expression_dict,
            lambda x: None,
            "expressions",
            equation_parser.generate_arithmetic_parser,
        )

    def add_variable(
        self,
        model_data: xr.Dataset,
        name: str,
        variable_dict: parsing.UnparsedVariableDict,
    ) -> None:
        self.valid_arithmetic_components.add(name)
        self._raise_error_on_preexistence(name, "variables")

        parsed_variable = parsing.ParsedBackendComponent(name, variable_dict)
        foreach_imask = parsed_variable.evaluate_foreach(model_data)

        parsed_variable.parse_top_level_where()
        imask = parsed_variable.evaluate_where(model_data, initial_imask=foreach_imask)
        imask = parsed_variable.align_imask_with_sets(imask)
        imask_latex = parsed_variable.evaluate_where(model_data, as_latex=True)
        imask.attrs["latex_strings"] = {1: {"where": imask_latex}}
        self._add_to_dataset(name, imask.where(imask), "variables")

    def add_objective(
        self,
        model_data: xr.Dataset,
        name: str,
        objective_dict: parsing.UnparsedObjectiveDict,
    ) -> None:
        self._raise_error_on_preexistence(name, "objectives")
        sense_dict = {"minimize": r"\min{}", "maximize": r"\max{}"}
        parsed_objective = parsing.ParsedBackendComponent(name, objective_dict)
        equations = parsed_objective.parse_equations(
            equation_parser.generate_arithmetic_parser, self.valid_arithmetic_components
        )
        latex_strings = {}
        valid_expressions = []
        for element in equations:
            imask = element.evaluate_where(model_data)
            if imask.any():
                valid_expressions.append(element.name)
            imask_latex = element.evaluate_where(model_data, as_latex=True)
            expr = element.evaluate_expression(model_data, self, as_latex=True)
            latex_strings[element.name] = {
                "expression": sense_dict[objective_dict["sense"]] + expr,
                "where": imask_latex,
            }

        self._add_to_dataset(
            name,
            xr.DataArray(valid_expressions).assign_attrs(latex_strings=latex_strings),
            "objectives",
        )

    def get_parameter(
        self, parameter_name: str, as_backend_objs: bool = True
    ) -> Optional[xr.DataArray]:
        return self.parameters.get(parameter_name, None)

    def create_obj_list(self, key: str, component_type: backends._COMPONENTS_T) -> None:
        return None

    def get_constraint(
        self,
        constraint_name: str,
        as_backend_objs: bool = True,
        eval_body: bool = False,
    ) -> Optional[Union[xr.DataArray, xr.Dataset]]:
        return self.constraints.get(constraint_name, None)

    def get_variable(
        self, variable_name: str, as_backend_objs: bool = True
    ) -> Optional[xr.DataArray]:
        return self.variables.get(variable_name, None)

    def get_expression(
        self, expression_name: str, as_backend_objs: bool = True, eval_body: bool = True
    ) -> Optional[xr.DataArray]:
        return self.expressions.get(expression_name, None)

    def solve(
        self,
        solver: str,
        solver_io: Optional[str] = None,
        solver_options: Optional[dict] = None,
        save_logs: Optional[str] = None,
        warmstart: bool = False,
        **solve_kwargs,
    ):
        raise BackendError(
            "Cannot solve a LaTex backend model - this only exists to produce a string representation of the model math"
        )

    def _add_constraint_or_expression(
        self,
        model_data: xr.Dataset,
        name: str,
        component_dict: parsing.UnparsedConstraintDict,
        component_setter: Callable,
        component_type: Literal["constraints", "expressions"],
        parser: Callable,
    ) -> None:
        references: set[str] = set()

        parsed_component = parsing.ParsedBackendComponent(name, component_dict)
        foreach_imask = parsed_component.evaluate_foreach(model_data)
        parsed_component.parse_top_level_where()
        top_level_imask = parsed_component.evaluate_where(
            model_data, initial_imask=foreach_imask
        )
        top_level_imask_latex = parsed_component.evaluate_where(
            model_data, as_latex=True
        )
        self._raise_error_on_preexistence(name, component_type)

        component_da = (
            xr.DataArray()
            .where(parsed_component.align_imask_with_sets(top_level_imask))
            .assign_attrs(latex_strings={"where": top_level_imask_latex})
        )

        equations = parsed_component.parse_equations(
            parser, self.valid_arithmetic_components
        )
        for element in equations:
            imask = element.evaluate_where(model_data, initial_imask=top_level_imask)
            imask = parsed_component.align_imask_with_sets(imask)

            if component_da.where(imask).notnull().any():
                subset_overlap = component_da.where(imask).to_series().dropna().index

                raise BackendError(
                    "Trying to set two equations for the same index of "
                    f"{component_type.removesuffix('s')} `{name}`:\n{subset_overlap}"
                )

            expr = element.evaluate_expression(
                model_data, self, as_latex=True, references=references
            )
            imask_latex = element.evaluate_where(model_data, as_latex=True)
            component_da = component_da.fillna(xr.DataArray(element.name).where(imask))
            component_da.latex_strings[element.name] = {
                "expression": expr,
                "where": imask_latex,
            }

        self._add_to_dataset(name, component_da, component_type, references)
