from __future__ import annotations

from typing import Any, Callable, Optional, Literal, TypeVar, Generic, Union, Iterable
import textwrap

import xarray as xr
import numpy as np
import jinja2

from calliope.backend import backends
from calliope.backend import parsing, equation_parser
from calliope.exceptions import BackendError


class LatexBackendModel(backends.BackendModel):
    LATEX_EQUATION_ELEMENT = textwrap.dedent(
        r"""
        \begin{array}{r}
        {% if sets is defined and sets%}
            \forall{}
        {% for set in sets %}
            \text{\,{{set|removesuffix("s")}}\,} \in \text{\,{{set + ", " if not loop.last else set }}\,}
        {% endfor %}
            \\
        {% endif %}
        {% if sense is defined %}
            {{sense}}
        {% endif %}
        {% if where is defined and where != "" %}
            if {{where}}
        {% endif %}
        \end{array}
        \begin{cases}
        {% for equation in equations %}
            {{equation["expression"]}}&\quad
        {% if "where" in equation and equation["where"] != "" %}
            if {{equation["where"]}}
        {% endif %}
            \\
        {% endfor %}
        \end{cases}
        """
    )
    RST_DOC = textwrap.dedent(
        r"""
    {% for component_type, equations in components.items() %}
    {% if component_type == "objectives" %}
    Objective
    #########
    {% elif component_type == "constraints" %}

    Subject to
    ##########
    {% elif component_type == "expressions" %}

    Where
    #####
    {% elif component_type == "variables" %}

    Decision Variables
    ##################
    {% endif %}
    {% for equation in equations %}

    {{ equation.name }}
    {{ "=" * equation.name|length }}

    .. container:: scrolling-wrapper

        .. math::{{ equation.expression | indent(8) }}
    {% endfor %}
    {% endfor %}
    """
    )

    TEX_DOC = textwrap.dedent(
        r"""
    \documentclass{article}

    \usepackage{amsmath}
    \usepackage[T1]{fontenc}
    \usepackage{graphicx}
    \usepackage[landscape, margin=2mm]{geometry}

    \AtBeginDocument{ % escape underscores that are not in math
        \catcode`_=12
        \begingroup\lccode`~=`_
        \lowercase{\endgroup\let~}\sb
        \mathcode`_="8000
    }

    \begin{document}
    {% for component_type, equations in components.items() %}
    {% if component_type == "objectives" %}
    \section{Objective}
    {% elif component_type == "constraints" %}
    \section{Subject to}
    {% elif component_type == "expressions" %}
    \section{Where}
    {% elif component_type == "variables" %}
    \section{Decision Variables}
    {% endif %}
    {% for equation in equations %}

    \paragraph{ {{ equation.name }} }
    \begin{equation}
    \resizebox{\ifdim\width>\linewidth0.95\linewidth\else\width\fi}{!}{${{ equation.expression }}
    $}
    \end{equation}
    {% endfor %}
    {% endfor %}
    \end{document}
    """
    )
    MD_DOC = textwrap.dedent(
        r"""
    {% for component_type, equations in components.items() %}
    {% if component_type == "objectives" %}
    # Objective
    {% elif component_type == "constraints" %}

    # Subject to
    {% elif component_type == "expressions" %}

    # Where
    {% elif component_type == "variables" %}

    # Decision Variables
    {% endif %}
    {% for equation in equations %}

    ## {{ equation.name }}
        ```math{{ equation.expression | indent(4) }}
        ```
    {% endfor %}
    {% endfor %}
    """
    )
    FORMAT_STRINGS = {"rst": RST_DOC, "tex": TEX_DOC, "md": MD_DOC}

    def __init__(
        self,
        include: Literal["all", "valid"] = "all",
        format: Literal["tex", "rst", "md"] = "tex",
    ):
        """Interface to build a string representation of the mathematical formulation using LaTeX math notation.

        Args:
            include (Literal["all", "valid"], optional):
                Defines whether to include all possible math equations ("all") or only those for which at least one index item in the "where" string is valid ("valid"). Defaults to "all".
            format (Optional["tex", "rst", "md"], optional):
                Not required if filename is given (as the format will be automatically inferred). Required if expecting a string return from calling this function. The LaTeX math will be embedded in a document of the given format (tex=LaTeX, rst=reStructuredText, md=Markdown). Defaults to None.
        """
        backends.BackendModel.__init__(self, instance=dict())
        self.include = include
        for component in ["objectives", "constraints", "expressions", "variables"]:
            self._instance[component] = []
        self._doctemplate = self.FORMAT_STRINGS[format]

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
        imask = parsed_variable.generate_top_level_where_array(
            model_data, break_early=False
        )

        # add early to be accessed when creating bound strings.
        self._add_to_dataset(name, imask, "variables")

        imask_latex = parsed_variable.evaluate_where(model_data, as_latex=True)
        lb, ub = self._get_capacity_bounds(variable_dict["bounds"], name, model_data)

        if self.include == "all" or (self.include == "valid" and imask.any()):
            self._generate_math_string(
                name,
                imask,
                "variables",
                sets=parsed_variable.sets,
                where=imask_latex if imask_latex != "" else None,
                equations=[lb, ub],
            )
        # add again to ensure "math_string" attribute is there.
        self._add_to_dataset(name, imask, "variables")

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
        equation_strings = []
        for element in equations:
            if self.include == "valid":
                imask = element.evaluate_where(model_data)
            imask_latex = element.evaluate_where(model_data, as_latex=True)
            expr = element.evaluate_expression(model_data, self, as_latex=True)

            if self.include == "all" or (self.include == "valid" and imask.any()):
                equation_strings.append({"expression": expr, "where": imask_latex})
        objective_da = xr.DataArray()
        if equation_strings:
            self._generate_math_string(
                name,
                objective_da,
                "objectives",
                sense=sense_dict[objective_dict["sense"]],
                equations=equation_strings,
            )
        self._add_to_dataset(name, objective_da, "objectives")

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

    def generate_math_doc(self):
        return self._render(self._doctemplate, components=self._instance)

    def _add_constraint_or_expression(
        self,
        model_data: xr.Dataset,
        name: str,
        component_dict: parsing.UnparsedConstraintDict,
        component_setter: Callable,
        component_type: Literal["constraints", "expressions"],
        parser: Callable,
    ) -> None:
        parsed_component = parsing.ParsedBackendComponent(name, component_dict)

        top_level_imask = parsed_component.generate_top_level_where_array(
            model_data, break_early=False
        )
        component_da = xr.DataArray().where(
            parsed_component.align_imask_with_foreach_sets(top_level_imask)
        )
        top_level_imask_latex = parsed_component.evaluate_where(
            model_data, as_latex=True
        )

        self._raise_error_on_preexistence(name, component_type)

        equations = parsed_component.parse_equations(
            parser, self.valid_arithmetic_components
        )
        equation_strings = []
        for element in equations:
            imask = element.evaluate_where(model_data, initial_imask=top_level_imask)
            imask = parsed_component.align_imask_with_foreach_sets(imask)

            expr = element.evaluate_expression(model_data, self, as_latex=True)
            imask_latex = element.evaluate_where(model_data, as_latex=True)

            if self.include == "all" or (self.include == "valid" and imask.any()):
                equation_strings.append({"expression": expr, "where": imask_latex})
            component_da = component_da.fillna(imask.where(imask))
        if equation_strings:
            self._generate_math_string(
                name,
                component_da,
                component_type,
                sets=parsed_component.sets,
                where=top_level_imask_latex,
                equations=equation_strings,
            )
        self._add_to_dataset(name, component_da.fillna(0), component_type)

    def _generate_math_string(
        self,
        name: str,
        da: xr.DataArray,
        component_type: backends._COMPONENTS_T,
        **kwargs,
    ) -> None:
        equation_element_string = self._render(self.LATEX_EQUATION_ELEMENT, **kwargs)
        da.attrs.update({"math_string": equation_element_string})
        self._instance[component_type].append(
            {"expression": equation_element_string, "name": name}
        )
        return None

    @staticmethod
    def _render(template: str, **kwargs) -> str:
        jinja_env = jinja2.Environment(trim_blocks=True, autoescape=False)
        jinja_env.filters["removesuffix"] = lambda val, remove: val.removesuffix(remove)
        return jinja_env.from_string(template).render(**kwargs)

    def _get_capacity_bounds(
        self,
        bounds: parsing.UnparsedVariableBoundDict,
        name: str,
        model_data: xr.Dataset,
    ) -> tuple[dict[str, str], ...]:
        bound_dict: parsing.UnparsedConstraintDict = {
            "foreach": [],
            "where": "True",
            "equations": [
                {"expression": f"{bounds['min']} <= {name}"},
                {"expression": f"{name} <= {bounds['max']}"},
            ],
        }
        parsed_bounds = parsing.ParsedBackendComponent(name, bound_dict)
        equations = parsed_bounds.parse_equations(
            equation_parser.generate_equation_parser,
            self.valid_arithmetic_components,
        )
        return tuple(
            {"expression": eq.evaluate_expression(model_data, self, as_latex=True)}
            for eq in equations
        )
