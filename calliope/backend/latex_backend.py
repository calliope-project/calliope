from __future__ import annotations

import textwrap
import typing
from pathlib import Path
from typing import Any, Callable, Literal, Optional, Union, overload

import jinja2
import numpy as np
import xarray as xr

from calliope.backend import backends, parsing
from calliope.exceptions import BackendError, ModelError

_ALLOWED_MATH_FILE_FORMATS = Literal["tex", "rst", "md"]


class MathDocumentation:
    def __init__(self, backend_builder: Callable) -> None:
        """Math documentation builder/writer

        Args:
            backend_builder (Callable):
                Method to generate all optimisation problem components on a calliope.backends.BackendModel object.
        """
        self._builder = backend_builder

    def build(
        self,
        include: Literal["all", "valid"] = "all",
    ) -> None:
        """Build string representations of the mathematical formulation using LaTeX math notation, ready to be written with `write`.

        Args:
            include (Literal["all", "valid"], optional):
                Defines whether to include all possible math equations ("all") or only those for which at least one index item in the "where" string is valid ("valid"). Defaults to "all".
        """

        backend = LatexBackendModel(include=include)

        self._instance = self._builder(backend)

    @overload  # noqa: F811
    def write(  # noqa: F811
        self,
        filename: Literal[None] = None,
        format: Optional[_ALLOWED_MATH_FILE_FORMATS] = None,
    ) -> str:
        "Expecting string if not giving filename"

    @overload  # noqa: F811
    def write(  # noqa: F811
        self,
        filename: Union[str, Path],
    ) -> None:
        "Expecting None (and format arg is not needed) if giving filename"

    def write(  # noqa: F811
        self,
        filename: Optional[Union[str, Path]] = None,
        format: Optional[_ALLOWED_MATH_FILE_FORMATS] = None,
    ) -> Optional[str]:
        """_summary_

        Args:
            filename (Optional[str], optional):
                If given, will write the built mathematical formulation to a file with the given extension as the file format. Defaults to None.

            format (Optional["tex", "rst", "md"], optional):
                Not required if filename is given (as the format will be automatically inferred). Required if expecting a string return from calling this function. The LaTeX math will be embedded in a document of the given format (tex=LaTeX, rst=reStructuredText, md=Markdown). Defaults to None.

        Raises:
            exceptions.ModelError: Math strings need to be built first (`build`)
            ValueError: The file format (inferred automatically from `filename` or given by `format`) must be one of ["tex", "rst", "md"].

        Returns:
            Optional[str]:
                If `filename` is None, the built mathematical formulation documentation will be returned as a string.
        """
        if not hasattr(self, "_instance"):
            raise ModelError(
                "Build the documentation (`build`) before trying to write it"
            )

        if format is None and filename is not None:
            format = Path(filename).suffix.removeprefix(".")  # type: ignore

        allowed_formats = typing.get_args(_ALLOWED_MATH_FILE_FORMATS)
        if format is None or format not in allowed_formats:
            raise ValueError(
                f"Math documentation style must be one of {allowed_formats}, received `{format}`"
            )
        populated_doc = self._instance.generate_math_doc(format)

        if filename is None:
            return populated_doc
        else:
            Path(filename).write_text(populated_doc)
            return None


class LatexBackendModel(backends.BackendModel):
    # \negthickspace used to counter the introduction of spaces to separate the curly braces
    # in \text. Curly braces need separating otherwise jinja2 gets confused.
    LATEX_EQUATION_ELEMENT = textwrap.dedent(
        r"""
        \begin{array}{r}
        {% if sets is defined and sets%}
            \forall{}
        {% for set in sets %}
            \text{ {{set|removesuffix("s")}} }\negthickspace \in \negthickspace\text{ {{set + "," if not loop.last else set }} }
        {% endfor %}
            \\
        {% endif %}
        {% if sense is defined %}
            {{sense}}
        {% endif %}
        {% if where is defined and where != "" %}
            \text{if } {{where}}
        {% endif %}
        \end{array}
        \begin{cases}
        {% for equation in equations %}
            {{equation["expression"]}}&\quad
        {% if "where" in equation and equation["where"] != "" %}
            \text{if } {{equation["where"]}}
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
    ---------
    {% elif component_type == "constraints" %}

    Subject to
    ----------
    {% elif component_type == "global_expressions" %}

    Where
    -----
    {% elif component_type == "variables" %}

    Decision Variables
    ------------------
    {% endif %}
    {% for equation in equations %}

    {{ equation.name }}
    {{ "^" * equation.name|length }}

    {% if equation.description is not none %}
    {{ equation.description }}
    {% endif %}

    {% if equation.expression != "" %}
    .. container:: scrolling-wrapper

        .. math::{{ equation.expression | indent(8) }}
    {% endif %}
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
    {% elif component_type == "global_expressions" %}
    \section{Where}
    {% elif component_type == "variables" %}
    \section{Decision Variables}
    {% endif %}
    {% for equation in equations %}

    \paragraph{ {{ equation.name }} }
    {% if equation.description is not none %}
    {{ equation.description }}
    {% endif %}
    {% if equation.expression != "" %}
    \begin{equation}
    \resizebox{\ifdim\width>\linewidth0.95\linewidth\else\width\fi}{!}{${{ equation.expression }}
    $}
    \end{equation}
    {% endif %}
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
    {% elif component_type == "global_expressions" %}

    # Where
    {% elif component_type == "variables" %}

    # Decision Variables
    {% endif %}
    {% for equation in equations %}

    ## {{ equation.name }}
    {% if equation.description is not none %}
    {{ equation.description }}
    {% endif %}
    {% if equation.expression != "" %}

        ```math{{ equation.expression | indent(4) }}
        ```
    {% endif %}
    {% endfor %}
    {% endfor %}
    """
    )
    FORMAT_STRINGS = {"rst": RST_DOC, "tex": TEX_DOC, "md": MD_DOC}

    def __init__(self, include: Literal["all", "valid"] = "all") -> None:
        """Interface to build a string representation of the mathematical formulation using LaTeX math notation.

        Args:
            include (Literal["all", "valid"], optional):
                Defines whether to include all possible math equations ("all") or only those for which at least one index item in the "where" string is valid ("valid"). Defaults to "all".
        """
        backends.BackendModel.__init__(self, instance=dict())
        self.include = include

    def add_parameter(
        self,
        parameter_name: str,
        parameter_values: xr.DataArray,
        default: Any = np.nan,
        use_inf_as_na: bool = False,
    ) -> None:
        self._add_to_dataset(parameter_name, parameter_values, "parameters", {})
        self.valid_math_element_names.add(parameter_name)

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
        )

    def add_global_expression(
        self,
        model_data: xr.Dataset,
        name: str,
        expression_dict: parsing.UnparsedConstraintDict,
    ) -> None:
        self.valid_math_element_names.add(name)

        self._add_constraint_or_expression(
            model_data,
            name,
            expression_dict,
            lambda x: None,
            "global_expressions",
        )

    def add_variable(
        self,
        model_data: xr.Dataset,
        name: str,
        variable_dict: parsing.UnparsedVariableDict,
    ) -> None:
        self.valid_math_element_names.add(name)
        self._raise_error_on_preexistence(name, "variables")

        parsed_variable = parsing.ParsedBackendComponent(
            "variables", name, variable_dict
        )
        imask = parsed_variable.generate_top_level_where_array(
            model_data, break_early=False
        )

        # add early to be accessed when creating bound strings.
        self._add_to_dataset(name, imask, "variables", variable_dict)

        imask_latex = parsed_variable.evaluate_where(model_data, as_latex=True)
        lb, ub = self._get_capacity_bounds(variable_dict["bounds"], name, model_data)

        if self.include == "all" or (self.include == "valid" and imask.any()):
            self._generate_math_string(
                imask,
                sets=parsed_variable.sets,
                where=imask_latex if imask_latex != "" else None,
                equations=[lb, ub],
            )
        # add again to ensure "math_string" attribute is there.
        self._add_to_dataset(name, imask, "variables", variable_dict)

    def add_objective(
        self,
        model_data: xr.Dataset,
        name: str,
        objective_dict: parsing.UnparsedObjectiveDict,
    ) -> None:
        self._raise_error_on_preexistence(name, "objectives")
        sense_dict = {
            "minimize": r"\min{}",
            "maximize": r"\max{}",
            "minimise": r"\min{}",
            "maximise": r"\max{}",
        }
        parsed_objective = parsing.ParsedBackendComponent(
            "objectives", name, objective_dict
        )
        equations = parsed_objective.parse_equations(self.valid_math_element_names)
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
                objective_da,
                sense=sense_dict[objective_dict["sense"]],
                equations=equation_strings,
            )
        self._add_to_dataset(name, objective_da, "objectives", objective_dict)

    def get_parameter(
        self, parameter_name: str, as_backend_objs: bool = True
    ) -> Optional[xr.DataArray]:
        return self.parameters.get(parameter_name, None)

    def create_obj_list(self, key: str, component_type: backends._COMPONENTS_T) -> None:
        return None

    def delete_obj_list(self, key: str, component_type: backends._COMPONENTS_T) -> None:
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

    def get_global_expression(
        self, expression_name: str, as_backend_objs: bool = True, eval_body: bool = True
    ) -> Optional[xr.DataArray]:
        return self.global_expressions.get(expression_name, None)

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

    def verbose_strings(self):
        return None

    def generate_math_doc(self, format: _ALLOWED_MATH_FILE_FORMATS = "tex") -> str:
        """Generate the math documentation by embedding LaTeX math in a template.

        Args:
            format (Literal["tex", "rst", "md"]):
                The built LaTeX math will be embedded in a document of the given format (tex=LaTeX, rst=reStructuredText, md=Markdown). Defaults to "tex".

        Returns:
            str: Generated math documentation.
        """
        doc_template = self.FORMAT_STRINGS[format]
        components = {
            objtype: [
                {
                    "expression": da.attrs["math_string"],
                    "name": name,
                    "description": da.attrs.get("description", None),
                }
                for name, da in getattr(self, objtype).data_vars.items()
                if "math_string" in da.attrs
            ]
            for objtype in [
                "objectives",
                "constraints",
                "global_expressions",
                "variables",
            ]
            if getattr(self, objtype).data_vars
        }
        return self._render(doc_template, components=components)

    def _add_constraint_or_expression(
        self,
        model_data: xr.Dataset,
        name: str,
        component_dict: Union[
            parsing.UnparsedConstraintDict, parsing.UnparsedExpressionDict
        ],
        component_setter: Callable,
        component_type: Literal["constraints", "global_expressions"],
    ) -> None:
        parsed_component = parsing.ParsedBackendComponent(
            component_type, name, component_dict
        )

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

        equations = parsed_component.parse_equations(self.valid_math_element_names)
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
                component_da,
                sets=parsed_component.sets,
                where=top_level_imask_latex,
                equations=equation_strings,
            )
        self._add_to_dataset(
            name, component_da.fillna(0), component_type, component_dict
        )

    def _generate_math_string(
        self,
        da: xr.DataArray,
        **kwargs,
    ) -> None:
        equation_element_string = self._render(self.LATEX_EQUATION_ELEMENT, **kwargs)
        da.attrs.update({"math_string": equation_element_string})
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
        parsed_bounds = parsing.ParsedBackendComponent("constraints", name, bound_dict)
        equations = parsed_bounds.parse_equations(
            self.valid_math_element_names,
        )
        return tuple(
            {"expression": eq.evaluate_expression(model_data, self, as_latex=True)}
            for eq in equations
        )
