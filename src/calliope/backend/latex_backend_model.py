from __future__ import annotations

import logging
import re
import textwrap
import typing
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Optional, Union, overload

import jinja2
import numpy as np
import xarray as xr

from calliope.backend import backend_model, parsing
from calliope.exceptions import ModelError

if TYPE_CHECKING:
    pass

_ALLOWED_MATH_FILE_FORMATS = Literal["tex", "rst", "md"]


LOGGER = logging.getLogger(__name__)


class MathDocumentation:
    def __init__(self) -> None:
        """Math documentation builder/writer

        Args:
            backend_builder (Callable):
                Method to generate all optimisation problem components on a calliope.backend_model.BackendModel object.
        """
        self._inputs: xr.Dataset

    def build(self, include: Literal["all", "valid"] = "all", **kwargs) -> None:
        """Build string representations of the mathematical formulation using LaTeX math notation, ready to be written with `write`.

        Args:
            include (Literal["all", "valid"], optional):
                Defines whether to include all possible math equations ("all") or only those for which at least one index item in the "where" string is valid ("valid"). Defaults to "all".
        """

        backend = LatexBackendModel(self._inputs, include=include, **kwargs)
        backend._build()

        self._instance = backend

    @property
    def inputs(self):
        return self._inputs

    @inputs.setter
    def inputs(self, val: xr.Dataset):
        self._inputs = val

    @overload  # noqa: F811
    def write(  # noqa: F811
        self,
        filename: Literal[None] = None,
        format: Optional[_ALLOWED_MATH_FILE_FORMATS] = None,
    ) -> str:
        "Expecting string if not giving filename"

    @overload  # noqa: F811
    def write(self, filename: Union[str, Path]) -> None:  # noqa: F811
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
                Not required if filename is given (as the format will be automatically inferred).
                Required if expecting a string return from calling this function. The LaTeX math will be embedded in a document of the given format (tex=LaTeX, rst=reStructuredText, md=Markdown).
                Defaults to None.

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
            LOGGER.info(
                f"Inferring math documentation format from filename as `{format}`."
            )

        allowed_formats = typing.get_args(_ALLOWED_MATH_FILE_FORMATS)
        if format is None or format not in allowed_formats:
            raise ValueError(
                f"Math documentation format must be one of {allowed_formats}, received `{format}`"
            )
        populated_doc = self._instance.generate_math_doc(format)

        if filename is None:
            return populated_doc
        else:
            Path(filename).write_text(populated_doc)
            return None


class LatexBackendModel(backend_model.BackendModelGenerator):
    # \negthickspace used to counter the introduction of spaces to separate the curly braces
    # in \text. Curly braces need separating otherwise jinja2 gets confused.
    LATEX_EQUATION_ELEMENT = textwrap.dedent(
        r"""
        \begin{array}{r}
        {% if sets is defined and sets %}
            \forall{}
        {% for set in sets %}
            \text{ {{set|removesuffix("s")}} }\negthickspace \in \negthickspace\text{ {{set + "," if not loop.last else set }} }
        {% endfor %}
            \\
        {% endif %}
        {% if sense is defined and sense %}
            {{sense}}
        {% endif %}
        {% if where is defined and where and where != "" %}
            \text{if } {{where}}
        {% endif %}
        \end{array}
        \begin{cases}
        {% for equation in equations %}
            {{equation["expression"]}}&\quad
        {% if "where" in equation and equation.where != "" %}
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
    ## Objective
    {% elif component_type == "constraints" %}

    ## Subject to
    {% elif component_type == "global_expressions" %}

    ## Where
    {% elif component_type == "variables" %}

    ## Decision Variables
    {% endif %}
    {% for equation in equations %}

    ### {{ equation.name }}
    {% if equation.description is not none %}
    {{ equation.description }}
    {% endif %}
    {% if equation.expression != "" %}

    $$
    {{ equation.expression | trim | escape_underscores | mathify_text_in_text }}
    $$
    {% endif %}
    {% endfor %}
    {% endfor %}
    """
    )
    FORMAT_STRINGS = {"rst": RST_DOC, "tex": TEX_DOC, "md": MD_DOC}

    def __init__(
        self, inputs: xr.Dataset, include: Literal["all", "valid"] = "all", **kwargs
    ) -> None:
        """Interface to build a string representation of the mathematical formulation using LaTeX math notation.

        Args:
            include (Literal["all", "valid"], optional):
                Defines whether to include all possible math equations ("all") or only those for which at least one index item in the "where" string is valid ("valid"). Defaults to "all".
        """
        super().__init__(inputs, **kwargs)
        self.include = include

        self._add_all_inputs_as_parameters()

    def add_parameter(
        self,
        parameter_name: str,
        parameter_values: xr.DataArray,
        default: Any = np.nan,
        use_inf_as_na: bool = False,
    ) -> None:
        self._add_to_dataset(parameter_name, parameter_values, "parameters", {})

    def add_constraint(
        self,
        name: str,
        constraint_dict: Optional[parsing.UnparsedConstraintDict] = None,
    ) -> None:
        equation_strings: list = []

        def _constraint_setter(
            element: parsing.ParsedBackendEquation, where: xr.DataArray, references: set
        ) -> xr.DataArray:
            self._add_latex_strings(where, element, equation_strings)
            return where.where(where)

        parsed_component = self._add_component(
            name, constraint_dict, _constraint_setter, "constraints", break_early=False
        )

        self._generate_math_string(
            parsed_component, self.constraints[name], equations=equation_strings
        )

    def add_global_expression(
        self,
        name: str,
        expression_dict: Optional[parsing.UnparsedExpressionDict] = None,
    ) -> None:
        equation_strings: list = []

        def _expression_setter(
            element: parsing.ParsedBackendEquation, where: xr.DataArray, references: set
        ) -> xr.DataArray:
            self._add_latex_strings(where, element, equation_strings)
            return where.where(where)

        parsed_component = self._add_component(
            name,
            expression_dict,
            _expression_setter,
            "global_expressions",
            break_early=False,
        )

        self._generate_math_string(
            parsed_component, self.global_expressions[name], equations=equation_strings
        )

    def add_variable(
        self, name: str, variable_dict: Optional[parsing.UnparsedVariableDict] = None
    ) -> None:
        domain_dict = {"real": r"\mathbb{R}\;", "integer": r"\mathbb{Z}\;"}

        def _variable_setter(where: xr.DataArray) -> xr.DataArray:
            return where.where(where)

        if variable_dict is None:
            variable_dict = self.inputs.attrs["math"]["variables"][name]

        parsed_component = self._add_component(
            name, variable_dict, _variable_setter, "variables", break_early=False
        )
        where_array = self.variables[name]

        domain = domain_dict[variable_dict.get("domain", "real")]
        lb, ub = self._get_capacity_bounds(name, variable_dict["bounds"])

        self._generate_math_string(
            parsed_component, where_array, equations=[lb, ub], sense=r"\forall" + domain
        )

    def add_objective(
        self, name: str, objective_dict: Optional[parsing.UnparsedObjectiveDict] = None
    ) -> None:
        sense_dict = {
            "minimize": r"\min{}",
            "maximize": r"\max{}",
            "minimise": r"\min{}",
            "maximise": r"\max{}",
        }
        if objective_dict is None:
            objective_dict = self.inputs.attrs["math"]["objectives"][name]
        equation_strings: list = []

        def _objective_setter(
            element: parsing.ParsedBackendEquation, where: xr.DataArray, references: set
        ) -> None:
            self._add_latex_strings(where, element, equation_strings)
            return None

        parsed_component = self._add_component(
            name, objective_dict, _objective_setter, "objectives", break_early=False
        )

        self._generate_math_string(
            parsed_component,
            self.objectives[name],
            equations=equation_strings,
            sense=sense_dict[objective_dict["sense"]],
        )

    def _create_obj_list(
        self, key: str, component_type: backend_model._COMPONENTS_T
    ) -> None:
        return None

    def delete_component(
        self, key: str, component_type: backend_model._COMPONENTS_T
    ) -> None:
        if key in self._dataset and self._dataset[key].obj_type == component_type:
            del self._dataset[key]

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

    def _add_latex_strings(self, where, element, equation_strings):
        expr = element.evaluate_expression(self, return_type="math_string")
        where_latex = element.evaluate_where(self, return_type="math_string")

        if self.include == "all" or (self.include == "valid" and where.any()):
            equation_strings.append({"expression": expr, "where": where_latex})

    def _generate_math_string(
        self,
        parsed_component: Optional[parsing.ParsedBackendComponent],
        where_array: xr.DataArray,
        equations: Optional[list[dict[str, str]]] = None,
        sense: Optional[str] = None,
        where: Optional[str] = None,
        sets: Optional[list[str]] = None,
    ) -> None:
        if parsed_component is not None:
            where = parsed_component.evaluate_where(self, return_type="math_string")
            sets = parsed_component.sets
        if self.include == "all" or (
            self.include == "valid" and where_array.fillna(0).any()
        ):
            equation_element_string = self._render(
                self.LATEX_EQUATION_ELEMENT,
                equations=equations if equations is not None else [],
                sense=sense,
                where=where,
                sets=sets,
            )
            where_array.attrs.update({"math_string": equation_element_string})
        return None

    @staticmethod
    def _render(template: str, **kwargs) -> str:
        text_starter = r"\\text(?:bf|it)?"  # match one of `\text`, `\textit`, `\textbf`

        def __escape_underscore(instring):
            "KaTeX requires underscores in `\text{...}` blocks to be escaped."
            return re.sub(
                rf"{text_starter}{{.*?}}",
                lambda x: x.group(0).replace("_", r"\_"),
                instring,
            )

        def __mathify_text_in_text(instring):
            """KaTeX requires `\text{...}` blocks within `\text{...}` blocks to be placed within math blocks.

            We use `\\(` as the math block descriptor.
            """
            return re.sub(
                rf"{text_starter}{{(?:[^{{}}]*({text_starter}{{.*?}})[^{{}}]*?)?}}",
                lambda x: x.group(0).replace(x.group(1), rf"\({x.group(1)}\)"),
                instring,
            )

        jinja_env = jinja2.Environment(trim_blocks=True, autoescape=False)
        jinja_env.filters["removesuffix"] = lambda val, remove: val.removesuffix(remove)
        jinja_env.filters["escape_underscores"] = __escape_underscore
        jinja_env.filters["mathify_text_in_text"] = __mathify_text_in_text
        return jinja_env.from_string(template).render(**kwargs)

    def _get_capacity_bounds(
        self, name: str, bounds: parsing.UnparsedVariableBoundDict
    ) -> tuple[dict[str, str], ...]:
        bound_dict: parsing.UnparsedConstraintDict = {
            "equations": [
                {"expression": f"{bounds['min']} <= {name}"},
                {"expression": f"{name} <= {bounds['max']}"},
            ]
        }
        parsed_bounds = parsing.ParsedBackendComponent("constraints", name, bound_dict)
        equations = parsed_bounds.parse_equations(self.valid_component_names)
        return tuple(
            {"expression": eq.evaluate_expression(self, return_type="math_string")}
            for eq in equations
        )
