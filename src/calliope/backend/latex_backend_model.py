"""LaTeX backend functionality."""

from __future__ import annotations

import logging
import re
import textwrap
from typing import Literal

import jinja2
import xarray as xr

from calliope.backend import backend_model, parsing
from calliope.exceptions import ModelError
from calliope.io import to_yaml
from calliope.schemas import config_schema, math_schema

ALLOWED_MATH_FILE_FORMATS = Literal["tex", "rst", "md"]
LOGGER = logging.getLogger(__name__)


class LatexBackendModel(backend_model.BackendModelGenerator):
    """Calliope's LaTeX backend."""

    # \negthickspace used to counter the introduction of spaces to separate the curly braces
    # in \text. Curly braces need separating otherwise jinja2 gets confused.
    LATEX_EQUATION_ELEMENT = textwrap.dedent(
        r"""
        \begin{array}{l}
        {% if sets is defined and sets %}
            \forall{}
        {% for set in sets %}
            \text{ {{set|iterator}} }\negthickspace \in \negthickspace\text{ {{set + "," if not loop.last else set }} }
        {% endfor %}
        {% if (where is defined and where and where != "") or (sense is defined and sense) %}
            \!\!,\\
        {% else %}
            \!\!:\\[2em]
        {% endif %}
        {% endif %}
        {% if sense is defined and sense %}
            {{sense}}\!\!:\\[2em]
        {% endif %}
        {% if where is defined and where and where != "" %}
            \text{if } {{where}}\!\!:\\[2em]
        {% endif %}
        {% for equation in equations %}
        {% if "where" in equation and equation.where != "" %}
            \quad \text{if } {{equation["where"]}}\!\!:\\
            \qquad {{equation["expression"]}}\\[2em]
        {% else %}
            \quad {{equation["expression"]}}\\
        {% endif %}
        {% endfor %}
        \end{array}
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
    {% elif component_type == "piecewise_constraints" %}

    Subject to (piecewise)
    ----------------------
    {% elif component_type == "global_expressions" %}

    Where
    -----
    {% elif component_type == "variables" %}

    Decision Variables
    ------------------
    {% elif component_type == "parameters" %}

    Parameters
    ----------
    {% endif %}
    {% for equation in equations %}

    {{ equation.name }}
    {{ "^" * equation.name|length }}
    {% if equation.description != "" %}

    {{ equation.description }}
    {% endif %}
    {% if equation.used_in %}

    **Used in**:
    {% for ref in equation.used_in %}

    * {{ ref }}
    {% endfor %}
    {% endif %}
    {% if equation.uses %}

    **Uses**:
    {% for ref in equation.uses %}

    * {{ ref }}
    {% endfor %}
    {% endif %}
    {% if equation.unit != "" %}

    **Unit**: {{ equation.unit }}
    {% endif %}
    {% if equation.default is not none %}

    **Default**: {{ equation.default }}
    {% endif %}
    {% if equation.dtype is not none %}

    **Type**: {{ equation.dtype }}
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
    {% elif component_type == "piecewise_constraints" %}
    \section{Subject to (piecewise)}
    {% elif component_type == "global_expressions" %}
    \section{Where}
    {% elif component_type == "variables" %}
    \section{Decision Variables}
    {% elif component_type == "parameters" %}
    \section{Parameters}
    {% endif %}
    {% for equation in equations %}

    \paragraph{ {{ equation.name }} }
    {% if equation.description != "" %}

    {{ equation.description }}
    {% endif %}
    {% if equation.used_in %}

    \textbf{Used in}:
    {% for ref in equation.used_in %}
    \begin{itemize}
        \item {{ ref }}
    \end{itemize}
    {% endfor %}
    {% endif %}
    {% if equation.uses %}

    \textbf{Uses}:
    {% for ref in equation.uses %}
    \begin{itemize}
        \item {{ ref }}
    \end{itemize}
    {% endfor %}
    {% endif %}
    {% if equation.unit != "" %}

    \textbf{Unit}: {{ equation.unit }}
    {% endif %}
    {% if equation.default is not none %}

    \textbf{Default}: {{ equation.default }}
    {% endif %}
    {% if equation.dtype is not none %}

    \textbf{Type}: {{ equation.dtype }}
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
    {% elif component_type == "piecewise_constraints" %}

    ## Subject to (piecewise)
    {% elif component_type == "global_expressions" %}

    ## Where
    {% elif component_type == "variables" %}

    ## Decision Variables
    {% elif component_type == "parameters" %}

    ## Parameters
    {% endif %}
    {% for equation in equations %}

    ### {{ equation.name }}
    {% if equation.description != "" %}

    {{ equation.description }}
    {% endif %}
    {% if equation.used_in %}

    {% if mkdocs_features %}
    ??? info "Used in"
    {% else %}
    **Used in**:
    {% endif %}

    {% for ref in equation.used_in %}
    {{ "    " if mkdocs_features else "" }}* [{{ ref }}](#{{ ref }})
    {% endfor %}
    {% endif %}
    {% if equation.uses %}

    {% if mkdocs_features %}
    ??? info "Uses"
    {% else %}
    **Uses**:
    {% endif %}

    {% for ref in equation.uses %}
    {{ "    " if mkdocs_features else "" }}* [{{ ref }}](#{{ ref }})
    {% endfor %}
    {% endif %}
    {% if equation.unit != "" %}

    **Unit**: {{ equation.unit }}
    {% endif %}
    {% if equation.default is not none %}

    **Default**: {{ equation.default }}
    {% endif %}
    {% if equation.dtype is not none %}

    **Type**: {{ equation.dtype }}
    {% endif %}
    {% if equation.expression != "" %}
    {% if mkdocs_features and yaml_snippet is not none%}

    === "Math"

        $$
        {{ equation.expression | trim | escape_underscores | mathify_text_in_text | indent(width=4) }}
        $$

    === "YAML"

        ```yaml
        {{ equation.yaml_snippet | trim | indent(width=4) }}
        ```
    {% else %}

    $$
    {{ equation.expression | trim | escape_underscores | mathify_text_in_text}}
    $$
    {% endif %}
    {% endif %}
    {% endfor %}
    {% endfor %}
    """
    )
    FORMAT_STRINGS = {"rst": RST_DOC, "tex": TEX_DOC, "md": MD_DOC}

    def __init__(
        self,
        inputs: xr.Dataset,
        math: math_schema.CalliopeBuildMath,
        build_config: config_schema.Build,
        include: Literal["all", "valid"] = "all",
    ) -> None:
        """Interface to build a string representation of the mathematical formulation using LaTeX math notation.

        Args:
            inputs (xr.Dataset): model data.
            math (AttrDict): Calliope math.
            build_config (config_schema.Build): Build configuration options.
            include (Literal["all", "valid"], optional):
                Defines whether to include all possible math equations ("all") or only those for which at least one index item in the "where" string is valid ("valid"). Defaults to "all".
        """
        super().__init__(inputs, math, build_config)

        if include not in ["all", "valid"]:
            raise ValueError(f"Invalid `include` option: {include}")

        self.include = include

    def add_parameter(  # noqa: D102, override
        self, name: str, values: xr.DataArray, definition: math_schema.Parameter
    ) -> None:
        attrs = definition.model_dump() | {
            "math_repr": rf"\textit{{{name}}}" + self._dims_to_var_string(values)
        }

        self._add_to_dataset(name, values, "parameters", attrs)

    def add_lookup(  # noqa: D102, override
        self, name: str, values: xr.DataArray, definition: math_schema.Lookup
    ) -> None:
        super().add_lookup(name, values, definition)
        self._dataset[name].attrs["math_repr"] = (
            rf"\textit{{{name}}}" + self._dims_to_var_string(values)
        )

    def add_constraint(  # noqa: D102, override
        self, name: str, definition: math_schema.Constraint
    ) -> None:
        equation_strings: list = []

        def _constraint_setter(
            element: parsing.ParsedBackendEquation, where: xr.DataArray, references: set
        ) -> xr.DataArray:
            self._add_latex_strings(where, element, equation_strings, references)
            return where.where(where)

        parsed_component = self._add_component(
            name, definition, _constraint_setter, "constraints", break_early=False
        )

        self._generate_math_string(
            parsed_component, self.constraints[name], equations=equation_strings
        )

    def add_piecewise_constraint(  # noqa: D102, override
        self, name: str, definition: math_schema.PiecewiseConstraint
    ) -> None:
        non_where_refs: set = set()

        def _constraint_setter(where: xr.DataArray, references: set) -> xr.DataArray:
            return where.where(where)

        math_parts = {}
        for val in ["x_expression", "y_expression", "x_values", "y_values"]:
            val_name = definition[val]
            parsed_val = parsing.ParsedBackendComponent(
                "piecewise_constraints",
                name,
                math_schema.GlobalExpression.model_validate(
                    {"equations": [{"expression": val_name}]}
                ),
            )
            eq = parsed_val.parse_equations(self.valid_component_names)
            math_parts[val] = eq[0].evaluate_expression(
                self, return_type="math_string", references=non_where_refs
            )

        equation = {
            "expression": rf"{math_parts['y_expression']}\mathord{{=}}{math_parts['y_values']}",
            "where": rf"{math_parts['x_expression']}\mathord{{=}}{math_parts['x_values']}",
        }

        constraint_def_with_breakpoints = definition.update(
            {"foreach": definition.foreach + ["breakpoints"]}
        )

        parsed_component = self._add_component(
            name,
            constraint_def_with_breakpoints,
            _constraint_setter,
            "piecewise_constraints",
            break_early=False,
        )
        self._update_references(name, non_where_refs)
        self._generate_math_string(
            parsed_component, self.piecewise_constraints[name], equations=[equation]
        )

    def add_global_expression(  # noqa: D102, override
        self, name: str, definition: math_schema.GlobalExpression
    ) -> None:
        equation_strings: list = []

        def _expression_setter(
            element: parsing.ParsedBackendEquation, where: xr.DataArray, references: set
        ) -> xr.DataArray:
            self._add_latex_strings(where, element, equation_strings, references)
            return where.where(where)

        parsed_component = self._add_component(
            name,
            definition,
            _expression_setter,
            "global_expressions",
            break_early=False,
        )
        expr_da = self.global_expressions[name]
        expr_da.attrs["math_repr"] = rf"\textbf{{{name}}}" + self._dims_to_var_string(
            expr_da
        )

        self._generate_math_string(
            parsed_component, expr_da, equations=equation_strings
        )

    def add_variable(  # noqa: D102, override
        self, name: str, definition: math_schema.Variable
    ) -> None:
        domain_dict = {"real": r"\mathbb{R}\;", "integer": r"\mathbb{Z}\;"}
        bound_refs: set = set()

        def _variable_setter(where: xr.DataArray, references: set) -> xr.DataArray:
            return where.where(where)

        parsed_component = self._add_component(
            name, definition, _variable_setter, "variables", break_early=False
        )
        var_da = self.variables[name]
        var_da.attrs["math_repr"] = rf"\textbf{{{name}}}" + self._dims_to_var_string(
            var_da
        )

        domain = domain_dict[definition.domain]
        lb, ub = self._get_variable_bounds_string(name, definition.bounds, bound_refs)
        self._update_references(name, bound_refs.difference(name))

        self._generate_math_string(
            parsed_component, var_da, equations=[lb, ub], sense=r"\in" + domain
        )

    def add_objective(  # noqa: D102, override
        self, name: str, definition: math_schema.Objective
    ) -> None:
        sense_dict = {
            "minimize": r"\min{}",
            "maximize": r"\max{}",
            "minimise": r"\min{}",
            "maximise": r"\max{}",
        }
        equation_strings: list = []

        def _objective_setter(
            element: parsing.ParsedBackendEquation, where: xr.DataArray, references: set
        ) -> None:
            self._add_latex_strings(where, element, equation_strings, references)

        parsed_component = self._add_component(
            name, definition, _objective_setter, "objectives", break_early=False
        )

        self._generate_math_string(
            parsed_component,
            self.objectives[name],
            equations=equation_strings,
            sense=sense_dict[definition.sense],
        )

    def set_objective(self, name: str):  # noqa: D102, override
        self.objective = name
        self.log("objectives", name, "Objective activated.", level="info")

    def _create_obj_list(  # noqa: D102, override
        self, key: str, component_type: backend_model.ALL_COMPONENTS_T
    ) -> None:
        return None

    def delete_component(  # noqa: D102, override
        self, key: str, component_type: backend_model.ALL_COMPONENTS_T
    ) -> None:
        if key in self._dataset and self._dataset[key].obj_type == component_type:
            del self._dataset[key]

    def generate_math_doc(
        self, format: ALLOWED_MATH_FILE_FORMATS = "tex", mkdocs_features: bool = False
    ) -> str:
        """Generate the math documentation by embedding LaTeX math in a template.

        Args:
            format (Literal["tex", "rst", "md"]):
                The built LaTeX math will be embedded in a document of the given format (tex=LaTeX, rst=reStructuredText, md=Markdown). Defaults to "tex".
            mkdocs_features (bool, optional):
                If True and format is `md`, then:
                - the equations will be on a tab and the original YAML math definition will be on another tab;
                - the equation cross-references will be given in a drop-down list.

        Returns:
            str: Generated math documentation.
        """
        if mkdocs_features and format != "md":
            raise ModelError(
                "Cannot use MKDocs features when writing math to a non-Markdown file format."
            )

        doc_template = self.FORMAT_STRINGS[format]
        uses = {
            name: set(
                other
                for other, da_other in self._dataset.data_vars.items()
                if name in da_other.attrs.get("references", set())
            )
            for name in self._dataset.data_vars
        }
        components = {
            objtype: [
                {
                    "expression": da.attrs.get("math_string", ""),
                    "name": name,
                    "description": da.attrs.get("description", ""),
                    "used_in": sorted(
                        list(da.attrs.get("references", set()) - set([name]))
                    ),
                    "uses": sorted(list(uses[name] - set([name]))),
                    "default": da.attrs.get("default", None),
                    "dtype": da.attrs.get("dtype", None),
                    "unit": da.attrs.get("unit", ""),
                    "yaml_snippet": to_yaml(
                        self.math[objtype][name].model_dump(exclude_defaults=True)
                    ),
                }
                for name, da in sorted(getattr(self, objtype).data_vars.items())
                if ("math_string" in da.attrs)
                or (objtype in ["parameters", "lookups"] and da.attrs["references"])
            ]
            for objtype in [
                "objectives",
                "constraints",
                "piecewise_constraints",
                "global_expressions",
                "variables",
                "parameters",
                "lookups",
            ]
            if getattr(self, objtype).data_vars
        }
        if "parameters" in components and not components["parameters"]:
            del components["parameters"]
        if "lookups" in components and not components["lookups"]:
            del components["lookups"]
        for objective in components.get("objectives", []):
            if objective["name"] == self.objective:
                objective["name"] += " (active)"
            else:
                objective["name"] += " (inactive)"

        return self._render(
            doc_template, mkdocs_features=mkdocs_features, components=components
        )

    def _add_latex_strings(
        self,
        where: xr.DataArray,
        element: parsing.ParsedBackendEquation,
        equation_strings: list,
        references: set,
    ):
        expr = element.evaluate_expression(
            self, return_type="math_string", references=references
        )

        where_latex = element.evaluate_where(
            self, return_type="math_string", references=references
        )

        if self.include == "all" or (self.include == "valid" and where.any()):
            equation_strings.append({"expression": expr, "where": where_latex})

    def _generate_math_string(
        self,
        parsed_component: parsing.ParsedBackendComponent | None,
        where_array: xr.DataArray,
        equations: list[dict[str, str]] | None = None,
        sense: str | None = None,
        where: str | None = None,
        sets: list[str] | None = None,
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

    def _render(self, template: str, **kwargs) -> str:
        text_starter = r"\\text(?:bf|it)?"  # match one of `\text`, `\textit`, `\textbf`

        def __escape_underscore(instring):
            r"""KaTeX requires underscores in `\text{...}` blocks to be escaped."""
            return re.sub(
                rf"{text_starter}{{.*?}}",
                lambda x: x.group(0).replace("_", r"\_"),
                instring,
            )

        def __mathify_text_in_text(instring):
            r"""KaTeX requires `\text{...}` blocks within `\text{...}` blocks to be placed within math blocks.

            We use `\\(` as the math block descriptor.
            """
            return re.sub(
                rf"{text_starter}{{(?:[^{{}}]*({text_starter}{{.*?}})[^{{}}]*?)?}}",
                lambda x: x.group(0).replace(x.group(1), rf"\({x.group(1)}\)"),
                instring,
            )

        def __iterator(instring):
            """Get the iterator name for a given dimension name."""
            return self.math.dimensions[instring].iterator

        jinja_env = jinja2.Environment(trim_blocks=True, autoescape=False)
        jinja_env.filters["iterator"] = __iterator
        jinja_env.filters["escape_underscores"] = __escape_underscore
        jinja_env.filters["mathify_text_in_text"] = __mathify_text_in_text
        return jinja_env.from_string(template).render(**kwargs)

    def _get_variable_bounds_string(
        self, name: str, bounds: math_schema.Bounds, references: set
    ) -> tuple[dict[str, str], ...]:
        """Convert variable upper and lower bounds into math string expressions."""
        bound_dict = math_schema.Constraint.model_validate(
            {
                "equations": [
                    {"expression": f"{bounds.min} <= {name}"},
                    {"expression": f"{name} <= {bounds.max}"},
                ]
            }
        )
        parsed_bounds = parsing.ParsedBackendComponent("constraints", name, bound_dict)
        equations = parsed_bounds.parse_equations(self.valid_component_names)
        return tuple(
            {
                "expression": eq.evaluate_expression(
                    self, return_type="math_string", references=references
                )
            }
            for eq in equations
        )

    def _dims_to_var_string(self, da: xr.DataArray) -> str:
        if da.shape:
            iterators = ",".join(self.math.dimensions[dim].iterator for dim in da.dims)
            return rf"_\text{{{iterators}}}"
        else:
            return ""
