"""LaTeX backend functionality."""

from __future__ import annotations

import logging
import re
import textwrap
from collections import defaultdict
from collections.abc import Callable
from typing import Literal

import jinja2
import numpy as np
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
    {% elif component_type == "postprocessed" %}

    Postprocessed Statistics
    -------------------------
    {% elif component_type == "parameters" %}

    Parameters
    ----------
    {% elif component_type == "lookups" %}

    Lookup Arrays
    -------------
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
    {% if 'unit' in equation and equation.unit != "" %}

    **Unit**: {{ equation.unit }}
    {% endif %}
    {% if 'default' in equation and equation.default is not none %}

    **Default**: {{ equation.default }}
    {% endif %}
    {% if 'dtype' in equation and equation.dtype is not none %}

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
    {% elif component_type == "postprocessed" %}
    \section{Postprocessed Statistics}
    {% elif component_type == "parameters" %}
    \section{Parameters}
    {% elif component_type == "lookups" %}
    \section{Lookup arrays}
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
    {% if 'default' in equation and equation.default is not none %}

    \textbf{Default}: {{ equation.default }}
    {% endif %}
    {% if 'dtype' in equation and equation.dtype is not none %}

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
    {% elif component_type == "postprocessed" %}

    ## Postprocessed Statistics
    {% elif component_type == "parameters" %}

    ## Parameters
    {% elif component_type == "lookups" %}

    ## Lookup Arrays
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
    {% if 'unit' in equation and equation.unit != "" %}

    **Unit**: {{ equation.unit }}
    {% endif %}
    {% if 'default' in equation and equation.default is not none %}

    **Default**: {{ equation.default }}
    {% endif %}
    {% if 'dtype' in equation and equation.dtype is not none %}

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

    OBJECTIVE_SENSE_DICT = {
        "minimize": r"\min{}",
        "maximize": r"\max{}",
        "minimise": r"\min{}",
        "maximise": r"\max{}",
    }
    VARIABLE_DOMAIN_DICT = {"real": r"\mathbb{R}\;", "integer": r"\mathbb{Z}\;"}

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
        self.math_strings: dict[str, dict] = defaultdict(lambda: defaultdict(str))

    def add_parameter(  # noqa: D102, override
        self, name: str, values: xr.DataArray, definition: math_schema.Parameter
    ) -> None:
        attrs = {"math_repr": rf"\textit{{{name}}}" + self._dims_to_var_string(values)}

        self._add_to_dataset(
            name, values, "parameters", definition.model_dump(), extra_attrs=attrs
        )

        if name not in self.math["parameters"]:
            self.math = self.math.update(
                {f"parameters.{name}": definition.model_dump()}
            )

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
        super().add_constraint(name, definition)
        self._generate_math_string("constraints", name, self.constraints[name])

    def add_piecewise_constraint(  # noqa: D102, override
        self, name: str, definition: math_schema.PiecewiseConstraint
    ) -> None:
        if not definition.active:
            self.log("piecewise_constraints", name, "Component deactivated.")
            return

        references: set[str] = set()
        constraint_def_with_breakpoints = definition.update(
            {"foreach": definition.foreach + ["breakpoints"]}
        )

        parsed_component = parsing.ParsedBackendComponent(
            "piecewise_constraints",
            name,
            constraint_def_with_breakpoints,
            self.math.parsing_components,
        )
        component_da = self._eval_top_level_where(
            self._dataset, references, parsed_component
        )

        math_parts = {}
        for val in ["x_expression", "y_expression", "x_values", "y_values"]:
            val_name = definition[val]
            dummy_expression_dict = {"equations": [{"expression": val_name}]}
            parsed_val = parsing.ParsedBackendComponent(
                "piecewise_constraints",
                name,
                math_schema.GlobalExpression.model_validate(dummy_expression_dict),
                self.math.parsing_components,
            )
            eq = parsed_val.parse_equations()
            math_parts[val] = eq[0].evaluate_expression(
                self.inputs,
                self._dataset,
                self.math,
                return_type="math_string",
                references=references,
            )

        equation = {
            "expression": rf"{math_parts['y_expression']}\mathord{{=}}{math_parts['y_values']}",
            "where": rf"{math_parts['x_expression']}\mathord{{=}}{math_parts['x_values']}",
        }
        self._add_to_dataset(
            name,
            component_da,
            "piecewise_constraints",
            definition.model_dump(),
            references=references,
        )
        self._generate_math_string(
            "piecewise_constraints",
            name,
            self.piecewise_constraints[name].assign_attrs(
                {"equation_strings": [equation]}
            ),
        )

    def add_global_expression(  # noqa: D102, override
        self, name: str, definition: math_schema.GlobalExpression
    ) -> None:
        super().add_global_expression(name, definition)

        expr_da = self.global_expressions[name]
        expr_da.attrs["math_repr"] = rf"\textbf{{{name}}}" + self._dims_to_var_string(
            expr_da
        )
        self._generate_math_string("global_expressions", name, expr_da)

    def add_variable(  # noqa: D102, override
        self, name: str, definition: math_schema.Variable
    ) -> None:
        super().add_variable(name, definition)

        var_da = self.variables[name]
        var_da.attrs["math_repr"] = rf"\textbf{{{name}}}" + self._dims_to_var_string(
            var_da
        )
        bound_refs: set = set()

        domain = self.VARIABLE_DOMAIN_DICT[definition.domain]
        lb, ub = self._get_variable_bounds_string(name, definition.bounds, bound_refs)
        self._update_references(name, bound_refs.difference(name))

        self._generate_math_string(
            "variables",
            name,
            var_da.assign_attrs(
                {"equation_strings": [lb, ub], "sense_string": r"\in" + domain}
            ),
        )

    def add_objective(  # noqa: D102, override
        self, name: str, definition: math_schema.Objective
    ) -> None:
        super().add_objective(name, definition)

        self._generate_math_string(
            "objectives",
            name,
            self.objectives[name].assign_attrs(
                {"sense_string": self.OBJECTIVE_SENSE_DICT[definition.sense]}
            ),
        )

    def _add_postprocessed(  # noqa: D102, override
        self,
        name: str,
        definition: math_schema.PostprocessedExpression,
        dataset: xr.Dataset,
    ) -> xr.DataArray:
        expr_da = super()._add_postprocessed(name, definition, dataset)

        self._add_to_dataset(
            name,
            expr_da,
            "postprocessed",
            definition.model_dump(),
            references=expr_da.attrs["references"],
        )
        expr_da.attrs["math_repr"] = rf"\textbf{{{name}}}" + self._dims_to_var_string(
            expr_da
        )
        self._generate_math_string("postprocessed", name, expr_da)
        return expr_da

    @property
    def postprocessed(self):
        """Slice of backend dataset to show only built postprocessed expressions."""
        return self._dataset.filter_by_attrs(obj_type="postprocessed")

    def set_objective(self, name: str):  # noqa: D102, override
        self.objective = name
        self.log("objectives", name, "Objective activated.", level="info")

    def _add_component_passthrough(
        self, name: str, where: xr.DataArray, *args, **kwargs
    ):
        """Generic passthrough for _add_* methods that just return where.where(where)."""
        return where.where(where)

    _add_variable = _add_global_expression = _add_constraint = _add_objective = (
        _add_component_passthrough
    )

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
                    "expression": math_string,
                    "name": name,
                    "used_in": sorted(
                        list(da.attrs.get("references", set()) - set([name]))
                    ),
                    "uses": sorted(list(uses[name] - set([name]))),
                    "yaml_snippet": to_yaml(
                        self.math[objtype][name].model_dump(exclude_defaults=True)
                    ),
                    **self.math[objtype][name].model_dump(),
                }
                for name, da in sorted(getattr(self, objtype).data_vars.items())
                if (math_string := self.math_strings[objtype][name]) != ""
                or (objtype in ["parameters", "lookups"] and da.attrs["references"])
            ]
            for objtype in [
                "objectives",
                "constraints",
                "piecewise_constraints",
                "global_expressions",
                "variables",
                "postprocessed",
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

    def _eval_top_level_where(
        self,
        dataset: xr.Dataset,
        references: set[str],
        parsed_component: parsing.ParsedBackendComponent,
    ) -> xr.DataArray:
        top_level_where = parsed_component.generate_top_level_where_array(
            self.inputs,
            dataset,
            self.math,
            self.config,
            align_to_foreach_sets=True,
            break_early=False,
            references=references,
        )
        top_level_where_latex = parsed_component.evaluate_where(
            self.inputs, dataset, self.math, self.config, return_type="math_string"
        )
        top_level_where.attrs["where_string"] = top_level_where_latex
        return top_level_where

    def _eval_equations(
        self,
        name: str,
        parsed_component: parsing.ParsedBackendComponent,
        dataset: xr.Dataset,
        top_level_where: xr.DataArray,
        component_setter: Callable,
        references: set[str],
    ):
        component_da = (
            xr.DataArray(attrs=top_level_where.attrs)
            .where(parsed_component.drop_dims_not_in_foreach(top_level_where))
            .astype(np.dtype("O"))
        )
        equation_strings = []
        equations = parsed_component.parse_equations()
        for equation in equations:
            expr = equation.evaluate_expression(
                self.inputs,
                dataset,
                self.math,
                return_type="math_string",
                references=references,
            )

            where = equation.evaluate_where(
                self.inputs,
                dataset,
                self.math,
                self.config,
                initial_where=top_level_where,
                references=references,
            )
            where_latex = equation.evaluate_where(
                self.inputs,
                dataset,
                self.math,
                self.config,
                return_type="math_string",
                references=references,
            )
            if self.include == "all" or (self.include == "valid" and where.any()):
                equation_strings.append({"expression": expr, "where": where_latex})
                component_da = component_da.fillna(where.where(where))

        component_da.attrs["equation_strings"] = equation_strings
        return component_da

    def _generate_math_string(self, group: str, name: str, da: xr.DataArray) -> None:
        """If the component meets the conditions to be included, create a math string for it.

        Args:
            group (str): The component group (e.g., "constraints", "variables", etc.) the DataArray belongs to.
            name (str): The name of the component.
            da (xr.DataArray): The array to evaluate, whose attributes contain the component parts to build the string.
        """
        if self.include == "all" or (
            self.include == "valid" and da.fillna(0).astype(bool).any()
        ):
            where = da.attrs.pop("where_string", None)
            equations = da.attrs.pop("equation_strings", [])
            sense = da.attrs.pop("sense_string", None)
            equation_element_string = self._render(
                self.LATEX_EQUATION_ELEMENT,
                equations=equations,
                sense=sense,
                where=where,
                sets=list(da.dims),
            )
        else:
            equation_element_string = ""
        self.math_strings[group][name] = equation_element_string

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
        parsed_bounds = parsing.ParsedBackendComponent(
            "constraints", name, bound_dict, self.math.parsing_components
        )
        equations = parsed_bounds.parse_equations()
        return tuple(
            {
                "expression": eq.evaluate_expression(
                    self.inputs,
                    self._dataset,
                    self.math,
                    return_type="math_string",
                    references=references,
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
