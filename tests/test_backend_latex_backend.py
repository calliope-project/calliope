import logging
import textwrap

import pytest
import xarray as xr

from calliope import exceptions
from calliope.backend import latex_backend_model
from calliope.schemas import math_schema

from .common.util import check_error_or_warning


@pytest.fixture
def temp_dummy_latex_backend_model(dummy_model_data, dummy_model_math, default_config):
    """Function scoped model definition to avoid cross-test contamination."""
    return latex_backend_model.LatexBackendModel(
        dummy_model_data, dummy_model_math, default_config.build
    )


class TestLatexBackendModel:
    def test_inputs(self, dummy_latex_backend_model, dummy_model_data):
        assert dummy_latex_backend_model.inputs.equals(dummy_model_data)

    @pytest.mark.parametrize(
        "backend_obj", ["valid_latex_backend", "dummy_latex_backend_model"]
    )
    def test_add_parameter(self, request, backend_obj):
        latex_backend_model = request.getfixturevalue(backend_obj)
        latex_backend_model.add_parameter("param", xr.DataArray(1), {})
        assert latex_backend_model.parameters["param"] == xr.DataArray(1)
        assert "param" in latex_backend_model.valid_component_names

    @pytest.mark.parametrize(
        "backend_obj", ["valid_latex_backend", "dummy_latex_backend_model"]
    )
    def test_add_variable(self, request, dummy_model_data, backend_obj):
        latex_backend_model = request.getfixturevalue(backend_obj)
        latex_backend_model.add_variable(
            "var",
            {
                "foreach": ["nodes", "techs"],
                "where": "with_inf",
                "bounds": {"min": 0, "max": 1},
                "domain": "real",
            },
        )
        # some null values might be introduced by the foreach array, so we just check the upper bound
        assert (
            latex_backend_model.variables["var"].sum()
            <= dummy_model_data.with_inf_as_bool.sum()
        )
        assert "var" in latex_backend_model.valid_component_names
        assert "math_string" in latex_backend_model.variables["var"].attrs
        assert (
            latex_backend_model.variables["var"].attrs["math_repr"]
            == r"\textbf{var}_\text{node,tech}"
        )

    def test_add_variable_not_valid(self, valid_latex_backend):
        valid_latex_backend.add_variable(
            "invalid_var",
            {
                "foreach": ["nodes", "techs"],
                "where": "False",
                "bounds": {"min": 0, "max": 1},
                "domain": "real",
            },
        )
        # some null values might be introduced by the foreach array, so we just check the upper bound
        assert not valid_latex_backend.variables["invalid_var"].sum()
        assert "invalid_var" in valid_latex_backend.valid_component_names
        assert "math_string" not in valid_latex_backend.variables["invalid_var"].attrs

    @pytest.mark.parametrize(
        "backend_obj", ["valid_latex_backend", "dummy_latex_backend_model"]
    )
    def test_add_expression(self, request, dummy_model_data, backend_obj):
        latex_backend_model = request.getfixturevalue(backend_obj)
        latex_backend_model.add_global_expression(
            "expr",
            {
                "foreach": ["nodes", "techs"],
                "where": "multi_dim_var",
                "equations": [{"expression": "multi_dim_var + no_dims"}],
            },
        )
        # some null values might be introduced by the foreach array, so we just check the upper bound
        assert (
            latex_backend_model.global_expressions["expr"].sum()
            <= dummy_model_data.with_inf_as_bool.sum()
        )
        assert "expr" in latex_backend_model.valid_component_names
        assert "math_string" in latex_backend_model.global_expressions["expr"].attrs

    @pytest.mark.parametrize(
        "backend_obj", ["valid_latex_backend", "dummy_latex_backend_model"]
    )
    def test_add_expression_with_variable_in_where(
        self, request, dummy_model_data, backend_obj
    ):
        latex_backend_model = request.getfixturevalue(backend_obj)
        latex_backend_model.add_global_expression(
            "var_init_expr",
            {
                "foreach": ["nodes", "techs"],
                "where": "with_inf",
                "equations": [{"expression": "multi_dim_var + no_dims"}],
            },
        )
        # some null values might be introduced by the foreach array, so we just check the upper bound
        assert (
            latex_backend_model.global_expressions["var_init_expr"].sum()
            <= dummy_model_data.with_inf_as_bool.sum()
        )
        assert "var_init_expr" in latex_backend_model.valid_component_names
        assert (
            "math_string"
            in latex_backend_model.global_expressions["var_init_expr"].attrs
        )

    @pytest.mark.parametrize(
        "backend_obj", ["valid_latex_backend", "dummy_latex_backend_model"]
    )
    def test_add_constraint(self, request, dummy_model_data, backend_obj):
        latex_backend_model = request.getfixturevalue(backend_obj)
        latex_backend_model.add_constraint(
            "constr",
            {
                "foreach": ["nodes", "techs"],
                "where": "with_inf",
                "equations": [{"expression": "multi_dim_var >= no_dims"}],
                "active": True,
            },
        )
        # some null values might be introduced by the foreach array, so we just check the upper bound
        assert (
            latex_backend_model.constraints["constr"].sum()
            <= dummy_model_data.with_inf_as_bool.sum()
        )
        assert "constr" not in latex_backend_model.valid_component_names
        assert "math_string" in latex_backend_model.constraints["constr"].attrs

    @pytest.mark.parametrize(
        "backend_obj", ["valid_latex_backend", "dummy_latex_backend_model"]
    )
    def test_add_constraint_with_variable_and_expression_in_where(
        self, request, dummy_model_data, backend_obj
    ):
        latex_backend_model = request.getfixturevalue(backend_obj)
        latex_backend_model.add_constraint(
            "var_init_constr",
            {
                "foreach": ["nodes", "techs"],
                "where": "multi_dim_var and multi_dim_expr",
                "equations": [{"expression": "no_dim_var >= multi_dim_var"}],
            },
        )
        # some null values might be introduced by the foreach array, so we just check the upper bound
        assert (
            latex_backend_model.constraints["var_init_constr"].sum()
            <= dummy_model_data.with_inf_as_bool.sum()
        )
        assert "var_init_constr" not in latex_backend_model.valid_component_names
        assert "math_string" in latex_backend_model.constraints["var_init_constr"].attrs

    def test_add_constraint_not_valid(self, valid_latex_backend):
        valid_latex_backend.add_constraint(
            "invalid_constr",
            {
                "foreach": ["nodes", "techs"],
                "where": "False",
                "equations": [
                    {"expression": "multi_dim_var >= no_dims"},
                    {"expression": "multi_dim_var >= multi_dim_expr"},
                ],
            },
        )
        assert valid_latex_backend.constraints["invalid_constr"].isnull().all()
        assert (
            "math_string" not in valid_latex_backend.constraints["invalid_constr"].attrs
        )

    def test_add_constraint_one_not_valid(self, valid_latex_backend):
        valid_latex_backend.add_constraint(
            "valid_constr",
            {
                "foreach": ["nodes", "techs"],
                "where": "with_inf",
                "equations": [
                    {"expression": "multi_dim_var >= no_dims"},
                    {"expression": "multi_dim_var >= multi_dim_expr", "where": "False"},
                ],
            },
        )
        assert (
            "multi_dim_expr"
            not in valid_latex_backend.constraints["valid_constr"].attrs["math_string"]
        )

    def test_add_objective(self, dummy_latex_backend_model):
        dummy_latex_backend_model.add_objective(
            "obj",
            {
                "equations": [{"expression": "sum(no_dim_var, over=[nodes, techs])"}],
                "sense": "minimize",
            },
        )
        assert dummy_latex_backend_model.objectives["obj"].isnull().all()
        assert "obj" not in dummy_latex_backend_model.valid_component_names
        assert len(dummy_latex_backend_model.objectives.data_vars) == 1

    def test_default_objective_set(self, dummy_latex_backend_model):
        assert dummy_latex_backend_model.objective == "min_cost_optimisation"

    def test_new_objective_set(self, dummy_latex_backend_model):
        dummy_latex_backend_model.add_objective(
            "foo", {"equations": [{"expression": "no_dims"}], "sense": "minimise"}
        )
        dummy_latex_backend_model.set_objective("foo")
        assert dummy_latex_backend_model.objective == "foo"

    def test_new_objective_set_log(self, caplog, dummy_latex_backend_model):
        caplog.set_level(logging.INFO)
        dummy_latex_backend_model.set_objective("foo")
        assert ":foo | Objective activated." in caplog.text

    def test_add_piecewise_constraint(self, dummy_latex_backend_model):
        dummy_latex_backend_model.add_parameter(
            "piecewise_x",
            xr.DataArray(data=[0, 5, 10], coords={"breakpoints": [0, 1, 2]}),
            {},
        )
        dummy_latex_backend_model.add_parameter(
            "piecewise_y",
            xr.DataArray(data=[0, 1, 5], coords={"breakpoints": [0, 1, 2]}),
            {},
        )
        for param in ["piecewise_x", "piecewise_y"]:
            dummy_latex_backend_model.inputs[param] = (
                dummy_latex_backend_model._dataset[param]
            )
        dummy_latex_backend_model.add_piecewise_constraint(
            "p_constr",
            {
                "foreach": ["nodes", "techs"],
                "where": "piecewise_x AND piecewise_y",
                "x_values": "piecewise_x",
                "x_expression": "multi_dim_var + 1",
                "y_values": "piecewise_y",
                "y_expression": "no_dim_var",
                "description": "FOO",
            },
        )
        math_string = dummy_latex_backend_model.piecewise_constraints["p_constr"].attrs[
            "math_string"
        ]
        assert (
            r"\text{ breakpoint }\negthickspace \in \negthickspace\text{ breakpoints }"
            in math_string
        )
        assert (
            r"\text{if } \textbf{multi_dim_var}_\text{node,tech} + 1\mathord{=}\textit{piecewise_x}_\text{breakpoint}"
            in math_string
        )
        assert (
            r"\textbf{no_dim_var}\mathord{=}\textit{piecewise_y}_\text{breakpoint}"
            in math_string
        )

    def test_add_piecewise_constraint_no_foreach(self, dummy_latex_backend_model):
        dummy_latex_backend_model.add_parameter(
            "piecewise_x",
            xr.DataArray(data=[0, 5, 10], coords={"breakpoints": [0, 1, 2]}),
            {},
        )
        dummy_latex_backend_model.add_parameter(
            "piecewise_y",
            xr.DataArray(data=[0, 1, 5], coords={"breakpoints": [0, 1, 2]}),
            {},
        )
        for param in ["piecewise_x", "piecewise_y"]:
            dummy_latex_backend_model.inputs[param] = (
                dummy_latex_backend_model._dataset[param]
            )
        dummy_latex_backend_model.add_piecewise_constraint(
            "p_constr_no_foreach",
            {
                "where": "piecewise_x AND piecewise_y",
                "x_values": "piecewise_x",
                "x_expression": "sum(multi_dim_var, over=[nodes, techs])",
                "y_values": "piecewise_y",
                "y_expression": "no_dim_var",
                "description": "BAR",
            },
        )
        math_string = dummy_latex_backend_model.piecewise_constraints[
            "p_constr_no_foreach"
        ].attrs["math_string"]
        assert (
            r"\text{ breakpoint }\negthickspace \in \negthickspace\text{ breakpoints }"
            in math_string
        )

    def test_create_obj_list(self, dummy_latex_backend_model):
        assert dummy_latex_backend_model._create_obj_list("var", "variables") is None

    @pytest.mark.parametrize(
        ("format", "expected"),
        [
            (
                "tex",
                textwrap.dedent(
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
                    \section{Where}

                    \paragraph{ expr }

                    foobar

                    \textbf{Uses}:
                    \begin{itemize}
                        \item no_dims
                    \end{itemize}

                    \textbf{Default}: 0

                    \begin{equation}
                    \resizebox{\ifdim\width>\linewidth0.95\linewidth\else\width\fi}{!}{$
                    \begin{array}{l}
                        \quad \textit{no_dims} + 2\\
                    \end{array}
                    $}
                    \end{equation}
                    \section{Parameters}

                    \paragraph{ no_dims }

                    \textbf{Used in}:
                    \begin{itemize}
                        \item expr
                    \end{itemize}

                    \textbf{Default}: 0
                    \end{document}"""
                ),
            ),
            (
                "rst",
                textwrap.dedent(
                    r"""

                    Where
                    -----

                    expr
                    ^^^^

                    foobar

                    **Uses**:

                    * no_dims

                    **Default**: 0

                    .. container:: scrolling-wrapper

                        .. math::
                            \begin{array}{l}
                                \quad \textit{no_dims} + 2\\
                            \end{array}

                    Parameters
                    ----------

                    no_dims
                    ^^^^^^^

                    **Used in**:

                    * expr

                    **Default**: 0
                    """
                ),
            ),
            (
                "md",
                textwrap.dedent(
                    r"""

                    ## Where

                    ### expr

                    foobar

                    **Uses**:

                    * [no_dims](#no_dims)

                    **Default**: 0

                    $$
                    \begin{array}{l}
                        \quad \textit{no\_dims} + 2\\
                    \end{array}
                    $$

                    ## Parameters

                    ### no_dims

                    **Used in**:

                    * [expr](#expr)

                    **Default**: 0
                    """
                ),
            ),
        ],
    )
    def test_generate_math_doc(self, temp_dummy_latex_backend_model, format, expected):
        temp_dummy_latex_backend_model._load_inputs()
        temp_dummy_latex_backend_model.add_global_expression(
            "expr",
            {
                "equations": [{"expression": "no_dims + 2"}],
                "description": "foobar",
                "default": 0,
            },
        )
        doc = temp_dummy_latex_backend_model.generate_math_doc(format=format)
        assert doc == expected

    def test_generate_math_doc_no_params(self, temp_dummy_latex_backend_model):
        temp_dummy_latex_backend_model.add_global_expression(
            "expr",
            {
                "equations": [{"expression": "1 + 2"}],
                "description": "foobar",
                "default": 0,
            },
        )
        doc = temp_dummy_latex_backend_model.generate_math_doc(format="md")
        assert doc == textwrap.dedent(
            r"""

                    ## Where

                    ### expr

                    foobar

                    **Default**: 0

                    $$
                    \begin{array}{l}
                        \quad 1 + 2\\
                    \end{array}
                    $$
                    """
        )

    def test_generate_math_doc_mkdocs_features_tabs(
        self, temp_dummy_latex_backend_model
    ):
        temp_dummy_latex_backend_model.add_global_expression(
            "expr",
            {
                "equations": [{"expression": "1 + 2"}],
                "description": "foobar",
                "default": 0,
            },
        )
        doc = temp_dummy_latex_backend_model.generate_math_doc(
            format="md", mkdocs_features=True
        )
        assert doc == textwrap.dedent(
            r"""

                    ## Where

                    ### expr

                    foobar

                    **Default**: 0

                    === "Math"

                        $$
                        \begin{array}{l}
                            \quad 1 + 2\\
                        \end{array}
                        $$

                    === "YAML"

                        ```yaml
                        description: foobar
                        equations:
                        - expression: 1 + 2
                        default: 0
                        ```
                    """
        )

    def test_generate_math_doc_mkdocs_features_admonition(
        self, temp_dummy_latex_backend_model
    ):
        temp_dummy_latex_backend_model._load_inputs()
        temp_dummy_latex_backend_model.add_global_expression(
            "expr",
            {
                "equations": [{"expression": "no_dims + 1"}],
                "description": "foobar",
                "default": 0,
            },
        )
        doc = temp_dummy_latex_backend_model.generate_math_doc(
            format="md", mkdocs_features=True
        )
        assert doc == textwrap.dedent(
            r"""

                    ## Where

                    ### expr

                    foobar

                    ??? info "Uses"

                        * [no_dims](#no_dims)

                    **Default**: 0

                    === "Math"

                        $$
                        \begin{array}{l}
                            \quad \textit{no\_dims} + 1\\
                        \end{array}
                        $$

                    === "YAML"

                        ```yaml
                        description: foobar
                        equations:
                        - expression: no_dims + 1
                        default: 0
                        ```

                    ## Parameters

                    ### no_dims

                    ??? info "Used in"

                        * [expr](#expr)

                    **Default**: 0
                    """
        )

    def test_generate_math_doc_mkdocs_features_not_in_md(
        self, temp_dummy_latex_backend_model
    ):
        with pytest.raises(exceptions.ModelError) as excinfo:
            temp_dummy_latex_backend_model.generate_math_doc(
                format="rst", mkdocs_features=True
            )

        assert check_error_or_warning(
            excinfo,
            "Cannot use MKDocs features when writing math to a non-Markdown file format.",
        )

    @pytest.mark.parametrize(
        ("kwargs", "expected"),
        [
            (
                {"sets": ["nodes", "techs"]},
                textwrap.dedent(
                    r"""
                \begin{array}{l}
                    \forall{}
                    \text{ node }\negthickspace \in \negthickspace\text{ nodes, }
                    \text{ tech }\negthickspace \in \negthickspace\text{ techs }
                    \!\!:\\[2em]
                \end{array}"""
                ),
            ),
            (
                {"sense": r"\min{}"},
                textwrap.dedent(
                    r"""
                \begin{array}{l}
                    \min{}\!\!:\\[2em]
                \end{array}"""
                ),
            ),
            (
                {"where": r"foo \land bar"},
                textwrap.dedent(
                    r"""
                \begin{array}{l}
                    \text{if } foo \land bar\!\!:\\[2em]
                \end{array}"""
                ),
            ),
            (
                {"where": r""},
                textwrap.dedent(
                    r"""
                \begin{array}{l}
                \end{array}"""
                ),
            ),
            (
                {
                    "equations": [
                        {"expression": "foo", "where": "bar"},
                        {"expression": "foo + 1", "where": ""},
                    ]
                },
                textwrap.dedent(
                    r"""
                \begin{array}{l}
                    \quad \text{if } bar\!\!:\\
                    \qquad foo\\[2em]
                    \quad foo + 1\\
                \end{array}"""
                ),
            ),
        ],
    )
    def test_generate_math_string(self, dummy_latex_backend_model, kwargs, expected):
        da = xr.DataArray()
        dummy_latex_backend_model._generate_math_string(None, da, **kwargs)
        assert da.math_string == expected

    @pytest.mark.parametrize(
        ("instring", "kwargs", "expected"),
        [
            ("{{ foo }}", {"foo": 1}, "1"),
            ("{{ foo|iterator }}", {"foo": "techs"}, "tech"),
            ("{{ foo }} + {{ bar }}", {"foo": "1", "bar": "2"}, "1 + 2"),
            (
                "{{ foo|escape_underscores }}",
                {"foo": r"\text{foo_bar}_foo_{bar}"},
                r"\text{foo\_bar}_foo_{bar}",
            ),
            (
                "{{ foo|escape_underscores }}",
                {"foo": r"\textit{foo_bar}"},
                r"\textit{foo\_bar}",
            ),
            (
                "{{ foo|escape_underscores }}",
                {"foo": r"\textbf{foo_bar}"},
                r"\textbf{foo\_bar}",
            ),
            (
                "{{ foo|mathify_text_in_text }}",
                {"foo": r"\text{foo_bar} + \text{foo,\text{foo}_\textit{bar},bar}"},
                r"\text{foo_bar} + \text{foo,\(\text{foo}_\textit{bar}\),bar}",
            ),
        ],
    )
    def test_render(self, dummy_latex_backend_model, instring, kwargs, expected):
        rendered = dummy_latex_backend_model._render(instring, **kwargs)
        assert rendered == expected

    def test_get_variable_bounds_string(self, dummy_latex_backend_model):
        bounds = math_schema.Bounds.model_validate({"min": 1, "max": 2e6})
        refs = set()
        lb, ub = dummy_latex_backend_model._get_variable_bounds_string(
            "multi_dim_var", bounds, refs
        )
        assert lb == {"expression": r"1 \leq \textbf{multi_dim_var}_\text{node,tech}"}
        assert ub == {
            "expression": r"\textbf{multi_dim_var}_\text{node,tech} \leq 2\mathord{\times}10^{+06}"
        }
        assert refs == {"multi_dim_var"}

    def test_param_type(self, temp_dummy_latex_backend_model):
        temp_dummy_latex_backend_model._load_inputs()
        temp_dummy_latex_backend_model.add_global_expression(
            "expr",
            {
                "equations": [{"expression": "1 + with_inf"}],
                "description": "foobar",
                "default": 0,
            },
        )
        doc = temp_dummy_latex_backend_model.generate_math_doc(format="md")
        assert doc == textwrap.dedent(
            r"""

            ## Where

            ### expr

            foobar

            **Uses**:

            * [with_inf](#with_inf)

            **Default**: 0

            $$
            \begin{array}{l}
                \quad 1 + \textit{with\_inf}_\text{node,tech}\\
            \end{array}
            $$

            ## Parameters

            ### with_inf

            With infinity values.

            **Used in**:

            * [expr](#expr)

            **Unit**: power

            **Default**: 100
            """
        )
