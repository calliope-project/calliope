import pytest
import textwrap

import xarray as xr
from calliope.backend import latex_backend
from calliope import exceptions
from calliope.test.common.util import check_error_or_warning


class TestLatexBackendModel:
    @pytest.fixture(scope="class")
    def valid_latex_backend(self):
        return latex_backend.LatexBackendModel(include="valid")

    @pytest.mark.parametrize(
        "backend_obj", ["valid_latex_backend", "dummy_latex_backend_model"]
    )
    def test_add_parameter(self, request, backend_obj):
        latex_backend_model = request.getfixturevalue(backend_obj)
        latex_backend_model.add_parameter("param", xr.DataArray(1))
        assert latex_backend_model.parameters["param"] == xr.DataArray(1)
        assert "param" in latex_backend_model.valid_arithmetic_components

    @pytest.mark.parametrize(
        "backend_obj", ["valid_latex_backend", "dummy_latex_backend_model"]
    )
    def test_add_variable(self, request, dummy_model_data, backend_obj):
        latex_backend_model = request.getfixturevalue(backend_obj)
        latex_backend_model.add_variable(
            dummy_model_data,
            "var",
            {
                "foreach": ["nodes", "techs"],
                "where": "with_inf",
                "bounds": {"min": 0, "max": 1},
            },
        )
        # some null values might be introduced by the foreach array, so we just check the upper bound
        assert (
            latex_backend_model.variables["var"].sum()
            <= dummy_model_data.with_inf_as_bool.sum()
        )
        assert "var" in latex_backend_model.valid_arithmetic_components
        assert latex_backend_model._instance["variables"][-1]["name"] == "var"

    def test_add_variable_not_valid(self, dummy_model_data, valid_latex_backend):
        valid_latex_backend.add_variable(
            dummy_model_data,
            "invalid_var",
            {
                "foreach": ["nodes", "techs"],
                "where": "False",
                "bounds": {"min": 0, "max": 1},
            },
        )
        # some null values might be introduced by the foreach array, so we just check the upper bound
        assert not valid_latex_backend.variables["invalid_var"].sum()
        assert "invalid_var" in valid_latex_backend.valid_arithmetic_components
        assert valid_latex_backend._instance["variables"][-1]["name"] != "invalid_var"

    @pytest.mark.parametrize(
        "backend_obj", ["valid_latex_backend", "dummy_latex_backend_model"]
    )
    def test_add_expression(self, request, dummy_model_data, backend_obj):
        latex_backend_model = request.getfixturevalue(backend_obj)
        latex_backend_model.add_expression(
            dummy_model_data,
            "expr",
            {
                "foreach": ["nodes", "techs"],
                "where": "with_inf",
                "equation": "var + param",
            },
        )
        # some null values might be introduced by the foreach array, so we just check the upper bound
        assert (
            latex_backend_model.expressions["expr"].sum()
            <= dummy_model_data.with_inf_as_bool.sum()
        )
        assert "expr" in latex_backend_model.valid_arithmetic_components
        assert latex_backend_model._instance["expressions"][-1]["name"] == "expr"

    @pytest.mark.parametrize(
        "backend_obj", ["valid_latex_backend", "dummy_latex_backend_model"]
    )
    def test_add_constraint(self, request, dummy_model_data, backend_obj):
        latex_backend_model = request.getfixturevalue(backend_obj)
        latex_backend_model.add_constraint(
            dummy_model_data,
            "constr",
            {
                "foreach": ["nodes", "techs"],
                "where": "with_inf",
                "equation": "var >= param",
            },
        )
        # some null values might be introduced by the foreach array, so we just check the upper bound
        assert (
            latex_backend_model.constraints["constr"].sum()
            <= dummy_model_data.with_inf_as_bool.sum()
        )
        assert "constr" not in latex_backend_model.valid_arithmetic_components
        assert latex_backend_model._instance["constraints"][-1]["name"] == "constr"

    def test_add_constraint_not_valid(self, dummy_model_data, valid_latex_backend):
        valid_latex_backend.add_constraint(
            dummy_model_data,
            "invalid_constr",
            {
                "foreach": ["nodes", "techs"],
                "where": "False",
                "equations": [
                    {"expression": "var >= param"},
                    {"expression": "var >= expr"},
                ],
            },
        )
        assert not valid_latex_backend.constraints["invalid_constr"].any()
        assert (
            valid_latex_backend._instance["constraints"][-1]["name"] != "invalid_constr"
        )

    def test_add_constraint_one_not_valid(self, dummy_model_data, valid_latex_backend):
        valid_latex_backend.add_constraint(
            dummy_model_data,
            "valid_constr",
            {
                "foreach": ["nodes", "techs"],
                "where": "with_inf",
                "equations": [
                    {"expression": "var >= param"},
                    {"expression": "var <= expr", "where": "False"},
                ],
            },
        )
        assert (
            valid_latex_backend._instance["constraints"][-1]["name"] == "valid_constr"
        )
        assert (
            "expr" not in valid_latex_backend._instance["constraints"][-1]["expression"]
        )

    def test_add_objective(self, dummy_model_data, dummy_latex_backend_model):
        dummy_latex_backend_model.add_objective(
            dummy_model_data,
            "obj",
            {"equation": "sum(var, over=[nodes, techs])", "sense": "minimize"},
        )
        assert dummy_latex_backend_model.objectives["obj"].isnull().all()
        assert "obj" not in dummy_latex_backend_model.valid_arithmetic_components
        assert len(dummy_latex_backend_model._instance["objectives"]) == 1

    def test_get_parameter(self, dummy_latex_backend_model):
        param = dummy_latex_backend_model.get_parameter("param")
        assert param.equals(xr.DataArray(1))

    def test_create_obj_list(self, dummy_latex_backend_model):
        assert dummy_latex_backend_model.create_obj_list("var", "variables") is None

    def test_get_constraint(self, dummy_latex_backend_model):
        constr = dummy_latex_backend_model.get_constraint("constr")
        assert constr.equals(dummy_latex_backend_model.constraints["constr"])

    def test_get_variable(self, dummy_latex_backend_model):
        var = dummy_latex_backend_model.get_variable("var")
        assert var.equals(dummy_latex_backend_model.variables["var"])

    def test_get_expression(self, dummy_latex_backend_model):
        expr = dummy_latex_backend_model.get_expression("expr")
        assert expr.equals(dummy_latex_backend_model.expressions["expr"])

    def test_solve(self, dummy_latex_backend_model):
        with pytest.raises(exceptions.BackendError) as excinfo:
            dummy_latex_backend_model.solve("cbc")
        assert check_error_or_warning(excinfo, "Cannot solve a LaTex backend model")

    @pytest.mark.parametrize(
        ["format", "expected"],
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
                    \section{Objective}
                    \section{Subject to}
                    \section{Where}

                    \paragraph{ expr }
                    \begin{equation}
                    \resizebox{\ifdim\width>\linewidth0.95\linewidth\else\width\fi}{!}{$
                    \begin{array}{r}
                    \end{array}
                    \begin{cases}
                        1 + 2&\quad
                        \\
                    \end{cases}
                    $}
                    \end{equation}
                    \section{Decision Variables}
                    \end{document}"""
                ),
            ),
            (
                "rst",
                textwrap.dedent(
                    r"""
                    Objective
                    #########

                    Subject to
                    ##########

                    Where
                    #####

                    expr
                    ====

                    .. container:: scrolling-wrapper

                        .. math::
                            \begin{array}{r}
                            \end{array}
                            \begin{cases}
                                1 + 2&\quad
                                \\
                            \end{cases}

                    Decision Variables
                    ##################
                    """
                ),
            ),
            (
                "md",
                textwrap.dedent(
                    r"""
                    # Objective

                    # Subject to

                    # Where

                    ## expr
                        ```math
                        \begin{array}{r}
                        \end{array}
                        \begin{cases}
                            1 + 2&\quad
                            \\
                        \end{cases}
                        ```

                    # Decision Variables
                    """
                ),
            ),
        ],
    )
    def test_generate_math_doc(self, dummy_model_data, format, expected):
        latex_backend_model = latex_backend.LatexBackendModel(format=format)
        latex_backend_model.add_expression(
            dummy_model_data, "expr", {"equation": "1 + 2"}
        )
        doc = latex_backend_model.generate_math_doc()
        assert doc == expected

    @pytest.mark.parametrize(
        ["kwargs", "expected"],
        [
            (
                {"sets": ["nodes", "techs"]},
                textwrap.dedent(
                    r"""
                \begin{array}{r}
                    \forall{}
                    \text{ node }\negthickspace \in \negthickspace\text{ nodes, }
                    \text{ tech }\negthickspace \in \negthickspace\text{ techs }
                    \\
                \end{array}
                \begin{cases}
                \end{cases}"""
                ),
            ),
            (
                {"sense": r"\min{}"},
                textwrap.dedent(
                    r"""
                \begin{array}{r}
                    \min{}
                \end{array}
                \begin{cases}
                \end{cases}"""
                ),
            ),
            (
                {"where": r"foo \land bar"},
                textwrap.dedent(
                    r"""
                \begin{array}{r}
                    \text{if } foo \land bar
                \end{array}
                \begin{cases}
                \end{cases}"""
                ),
            ),
            (
                {"where": r""},
                textwrap.dedent(
                    r"""
                \begin{array}{r}
                \end{array}
                \begin{cases}
                \end{cases}"""
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
                \begin{array}{r}
                \end{array}
                \begin{cases}
                    foo&\quad
                    \text{if } bar
                    \\
                    foo + 1&\quad
                    \\
                \end{cases}"""
                ),
            ),
        ],
    )
    def test_generate_math_string(self, dummy_latex_backend_model, kwargs, expected):
        da = xr.DataArray()
        dummy_latex_backend_model._generate_math_string(
            "foo", da, "constraints", **kwargs
        )
        assert da.math_string == expected
        assert dummy_latex_backend_model._instance["constraints"][-1] == {
            "expression": expected,
            "name": "foo",
        }

    @pytest.mark.parametrize(
        ["instring", "kwargs", "expected"],
        [
            ("{{ foo }}", {"foo": 1}, "1"),
            ("{{ foo|removesuffix('s') }}", {"foo": "bars"}, "bar"),
            ("{{ foo }} + {{ bar }}", {"foo": "1", "bar": "2"}, "1 + 2"),
        ],
    )
    def test_render(self, dummy_latex_backend_model, instring, kwargs, expected):
        rendered = dummy_latex_backend_model._render(instring, **kwargs)
        assert rendered == expected

    def test_get_capacity_bounds(self, dummy_latex_backend_model, dummy_model_data):
        bounds = {"min": 1, "max": 2e6}
        lb, ub = dummy_latex_backend_model._get_capacity_bounds(
            bounds, "var", dummy_model_data
        )
        assert lb == {"expression": r"1 \leq \textbf{var}_\text{node,tech}"}
        assert ub == {
            "expression": r"\textbf{var}_\text{node,tech} \leq 2\mathord{\times}10^{+06}"
        }
