import textwrap
from pathlib import Path

import pytest
import xarray as xr

from calliope import exceptions
from calliope.backend import latex_backend_model
from calliope.test.common.util import build_test_model, check_error_or_warning


class TestMathDocumentation:
    @pytest.fixture(scope="class")
    def no_build(self):
        return build_test_model({}, "simple_supply,two_hours,investment_costs")

    @pytest.fixture(scope="class")
    def build_all(self):
        model = build_test_model({}, "simple_supply,two_hours,investment_costs")
        model.math_documentation.build(include="all")
        return model

    @pytest.fixture(scope="class")
    def build_valid(self):
        model = build_test_model({}, "simple_supply,two_hours,investment_costs")
        model.math_documentation.build(include="valid")
        return model

    def test_write_before_build(self, no_build, tmpdir_factory):
        filepath = tmpdir_factory.mktemp("custom_math").join("foo.tex")
        with pytest.raises(exceptions.ModelError) as excinfo:
            no_build.math_documentation.write(filepath)
        check_error_or_warning(
            excinfo, "Build the documentation (`build`) before trying to write it"
        )

    @pytest.mark.parametrize(
        ["format", "startswith"],
        [
            ("tex", "\n\\documentclass{article}"),
            ("rst", "\nObjective"),
            ("md", "\n# Objective"),
        ],
    )
    @pytest.mark.parametrize("include", ["build_all", "build_valid"])
    def test_string_return(self, request, format, startswith, include):
        model = request.getfixturevalue(include)
        string_math = model.math_documentation.write(format=format)
        assert string_math.startswith(startswith)

    def test_to_file(self, build_all, tmpdir_factory):
        filepath = tmpdir_factory.mktemp("custom_math").join("custom-math.tex")
        build_all.math_documentation.write(filename=filepath)
        assert Path(filepath).exists()

    @pytest.mark.parametrize(
        ["filepath", "format"],
        [(None, "foo"), ("myfile.foo", None), ("myfile.tex", "foo")],
    )
    def test_invalid_format(self, build_all, tmpdir_factory, filepath, format):
        if filepath is not None:
            filepath = tmpdir_factory.mktemp("custom_math").join(filepath)
        with pytest.raises(ValueError) as excinfo:
            build_all.math_documentation.write(filename="foo", format=format)
        check_error_or_warning(excinfo, "Math documentation style must be one of")


class TestLatexBackendModel:
    @pytest.fixture(scope="class")
    def valid_latex_backend(self, dummy_model_data):
        return latex_backend_model.LatexBackendModel(dummy_model_data, include="valid")

    @pytest.mark.parametrize(
        "backend_obj", ["valid_latex_backend", "dummy_latex_backend_model"]
    )
    def test_add_parameter(self, request, backend_obj):
        latex_backend_model = request.getfixturevalue(backend_obj)
        latex_backend_model.add_parameter("param", xr.DataArray(1))
        assert latex_backend_model.parameters["param"] == xr.DataArray(1)
        assert "param" in latex_backend_model.valid_math_element_names

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
            },
        )
        # some null values might be introduced by the foreach array, so we just check the upper bound
        assert (
            latex_backend_model.variables["var"].sum()
            <= dummy_model_data.with_inf_as_bool.sum()
        )
        assert "var" in latex_backend_model.valid_math_element_names
        assert "math_string" in latex_backend_model.variables["var"].attrs

    def test_add_variable_not_valid(self, valid_latex_backend):
        valid_latex_backend.add_variable(
            "invalid_var",
            {
                "foreach": ["nodes", "techs"],
                "where": "False",
                "bounds": {"min": 0, "max": 1},
            },
        )
        # some null values might be introduced by the foreach array, so we just check the upper bound
        assert not valid_latex_backend.variables["invalid_var"].sum()
        assert "invalid_var" in valid_latex_backend.valid_math_element_names
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
                "where": "with_inf",
                "equation": "var + param",
            },
        )
        # some null values might be introduced by the foreach array, so we just check the upper bound
        assert (
            latex_backend_model.global_expressions["expr"].sum()
            <= dummy_model_data.with_inf_as_bool.sum()
        )
        assert "expr" in latex_backend_model.valid_math_element_names
        assert "math_string" in latex_backend_model.global_expressions["expr"].attrs

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
                "equation": "var >= param",
            },
        )
        # some null values might be introduced by the foreach array, so we just check the upper bound
        assert (
            latex_backend_model.constraints["constr"].sum()
            <= dummy_model_data.with_inf_as_bool.sum()
        )
        assert "constr" not in latex_backend_model.valid_math_element_names
        assert "math_string" in latex_backend_model.constraints["constr"].attrs

    def test_add_constraint_not_valid(self, valid_latex_backend):
        valid_latex_backend.add_constraint(
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
                    {"expression": "var >= param"},
                    {"expression": "var <= expr", "where": "False"},
                ],
            },
        )
        assert (
            "expr"
            not in valid_latex_backend.constraints["valid_constr"].attrs["math_string"]
        )

    def test_add_objective(self, dummy_latex_backend_model):
        dummy_latex_backend_model.add_objective(
            "obj",
            {"equation": "sum(var, over=[nodes, techs])", "sense": "minimize"},
        )
        assert dummy_latex_backend_model.objectives["obj"].isnull().all()
        assert "obj" not in dummy_latex_backend_model.valid_math_element_names
        assert len(dummy_latex_backend_model.objectives.data_vars) == 1

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
        expr = dummy_latex_backend_model.get_global_expression("expr")
        assert expr.equals(dummy_latex_backend_model.global_expressions["expr"])

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
                    \section{Where}

                    \paragraph{ expr }
                    foobar
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

                    .. container:: scrolling-wrapper

                        .. math::
                            \begin{array}{r}
                            \end{array}
                            \begin{cases}
                                1 + 2&\quad
                                \\
                            \end{cases}
                    """
                ),
            ),
            (
                "md",
                textwrap.dedent(
                    r"""

                    # Where

                    ## expr
                    foobar

                        ```math
                        \begin{array}{r}
                        \end{array}
                        \begin{cases}
                            1 + 2&\quad
                            \\
                        \end{cases}
                        ```
                    """
                ),
            ),
        ],
    )
    def test_generate_math_doc(self, dummy_model_data, format, expected):
        backend_model = latex_backend_model.LatexBackendModel(dummy_model_data)
        backend_model.add_global_expression(
            "expr", {"equation": "1 + 2", "description": "foobar"}
        )
        doc = backend_model.generate_math_doc(format=format)
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
        dummy_latex_backend_model._generate_math_string(None, da, **kwargs)
        assert da.math_string == expected

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

    def test_get_capacity_bounds(self, dummy_latex_backend_model):
        bounds = {"min": 1, "max": 2e6}
        lb, ub = dummy_latex_backend_model._get_capacity_bounds("var", bounds)
        assert lb == {"expression": r"1 \leq \textbf{var}_\text{node,tech}"}
        assert ub == {
            "expression": r"\textbf{var}_\text{node,tech} \leq 2\mathord{\times}10^{+06}"
        }
