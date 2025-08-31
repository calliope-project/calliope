from pathlib import Path

import pytest

from calliope.postprocess import MathDocumentation

from .common.util import build_test_model, check_error_or_warning


class TestMathDocumentation:
    @pytest.fixture(scope="class")
    def no_build(self):
        model = build_test_model({}, "simple_supply,two_hours,investment_costs")
        model.build()
        return model

    @pytest.fixture(scope="class")
    def build_all(self):
        model = build_test_model({}, "simple_supply,two_hours,investment_costs")
        model.build()
        return MathDocumentation(model, include="all")

    @pytest.fixture(scope="class")
    def build_valid(self):
        model = build_test_model({}, "simple_supply,two_hours,investment_costs")
        model.build()
        return MathDocumentation(model, include="valid")

    @pytest.mark.parametrize(
        ("format", "startswith"),
        [
            ("tex", "\n\\documentclass{article}"),
            ("rst", "\nObjective"),
            ("md", "\n## Objective"),
        ],
    )
    @pytest.mark.parametrize("include", ["build_all", "build_valid"])
    def test_string_return(self, request, format, startswith, include):
        math_documentation = request.getfixturevalue(include)
        string_math = math_documentation.write(format=format)
        assert string_math.startswith(startswith)

    def test_to_file(self, build_all, tmpdir_factory):
        filepath = tmpdir_factory.mktemp("custom_math").join("custom-math.tex")
        build_all.write(filename=filepath)
        assert Path(filepath).exists()

    @pytest.mark.parametrize(
        ("filepath", "format"),
        [(None, "foo"), ("myfile.foo", None), ("myfile.tex", "foo")],
    )
    def test_invalid_format(self, build_all, tmpdir_factory, filepath, format):
        if filepath is not None:
            filepath = tmpdir_factory.mktemp("custom_math").join(filepath)
        with pytest.raises(ValueError) as excinfo:  # noqa: PT011
            build_all.write(filename="foo", format=format)
        assert check_error_or_warning(
            excinfo, "Math documentation format must be one of"
        )
