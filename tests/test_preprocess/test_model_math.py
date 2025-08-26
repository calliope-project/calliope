"""Test the model math handler."""

import logging
from pathlib import Path

import pytest

import calliope
from calliope import exceptions
from calliope.io import read_rich_yaml, to_yaml
from calliope.preprocess import model_math
from calliope.schemas import math_schema

from ..common.util import build_test_model as build_model
from ..common.util import check_error_or_warning

PRE_DEFINED_MATH = ["base", "operate", "spores", "storage_inter_cluster", "milp"]


@pytest.fixture(scope="module")
def def_path(tmp_path_factory):
    return tmp_path_factory.mktemp("test_model_math")


@pytest.fixture(scope="module")
def user_math(dummy_int):
    new_vars = {"variables": {"storage": {"bounds": {"min": dummy_int}}}}
    new_constr = {
        "constraints": {
            "foobar": {"foreach": [], "where": "", "equations": [{"expression": ""}]}
        }
    }
    return calliope.AttrDict(new_vars | new_constr)


@pytest.fixture(scope="module")
def user_math_path(def_path, user_math):
    file_path = def_path / "custom-math.yaml"
    to_yaml(user_math, path=def_path / file_path)
    return "custom-math.yaml"


class TestInitMath:
    @pytest.fixture(scope="class", params=["default", "w_extra"])
    def extra_math(self, request, user_math_path):
        extra = {}
        if request.param == "w_extra":
            extra["user_math"] = user_math_path
        return extra

    @pytest.fixture(scope="class")
    def math_data(self, extra_math, def_path):
        math_paths = model_math.initialise_math_paths(extra_math, def_path)
        return model_math.load_math(math_paths)

    def test_loaded_internal(self, math_data, extra_math):
        """Loaded math should contain both user defined and internal files."""
        assert not math_data.keys() - (set(PRE_DEFINED_MATH) | extra_math.keys())

    def test_pre_defined_load(self, math_data):
        """Internal files should be loaded correctly."""
        for file in model_math.MATH_FILE_DIR.iterdir():
            expected = read_rich_yaml(file)
            math_data[file.stem] == expected

    def test_extra_load(self, math_data, extra_math, user_math):
        """Extra math should be loaded with no alterations."""
        if extra_math:
            assert math_data["user_math"] == user_math

    def test_overwrite_warning(self, user_math_path, def_path):
        """Users should be warned when overwritting pre-defined math."""
        extra_math = {"base": user_math_path}
        with pytest.raises(
            exceptions.ModelWarning, match="Overwriting pre-defined 'base' math."
        ):
            model_math.initialise_math_paths(extra_math, def_path)


class TestBuildMath:
    @pytest.fixture(scope="class")
    def test_model(self):
        """Simulate users adding extra math using the urban example model."""
        calliope_dir = Path(calliope.__file__).parent
        additional = calliope_dir / "example_models/urban_scale/additional_math.yaml"
        alternative_base = calliope_dir / "math/base.yaml"

        return calliope.examples.urban_scale(
            override_dict={
                "config": {
                    "init": {
                        "math_paths": {
                            "additional_math": str(additional.absolute()),
                            "alternative_base": str(alternative_base.absolute()),
                        },
                        "mode": "base",
                        "extra_math": [],
                    }
                }
            }
        )

    @pytest.fixture(scope="class")
    def config(self, test_model):
        return test_model.config

    @pytest.fixture(scope="class")
    def math_options(self, test_model: calliope.Model):
        return test_model.math.init.model_dump()

    @pytest.mark.parametrize(
        "math_order",
        [
            # Default
            ["base"],
            # Default and extra
            ["base", "additional_math"],
            # Default, mode and extra
            ["base", "operate", "additional_math"],
            # Alternative base, mode and extra
            ["alternative_base", "operate", "additional_math"],
        ],
    )
    def test_build_math(self, caplog, math_options, math_order):
        """Math builds must respect the order: base -> mode -> extra."""
        math = calliope.AttrDict()
        for i in math_order:
            math.union(math_options[i], allow_override=True)
        expected_math = math_schema.CalliopeBuildMath(**math).model_dump()
        with caplog.at_level(logging.INFO):
            built_math = model_math.build_applied_math(math_order, math_options)
        assert expected_math == built_math
        assert str(math_order) in caplog.text

    def test_math_name_error(self, math_options):
        """Incorrect math name errors must be user-friendly."""
        wrong_names = ["foobar_fail"]
        with pytest.raises(
            exceptions.ModelError,
            match="Requested math 'foobar_fail' was not initialised.",
        ):
            model_math.build_applied_math(wrong_names, math_options)


class TestValidateMathDict:
    LOGGER = "calliope.backend.backend_model"

    @pytest.fixture(scope="class")
    def test_model(self):
        return build_model({}, "simple_supply,investment_costs")

    @pytest.fixture
    def validate_math(self, test_model):
        def _validate_math(math_dict: dict):
            model_math.build_applied_math(
                ["base"], test_model.math.init.model_dump(), math_dict, validate=True
            )

        return _validate_math

    def test_base_math(self, caplog, validate_math):
        with caplog.at_level(logging.INFO, logger=self.LOGGER):
            validate_math({})
        assert "Math build | Validated math strings." in caplog.text

    @pytest.mark.parametrize(
        ("equation", "where"),
        [
            ("1 == 1", "True"),
            (
                "sum(flow_out * flow_out_eff, over=[nodes, carriers, techs, timesteps]) <= .inf",
                "base_tech==supply and flow_out_eff>0",
            ),
        ],
    )
    def test_add_math(self, caplog, validate_math, equation, where):
        with caplog.at_level(logging.INFO, logger=self.LOGGER):
            validate_math(
                {
                    "constraints": {
                        "foo": {"equations": [{"expression": equation}], "where": where}
                    }
                }
            )
        assert "Optimisation Model | Validated math strings." in [
            rec.message for rec in caplog.records
        ]

    @pytest.mark.parametrize(
        "component_dict",
        [
            {"equations": [{"expression": "1 = 1"}]},
            {"equations": [{"expression": "1 = 1"}], "where": "foo[bar]"},
        ],
    )
    @pytest.mark.parametrize("both_fail", [True, False])
    def test_add_math_fails(self, validate_math, component_dict, both_fail):
        math_dict = {"constraints": {"foo": component_dict}}
        errors_to_check = [
            "math string parsing (marker indicates where parsing stopped, but may not point to the root cause of the issue)",
            " * constraints:foo:",
            "equations[0].expression",
            "where",
        ]
        if both_fail:
            math_dict["constraints"]["bar"] = component_dict
            errors_to_check.append("* constraints:bar:")
        else:
            math_dict["constraints"]["bar"] = {"equations": [{"expression": "1 == 1"}]}

        with pytest.raises(calliope.exceptions.ModelError) as excinfo:
            validate_math(math_dict)
        assert check_error_or_warning(excinfo, errors_to_check)

    @pytest.mark.parametrize("eq_string", ["1 = 1", "1 ==\n1[a]"])
    def test_add_math_fails_marker_correct_position(self, validate_math, eq_string):
        math_dict = {"constraints": {"foo": {"equations": [{"expression": eq_string}]}}}

        with pytest.raises(calliope.exceptions.ModelError) as excinfo:
            validate_math(math_dict)
        errorstrings = str(excinfo.value).split("\n")
        # marker should be at the "=" sign, i.e., 2 characters from the end
        assert len(errorstrings[-2]) - 2 == len(errorstrings[-1])
