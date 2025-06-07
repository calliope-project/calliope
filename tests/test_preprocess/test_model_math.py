"""Test the model math handler."""

import logging
from pathlib import Path

import pytest

import calliope
from calliope import exceptions
from calliope.io import to_yaml
from calliope.preprocess import model_math
from calliope.schemas import config_schema, math_schema


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
    def math_config(self, request, user_math_path):
        config = config_schema.InitMath().model_dump()
        if request.param == "w_extra":
            config["extra"] = {"user_math": user_math_path}
        return config_schema.InitMath(**config)

    @pytest.fixture(scope="class")
    def math_data(self, math_config, def_path):
        return model_math.initialise_math(math_config, def_path)

    def test_loaded_internal(self, math_data, math_config):
        """Loaded math should contain both user defined and internal files."""
        assert not math_data.keys() - (
            set(model_math.PRE_DEFINED_MATH) | math_config.extra.keys()
        )

    def test_pre_defined_load(self, math_data):
        """Internal files should be loaded correctly."""
        for filename in model_math.PRE_DEFINED_MATH:
            expected = model_math._load_internal_math(filename)
            math_data[filename] == expected

    def test_extra_load(self, math_data, math_config, user_math):
        """Extra math should be loaded with no alterations."""
        if math_config.extra:
            assert math_data["user_math"] == user_math

    def test_overwrite_warning(self, user_math_path, def_path):
        """Users should be warned when overwritting pre-defined math."""
        config = config_schema.InitMath()
        config = config.update({"extra": {"plan": user_math_path}})
        with pytest.raises(
            exceptions.ModelWarning, match="Overwriting pre-defined 'plan' math."
        ):
            model_math.initialise_math(config, def_path)


class TestBuildMath:
    @pytest.fixture(scope="class")
    def test_model(self):
        """Simulate users adding extra math using the urban example model."""
        calliope_dir = Path(calliope.__file__).parent
        additional = calliope_dir / "example_models/urban_scale/additional_math.yaml"
        alternative_base = calliope_dir / "math/plan.yaml"

        return calliope.examples.urban_scale(
            override_dict={
                "config": {
                    "init.math": {
                        "base": "plan",
                        "extra": {
                            "additional_math": str(additional.absolute()),
                            "alternative_base": str(alternative_base.absolute()),
                        },
                    },
                    "build": {"mode": "base", "extra_math": []},
                }
            }
        )

    @pytest.fixture(scope="class")
    def config(self, test_model):
        return test_model.config

    @pytest.fixture(scope="class")
    def math_options(self, test_model):
        return test_model._def.math

    @pytest.mark.parametrize(
        "math_order",
        [
            # Default
            ["plan"],
            # Default and extra
            ["plan", "additional_math"],
            # Default, mode and extra
            ["plan", "operate", "additional_math"],
            # Alternative base, mode and extra
            ["alternative_base", "operate", "additional_math"],
        ],
    )
    def test_build_math(self, caplog, math_options, math_order):
        """Math builds must respect the order: base -> mode -> extra."""
        math = calliope.AttrDict()
        for i in math_order:
            math.union(math_options[i], allow_override=True)
        expected_math = math_schema.MathSchema(**math).model_dump()
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
