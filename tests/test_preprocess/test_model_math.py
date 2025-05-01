"""Test the model math handler."""

import pytest

import calliope
from calliope.io import to_yaml
from calliope.preprocess import model_math
from calliope.schemas import config_schema


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


class TestLoadMathModes:
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
            set(math_config.pre_defined) | math_config.extra.keys()
        )

    def test_pre_defined_load(self, math_data, math_config):
        """Internal files should be loaded correctly."""
        for filename in math_config.pre_defined:
            expected = model_math._load_internal_math(filename)
            math_data[filename] == expected

    def test_extra_load(self, math_data, math_config, user_math):
        """Extra math should be loaded with no alterations."""
        if math_config.extra:
            assert math_data["user_math"] == user_math
