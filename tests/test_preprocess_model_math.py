"""Test the model math handler."""

import logging
from copy import deepcopy
from pathlib import Path
from random import shuffle

import calliope
import pytest
from calliope.exceptions import ModelError
from calliope.preprocess import ModelMath


def _shuffle_modes(modes: list):
    shuffle(modes)
    return modes


@pytest.fixture(scope="module")
def model_math_default():
    return ModelMath()


@pytest.fixture(scope="module")
def def_path(tmpdir_factory):
    return tmpdir_factory.mktemp("test_model_math")


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
    file_path = def_path.join("custom-math.yaml")
    user_math.to_yaml(file_path)
    return str(file_path)


@pytest.mark.parametrize("invalid_obj", [1, "foo", {"foo": "bar"}, True, ModelMath])
def test_invalid_eq(model_math_default, invalid_obj):
    """Comparisons should not work with invalid objects."""
    assert not model_math_default == invalid_obj


@pytest.mark.parametrize("modes", [[], ["storage_inter_cluster"]])
class TestInit:
    def test_init_order(self, caplog, modes, model_math_default):
        """Math should be added in order, keeping defaults."""
        with caplog.at_level(logging.INFO):
            model_math = ModelMath(modes)
        assert all(f"ModelMath: added file '{i}'." in caplog.messages for i in modes)
        assert model_math_default._history + modes == model_math._history

    def test_init_order_user_math(
        self, modes, user_math_path, def_path, model_math_default
    ):
        """User math order should be respected."""
        modes = _shuffle_modes(modes + [user_math_path])
        model_math = ModelMath(modes, def_path)
        assert model_math_default._history + modes == model_math._history

    def test_init_user_math_invalid(self, modes, user_math_path):
        """Init with user math should fail if model definition path is not given."""
        with pytest.raises(ModelError):
            ModelMath(modes + [user_math_path])

    def test_init_dict(self, modes, user_math_path, def_path):
        """Math dictionary reload should lead to no alterations."""
        modes = _shuffle_modes(modes + [user_math_path])
        model_math = ModelMath(modes, def_path)
        saved = model_math.to_dict()
        reloaded = ModelMath(saved)
        assert model_math == reloaded


class TestMathLoading:
    @pytest.fixture(scope="class")
    def pre_defined_mode(self):
        return "storage_inter_cluster"

    @pytest.fixture(scope="class")
    def model_math_w_mode(self, model_math_default, pre_defined_mode):
        model_math_default.add_pre_defined_math(pre_defined_mode)
        return model_math_default

    @pytest.fixture(scope="class")
    def predefined_mode_data(self, pre_defined_mode):
        path = Path(calliope.__file__).parent / "math" / f"{pre_defined_mode}.yaml"
        math = calliope.AttrDict.from_yaml(path)
        return math

    def test_predefined_add(self, model_math_w_mode, predefined_mode_data):
        """Added mode should be in data."""
        flat = predefined_mode_data.as_dict_flat()
        assert all(model_math_w_mode._data.get_key(i) == flat[i] for i in flat.keys())

    def test_predefined_add_history(self, pre_defined_mode, model_math_w_mode):
        """Added modes should be recorded."""
        assert model_math_w_mode.check_in_history(pre_defined_mode)

    def test_predefined_add_duplicate(self, pre_defined_mode, model_math_w_mode):
        """Adding the same mode twice is invalid."""
        with pytest.raises(ModelError):
            model_math_w_mode.add_pre_defined_math(pre_defined_mode)

    @pytest.mark.parametrize("invalid_mode", ["foobar", "foobar.yaml", "operate.yaml"])
    def test_predefined_add_fail(self, invalid_mode, model_math_w_mode):
        """Requesting inexistent modes or modes with suffixes should fail."""
        with pytest.raises(ModelError):
            model_math_w_mode.add_pre_defined_math(invalid_mode)

    @pytest.fixture(scope="class")
    def model_math_w_mode_user(self, model_math_w_mode, user_math_path, def_path):
        model_math_w_mode.add_user_defined_math(user_math_path, def_path)
        return model_math_w_mode

    def test_user_math_add(
        self, model_math_w_mode_user, predefined_mode_data, user_math
    ):
        """Added user math should be in data."""
        expected_math = deepcopy(predefined_mode_data)
        expected_math.union(user_math, allow_override=True)
        flat = expected_math.as_dict_flat()
        assert all(
            model_math_w_mode_user._data.get_key(i) == flat[i] for i in flat.keys()
        )

    def test_user_math_add_history(self, model_math_w_mode_user, user_math_path):
        """Added user math should be recorded."""
        assert model_math_w_mode_user.check_in_history(user_math_path)

    def test_user_math_add_duplicate(
        self, model_math_w_mode_user, user_math_path, def_path
    ):
        """Adding the same user math file twice should fail."""
        with pytest.raises(ModelError):
            model_math_w_mode_user.add_user_defined_math(user_math_path, def_path)

    @pytest.mark.parametrize("invalid_mode", ["foobar", "foobar.yaml", "operate.yaml"])
    def test_user_math_add_fail(self, invalid_mode, model_math_w_mode_user, def_path):
        """Requesting inexistent user modes should fail."""
        with pytest.raises(ModelError):
            model_math_w_mode_user.add_user_defined_math(invalid_mode, def_path)


class TestValidate:
    def test_validate_math_fail(self, model_math_default):
        """Invalid math keys must trigger a failure."""
        with pytest.raises(ModelError):
            # TODO: remove AttrDict once https://github.com/calliope-project/calliope/issues/640 is solved
            model_math_default.validate(calliope.AttrDict({"foo": "bar"}))

    def test_math_default(self, caplog, model_math_default):
        with caplog.at_level(logging.INFO):
            model_math_default.validate()
        assert "ModelMath: validated math against schema." in caplog.messages
