"""Test the model math handler."""

import logging
from copy import deepcopy
from pathlib import Path
from random import shuffle

import pytest

import calliope
from calliope.exceptions import ModelError
from calliope.preprocess import CalliopeMath


@pytest.fixture
def model_math_default():
    return CalliopeMath([])


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
    user_math.to_yaml(def_path / file_path)
    return "custom-math.yaml"


@pytest.mark.parametrize("invalid_obj", [1, "foo", {"foo": "bar"}, True, CalliopeMath])
def test_invalid_eq(model_math_default, invalid_obj):
    """Comparisons should not work with invalid objects."""
    assert not model_math_default == invalid_obj


@pytest.mark.parametrize("modes", [[], ["storage_inter_cluster"]])
class TestInit:
    def test_init_order(self, caplog, modes, model_math_default):
        """Math should be added in order, keeping defaults."""
        with caplog.at_level(logging.INFO):
            model_math = CalliopeMath(modes)
        assert all(
            f"Math preprocessing | added file '{i}'." in caplog.messages for i in modes
        )
        assert model_math_default.history + modes == model_math.history

    def test_init_order_user_math(
        self, modes, user_math_path, def_path, model_math_default
    ):
        """User math order should be respected."""
        modes = modes + [user_math_path]
        shuffle(modes)
        model_math = CalliopeMath(modes, def_path)
        assert model_math_default.history + modes == model_math.history

    def test_init_user_math_invalid_relative(self, modes, user_math_path):
        """Init with user math should fail if model definition path is not given for a relative path."""
        with pytest.raises(ModelError):
            CalliopeMath(modes + [user_math_path])

    def test_init_user_math_valid_absolute(self, modes, def_path, user_math_path):
        """Init with user math should succeed if user math is an absolute path."""
        abs_path = str((def_path / user_math_path).absolute())
        model_math = CalliopeMath(modes + [abs_path])
        assert model_math.in_history(abs_path)

    def test_init_dict(self, modes, user_math_path, def_path):
        """Math dictionary reload should lead to no alterations."""
        modes = modes + [user_math_path]
        shuffle(modes)
        model_math = CalliopeMath(modes, def_path)
        saved = dict(model_math)
        reloaded = CalliopeMath.from_dict(saved)
        assert model_math == reloaded


class TestMathLoading:
    @pytest.fixture(scope="class")
    def pre_defined_mode(self):
        return "storage_inter_cluster"

    @pytest.fixture
    def model_math_w_mode(self, model_math_default, pre_defined_mode):
        model_math_default._add_pre_defined_file(pre_defined_mode)
        return model_math_default

    @pytest.fixture
    def model_math_w_mode_user(self, model_math_w_mode, user_math_path, def_path):
        model_math_w_mode._add_user_defined_file(user_math_path, def_path)
        return model_math_w_mode

    @pytest.fixture(scope="class")
    def predefined_mode_data(self, pre_defined_mode):
        path = Path(calliope.__file__).parent / "math" / f"{pre_defined_mode}.yaml"
        math = calliope.AttrDict.from_yaml(path)
        return math

    def test_predefined_add(self, model_math_w_mode, predefined_mode_data):
        """Added mode should be in data."""
        flat = predefined_mode_data.as_dict_flat()
        assert all(model_math_w_mode.data.get_key(i) == flat[i] for i in flat.keys())

    def test_predefined_add_history(self, pre_defined_mode, model_math_w_mode):
        """Added modes should be recorded."""
        assert model_math_w_mode.in_history(pre_defined_mode)

    def test_predefined_add_duplicate(self, pre_defined_mode, model_math_w_mode):
        """Adding the same mode twice is invalid."""
        with pytest.raises(ModelError):
            model_math_w_mode._add_pre_defined_file(pre_defined_mode)

    @pytest.mark.parametrize("invalid_mode", ["foobar", "foobar.yaml", "operate.yaml"])
    def test_predefined_add_fail(self, invalid_mode, model_math_w_mode):
        """Requesting inexistent modes or modes with suffixes should fail."""
        with pytest.raises(ModelError):
            model_math_w_mode._add_pre_defined_file(invalid_mode)

    def test_user_math_add(
        self, model_math_w_mode_user, predefined_mode_data, user_math
    ):
        """Added user math should be in data."""
        expected_math = deepcopy(predefined_mode_data)
        expected_math.union(user_math, allow_override=True)
        flat = expected_math.as_dict_flat()
        assert all(
            model_math_w_mode_user.data.get_key(i) == flat[i] for i in flat.keys()
        )

    def test_user_math_add_history(self, model_math_w_mode_user, user_math_path):
        """Added user math should be recorded."""
        assert model_math_w_mode_user.in_history(user_math_path)

    def test_repr(self, model_math_w_mode):
        expected_repr_content = """Calliope math definition dictionary with:
    4 decision variable(s)
    0 global expression(s)
    9 constraint(s)
    0 piecewise constraint(s)
    0 objective(s)
        """
        assert expected_repr_content == str(model_math_w_mode)

    def test_add_dict(self, model_math_w_mode, model_math_w_mode_user, user_math):
        model_math_w_mode.add(user_math)
        assert model_math_w_mode_user == model_math_w_mode

    def test_user_math_add_duplicate(
        self, model_math_w_mode_user, user_math_path, def_path
    ):
        """Adding the same user math file twice should fail."""
        with pytest.raises(ModelError):
            model_math_w_mode_user._add_user_defined_file(user_math_path, def_path)

    @pytest.mark.parametrize("invalid_mode", ["foobar", "foobar.yaml", "operate.yaml"])
    def test_user_math_add_fail(self, invalid_mode, model_math_w_mode_user, def_path):
        """Requesting inexistent user modes should fail."""
        with pytest.raises(ModelError):
            model_math_w_mode_user._add_user_defined_file(invalid_mode, def_path)


class TestValidate:
    def test_validate_math_fail(self):
        """Invalid math keys must trigger a failure."""
        model_math = CalliopeMath([{"foo": "bar"}])
        with pytest.raises(ModelError):
            model_math.validate()

    def test_math_default(self, caplog, model_math_default):
        with caplog.at_level(logging.INFO):
            model_math_default.validate()
        assert "Math preprocessing | validated math against schema." in caplog.messages
