"""Test the model math handler."""

from copy import deepcopy
from pathlib import Path

import calliope
import pytest
from calliope.exceptions import ModelError
from calliope.preprocess import ModelMath


class TestModelMath:
    @pytest.fixture(scope="class")
    def def_path(self, tmpdir_factory):
        return tmpdir_factory.mktemp("test_model_math")

    @pytest.fixture(scope="class")
    def user_math(self, dummy_int):
        return calliope.AttrDict(
            {"variables": {"storage": {"bounds": {"min": dummy_int}}}}
        )

    @pytest.fixture(scope="class")
    def user_math_path(self, def_path, user_math):
        file_path = def_path.join("custom-math.yaml")
        user_math.to_yaml(file_path)
        return str(file_path)

    @pytest.fixture(scope="class")
    def model_math_base(self):
        return ModelMath()

    def test_validate_fail(self, model_math_base):
        """Invalid math keys must trigger a math failure."""
        model_math_base.math["foo"] = "bar"
        with pytest.raises(ModelError):
            model_math_base.validate()

    @pytest.mark.parametrize(
        "modes", [[], ["operate"], ["operate", "storage_inter_cluster"]]
    )
    class TestInit:
        def custom_init(self, math_to_add, model_def_path):
            return ModelMath(math_to_add, model_def_path)

        def test_regular_init(self, modes, user_math_path, def_path, user_math):
            """Math must be loaded in order with base math first."""
            math_to_add = modes + [user_math_path]
            model_math = ModelMath(math_to_add, def_path)
            flat = user_math.as_dict_flat()
            assert ["base"] + math_to_add == model_math._history
            assert all(model_math.math.get_key(i) == flat[i] for i in flat.keys())

        def test_failed_init(self, modes, user_math_path):
            """Init with user math should trigger errors if model definition path is not specified."""
            math_to_add = modes + [user_math_path]
            with pytest.raises(ModelError):
                ModelMath(math_to_add)

        def test_reload_from_dict(self, modes, user_math_path, def_path):
            """Math dictionary reload should lead to no alterations."""
            model_math = ModelMath(modes + [user_math_path], def_path)
            saved = model_math.to_dict()
            reloaded = ModelMath(saved)
            assert model_math.math == reloaded.math
            assert model_math._history == reloaded._history

    @pytest.mark.parametrize("mode", ["spores", "operate", "storage_inter_cluster"])
    class TestPreDefinedMathLoading:
        @staticmethod
        def get_expected_math(mode):
            path = Path(calliope.__file__).parent / "math"
            base_math = calliope.AttrDict.from_yaml(path / "base.yaml")
            mode_math = calliope.AttrDict.from_yaml(path / f"{mode}.yaml")
            base_math.union(mode_math, allow_override=True)
            return base_math

        def test_math_addition(self, model_math_base, mode):
            """Pre-defined math should be loaded and recorded only once."""
            expected = self.get_expected_math(mode)
            model_math_base.add_pre_defined_math(mode)
            assert expected == model_math_base.math
            assert model_math_base.check_in_history(mode)
            with pytest.raises(ModelError):
                model_math_base.add_pre_defined_math(mode)

    class TestUserMathLoading:
        @pytest.fixture(scope="class")
        def model_math_custom_add(self, model_math_base, user_math_path, def_path):
            model_math_base.add_user_defined_math(user_math_path, def_path)
            return model_math_base

        def test_user_math_addition(
            self, model_math_base, model_math_custom_load, user_math
        ):
            """User math must be loaded correctly."""
            base_math = deepcopy(model_math_base.math)
            base_math.union(user_math, allow_override=True)
            assert base_math == model_math_custom_load.math

        def test_user_math_history(self, model_math_custom_load, user_math_path):
            """User math additions should be recorded."""
            assert model_math_custom_load.check_in_history(user_math_path)

        def test_user_math_fail(self, model_math_custom_load, user_math_path, def_path):
            """User math should fail if loaded twice."""
            with pytest.raises(ModelError):
                model_math_custom_load.add_user_defined_math(user_math_path, def_path)
