import pytest

from calliope.preprocess import prepare_model_definition
from calliope.schemas.config_schema import CalliopeConfig

from . import utils


class TestCalliopeConfig:
    @pytest.mark.parametrize("model_path", utils.EXAMPLE_MODELS + utils.TEST_MODELS)
    def test_example_models(self, model_path):
        """Test the schema against example and test model definitions."""
        model_def, _ = prepare_model_definition(model_path)
        CalliopeConfig(**model_def["config"])
