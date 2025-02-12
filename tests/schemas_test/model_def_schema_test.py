import pytest

from calliope.preprocess import prepare_model_definition
from calliope.schemas.model_def_schema import CalliopeModelDef

from . import utils


class TestCalliopeModelDef:
    @pytest.mark.parametrize("model_path", utils.EXAMPLE_MODELS + utils.TEST_MODELS)
    def test_example_models(self, model_path):
        """Test the schema against example and test model definitions."""
        model_def, _ = prepare_model_definition(model_path)
        CalliopeModelDef(**model_def)
