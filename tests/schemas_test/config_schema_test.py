import importlib

import pytest

from calliope.preprocess import prepare_model_definition
from calliope.schemas.config_schema import CalliopeConfig

EXAMPLES_DIR = importlib.resources.files("calliope") / "example_models"


class TestCalliopeConfig:
    @pytest.mark.parametrize("example", ["national_scale", "urban_scale"])
    def test_example_config(self, example):
        """Test the schema against generic model configurations."""
        model_file_path = EXAMPLES_DIR / example / "model.yaml"
        model_def, _ = prepare_model_definition(model_file_path)
        CalliopeConfig(**model_def["config"])
