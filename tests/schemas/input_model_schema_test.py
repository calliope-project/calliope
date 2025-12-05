import importlib
import importlib.resources
from pathlib import Path

import pytest

from calliope.io import read_rich_yaml
from calliope.preprocess import prepare_model_definition

EXAMPLE_MODEL_DIR = Path(importlib.resources.files("calliope") / "example_models")
TEST_MODELS_DIR = Path(__file__).parent.parent

EXAMPLE_MODELS = [f / "model.yaml" for f in EXAMPLE_MODEL_DIR.iterdir() if f.is_dir()]

TEST_MODELS = [
    TEST_MODELS_DIR / i
    for i in [
        "example_models/national_scale_from_data_tables/model.yaml",
        "common/test_model/model.yaml",
    ]
]


@pytest.mark.parametrize("model_path", EXAMPLE_MODELS + TEST_MODELS)
def test_prepare_example_models(model_path):
    """Test the node schema against example and test model definitions."""
    prepare_model_definition(read_rich_yaml(model_path), definition_path=model_path)
