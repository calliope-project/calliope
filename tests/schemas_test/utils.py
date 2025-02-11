import importlib
import importlib.resources
from pathlib import Path

EXAMPLE_MODEL_DIR = Path(importlib.resources.files("calliope") / "example_models")
TEST_MODELS_DIR = Path(__file__).parent.parent / "common"

EXAMPLE_MODELS = [f / "model.yaml" for f in EXAMPLE_MODEL_DIR.iterdir() if f.is_dir()]

TEST_MODELS = [
    TEST_MODELS_DIR / i
    for i in [
        "national_scale_from_data_tables/model.yaml",
        "test_model/model.yaml",
        "test_model/model_minimal.yaml",
    ]
]
