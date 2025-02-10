import importlib
from pathlib import Path

EXAMPLE_MODELS = [
    f / "model.yaml"
    for f in Path(importlib.resources.files("calliope") / "example_models").iterdir()
    if f.is_dir()
]

COMMON_TEST_MODELS = [
    Path(__file__).parent.parent / "common" / i
    for i in [
        "national_scale_from_data_tables/model.yaml",
        "test_model/model.yaml",
        "test_model/model_minimal.yaml",
    ]
]
