import os
import pdb

from calliope.test.common.util import (
    build_test_model,
)

_MODEL_NATIONAL = os.path.join(
    os.path.dirname(__file__), "..", "example_models",
    "national_scale", "model.yaml"
)

_MODEL_URBAN = os.path.join(
    os.path.dirname(__file__), "..", "example_models",
    "urban_scale", "model.yaml"
)


def _dev_test():
    model = build_test_model(model_file="model_operate.yaml")
    model.run()


if __name__ == '__main__':
    _dev_test()
