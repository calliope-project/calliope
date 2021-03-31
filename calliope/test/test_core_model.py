import os

import pytest
import tempfile
import pandas as pd

import calliope
from calliope.test.common.util import check_error_or_warning


class TestModel:
    @pytest.fixture(scope="module")
    def national_scale_example(self):
        model = calliope.examples.national_scale(
            override_dict={"model.subset_time": ["2005-01-01", "2005-01-01"]}
        )
        model.run()
        return model

    def test_info(self, national_scale_example):
        model = national_scale_example

        model.info()

    def test_info_minimal_model(self):
        this_path = os.path.dirname(__file__)
        model_location = os.path.join(
            this_path, "common", "test_model", "model_minimal.yaml"
        )
        model = calliope.Model(model_location)

        model.info()
