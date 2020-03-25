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
            override_dict={"model.subset_time": "2005-01-01"}
        )
        model.run()
        return model

    def test_save_commented_model_yaml(self, national_scale_example):
        model = national_scale_example

        with tempfile.TemporaryDirectory() as tempdir:
            out_path = os.path.join(tempdir, "model_debug.yaml")
            model.save_commented_model_yaml(out_path)
            assert os.path.isfile(out_path)

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

    def test_get_formatted_array_index(self, national_scale_example):
        array = national_scale_example.get_formatted_array(
            "resource", index_format="index"
        )

        assert array.dims == ("locs", "techs", "timesteps")

    def test_get_formatted_array_multiindex(self, national_scale_example):
        array = national_scale_example.get_formatted_array(
            "resource", index_format="multiindex"
        )

        assert array.dims == ("loc_techs_finite_resource", "timesteps")
        assert isinstance(array.loc_techs_finite_resource.to_index(), pd.MultiIndex)

    def test_get_formatted_array_unknown_format(self, national_scale_example):
        with pytest.raises(ValueError) as excinfo:
            national_scale_example.get_formatted_array("resource", index_format="foo")

        assert check_error_or_warning(
            excinfo, "Argument 'index_format' must be one of 'index' or 'multiindex'"
        )

    def test_get_formatted_array_unknown_var(self, national_scale_example):
        with pytest.raises(KeyError) as excinfo:
            national_scale_example.get_formatted_array("foo")

        assert check_error_or_warning(excinfo, "Variable foo not in Model data")
