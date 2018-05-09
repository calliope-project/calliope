import os

import pytest  # pylint: disable=unused-import
import tempfile

import calliope


class TestModel:
    @pytest.fixture(scope="module")
    def national_scale_example(self):
        model = calliope.examples.national_scale(
            override_dict={'model.subset_time': '2005-01-01'}
        )
        model.run()
        return model

    def test_save_commented_model_yaml(self, national_scale_example):
        model = national_scale_example

        with tempfile.TemporaryDirectory() as tempdir:
            out_path = os.path.join(tempdir, 'model_debug.yaml')
            model.save_commented_model_yaml(out_path)
            assert os.path.isfile(out_path)

    def test_info(self, national_scale_example):
        model = national_scale_example

        model.info()

    def test_info_minimal_model(self):
        this_path = os.path.dirname(__file__)
        model_location = os.path.join(this_path, 'common', 'test_model', 'model_minimal.yaml')
        model = calliope.Model(model_location)

        model.info()
