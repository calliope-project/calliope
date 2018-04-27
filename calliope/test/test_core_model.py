import os

import pytest  # pylint: disable=unused-import
import tempfile

import calliope


class TestModel:
    def test_save_commented_model_yaml(self):
        model = calliope.examples.national_scale()
        with tempfile.TemporaryDirectory() as tempdir:
            out_path = os.path.join(tempdir, 'model_debug.yaml')
            model.save_commented_model_yaml(out_path)
            assert os.path.isfile(out_path)
