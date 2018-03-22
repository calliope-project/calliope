import os

import pytest  # pylint: disable=unused-import
import tempfile

import calliope


class TestDebug:
    def test_save_debug(self):
        model = calliope.examples.national_scale()
        with tempfile.TemporaryDirectory() as tempdir:
            out_path = os.path.join(tempdir, 'debug.yaml')
            model.save_debug_data(out_path)
            assert os.path.isfile(out_path)
