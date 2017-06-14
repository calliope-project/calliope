import os
import tempfile

import pytest  # pylint: disable=unused-import
from click.testing import CliRunner

from calliope import cli


class TestCLI:
    def test_new(self):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tempdir:
            new_path = os.path.join(tempdir, 'test')
            result = runner.invoke(cli.new, [new_path])
            assert result.exit_code == 0
            # Assert that `run.yaml` in the target dir exists
            assert os.path.isfile(os.path.join(tempdir, 'test', 'run.yaml'))

    def test_run(self):
        runner = CliRunner()
        this_dir = os.path.dirname(__file__)
        run_config = os.path.join(this_dir, '..', 'example_models', 'national_scale', 'run.yaml')
        with runner.isolated_filesystem() as tempdir:
            result = runner.invoke(cli.run, [run_config])
            assert result.exit_code == 0
            assert os.path.isfile(os.path.join(tempdir, 'Output', 'r.csv'))

    def test_generate(self):
        runner = CliRunner()
        this_dir = os.path.dirname(__file__)
        run_config = os.path.join(this_dir, '..', 'example_models', 'national_scale', 'run.yaml')
        with runner.isolated_filesystem() as tempdir:
            result = runner.invoke(cli.generate, [run_config])
            assert result.exit_code == 0
            assert os.path.isfile(os.path.join(tempdir, 'runs', 'example-model-national', 'submit_array.sh'))
