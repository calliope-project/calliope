import os
import tempfile

import pytest  # pylint: disable=unused-import
from click.testing import CliRunner

import calliope
from calliope import cli


class TestCLI:
    def test_new(self):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tempdir:
            new_path = os.path.join(tempdir, 'test')
            result = runner.invoke(cli.new, [new_path])
            assert result.exit_code == 0
            # Assert that `run.yaml` in the target dir exists
            assert os.path.isfile(os.path.join(tempdir, 'test', 'model.yaml'))

    def test_run_from_yaml(self):
        runner = CliRunner()
        this_dir = os.path.dirname(__file__)
        model_config = os.path.join(this_dir, '..', 'example_models', 'national_scale', 'model.yaml')
        with runner.isolated_filesystem() as tempdir:
            result = runner.invoke(cli.run, [model_config, '--save_netcdf=output.nc'])
            assert result.exit_code == 0
            assert os.path.isfile(os.path.join(tempdir, 'output.nc'))

    def test_run_from_netcdf(self):
        runner = CliRunner()
        model = calliope.examples.national_scale()
        with runner.isolated_filesystem() as tempdir:
            model_file = os.path.join(tempdir, 'model.nc')
            out_file = os.path.join(tempdir, 'output.nc')
            model.to_netcdf(model_file)
            result = runner.invoke(cli.run, [model_file, '--debug', '--save_netcdf=output.nc'])
            assert result.exit_code == 0
            assert os.path.isfile(out_file)

    def test_generate_runs_bash(self):
        runner = CliRunner()
        this_dir = os.path.dirname(__file__)
        model_config = os.path.join(this_dir, '..', 'example_models', 'national_scale', 'model.yaml')
        override_file = os.path.join(this_dir, '..', 'example_models', 'national_scale', 'overrides.yaml')
        # test.sh '
        with runner.isolated_filesystem() as tempdir:
            result = runner.invoke(cli.generate_runs, [
                model_config, 'test.sh', '--kind=bash',
                '--groups="run1,run2,run3,run4"',
                '--override_file={}'.format(override_file)
            ])
            assert result.exit_code == 0
            assert os.path.isfile(os.path.join(tempdir, 'test.sh'))

    def test_generate_runs_windows(self):
        runner = CliRunner()
        this_dir = os.path.dirname(__file__)
        model_config = os.path.join(this_dir, '..', 'example_models', 'national_scale', 'model.yaml')
        override_file = os.path.join(this_dir, '..', 'example_models', 'national_scale', 'overrides.yaml')
        # test.sh '
        with runner.isolated_filesystem() as tempdir:
            result = runner.invoke(cli.generate_runs, [
                model_config, 'test.bat', '--kind=windows',
                '--groups="run1,run2,run3,run4"',
                '--override_file={}'.format(override_file)
            ])
            assert result.exit_code == 0
            assert os.path.isfile(os.path.join(tempdir, 'test.bat'))

    def test_generate_runs_bsub(self):
        runner = CliRunner()
        this_dir = os.path.dirname(__file__)
        model_config = os.path.join(this_dir, '..', 'example_models', 'national_scale', 'model.yaml')
        override_file = os.path.join(this_dir, '..', 'example_models', 'national_scale', 'overrides.yaml')
        with runner.isolated_filesystem() as tempdir:
            result = runner.invoke(cli.generate_runs, [
                model_config, 'test.sh', '--kind=bsub',
                '--groups="run1,run2,run3,run4"',
                '--cluster_mem=1G', '--cluster_time=100',
                '--override_file={}'.format(override_file)
            ])
            assert result.exit_code == 0
            assert os.path.isfile(os.path.join(tempdir, 'test.sh'))
            assert os.path.isfile(os.path.join(tempdir, 'test.sh.array.sh'))
