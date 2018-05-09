import os
import tempfile

import pytest  # pylint: disable=unused-import
from click.testing import CliRunner

import calliope
from calliope import cli

_THIS_DIR = os.path.dirname(__file__)
_MODEL_NATIONAL = os.path.join(
    _THIS_DIR, '..', 'example_models', 'national_scale', 'model.yaml')
_OVERRIDES_NATIONAL = os.path.join(
    _THIS_DIR, '..', 'example_models', 'national_scale', 'overrides.yaml')


class TestCLI:
    def test_new(self):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tempdir:
            new_path = os.path.join(tempdir, 'test')
            result = runner.invoke(cli.new, [new_path])
            assert result.exit_code == 0
            # Assert that `model.yaml` in the target dir exists
            assert os.path.isfile(os.path.join(tempdir, 'test', 'model.yaml'))

    def test_run_from_yaml(self):
        runner = CliRunner()

        with runner.isolated_filesystem() as tempdir:
            result = runner.invoke(cli.run, [_MODEL_NATIONAL, '--save_netcdf=output.nc', '--save_plots=results.html'])
            assert calliope._logger.level == 19
            assert result.exit_code == 0
            assert os.path.isfile(os.path.join(tempdir, 'output.nc'))
            assert os.path.isfile(os.path.join(tempdir, 'results.html'))

    def test_run_from_netcdf(self):
        runner = CliRunner()
        model = calliope.examples.national_scale()
        with runner.isolated_filesystem() as tempdir:
            model_file = os.path.join(tempdir, 'model.nc')
            out_file = os.path.join(tempdir, 'output.nc')
            model.to_netcdf(model_file)
            result = runner.invoke(cli.run, [model_file, '--save_netcdf=output.nc'])
            assert result.exit_code == 0
            assert os.path.isfile(out_file)

    def test_generate_runs_bash(self):
        runner = CliRunner()

        with runner.isolated_filesystem() as tempdir:
            result = runner.invoke(cli.generate_runs, [
                _MODEL_NATIONAL, 'test.sh', '--kind=bash',
                '--groups="run1,run2,run3,run4"',
                '--override_file={}'.format(_OVERRIDES_NATIONAL)
            ])
            assert result.exit_code == 0
            assert os.path.isfile(os.path.join(tempdir, 'test.sh'))

    def test_generate_runs_windows(self):
        runner = CliRunner()

        with runner.isolated_filesystem() as tempdir:
            result = runner.invoke(cli.generate_runs, [
                _MODEL_NATIONAL, 'test.bat', '--kind=windows',
                '--groups="run1,run2,run3,run4"',
                '--override_file={}'.format(_OVERRIDES_NATIONAL)
            ])
            assert result.exit_code == 0
            assert os.path.isfile(os.path.join(tempdir, 'test.bat'))

    def test_generate_runs_bsub(self):
        runner = CliRunner()

        with runner.isolated_filesystem() as tempdir:
            result = runner.invoke(cli.generate_runs, [
                _MODEL_NATIONAL, 'test.sh', '--kind=bsub',
                '--groups="run1,run2,run3,run4"',
                '--cluster_mem=1G', '--cluster_time=100',
                '--override_file={}'.format(_OVERRIDES_NATIONAL)
            ])
            assert result.exit_code == 0
            assert os.path.isfile(os.path.join(tempdir, 'test.sh'))
            assert os.path.isfile(os.path.join(tempdir, 'test.sh.array.sh'))

    def test_debug(self):
        runner = CliRunner()
        result = runner.invoke(cli.run, ['foo.yaml', '--debug'])
        assert result.exit_code == 1
        assert 'Traceback (most recent call last)' in result.output

        result = runner.invoke(cli.run, ['foo.yaml'])
        assert result.exit_code == 1
        assert 'Traceback (most recent call last)' not in result.output

    def test_convert(self):
        runner = CliRunner()
        run_config = os.path.join(
            _THIS_DIR, 'model_conversion', 'national_scale', 'run.yaml'
        )
        model_config = os.path.join(
            _THIS_DIR, 'model_conversion', 'national_scale', 'model_config', 'model.yaml'
        )

        with runner.isolated_filesystem() as tempdir:
            result = runner.invoke(
                cli.convert, [run_config, model_config, tempdir]
            )

            # Conversion ran without errors
            assert result.exit_code == 0

            # FIXME to add:
            # Ensure that 'override' is in __disabled
            # Ensure that tech groups are converted
            # Check that only allowed top-level things are at top level in model
            # Check that time series files have converted successfully
