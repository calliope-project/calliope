import importlib.resources
import os
import subprocess
import tempfile
from pathlib import Path

import pytest  # noqa: F401
from click.testing import CliRunner

import calliope
from calliope import cli, io

with importlib.resources.as_file(importlib.resources.files("calliope")) as f:
    _MODEL_NATIONAL = (
        f / "example_models" / "national_scale" / "model.yaml"
    ).as_posix()


class TestCLI:
    def test_new(self):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tempdir:
            new_path = os.path.join(tempdir, "test")
            result = runner.invoke(cli.new, [new_path])
            assert result.exit_code == 0
            # Assert that `model.yaml` in the target dir exists
            assert os.path.isfile(os.path.join(tempdir, "test", "model.yaml"))

    def test_run_from_yaml(self):
        runner = CliRunner()

        with runner.isolated_filesystem() as tempdir:
            result = runner.invoke(
                cli.run,
                [
                    _MODEL_NATIONAL,
                    "--save_netcdf=output.nc",
                    # FIXME: should be CBC but a timeout error causes issues on OSX CI
                    # (likely fixed by updating pyomo)
                    "--override_dict={'config.solve.solver': 'glpk'}",
                ],
            )
            assert result.exit_code == 0
            assert os.path.isfile(os.path.join(tempdir, "output.nc"))

    @pytest.mark.skip(
        reason="SPORES mode will fail until the cost max group constraint can be reproduced"
    )
    def test_save_per_spore(self):
        runner = CliRunner()

        with runner.isolated_filesystem() as tempdir:
            os.mkdir(os.path.join(tempdir, "output"))
            print(os.listdir(tempdir))
            result = runner.invoke(
                cli.run,
                [
                    _MODEL_NATIONAL,
                    "--save_netcdf=output.nc",
                    "--scenario=spores",
                    "--override_dict={'config.solve.spores_save_per_spore': True}",
                ],
            )
            print(os.listdir(os.path.join(tempdir, "output")))
            assert result.exit_code == 0
            for i in ["0", "1", "2", "3"]:
                assert os.path.isfile(os.path.join(tempdir, "output", f"spore_{i}.nc"))
            assert os.path.isfile(os.path.join(tempdir, "output.nc"))

    def test_incorrect_file_format(self):
        runner = CliRunner()

        result = runner.invoke(cli.run, ["test_model.txt", "--save_netcdf=output.nc"])
        assert "Cannot determine model file format" in result.output
        assert result.exit_code == 1

    def test_incorrect_model_format(self):
        runner = CliRunner()

        result = runner.invoke(
            cli.run, ["test_model.txt", "--model_format=yml", "--save_netcdf=output.nc"]
        )
        assert "Invalid model format" in result.output
        assert result.exit_code == 1

    @pytest.mark.parametrize(
        "arg", [("--scenario=test"), ("--override_dict={'config.init.name': 'test'}")]
    )
    def test_unavailable_arguments(self, arg):
        runner = CliRunner()

        result = runner.invoke(
            cli.run, ["test_model.nc", arg, "--save_netcdf=output.nc"]
        )
        assert (
            "the --scenario and --override_dict options are not available"
            in result.output
        )
        assert result.exit_code == 1

    def test_run_from_netcdf(self):
        runner = CliRunner()
        model = calliope.examples.national_scale()

        model_file = "model.nc"
        out_file = "output.nc"

        with runner.isolated_filesystem() as tempdir:
            model.to_netcdf(model_file)
            result = runner.invoke(cli.run, [model_file, f"--save_netcdf={out_file}"])
            assert result.exit_code == 0
            assert (Path(tempdir) / out_file).is_file()

    def test_run_save_lp(self):
        runner = CliRunner()

        with runner.isolated_filesystem() as tempdir:
            result = runner.invoke(cli.run, [_MODEL_NATIONAL, "--save_lp=output.lp"])
            assert result.exit_code == 0
            assert os.path.isfile(os.path.join(tempdir, "output.lp"))

    def test_generate_runs_bash(self):
        runner = CliRunner()

        with runner.isolated_filesystem() as tempdir:
            result = runner.invoke(
                cli.generate_runs,
                [
                    _MODEL_NATIONAL,
                    "test.sh",
                    "--kind=bash",
                    '--scenarios="run1;run2;run3;run4"',
                ],
            )
            assert result.exit_code == 0
            assert os.path.isfile(os.path.join(tempdir, "test.sh"))

    def test_generate_runs_windows(self):
        runner = CliRunner()

        with runner.isolated_filesystem() as tempdir:
            result = runner.invoke(
                cli.generate_runs,
                [
                    _MODEL_NATIONAL,
                    "test.bat",
                    "--kind=windows",
                    '--scenarios="run1;run2;run3;run4"',
                ],
            )
            assert result.exit_code == 0
            assert os.path.isfile(os.path.join(tempdir, "test.bat"))

    def test_generate_runs_bsub(self):
        runner = CliRunner()

        with runner.isolated_filesystem() as tempdir:
            result = runner.invoke(
                cli.generate_runs,
                [
                    _MODEL_NATIONAL,
                    "test.sh",
                    "--kind=bsub",
                    '--scenarios="run1;run2;run3;run4"',
                    "--cluster_mem=1G",
                    "--cluster_time=100",
                ],
            )
            assert result.exit_code == 0
            assert os.path.isfile(os.path.join(tempdir, "test.sh"))
            assert os.path.isfile(os.path.join(tempdir, "test.sh.array.sh"))

    def test_generate_runs_sbatch(self):
        runner = CliRunner()

        with runner.isolated_filesystem() as tempdir:
            result = runner.invoke(
                cli.generate_runs,
                [
                    _MODEL_NATIONAL,
                    "test.sh",
                    "--kind=sbatch",
                    '--scenarios="run1;run2;run3;run4"',
                    "--cluster_mem=1G",
                    "--cluster_time=100",
                ],
            )
            assert result.exit_code == 0
            assert os.path.isfile(os.path.join(tempdir, "test.sh"))
            assert os.path.isfile(os.path.join(tempdir, "test.sh.array.sh"))

    def test_debug(self):
        """Trackeback should only be printed in debug mode."""
        # FIXME: revert back to CliRunner when error handling is made consistent with terminal output
        # See https://github.com/pallets/click/issues/2682
        shell_cmd = "calliope run foo.yaml"
        result = subprocess.run(shell_cmd, shell=True, capture_output=True)
        assert result.returncode == 1
        assert not result.stderr

        result = subprocess.run(shell_cmd + " --debug", shell=True, capture_output=True)
        assert result.returncode == 1
        assert "Traceback (most recent call last)" in result.stderr.decode()

    def test_generate_scenarios(self):
        runner = CliRunner()
        with runner.isolated_filesystem() as tempdir:
            out_file = os.path.join(tempdir, "scenarios.yaml")
            result = runner.invoke(
                cli.generate_scenarios,
                [
                    _MODEL_NATIONAL,
                    out_file,
                    "cold_fusion",
                    "run1;run2",
                    "cold_fusion_cap_share;cold_fusion_prod_share",
                ],
            )
            assert result.exit_code == 0
            assert os.path.isfile(out_file)
            scenarios = io.read_rich_yaml(out_file)
            assert "scenario_0" not in scenarios["scenarios"]
            assert scenarios["scenarios"]["scenario_1"] == [
                "cold_fusion",
                "run1",
                "cold_fusion_cap_share",
            ]

    @pytest.mark.filterwarnings(
        "ignore:(?s).*Model solution was non-optimal:calliope.exceptions.BackendWarning"
    )
    def test_no_success_exit_code_when_infeasible(self, minimal_test_model_path):
        runner = CliRunner()
        result = runner.invoke(
            cli.run,
            [
                minimal_test_model_path,
                "--scenario=simple_supply",  # without these, the model cannot run
                "--override_dict={techs.test_supply_elec.flow_cap_max: 1}",  # elec supply too low
            ],
        )
        assert result.exit_code != 0

    @pytest.mark.filterwarnings(
        "ignore:(?s).*Model solution was non-optimal:calliope.exceptions.BackendWarning"
    )
    def test_success_exit_code_when_infeasible_and_demanded(
        self, minimal_test_model_path
    ):
        runner = CliRunner()
        result = runner.invoke(
            cli.run,
            [
                minimal_test_model_path,
                "--no_fail_when_infeasible",
                "--scenario=simple_supply",  # without these, the model cannot run
                "--override_dict={techs.test_supply_elec.flow_cap_max: 1}",  # elec supply too low
            ],
        )
        assert result.exit_code == 0
