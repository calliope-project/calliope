import os
import tempfile

import pytest  # pylint: disable=unused-import
from click.testing import CliRunner

import calliope
from calliope import cli, AttrDict

_THIS_DIR = os.path.dirname(__file__)
_MODEL_NATIONAL = os.path.join(
    _THIS_DIR, "..", "example_models", "national_scale", "model.yaml"
)
_MINIMAL_TEST_MODEL = os.path.join(
    _THIS_DIR, "common", "test_model", "model_minimal.yaml"
)


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
                    "--save_plots=results.html",
                ],
            )
            assert result.exit_code == 0
            assert os.path.isfile(os.path.join(tempdir, "output.nc"))
            assert os.path.isfile(os.path.join(tempdir, "results.html"))

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
                    "--override_dict={'run.spores_options.save_per_spore': True}",
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
        "arg", (("--scenario=test"), ("--override_dict={'model.name': 'test'}"))
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
        with runner.isolated_filesystem() as tempdir:
            model_file = os.path.join(tempdir, "model.nc")
            out_file = os.path.join(tempdir, "output.nc")
            model.to_netcdf(model_file)
            result = runner.invoke(cli.run, [model_file, "--save_netcdf=output.nc"])
            assert result.exit_code == 0
            assert os.path.isfile(out_file)

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
        runner = CliRunner()
        result = runner.invoke(cli.run, ["foo.yaml", "--debug"])
        assert result.exit_code == 1
        assert "Traceback (most recent call last)" in result.output

        result = runner.invoke(cli.run, ["foo.yaml"])
        assert result.exit_code == 1
        assert "Traceback (most recent call last)" not in result.output

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
            scenarios = AttrDict.from_yaml(out_file)
            assert "scenario_0" not in scenarios["scenarios"]
            assert scenarios["scenarios"]["scenario_1"] == [
                "cold_fusion",
                "run1",
                "cold_fusion_cap_share",
            ]

    def test_no_success_exit_code_when_infeasible(self):
        runner = CliRunner()
        result = runner.invoke(
            cli.run,
            [
                _MINIMAL_TEST_MODEL,
                "--scenario=investment_costs",  # without these, the model cannot run
                "--override_dict={techs.test_supply_elec.constraints.energy_cap_max: 1}",  # elec supply too low
            ],
        )
        assert result.exit_code != 0

    def test_success_exit_code_when_infeasible_and_demanded(self):
        runner = CliRunner()
        result = runner.invoke(
            cli.run,
            [
                _MINIMAL_TEST_MODEL,
                "--no_fail_when_infeasible",
                "--scenario=investment_costs",  # without these, the model cannot run
                "--override_dict={techs.test_supply_elec.constraints.energy_cap_max: 1}",  # elec supply too low
            ],
        )
        assert result.exit_code == 0
