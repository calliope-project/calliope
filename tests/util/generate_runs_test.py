"""Tests for util.generate_runs module."""

import pytest

from calliope.util import generate_runs


@pytest.fixture
def simple_model_file(tmp_path):
    """Create a simple model file with scenarios."""
    model = tmp_path / "model.yaml"
    model.write_text("""
scenarios:
  scenario1: [override1]
  scenario2: [override2]
overrides:
  override1:
    config.init.name: test1
  override2:
    config.init.name: test2
""")
    return str(model)


@pytest.fixture
def override_file(tmp_path):
    """Create an override file."""
    override = tmp_path / "override.yaml"
    override.write_text("""
config:
  solve:
    solver: glpk
""")
    return str(override)


class TestGenerateRuns:
    def test_generate_from_scenarios(self, simple_model_file):
        """Test command generation from scenarios in model file."""
        commands = generate_runs.generate_runs(simple_model_file)
        assert len(commands) == 2
        assert (
            commands[0]
            == f"calliope run {simple_model_file} --scenario scenario1 --save_netcdf out_1_scenario1.nc"
        )
        assert (
            commands[1]
            == f"calliope run {simple_model_file} --scenario scenario2 --save_netcdf out_2_scenario2.nc"
        )

    def test_generate_with_explicit_scenarios(self, simple_model_file):
        """Test command generation with explicitly specified scenarios."""
        commands = generate_runs.generate_runs(simple_model_file, scenarios="scenario1")
        assert len(commands) == 1
        assert (
            commands[0]
            == f"calliope run {simple_model_file} --scenario scenario1 --save_netcdf out_1_scenario1.nc"
        )

    def test_generate_with_override_dict(self, simple_model_file, override_file):
        """Test command generation with override dictionary."""
        commands = generate_runs.generate_runs(
            simple_model_file, override_dict=override_file
        )
        assert len(commands) == 2
        assert all(f'--override_dict="{override_file}"' in cmd for cmd in commands)

    def test_generate_with_additional_args(self, simple_model_file):
        """Test command generation with additional arguments."""
        commands = generate_runs.generate_runs(
            simple_model_file, additional_args="--debug"
        )
        assert len(commands) == 2
        assert all("--debug" in cmd for cmd in commands)

    def test_generate_padding(self, tmp_path):
        """Test that output filenames are zero-padded correctly."""
        model = tmp_path / "model.yaml"
        overrides = "\n".join(
            [f"  override{i}:\n    config.init.name: test{i}" for i in range(12)]
        )
        model.write_text(f"overrides:\n{overrides}")

        commands = generate_runs.generate_runs(str(model))
        assert "out_01_" in commands[0]
        assert "out_12_" in commands[11]


class TestGenerateBashScript:
    def test_generate_script(self, simple_model_file, tmp_path):
        """Test bash script generation."""
        output = tmp_path / "run.sh"
        generate_runs.generate_bash_script(
            str(output), simple_model_file, scenarios=None
        )

        assert output.exists()
        content = output.read_text()
        assert "#!/bin/sh" in content
        assert "scenario1" in content or "scenario2" in content

    def test_generate_script_with_additional_args(self, simple_model_file, tmp_path):
        """Test bash script generation with additional arguments."""
        output = tmp_path / "run.sh"
        generate_runs.generate_bash_script(
            str(output), simple_model_file, scenarios=None, additional_args="--debug"
        )

        content = output.read_text()
        assert "--debug" in content
