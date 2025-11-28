import importlib.resources
import logging

import pytest

import calliope
from calliope.util.generate_runs import generate_runs
from calliope.util.logging import log_time

with importlib.resources.as_file(importlib.resources.files("calliope")) as f:
    _MODEL_NATIONAL = (
        f / "example_models" / "national_scale" / "model.yaml"
    ).as_posix()
    _MODEL_URBAN = (f / "example_models" / "urban_scale" / "model.yaml").as_posix()


class TestLogging:
    @pytest.mark.parametrize(
        ("level", "include_solver_output", "expected_level", "expected_solver_level"),
        [
            ("CRITICAL", True, 50, 10),
            ("CRITICAL", False, 50, 50),
            ("info", True, 20, 10),
            (20, True, 20, 10),
        ],
    )
    def test_set_log_verbosity(
        self, level, include_solver_output, expected_level, expected_solver_level
    ):
        calliope.set_log_verbosity(level, include_solver_output=include_solver_output)

        assert logging.getLogger("calliope").getEffectiveLevel() == expected_level
        assert logging.getLogger("py.warnings").getEffectiveLevel() == expected_level
        assert (
            logging.getLogger("calliope.backend.backend_model").getEffectiveLevel()
            == expected_level
        )
        assert (
            logging.getLogger(
                "calliope.backend.backend_model.<solve>"
            ).getEffectiveLevel()
            == expected_solver_level
        )

    def test_timing_log(self):
        timings = {}
        logger = logging.getLogger("calliope.testlogger")

        # TODO: capture logging output and check that comment is in string
        log_time(logger, timings, "test", comment="test_comment", level="info")
        assert isinstance(timings["test"], float)

        log_time(logger, timings, "test2", comment=None, level="info")
        assert isinstance(timings["test2"], float)

        # TODO: capture logging output and check that time_since_solve_start is in the string
        log_time(
            logger,
            timings,
            "test",
            comment=None,
            level="info",
            time_since_solve_start=True,
        )

    @pytest.mark.parametrize(
        ("capture", "expected_level", "n_handlers"), [(True, 20, 1), (False, 30, 0)]
    )
    def test_capture_warnings(self, capture, expected_level, n_handlers):
        calliope.set_log_verbosity("info", capture_warnings=capture)

        assert logging.getLogger("py.warnings").getEffectiveLevel() == expected_level
        assert len(logging.getLogger("py.warnings").handlers) == n_handlers

    def test_capture_warnings_handlers_dont_append(self):
        for level in ["critical", "warning", "info", "debug"]:
            calliope.set_log_verbosity(level, capture_warnings=True)
            assert len(logging.getLogger("py.warnings").handlers) == 1


class TestGenerateRuns:
    def test_generate_runs_scenarios(self):
        runs = generate_runs(
            _MODEL_NATIONAL, scenarios="time_resampling;profiling;time_clustering"
        )
        assert len(runs) == 3
        assert runs[0].endswith(
            "--scenario time_resampling --save_netcdf out_1_time_resampling.nc"
        )

    def test_generate_runs_scenarios_none_with_scenarios(self):
        runs = generate_runs(_MODEL_NATIONAL, scenarios=None)
        assert len(runs) == 2
        assert runs[0].endswith(
            "--scenario cold_fusion_with_production_share --save_netcdf out_1_cold_fusion_with_production_share.nc"
        )

    def test_generate_runs_scenarios_none_with_overrides(self):
        runs = generate_runs(_MODEL_URBAN, scenarios=None)
        assert len(runs) == 2
        assert runs[0].endswith("--scenario milp --save_netcdf out_1_milp.nc")


class TestPandasExport:
    MODEL = calliope.examples.national_scale()

    @pytest.mark.parametrize(
        "variable_name", sorted([i for i in MODEL.inputs.data_vars.keys()])
    )
    def test_data_variables_can_be_exported_to_pandas(self, variable_name):
        if self.MODEL.inputs[variable_name].shape:
            self.MODEL.inputs[variable_name].to_dataframe()
        else:
            pass
