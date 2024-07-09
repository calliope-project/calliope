"""Test logging functionality."""

import logging
from datetime import datetime

import calliope
import pytest
from calliope.util.logging import log_time


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
        timings = {"model_creation": datetime.now().timestamp()}
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
