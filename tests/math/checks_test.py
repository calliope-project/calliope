"""
Tests for all math checks defined in base.yaml.

Each test validates that the appropriate error or warning is raised
when the check condition is triggered.

Test configurations are defined in tests/common/base_math_checks_config.yaml
and loaded via parametrization for maintainability.
"""

import re
from abc import ABC, abstractmethod
from pathlib import Path

import pytest

from calliope import io
from calliope.exceptions import ModelError, ModelWarning

from ..common.util import build_test_model

# Load test configuration from YAML

CONFIG = io.read_rich_yaml(Path(__file__).parent / "math_checks_config.yaml")


class Checks(ABC):
    """Test class for base.yaml math checks using parametrized test data."""

    @pytest.fixture(scope="class", params=[])
    @abstractmethod
    def check(self, request):
        return request.param

    def test_check_raises_error_or_warning(self, check):
        """Test that checks raise appropriate errors or warnings."""
        model = build_test_model(
            override_dict=check["override_dict"], scenario=check["scenario"]
        )
        check_being_tested = model.math.build.checks[check["check_name"]]
        error_type = check_being_tested.errors
        error_message = re.escape(check_being_tested.message)

        # Determine if this is an error (raise) or warning (warn)
        if error_type == "raise":
            func = pytest.raises
            error_type = ModelError
        elif error_type == "warn":
            func = pytest.warns
            error_type = ModelWarning
        with func(error_type, match=error_message):
            model.build()


class TestBaseMathChecks(Checks):
    """Test class for base.yaml math checks using parametrized test data."""

    @pytest.fixture(scope="class", params=CONFIG["base"])
    def check(self, request):
        return request.param


class TestMILPMathChecks(Checks):
    """Test class for base.yaml math checks using parametrized test data."""

    @pytest.fixture(scope="class", params=CONFIG["milp"])
    def check(self, request):
        entry = request.param
        entry["override_dict"]["config.init.extra_math"] = ["milp"]
        return entry


class TestOperateMathChecks(Checks):
    """Test class for base.yaml math checks using parametrized test data."""

    @pytest.fixture(scope="class", params=CONFIG["operate"])
    def check(self, request):
        entry = request.param
        entry["override_dict"]["config.init.extra_math"] = ["operate"]
        return entry
