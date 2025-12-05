"""
Tests for all math checks defined in base.yaml.

Each test validates that the appropriate error or warning is raised
when the check condition is triggered.

Test configurations are defined in tests/math/math_checks_config.yaml
and loaded via parametrization for maintainability.
"""

import copy
import re
from abc import ABC, abstractmethod
from pathlib import Path

import pytest

from calliope import io
from calliope.exceptions import ModelError, ModelWarning

from ..common.util import build_test_model

# Load test configuration from YAML
CONFIG = io.read_rich_yaml(Path(__file__).parent / "math_checks_config.yaml")


def _iter_cases(section_cfg: dict):
    """Iterate over sections in the checks file.

    section_cfg is a mapping:
      check_name -> {scenario, override_dict} OR
      check_name -> [{scenario, override_dict}, ...]
    Yields: (check_name, case, case_index_or_none)
    """
    for check_name, payload in section_cfg.items():
        if isinstance(payload, list):
            for idx, case in enumerate(payload):
                yield check_name, case, idx
        else:
            yield check_name, payload, None


def _make_params(
    section_name: str, section_config: dict, extra_math: str | None = None
):
    params = []
    for check_name, case, idx in _iter_cases(section_config):
        entry = copy.deepcopy(case)
        entry["check_name"] = check_name

        if extra_math is not None:
            entry.setdefault("override_dict", {})
            entry["override_dict"]["config.init.extra_math"] = [extra_math]

        case_id = (
            f"{section_name}::{check_name}"
            if idx is None
            else f"{section_name}::{check_name}__{idx:02d}"
        )
        params.append(pytest.param(entry, id=case_id))
    return params


BASE_PARAMS = _make_params("base", CONFIG["base"])
MILP_PARAMS = _make_params("milp", CONFIG["milp"], extra_math="milp")
OPERATE_PARAMS = _make_params("operate", CONFIG["operate"], extra_math="operate")


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

        if error_type == "raise":
            ctx = pytest.raises
            exc = ModelError
        elif error_type == "warn":
            ctx = pytest.warns
            exc = ModelWarning
        else:
            raise AssertionError(f"Unknown check error type: {error_type!r}")

        with ctx(exc, match=error_message):
            model.build()


class TestBaseMathChecks(Checks):
    """Test class for base.yaml math checks using parametrized test data."""

    @pytest.fixture(scope="class", params=BASE_PARAMS)
    def check(self, request):
        return request.param


class TestMILPMathChecks(Checks):
    """Test class for milp.yaml math checks using parametrized test data."""

    @pytest.fixture(scope="class", params=MILP_PARAMS)
    def check(self, request):
        return request.param


class TestOperateMathChecks(Checks):
    """Test class for operate.yaml math checks using parametrized test data."""

    @pytest.fixture(scope="class", params=OPERATE_PARAMS)
    def check(self, request):
        return request.param
