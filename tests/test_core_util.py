import datetime
import glob
import logging
from pathlib import Path

import calliope
import importlib_resources
import jsonschema
import pytest
from calliope.core.util.generate_runs import generate_runs
from calliope.core.util.logging import log_time
from calliope.core.util.tools import memoize, memoize_instancemethod, validate_dict

from .common.util import check_error_or_warning, python36_or_higher

_EXAMPLES_DIR = importlib_resources.files("calliope") / "example_models"
_MODEL_NATIONAL = (_EXAMPLES_DIR / "national_scale" / "model.yaml").as_posix()
_MODEL_URBAN = (_EXAMPLES_DIR / "urban_scale" / "model.yaml").as_posix()


class TestMemoization:
    @memoize_instancemethod
    def instance_method(self, a, b):
        return a + b

    def test_memoize_one_arg(self):
        @memoize
        def test(a):
            return a + 1

        assert test(1) == 2
        assert test(1) == 2

    def test_memoize_two_args(self):
        @memoize
        def test(a, b):
            return a + b

        assert test(1, 2) == 3
        assert test(1, 2) == 3

    def test_memoize_instancemethod(self):
        assert self.instance_method(1, 2) == 3
        assert self.instance_method(1, 2) == 3


class TestLogging:
    @pytest.mark.parametrize(
        ["level", "include_solver_output", "expected_level", "expected_solver_level"],
        [
            ("CRITICAL", True, 50, 10),
            ("CRITICAL", False, 50, 50),
            ("info", True, 20, 10),
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
        timings = {"model_creation": datetime.datetime.now()}
        logger = logging.getLogger("calliope.testlogger")

        # TODO: capture logging output and check that comment is in string
        log_time(logger, timings, "test", comment="test_comment", level="info")
        assert isinstance(timings["test"], datetime.datetime)

        log_time(logger, timings, "test2", comment=None, level="info")
        assert isinstance(timings["test2"], datetime.datetime)

        # TODO: capture logging output and check that time_since_solve_start is in the string
        log_time(
            logger,
            timings,
            "test",
            comment=None,
            level="info",
            time_since_solve_start=True,
        )


class TestGenerateRuns:
    @python36_or_higher
    def test_generate_runs_scenarios(self):
        runs = generate_runs(
            _MODEL_NATIONAL, scenarios="time_resampling;profiling;time_clustering"
        )
        assert len(runs) == 3
        assert runs[0].endswith(
            "--scenario time_resampling --save_netcdf out_1_time_resampling.nc"
        )

    @python36_or_higher
    def test_generate_runs_scenarios_none_with_scenarios(self):
        runs = generate_runs(_MODEL_NATIONAL, scenarios=None)
        assert len(runs) == 2
        assert runs[0].endswith(
            "--scenario cold_fusion_with_production_share --save_netcdf out_1_cold_fusion_with_production_share.nc"
        )

    @python36_or_higher
    def test_generate_runs_scenarios_none_with_overrides(self):
        runs = generate_runs(
            _MODEL_URBAN,
            scenarios=None,
        )
        assert len(runs) == 4
        assert runs[0].endswith("--scenario milp --save_netcdf out_1_milp.nc")


class TestPandasExport:
    @pytest.fixture(scope="module")
    def model(self):
        return calliope.examples.national_scale()

    @pytest.mark.parametrize(
        "variable_name",
        sorted(
            [i for i in calliope.examples.national_scale()._model_data.data_vars.keys()]
        ),
    )
    def test_data_variables_can_be_exported_to_pandas(self, model, variable_name):
        if model.inputs[variable_name].shape:
            model.inputs[variable_name].to_dataframe()
        else:
            pass


class TestValidateDict:
    @pytest.mark.parametrize(
        ["schema", "expected_path"],
        [
            ({"foo": 2}, ""),
            ({"properties": {"bar": {"foo": "string"}}}, " at `properties.bar`"),
            (
                {
                    "definitions": {"baz": {"foo": "string"}},
                    "properties": {"bar": {"$ref": "#definitions/baz"}},
                },
                " at `definitions.baz`",
            ),
        ],
    )
    def test_malformed_schema(self, schema, expected_path):
        to_validate = {"bar": [1, 2, 3]}
        with pytest.raises(jsonschema.SchemaError) as err:
            validate_dict(to_validate, schema, "foobar")
        assert check_error_or_warning(
            err,
            f"The foobar schema is malformed{expected_path}: Additional properties are not allowed ('foo' was unexpected)",
        )

    @pytest.mark.parametrize(
        ["to_validate", "expected_path"],
        [
            ({"invalid": {"foo": 2}}, ""),
            ({"valid": {"foo": 2, "invalid": 3}}, "valid: "),
        ],
    )
    def test_invalid_dict(self, to_validate, expected_path):
        schema = {
            "properties": {
                "valid": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {"foo": {"type": "number"}},
                }
            },
            "additionalProperties": False,
        }
        with pytest.raises(calliope.exceptions.ModelError) as err:
            validate_dict(to_validate, schema, "foobar")
        assert check_error_or_warning(
            err,
            [
                "Errors during validation of the foobar dictionary",
                f"* {expected_path}Additional properties are not allowed ('invalid' was unexpected)",
            ],
        )

    @pytest.fixture
    def base_math(self):
        return calliope.AttrDict.from_yaml(
            Path(calliope.__file__).parent / "math" / "base.yaml"
        )

    @pytest.mark.parametrize(
        "dict_path", glob.glob(str(Path(calliope.__file__).parent / "math" / "*.yaml"))
    )
    def test_validate_math(self, base_math, dict_path):
        math_schema = calliope.AttrDict.from_yaml(
            Path(calliope.__file__).parent / "config" / "math_schema.yaml"
        )
        to_validate = base_math.union(
            calliope.AttrDict.from_yaml(dict_path), allow_override=True
        )
        validate_dict(to_validate, math_schema, "")
