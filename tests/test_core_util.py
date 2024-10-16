import datetime
import glob
import logging
from pathlib import Path

import importlib_resources
import jsonschema
import numpy as np
import pandas as pd
import pytest

import calliope
from calliope.util import schema
from calliope.util.generate_runs import generate_runs
from calliope.util.logging import log_time
from calliope.util.tools import climb_template_tree

from .common.util import check_error_or_warning

_EXAMPLES_DIR = importlib_resources.files("calliope") / "example_models"
_MODEL_NATIONAL = (_EXAMPLES_DIR / "national_scale" / "model.yaml").as_posix()
_MODEL_URBAN = (_EXAMPLES_DIR / "urban_scale" / "model.yaml").as_posix()


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
    @pytest.mark.xfail(
        reason="Checking the schema itself doesn't seem to be working properly; no clear idea of _why_ yet..."
    )
    @pytest.mark.parametrize(
        ("schema_dict", "expected_path"),
        [
            ({"foo": 2}, ""),
            ({"properties": {"bar": {"foo": "string"}}}, " at `properties.bar`"),
            (
                {
                    "definitions": {"baz": {"foo": "string"}},
                    "properties": {"bar": {"$ref": "#/definitions/baz"}},
                },
                " at `definitions.baz`",
            ),
        ],
    )
    def test_malformed_schema(self, schema_dict, expected_path):
        to_validate = {"bar": [1, 2, 3]}
        with pytest.raises(jsonschema.SchemaError) as err:
            schema.validate_dict(to_validate, schema_dict, "foobar")
        assert check_error_or_warning(
            err,
            f"The foobar schema is malformed{expected_path}: Unevaluated properties are not allowed ('foo' was unexpected)",
        )

    @pytest.mark.parametrize(
        ("to_validate", "expected_path"),
        [
            ({"invalid": {"foo": 2}}, ""),
            ({"valid": {"foo": 2, "invalid": 3}}, "valid: "),
        ],
    )
    def test_invalid_dict(self, to_validate, expected_path):
        schema_dict = {
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
            schema.validate_dict(to_validate, schema_dict, "foobar")
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
            Path(calliope.__file__).parent / "math" / "plan.yaml"
        )

    @pytest.mark.parametrize(
        "dict_path", glob.glob(str(Path(calliope.__file__).parent / "math" / "*.yaml"))
    )
    def test_validate_math(self, base_math, dict_path):
        math_schema = calliope.AttrDict.from_yaml(
            Path(calliope.__file__).parent / "config" / "math_schema.yaml"
        )
        to_validate = base_math.union(
            calliope.AttrDict.from_yaml(dict_path, allow_override=True),
            allow_override=True,
        )
        schema.validate_dict(to_validate, math_schema, "")


class TestExtractFromSchema:
    @pytest.fixture(scope="class")
    def sample_config_schema(self):
        schema_string = r"""
        $schema: https://json-schema.org/draft/2020-12/schema#
        title: All options available to configure a Calliope model.
        type: object
        additionalProperties: false
        properties:
          config:
            type: object
            description: All configuration options used for a Calliope model
            additionalProperties: false
            properties:
              init:
                type: object
                description: init config.
                additionalProperties: false
                properties:
                  name:
                    type: ["null", string]
                    default: null
                    description: Model name
              build:
                type: object
                description: build config.
                additionalProperties: true
                properties:
                  backend:
                    type: string
                    default: pyomo
                    description: backend.
              solve:
                type: object
                description: solve config.
                additionalProperties: false
                properties:
                  operate_window:
                    type: integer
                    description: operate window.
                  operate_use_cap_results:
                    type: boolean
                    default: false
                    description: operate use cap results.
        """
        return calliope.AttrDict.from_yaml_string(schema_string)

    @pytest.fixture(scope="class")
    def sample_model_def_schema(self):
        schema_string = r"""
        $schema: https://json-schema.org/draft/2020-12/schema#
        title: All options available to configure a Calliope model.
        type: object
        additionalProperties: false
        properties:
          parameters:
            type: object
            description: Calliope model arbitrary parameter definitions.
            additionalProperties: false
            properties:
              objective_cost_weights:
                type: string
                default: 1
                description: foo.
          nodes:
            type: object
            description: Calliope model node definitions.
            additionalProperties: false
            patternProperties:
              '^[^_^\d][\w]*$':
                type: object
                title: A named node.
                properties:
                  latitude:
                    type: number
                    title: Latitude (WGS84 / EPSG4326).
                  available_area:
                    type: number
                    default: .inf
                    minimum: 0
          techs:
            type: object
            description: Calliope model technology definitions.
            additionalProperties: false
            properties:
              color:
                type: ["null", string]
                default: .nan

              carrier_export:
                oneOf:
                  - type: string
                  - type: array
                    uniqueItems: true
                    minItems: 2

              base_tech:
                type: string
                enum: [demand, supply, conversion, storage, transmission]

              cap_method:
                type: string
                default: continuous
                title: Foo.
                description: foo.
                enum: [continuous, binary, integer]

              include_storage:
                type: boolean
                default: false
                title: Bar.
                description: bar.

              flow_cap_per_storage_cap_min:
                $ref: "#/$defs/TechParamNullNumberFixed"
                default: 0
                title: Foobar.
                description: foobar.
        """
        return calliope.AttrDict.from_yaml_string(schema_string)

    @pytest.fixture
    def expected_config_defaults(self):
        return pd.Series(
            {
                "init.name": None,
                "build.backend": "pyomo",
                "solve.operate_use_cap_results": False,
            }
        ).sort_index()

    @pytest.fixture
    def expected_model_def_defaults(self):
        return pd.Series(
            {
                "objective_cost_weights": 1,
                "available_area": np.inf,
                "color": np.nan,
                "cap_method": "continuous",
                "include_storage": False,
                "flow_cap_per_storage_cap_min": 0,
            }
        ).sort_index()

    def test_extract_config_defaults(
        self, sample_config_schema, expected_config_defaults
    ):
        extracted_defaults = pd.Series(
            schema.extract_from_schema(sample_config_schema, "default")
        )
        pd.testing.assert_series_equal(
            extracted_defaults.sort_index(), expected_config_defaults
        )

    def test_extract_model_def_defaults(
        self, sample_model_def_schema, expected_model_def_defaults
    ):
        extracted_defaults = pd.Series(
            schema.extract_from_schema(sample_model_def_schema, "default")
        )
        pd.testing.assert_series_equal(
            extracted_defaults.sort_index(), expected_model_def_defaults
        )

    @pytest.mark.parametrize(
        ("schema_key", "prop_keys"),
        [
            ("parameters", ["objective_cost_weights"]),
            ("nodes", ["available_area"]),
            (
                "techs",
                [
                    "color",
                    "cap_method",
                    "include_storage",
                    "flow_cap_per_storage_cap_min",
                ],
            ),
        ],
    )
    def test_extract_defaults_subset(
        self,
        sample_model_def_schema,
        expected_model_def_defaults,
        schema_key,
        prop_keys,
    ):
        extracted_defaults = pd.Series(
            schema.extract_from_schema(sample_model_def_schema, "default", schema_key)
        )
        pd.testing.assert_series_equal(
            expected_model_def_defaults.loc[prop_keys].sort_index(),
            extracted_defaults.sort_index(),
            check_dtype=False,
        )


class TestUpdateSchema:
    @pytest.mark.parametrize("top_level", ["parameters", "nodes", "techs"])
    def test_add_new_schema(self, top_level):
        schema.update_model_schema(
            top_level,
            {
                f"{top_level}_foo": {
                    "type": "number",
                    "description": "bar",
                    "default": 1,
                }
            },
            allow_override=False,
        )

        extracted_defaults = schema.extract_from_schema(schema.MODEL_SCHEMA, "default")
        assert extracted_defaults[f"{top_level}_foo"] == 1
        extracted_descriptions = schema.extract_from_schema(
            schema.MODEL_SCHEMA, "description"
        )
        assert extracted_descriptions[f"{top_level}_foo"] == "bar"

        schema.reset()

    @pytest.mark.parametrize("top_level", ["parameters", "nodes", "techs"])
    def test_update_schema(self, top_level):
        schema.update_model_schema(
            top_level, {f"{top_level}_foo": {"default": 1}}, allow_override=False
        )

        extracted_defaults = schema.extract_from_schema(schema.MODEL_SCHEMA, "default")
        assert extracted_defaults[f"{top_level}_foo"] == 1

        schema.update_model_schema(
            top_level, {f"{top_level}_foo": {"default": 2}}, allow_override=True
        )

        extracted_defaults = pd.Series(
            schema.extract_from_schema(schema.MODEL_SCHEMA, "default")
        )
        assert extracted_defaults[f"{top_level}_foo"] == 2

        schema.reset()

    @pytest.mark.parametrize("top_level", ["parameters", "nodes", "techs"])
    def test_update_schema_malformed(self, top_level):
        with pytest.raises(jsonschema.SchemaError):
            schema.update_model_schema(
                top_level,
                {f"{top_level}_foo": {"type": "i_am_not_a_type"}},
                allow_override=True,
            )
        schema.reset()

    def test_reset_schema(self):
        schema.update_model_schema(
            "techs",
            {"foo": {"type": "number", "description": "bar", "default": 1}},
            allow_override=False,
        )
        schema.reset()
        assert (
            "foo"
            not in schema.MODEL_SCHEMA["properties"]["techs"]["patternProperties"][
                "^[^_^\\d][\\w]*$"
            ]["properties"]
        )


class TestClimbTemplateTree:
    @pytest.fixture
    def templates(self) -> "calliope.AttrDict":
        return calliope.AttrDict(
            {
                "foo_group": {"template": "bar_group", "my_param": 1},
                "bar_group": {"my_param": 2, "my_other_param": 2},
                "data_table_group": {"rows": ["foobar"]},
            }
        )

    @pytest.mark.parametrize(
        ("starting_dict", "expected_dict", "expected_inheritance"),
        [
            ({"my_param": 1}, {"my_param": 1}, None),
            (
                {"template": "foo_group"},
                {"my_param": 1, "my_other_param": 2, "template": "foo_group"},
                ["bar_group", "foo_group"],
            ),
            (
                {"template": "bar_group"},
                {"my_param": 2, "my_other_param": 2, "template": "bar_group"},
                ["bar_group"],
            ),
            (
                {"template": "bar_group", "my_param": 3, "my_own_param": 1},
                {
                    "my_param": 3,
                    "my_other_param": 2,
                    "my_own_param": 1,
                    "template": "bar_group",
                },
                ["bar_group"],
            ),
            (
                {"template": "data_table_group", "columns": "techs"},
                {
                    "columns": "techs",
                    "rows": ["foobar"],
                    "template": "data_table_group",
                },
                ["data_table_group"],
            ),
        ],
    )
    def test_climb_template_tree(
        self, templates, starting_dict, expected_dict, expected_inheritance
    ):
        """Templates should be found and applied in order of 'ancestry' (newer dict keys replace older ones if they overlap)."""

        new_dict, inheritance = climb_template_tree(
            calliope.AttrDict(starting_dict), templates, "A"
        )
        assert new_dict == expected_dict
        assert inheritance == expected_inheritance

    @pytest.mark.parametrize(
        ("item_name", "expected_message_prefix"), [("A", "A | "), (None, "")]
    )
    def test_climb_template_tree_missing_ancestor(
        self, templates, item_name, expected_message_prefix
    ):
        """Referencing a template that doesn't exist in `templates` raises an error."""
        with pytest.raises(KeyError) as excinfo:
            climb_template_tree(
                calliope.AttrDict({"template": "not_there"}), templates, item_name
            )

        assert check_error_or_warning(
            excinfo,
            f"{expected_message_prefix}Cannot find `not_there` in template inheritance tree.",
        )
