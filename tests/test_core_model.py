import logging
import os

import calliope
import numpy as np
import pytest

from .common.util import build_test_model as build_model
from .common.util import check_error_or_warning

LOGGER = "calliope.core.model"


class TestModel:
    @pytest.fixture(scope="module")
    def national_scale_example(self):
        model = calliope.examples.national_scale(
            override_dict={"config.init.time_subset": ["2005-01-01", "2005-01-01"]}
        )
        return model

    @pytest.fixture(params=[dict, calliope.AttrDict])
    def dict_to_add(self, request):
        return request.param({"a": {"b": 1}})

    def test_info(self, national_scale_example):
        national_scale_example.info()

    def test_info_minimal_model(self):
        this_path = os.path.dirname(__file__)
        model_location = os.path.join(
            this_path, "common", "test_model", "model_minimal.yaml"
        )
        model = calliope.Model(model_location)

        model.info()

    def test_update_observed_dict(self, national_scale_example):
        national_scale_example.config.build["backend"] = "foo"
        assert national_scale_example._model_data.attrs["config"].build.backend == "foo"

    def test_add_observed_dict_from_model_data(
        self, national_scale_example, dict_to_add
    ):
        national_scale_example._model_data.attrs["foo"] = dict_to_add
        national_scale_example._add_observed_dict("foo")
        assert national_scale_example.foo == dict_to_add
        assert national_scale_example._model_data.attrs["foo"] == dict_to_add

    def test_add_observed_dict_from_dict(self, national_scale_example, dict_to_add):
        national_scale_example._add_observed_dict("bar", dict_to_add)
        assert national_scale_example.bar == dict_to_add
        assert national_scale_example._model_data.attrs["bar"] == dict_to_add

    def test_add_observed_dict_not_available(self, national_scale_example):
        with pytest.raises(calliope.exceptions.ModelError) as excinfo:
            national_scale_example._add_observed_dict("baz")
        assert check_error_or_warning(
            excinfo,
            "Expected the model property `baz` to be a dictionary attribute of the model dataset",
        )
        assert not hasattr(national_scale_example, "baz")

    def test_add_observed_dict_not_dict(self, national_scale_example):
        with pytest.raises(TypeError) as excinfo:
            national_scale_example._add_observed_dict("baz", "bar")
        assert check_error_or_warning(
            excinfo,
            "Attempted to add dictionary property `baz` to model, but received argument of type `str`",
        )


class TestCustomMath:
    @pytest.fixture
    def storage_inter_cluster(self):
        return build_model(
            {"config.init.custom_math": ["storage_inter_cluster"]},
            "simple_supply,two_hours,investment_costs",
        )

    @pytest.fixture
    def temp_path(self, tmpdir_factory):
        return tmpdir_factory.mktemp("custom_math")

    def test_internal_override(self, storage_inter_cluster):
        assert "storage_intra_max" in storage_inter_cluster.math["constraints"].keys()

    def test_variable_bound(self, storage_inter_cluster):
        assert (
            storage_inter_cluster.math["variables"]["storage"]["bounds"]["min"]
            == -np.inf
        )

    @pytest.mark.parametrize(
        ["override", "expected"],
        [
            (["foo"], ["foo"]),
            (["bar", "foo"], ["bar", "foo"]),
            (["foo", "storage_inter_cluster"], ["foo"]),
            (["foo.yaml"], ["foo.yaml"]),
        ],
    )
    def test_allowed_internal_constraint(self, override, expected):
        with pytest.raises(calliope.exceptions.ModelError) as excinfo:
            build_model(
                {"config.init.custom_math": override},
                "simple_supply,two_hours,investment_costs",
            )
        assert check_error_or_warning(
            excinfo, f"Attempted to load custom math that does not exist: {expected}"
        )

    def test_internal_override_from_yaml(self, temp_path):
        new_constraint = calliope.AttrDict(
            {
                "constraints": {
                    "constraint_name": {
                        "foreach": [],
                        "where": "",
                        "equations": [{"expression": ""}],
                    }
                }
            }
        )
        new_constraint.to_yaml(temp_path.join("custom-math.yaml"))
        m = build_model(
            {"config.init.custom_math": [str(temp_path.join("custom-math.yaml"))]},
            "simple_supply,two_hours,investment_costs",
        )
        assert "constraint_name" in m.math["constraints"].keys()

    def test_override_existing_internal_constraint(self, temp_path, simple_supply):
        file_path = temp_path.join("custom-math.yaml")
        new_constraint = calliope.AttrDict(
            {
                "constraints": {
                    "flow_capacity_per_storage_capacity_min": {"foreach": ["nodes"]}
                }
            }
        )
        new_constraint.to_yaml(file_path)
        m = build_model(
            {"config.init.custom_math": [str(file_path)]},
            "simple_supply,two_hours,investment_costs",
        )
        base = simple_supply.math["constraints"][
            "flow_capacity_per_storage_capacity_min"
        ]
        new = m.math["constraints"]["flow_capacity_per_storage_capacity_min"]

        for i in base.keys():
            if i == "foreach":
                assert new[i] == ["nodes"]
            else:
                assert base[i] == new[i]

    def test_override_order(self, temp_path, simple_supply):
        to_add = []
        for path_suffix, foreach in [(1, "nodes"), (2, "techs")]:
            constr = calliope.AttrDict(
                {
                    "constraints.flow_capacity_per_storage_capacity_min.foreach": [
                        foreach
                    ]
                }
            )
            filepath = temp_path.join(f"custom-math-{path_suffix}.yaml")
            constr.to_yaml(filepath)
            to_add.append(str(filepath))

        m = build_model(
            {"config.init.custom_math": to_add},
            "simple_supply,two_hours,investment_costs",
        )

        base = simple_supply.math["constraints"][
            "flow_capacity_per_storage_capacity_min"
        ]
        new = m.math["constraints"]["flow_capacity_per_storage_capacity_min"]

        for i in base.keys():
            if i == "foreach":
                assert new[i] == ["techs"]
            else:
                assert base[i] == new[i]

    def test_override_existing_internal_constraint_merge(
        self, temp_path, simple_supply
    ):
        new_constraint = calliope.AttrDict(
            {"variables": {"storage": {"bounds": {"min": -1}}}}
        )
        file_path = temp_path.join("custom-math.yaml")
        new_constraint.to_yaml(file_path)
        m = build_model(
            {"config.init.custom_math": ["storage_inter_cluster", str(file_path)]},
            "simple_supply,two_hours,investment_costs",
        )
        base = simple_supply.math["variables"]["storage"]
        new = m.math["variables"]["storage"]

        for i in base.keys():
            if i == "bounds":
                assert new[i]["min"] == -1
                assert new[i]["max"] == new[i]["max"]
            elif i == "description":
                assert new[i].startswith(
                    "The virtual carrier stored by a `supply_plus` or `storage` technology"
                )
            else:
                assert base[i] == new[i]


class TestValidateMathDict:
    def test_base_math(self, caplog, simple_supply):
        with caplog.at_level(logging.INFO, logger=LOGGER):
            simple_supply.validate_math_strings(simple_supply.math)
        assert "Model: validated math strings" in [
            rec.message for rec in caplog.records
        ]

    @pytest.mark.parametrize(
        ["equation", "where"],
        [
            ("1 == 1", "True"),
            (
                "flow_out * flow_out_eff + sum(cost, over=costs) <= .inf",
                "parent=supply and flow_out_eff>0",
            ),
        ],
    )
    def test_custom_math(self, caplog, simple_supply, equation, where):
        with caplog.at_level(logging.INFO, logger=LOGGER):
            simple_supply.validate_math_strings(
                {
                    "constraints": {
                        "foo": {"equations": [{"expression": equation}], "where": where}
                    }
                }
            )
        assert "Model: validated math strings" in [
            rec.message for rec in caplog.records
        ]

    @pytest.mark.parametrize(
        "component_dict",
        [
            {"equations": [{"expression": "1 = 1"}]},
            {"equations": [{"expression": "1 = 1"}], "where": "foo[bar]"},
        ],
    )
    @pytest.mark.parametrize("both_fail", [True, False])
    def test_custom_math_fails(self, simple_supply, component_dict, both_fail):
        math_dict = {"constraints": {"foo": component_dict}}
        errors_to_check = [
            "math string parsing (marker indicates where parsing stopped, which might not be the root cause of the issue; sorry...)",
            " * constraints:foo:",
            "equations[0].expression",
            "where",
        ]
        if both_fail:
            math_dict["constraints"]["bar"] = component_dict
            errors_to_check.append("* constraints:bar:")
        else:
            math_dict["constraints"]["bar"] = {"equations": [{"expression": "1 == 1"}]}

        with pytest.raises(calliope.exceptions.ModelError) as excinfo:
            simple_supply.validate_math_strings(math_dict)
        assert check_error_or_warning(excinfo, errors_to_check)

    @pytest.mark.parametrize("eq_string", ["1 = 1", "1 ==\n1[a]"])
    def test_custom_math_fails_marker_correct_position(self, simple_supply, eq_string):
        math_dict = {"constraints": {"foo": {"equations": [{"expression": eq_string}]}}}

        with pytest.raises(calliope.exceptions.ModelError) as excinfo:
            simple_supply.validate_math_strings(math_dict)
        errorstrings = str(excinfo.value).split("\n")
        # marker should be at the "=" sign, i.e., 2 characters from the end
        assert len(errorstrings[-2]) - 2 == len(errorstrings[-1])
