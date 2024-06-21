import importlib
import logging

import calliope
import numpy as np
import pandas as pd
import pytest

from .common.util import build_test_model as build_model
from .common.util import check_error_or_warning

LOGGER = "calliope.model"


@pytest.fixture(scope="module")
def temp_path(tmpdir_factory):
    return tmpdir_factory.mktemp("custom_math")


@pytest.fixture(scope="module")
def initialised_model():
    m = build_model({}, "simple_supply,two_hours,investment_costs")
    return m


class TestModel:
    @pytest.fixture(scope="module")
    def national_scale_example(self):
        model = calliope.examples.national_scale(
            time_subset=["2005-01-01", "2005-01-01"]
        )
        return model

    @pytest.fixture(params=[dict, calliope.AttrDict])
    def dict_to_add(self, request):
        return request.param({"a": {"b": 1}})

    def test_info(self, national_scale_example):
        national_scale_example.info()

    def test_info_simple_model(self, simple_supply):
        simple_supply.info()


class TestInitMath:
    def test_apply_base_math_by_default(self, initialised_model):
        """Base math should be applied by default."""
        _, applied_math = initialised_model._math_init_from_yaml(
            initialised_model.config.init
        )
        assert "base" in applied_math
        assert not initialised_model.config.init.add_math

    def test_base_math_deactivation(self, initialised_model):
        """Base math should be deactivated if requested in config."""
        init_config = initialised_model.config.init.copy()
        init_config["base_math"] = False
        math, applied_math = initialised_model._math_init_from_yaml(init_config)
        assert not math
        assert not applied_math

    @pytest.mark.parametrize(
        ("add_math", "in_error"),
        [
            (["foo"], ["foo"]),
            (["bar", "foo"], ["bar", "foo"]),
            (["foo", "storage_inter_cluster"], ["foo"]),
            (["foo.yaml"], ["foo.yaml"]),
        ],
    )
    def test_math_loading_invalid(self, initialised_model, add_math, in_error):
        init_config = initialised_model.config.init.copy()
        init_config.add_math = add_math
        with pytest.raises(calliope.exceptions.ModelError) as excinfo:
            initialised_model._math_init_from_yaml(init_config)
        assert check_error_or_warning(
            excinfo,
            f"Attempted to load additional math that does not exist: {in_error}",
        )

    def test_internal_override_from_yaml(self, initialised_model, temp_path):
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
        added_math_path = temp_path.join("custom-math.yaml")
        new_constraint.to_yaml(added_math_path)
        init_config = initialised_model.config.init
        init_config["add_math"] = [str(added_math_path)]

        math, applied_math = initialised_model._math_init_from_yaml(init_config)
        assert "constraint_name" in math.constraints.keys()
        assert str(added_math_path) in applied_math


class TestModelMathOverrides:
    @pytest.fixture()
    def storage_inter_cluster(self):
        return build_model(
            {"config.init.add_math": ["storage_inter_cluster"]},
            "simple_supply,two_hours,investment_costs",
        )

    def test_internal_override(self, storage_inter_cluster):
        assert "storage_intra_max" in storage_inter_cluster.math["constraints"].keys()

    def test_variable_bound(self, storage_inter_cluster):
        assert (
            storage_inter_cluster.math["variables"]["storage"]["bounds"]["min"]
            == -np.inf
        )

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
            {"config.init.add_math": [str(file_path)]},
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
            {"config.init.add_math": to_add}, "simple_supply,two_hours,investment_costs"
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
            {"config.init.add_math": ["storage_inter_cluster", str(file_path)]},
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
        ("equation", "where"),
        [
            ("1 == 1", "True"),
            (
                "flow_out * flow_out_eff + sum(cost, over=costs) <= .inf",
                "base_tech=supply and flow_out_eff>0",
            ),
        ],
    )
    def test_add_math(self, caplog, simple_supply, equation, where):
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
    def test_add_math_fails(self, simple_supply, component_dict, both_fail):
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
    def test_add_math_fails_marker_correct_position(self, simple_supply, eq_string):
        math_dict = {"constraints": {"foo": {"equations": [{"expression": eq_string}]}}}

        with pytest.raises(calliope.exceptions.ModelError) as excinfo:
            simple_supply.validate_math_strings(math_dict)
        errorstrings = str(excinfo.value).split("\n")
        # marker should be at the "=" sign, i.e., 2 characters from the end
        assert len(errorstrings[-2]) - 2 == len(errorstrings[-1])


class TestOperateMode:
    @pytest.fixture(scope="class")
    def plan_model(self):
        model = build_model({}, "simple_supply,operate,var_costs,investment_costs")
        model.build(mode="plan")
        model.solve()
        return model

    @pytest.fixture(scope="class")
    def operate_model(self):
        model = build_model({}, "simple_supply,operate,var_costs,investment_costs")
        model.build(mode="plan")
        model.solve()
        model.build(force=True, mode="operate", operate_use_cap_results=True)
        model.solve(force=True)

        return model

    def test_use_cap_results(self, plan_model, operate_model):
        assert plan_model.backend.inputs.attrs["config"]["build"]["mode"] == "plan"
        assert (
            operate_model.backend.inputs.attrs["config"]["build"]["mode"] == "operate"
        )
        assert operate_model.results.attrs["termination_condition"] == "optimal"

        assert plan_model.results.flow_cap.equals(operate_model.inputs.flow_cap)

    def test_rerun_operate(self, caplog, operate_model):
        with caplog.at_level(logging.INFO):
            operate_model.solve(force=True)
        assert "Resetting model to first time window." in caplog.text
        assert "Reaching the end of the timeseries." in caplog.text

    def test_operate_timeseries(self, operate_model):
        assert all(
            operate_model.results.timesteps
            == pd.date_range("2005-01", "2005-01-02 23:00:00", freq="h")
        )


class TestMathUpdate:

    TEST_MODES = ["operate", "spores"]

    @pytest.mark.parametrize("mode", TEST_MODES)
    def test_math_addition(self, initialised_model, mode):
        """Run mode math must be added to the model if not present."""
        mode_custom_math = calliope.AttrDict.from_yaml(
            importlib.resources.files("calliope") / "math" / f"{mode}.yaml"
        )
        new_math = initialised_model.math.copy()
        new_math.union(mode_custom_math, allow_override=True)

        updated_math, applied_math = initialised_model._math_update_with_mode(
            {**initialised_model.config.build, **{"mode": mode}}
        )
        assert new_math == updated_math
        assert mode in applied_math

    def test_no_update(self, initialised_model):
        updated_math, applied_math = initialised_model._math_update_with_mode(
            initialised_model.config.build
        )
        assert updated_math is None
        assert "base" in applied_math

    @pytest.mark.parametrize("mode", TEST_MODES)
    def test_mismatch_warning(self, mode):
        """Warn users is they load unused pre-defined math."""
        m = build_model({}, "simple_supply,two_hours,investment_costs", add_math=[mode])
        with pytest.warns(calliope.exceptions.ModelWarning):
            m._math_update_with_mode(m.config.build)
