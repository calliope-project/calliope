import logging

import calliope
import pandas as pd
import pytest

from .common.util import build_test_model as build_model
from .common.util import check_error_or_warning

LOGGER = "calliope.model"


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


class TestValidateMathDict:
    def test_base_math(self, caplog, simple_supply):
        with caplog.at_level(logging.INFO, logger=LOGGER):
            simple_supply.validate_math_strings(simple_supply.math.data)
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

    def test_build_operate_not_allowed_build(self):
        """Cannot build in operate mode if the `allow_operate_mode` attribute is False"""

        m = build_model({}, "simple_supply,two_hours,investment_costs")
        m._model_data.attrs["allow_operate_mode"] = False
        with pytest.raises(
            calliope.exceptions.ModelError, match="Unable to run this model in op"
        ):
            m.build(mode="operate")


class TestSolve:
    def test_solve_before_build(self):
        m = build_model({}, "simple_supply,two_hours,investment_costs")
        with pytest.raises(
            calliope.exceptions.ModelError, match="You must build the optimisation"
        ):
            m.solve()

    def test_solve_after_solve(self, simple_supply):
        with pytest.raises(
            calliope.exceptions.ModelError,
            match="This model object already has results.",
        ):
            simple_supply.solve()
