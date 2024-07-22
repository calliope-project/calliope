import logging
from contextlib import contextmanager

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
    @contextmanager
    def caplog_session(self, request):
        """caplog for class/session-scoped fixtures.

        See https://github.com/pytest-dev/pytest/discussions/11177
        """
        request.node.add_report_section = lambda *args: None
        logging_plugin = request.config.pluginmanager.getplugin("logging-plugin")
        for _ in logging_plugin.pytest_runtest_setup(request.node):
            yield pytest.LogCaptureFixture(request.node, _ispytest=True)

    @pytest.fixture(scope="class")
    def plan_model(self):
        """Solve in plan mode for the same overrides, to check against operate mode model."""
        model = build_model({}, "simple_supply,operate,var_costs,investment_costs")
        model.build(mode="plan")
        model.solve()
        return model

    @pytest.fixture(
        scope="class", params=[("6h", "12h"), ("12h", "12h"), ("16h", "20h")]
    )
    def operate_model_and_log(self, request):
        """Solve in plan mode, then use plan mode results to set operate mode inputs, then solve in operate mode.

        Three different operate/horizon windows chosen:
        ("6h", "12h"): Both window and horizon fit completely into the model time range (48hr)
        ("12h", "12h"): Both window and horizon are the same length, so there is no need to rebuild the optimisation problem towards the end of horizon
        ("16h", "20h"): Neither window or horizon fit completely into the model time range (48hr)
        """
        model = build_model({}, "simple_supply,operate,var_costs,investment_costs")
        model.build(mode="plan")
        model.solve()
        model.build(
            force=True,
            mode="operate",
            operate_use_cap_results=True,
            operate_window=request.param[0],
            operate_horizon=request.param[1],
        )

        with self.caplog_session(request) as caplog:
            with caplog.at_level(logging.INFO):
                model.solve(force=True)
            log = caplog.text

        return model, log

    @pytest.fixture(scope="class")
    def rerun_operate_log(self, request, operate_model_and_log):
        """Solve in operate mode a second time, to trigger new log messages."""
        with self.caplog_session(request) as caplog:
            with caplog.at_level(logging.INFO):
                operate_model_and_log[0].solve(force=True)
            return caplog.text

    def test_backend_build_mode(self, operate_model_and_log):
        """Verify that we have run in operate mode"""
        operate_model, _ = operate_model_and_log
        assert (
            operate_model.backend.inputs.attrs["config"]["build"]["mode"] == "operate"
        )

    def test_operate_mode_success(self, operate_model_and_log):
        """Solving in operate mode should lead to an optimal solution."""
        operate_model, _ = operate_model_and_log
        assert operate_model.results.attrs["termination_condition"] == "optimal"

    def test_use_cap_results(self, plan_model, operate_model_and_log):
        """Operate mode uses plan mode outputs as inputs."""
        operate_model, _ = operate_model_and_log
        assert plan_model.results.flow_cap.equals(operate_model.inputs.flow_cap)

    def test_not_reset_model_window(self, operate_model_and_log):
        """We do not expect the first time window to need resetting on solving in operate mode for the first time."""
        _, log = operate_model_and_log
        assert "Resetting model to first time window." not in log

    def test_reset_model_window(self, rerun_operate_log):
        """The backend model time window needs resetting back to the start on rerunning in operate mode."""
        assert "Resetting model to first time window." in rerun_operate_log

    def test_end_of_horizon(self, operate_model_and_log):
        """Check that increasingly shorter time horizons are logged as model rebuilds."""
        operate_model, log = operate_model_and_log
        config = operate_model.backend.inputs.attrs["config"]["build"]
        if config["operate_window"] != config["operate_horizon"]:
            assert "Reaching the end of the timeseries." in log
        else:
            assert "Reaching the end of the timeseries." not in log

    def test_operate_backend_timesteps_align(self, operate_model_and_log):
        """Check that the timesteps in both backend xarray objects have updated together."""
        operate_model, _ = operate_model_and_log
        assert operate_model.backend.inputs.timesteps.equals(
            operate_model.backend._dataset.timesteps
        )

    def test_operate_timeseries(self, operate_model_and_log):
        """Check that the full timeseries exists in the operate model results."""
        operate_model, _ = operate_model_and_log
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
