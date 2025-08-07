import logging
from contextlib import contextmanager

import numpy as np
import numpy.testing
import pandas as pd
import pytest
import xarray as xr

import calliope
import calliope.backend
import calliope.preprocess

from .common.util import build_test_model as build_model

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
    def base_model(self):
        """Solve in base mode for the same overrides, to check against operate mode model."""
        model = build_model({}, "simple_supply,operate,var_costs,investment_costs")
        model.build(mode="base")
        model.solve()
        return model

    @pytest.fixture(
        scope="class", params=[("6h", "12h"), ("12h", "12h"), ("16h", "20h")]
    )
    def operate_model_and_log(self, request):
        """Solve in base mode, then use results to set operate mode inputs, then solve in operate mode.

        Three different operate/horizon windows chosen:
        ("6h", "12h"): Both window and horizon fit completely into the model time range (48hr)
        ("12h", "12h"): Both window and horizon are the same length, so there is no need to rebuild the optimisation problem towards the end of horizon
        ("16h", "20h"): Neither window or horizon fit completely into the model time range (48hr)
        """
        model = build_model({}, "simple_supply,operate,var_costs,investment_costs")
        model.build(mode="base")
        model.solve()
        model.build(
            force=True,
            mode="operate",
            operate={
                "use_cap_results": True,
                "window": request.param[0],
                "horizon": request.param[1],
            },
        )

        with self.caplog_session(request) as caplog:
            with caplog.at_level(logging.INFO):
                model.solve(force=True)
            log = caplog.text

        return model, log

    def test_backend_build_mode(self, operate_model_and_log):
        """Verify that we have run in operate mode"""
        operate_model, _ = operate_model_and_log
        assert operate_model.backend.config.mode == "operate"

    def test_operate_mode_success(self, operate_model_and_log):
        """Solving in operate mode should lead to an optimal solution."""
        operate_model, _ = operate_model_and_log
        assert operate_model.results.attrs["termination_condition"] == "optimal"

    def test_use_cap_results(self, base_model, operate_model_and_log):
        """Operate mode uses base mode outputs as inputs."""
        operate_model, _ = operate_model_and_log
        assert base_model.results.flow_cap.equals(operate_model.inputs.flow_cap)

    def test_not_reset_model_window(self, operate_model_and_log):
        """We do not expect the first time window to need resetting on solving in operate mode for the first time."""
        _, log = operate_model_and_log
        assert "Resetting model to first time window." not in log

    @pytest.fixture
    def rerun_operate_log(self, request, operate_model_and_log):
        """Solve in operate mode a second time, to trigger new log messages."""
        with self.caplog_session(request) as caplog:
            with caplog.at_level(logging.INFO):
                operate_model_and_log[0].solve(force=True)
            return caplog.text

    def test_reset_model_window(self, rerun_operate_log):
        """The backend model time window needs resetting back to the start on rerunning in operate mode."""
        assert "Resetting model to first time window." in rerun_operate_log

    def test_end_of_horizon(self, operate_model_and_log):
        """Check that increasingly shorter time horizons are logged as model rebuilds."""
        operate_model, log = operate_model_and_log
        config = operate_model.backend.config
        if config.operate.window != config.operate.horizon:
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

    def test_build_operate_use_cap_results_error(self):
        """Requesting to use capacity results should return an error if the model is not pre-solved."""
        m = build_model({}, "simple_supply,operate,var_costs,investment_costs")
        with pytest.raises(
            calliope.exceptions.ModelError,
            match="Cannot use base mode capacity results in operate mode if a solution does not yet exist for the model.",
        ):
            m.build(mode="operate", operate={"use_cap_results": True})


class TestSporesMode:
    SPORES_OVERRIDES = (
        "spores,var_costs,two_hours,simple_supply_spores_ready,investment_costs"
    )
    # Set the global numpy random seed to avoid occasional (random!) test failures with the "random" scoring algorithm.
    np.random.seed(0)

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
    def spores_model_and_log(self, request):
        """Iterate 2 times in SPORES mode."""
        model = build_model({}, self.SPORES_OVERRIDES)
        model.build(mode="spores")
        with self.caplog_session(request) as caplog:
            with caplog.at_level(logging.INFO):
                model.solve()
            log = caplog.text

        return model, log

    @pytest.fixture(
        scope="class",
        params=["integer", "relative_deployment", "random", "evolving_average"],
    )
    def spores_model_and_log_algorithms(self, request):
        """Iterate 2 times in SPORES mode using different scoring algorithms."""
        model = build_model({}, self.SPORES_OVERRIDES)
        model.build(mode="spores")

        with self.caplog_session(request) as caplog:
            with caplog.at_level(logging.INFO):
                model.solve(spores={"scoring_algorithm": request.param})
            log = caplog.text

        return model, log

    @pytest.fixture(scope="class")
    def spores_model_skip_baseline_and_log(self, request):
        """Iterate 2 times in SPORES mode having pre-computed the baseline results."""
        model = build_model({}, self.SPORES_OVERRIDES)
        model.build(mode="base")
        model.solve()

        model.build(mode="spores", force=True)
        with self.caplog_session(request) as caplog:
            with caplog.at_level(logging.INFO):
                model.solve(force=True, spores={"skip_baseline_run": True})
            log = caplog.text

        return model, log

    @pytest.fixture(scope="class")
    def spores_save_per_spore_path(self, tmp_path_factory):
        return tmp_path_factory.mktemp("outputs")

    @pytest.fixture(scope="class")
    def spores_model_save_per_spore_and_log(self, spores_save_per_spore_path, request):
        """Iterate 2 times in SPORES mode and save to file each time."""

        model = build_model({}, self.SPORES_OVERRIDES)
        model.build(mode="spores")

        with self.caplog_session(request) as caplog:
            with caplog.at_level(logging.INFO):
                model.solve(spores={"save_per_spore_path": spores_save_per_spore_path})
            log = caplog.text

        return model, log

    @pytest.fixture(
        scope="class",
        params=["integer", "relative_deployment", "random", "evolving_average"],
    )
    def spores_model_with_tracker(self, request):
        """Iterate 2 times in SPORES mode with a SPORES score tracking parameter."""
        model = build_model({}, f"{self.SPORES_OVERRIDES},spores_tech_tracking")
        model.build(mode="spores")
        model.solve(spores={"scoring_algorithm": request.param})

        return model

    @pytest.fixture(scope="class")
    def rerun_spores_log(self, request, spores_model_and_log):
        """Solve in spores mode a second time, to trigger new log messages."""
        with self.caplog_session(request) as caplog:
            with caplog.at_level(logging.INFO):
                spores_model_and_log[0].solve(force=True)
            return caplog.text

    @pytest.fixture
    def spores_infeasible_model(self):
        model = build_model({}, self.SPORES_OVERRIDES)
        # Add a negation, which makes the problem infeasible due to it being a minimisation problem that goes to minus infinity.
        infeasible_expression = (
            "sum(flow_cap * -1 * spores_score, over=[nodes, techs, carriers])"
        )
        model.build(
            add_math_dict={
                "objectives.min_spores.equations": [
                    {"expression": infeasible_expression}
                ]
            }
        )
        return model

    def test_backend_build_mode(self, spores_model_and_log):
        """Verify that we have run in spores mode"""
        spores_model, _ = spores_model_and_log
        assert spores_model.backend.config.mode == "spores"

    def test_io_save(self, spores_model_and_log, tmp_path):
        """Verify that we can save a model with SPORES results to file."""
        spores_model, _ = spores_model_and_log
        filepath = tmp_path / "test_io_save.nc"
        spores_model.to_netcdf(filepath)
        assert filepath.exists()

    def test_io_load(self, spores_model_and_log, tmp_path):
        """Verify that we can load a model with SPORES results from file."""
        spores_model, _ = spores_model_and_log
        filepath = tmp_path / "test_io_load.nc"
        spores_model.to_netcdf(filepath)
        new_model = calliope.read_netcdf(filepath)
        xr.testing.assert_allclose(spores_model._model_data, new_model._model_data)

    def test_spores_mode_success(self, spores_model_and_log_algorithms):
        """Solving in spores mode should lead to an optimal solution."""
        spores_model, _ = spores_model_and_log_algorithms
        assert spores_model.results.attrs["termination_condition"] == "optimal"

    def test_spores_fail_without_baseline(self):
        """You can't have SPORES without some initial, baseline results to work with."""
        model = build_model({}, self.SPORES_OVERRIDES)
        model.build()
        with pytest.raises(
            calliope.exceptions.ModelError,
            match="Cannot run SPORES without baseline results.",
        ):
            model.solve(spores={"skip_baseline_run": True})

    @pytest.mark.parametrize(
        "fixture",
        [
            "spores_model_and_log",
            "spores_model_skip_baseline_and_log",
            "spores_model_save_per_spore_and_log",
        ],
    )
    def test_spores_constraining_cost_is_baseline_obj(
        self, request, simple_supply_spores_ready, fixture
    ):
        """No matter how SPORES are initiated, the constraining cost (pre application of slack) should be the plan mode objective function value."""
        model, _ = request.getfixturevalue(fixture)
        assert (
            model.backend.get_parameter(
                "spores_baseline_cost", as_backend_objs=False
            ).item()
            == simple_supply_spores_ready._model_data["min_cost_optimisation"].item()
        )

    @pytest.mark.filterwarnings(
        "ignore:(?s).*Model solution was non-optimal:calliope.exceptions.BackendWarning"
    )
    def test_spores_break_on_infeasible(self, spores_infeasible_model):
        """On infeasibility, stop running and warn."""

        with pytest.warns(
            calliope.exceptions.ModelWarning, match="Stopping SPORES run after SPORE 1"
        ):
            spores_infeasible_model.solve()
        # We still get results, up to the point of infeasibility
        assert not set(
            spores_infeasible_model.results.spores.values
        ).symmetric_difference(["baseline"])

    def test_spores_mode_3_results(self, spores_model_and_log_algorithms):
        """Solving in spores mode should lead to 3 sets of results."""
        spores_model, _ = spores_model_and_log_algorithms
        assert not set(spores_model.results.spores.values).symmetric_difference(
            ["baseline", "1", "2"]
        )

    def test_spores_scores(self, spores_model_and_log_algorithms):
        """All techs should have a spores score defined."""
        spores_model, _ = spores_model_and_log_algorithms
        fill_gaps = ~spores_model._model_data.definition_matrix
        assert (
            spores_model._model_data.spores_score_cumulative.notnull() | fill_gaps
        ).all()

    def test_spores_caps(self, spores_model_and_log_algorithms):
        """There should be some changes in capacities between SPORES."""
        spores_model, _ = spores_model_and_log_algorithms
        n_spores = spores_model.config.solve.spores.number
        # as the spores dim is strings, it isn't ordered as one would expect
        order_dim = ["baseline"] + [f"{i}" for i in range(1, n_spores + 1)]
        cap_diffs = spores_model.results.flow_cap.sel(spores=order_dim).diff(
            dim="spores"
        )
        assert (cap_diffs != 0).any()

    def test_spores_algo_log(self, spores_model_and_log_algorithms):
        """The scoring algorithm being used should be logged correctly."""
        model, log = spores_model_and_log_algorithms
        assert (
            f"Running SPORES with `{model.config.solve.spores.scoring_algorithm}` scoring algorithm."
            in log
        )

    def test_spores_scores_never_decrease_integer_algo(self, spores_model_and_log):
        """SPORES scores can never decrease.

        This is not true for all algorithms (e.g. random scoring) so we test with integer scoring.
        """
        spores_model, _ = spores_model_and_log
        assert (
            spores_model._model_data.spores_score_cumulative.fillna(0).diff("spores")
            >= 0
        ).all()

    def test_spores_scores_increasing_with_cap_integer_algo(self, spores_model_and_log):
        """SPORES scores increase when a tech has a finite flow_cap in the previous iteration."""
        spores_model, _ = spores_model_and_log
        has_cap = spores_model.results.flow_cap > 0
        spores_score_increased = (
            spores_model._model_data.spores_score_cumulative.diff("spores") > 0
        )
        numpy.testing.assert_array_equal(
            has_cap.shift(spores=1).sel(spores=["1", "2"]), spores_score_increased
        )

    def test_use_tech_tracking(self, spores_model_with_tracker):
        """Tech tracking leads to only having spores scores for test_supply_elec."""
        sum_spores_score = (
            spores_model_with_tracker._model_data.spores_score_cumulative.groupby(
                "techs"
            ).sum(...)
        )
        assert (sum_spores_score.drop_sel(techs="test_supply_elec") == 0).all()

    @pytest.mark.usefixtures("spores_model_save_per_spore_and_log")
    def test_save_per_spore_file(self, spores_save_per_spore_path):
        """There are 4 files saved if saving per SPORE."""
        assert len(list(spores_save_per_spore_path.glob("*.nc"))) == 3

    @pytest.mark.usefixtures("spores_model_save_per_spore_and_log")
    @pytest.mark.parametrize("spore", ["baseline", "1", "2"])
    def test_save_per_spore_check_spore(self, spores_save_per_spore_path, spore):
        """We expect SPORES results to be saved to file once per iteration."""

        result = calliope.read_netcdf(
            (spores_save_per_spore_path / f"spore_{spore}").with_suffix(".nc")
        )
        assert result._model_data.spores.item() == spore

    @pytest.mark.usefixtures("spores_model_save_per_spore_and_log")
    @pytest.mark.parametrize("spore", ["baseline", "1", "2"])
    def test_save_per_spore_compare_results(
        self, spores_save_per_spore_path, spore, spores_model_and_log
    ):
        """We expect SPORES results saved per iteration to have the same results as those stored in memory."""

        result = calliope.read_netcdf(
            (spores_save_per_spore_path / f"spore_{spore}").with_suffix(".nc")
        )
        xr.testing.assert_equal(
            result.results.sel(spores=spore),
            spores_model_and_log[0].results.sel(spores=spore),
        )

    @pytest.mark.parametrize("spore", ["baseline", "1", "2"])
    def test_save_per_spore_log(self, spores_model_save_per_spore_and_log, spore):
        """We expect SPORES results saving to be logged."""
        _, log = spores_model_save_per_spore_and_log
        assert f"Saving SPORE {spore} to file." in log

    @pytest.mark.parametrize(
        "spores_model", ["spores_model_and_log", "spores_model_skip_baseline_and_log"]
    )
    def test_save_per_spore_without_path_log(self, request, spores_model):
        """We expect appropriate logs when SPORES results will not be saved due to lack of path."""
        _, log = request.getfixturevalue(spores_model)

        assert "Saving SPORE" not in log

    @pytest.mark.filterwarnings(
        "ignore:(?s).*Model solution was non-optimal:calliope.exceptions.BackendWarning"
    )
    @pytest.mark.filterwarnings(
        "ignore:(?s).*Stopping SPORES run after SPORE 1:calliope.exceptions.ModelWarning"
    )
    def test_save_per_spore_infeasible(self, caplog, tmp_path, spores_infeasible_model):
        """An infeasible SPORES objective should lead to no saved result file."""

        with caplog.at_level(logging.INFO):
            spores_infeasible_model.solve(spores={"save_per_spore_path": tmp_path})
        assert "No SPORE 1 results to save to file." in caplog.text
        assert (tmp_path / "spore_baseline.nc").exists()
        assert not (tmp_path / "spore_1.nc").exists()

    def test_skip_baseline_log(self, spores_model_skip_baseline_and_log):
        """Skipping baseline run should take existing results."""

        _, log = spores_model_skip_baseline_and_log

        assert "Using existing baseline model results." in log

    def test_save_per_spore_skip_cost_op(
        self, spores_model_and_log, spores_model_skip_baseline_and_log
    ):
        """Final result should be the same having skipped baseline."""

        model_all_solved_together, _ = spores_model_and_log
        model_baseline_solved_separately, _ = spores_model_skip_baseline_and_log
        assert model_all_solved_together._model_data.flow_cap.equals(
            model_baseline_solved_separately._model_data.flow_cap
        )

    def test_spores_relative_deployment_needs_max_param(self):
        """Can only run the `relative_deployment` algorithm if all techs have flow_cap_max."""
        model = build_model(
            {"techs.test_supply_elec.flow_cap_max": np.inf},
            f"{self.SPORES_OVERRIDES},spores_tech_tracking",
        )
        model.build(mode="spores")

        with pytest.raises(
            calliope.exceptions.BackendError,
            match="Cannot score SPORES with `relative_deployment`",
        ):
            model.solve(spores={"scoring_algorithm": "relative_deployment"})


class TestBuild:
    @pytest.fixture
    def init_model(self):
        return build_model({}, "simple_supply,two_hours,investment_costs")

    def test_add_math_dict_with_mode_math(self, init_model):
        init_model.build(
            add_math_dict={"constraints": {"system_balance": {"active": False}}},
            force=True,
        )
        assert len(init_model.backend.constraints) > 0
        assert "system_balance" not in init_model.backend.constraints


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
