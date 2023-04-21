import pytest
from pytest import approx

import calliope
from calliope.test.common.util import build_test_model as build_model


class TestNationalScaleExampleModelSenseChecks:
    @pytest.mark.xfail(
        reason="Group constraints no longer working: to be replaced by custom constraints"
    )
    def test_group_prod_min(self):
        model = calliope.examples.national_scale(
            scenario="cold_fusion_with_production_share"
        )
        model.build()
        model.solve()

        df_carrier_prod = (
            model.results.carrier_prod.loc[dict(carriers="power")]
            .sum("nodes")
            .sum("timesteps")
            .to_pandas()
        )

        prod_share = (
            df_carrier_prod.loc[["cold_fusion", "csp"]].sum()
            / df_carrier_prod.loc[["ccgt", "cold_fusion", "csp"]].sum()
        )

        assert prod_share == approx(0.85)

    @pytest.mark.xfail(
        reason="Group constraints no longer working: to be replaced by custom constraints"
    )
    def test_group_cap_max(self):
        model = calliope.examples.national_scale(
            scenario="cold_fusion_with_capacity_share"
        )
        model.build()
        model.solve()

        cap_share = (
            model._model_data.energy_cap.loc[{"techs": ["cold_fusion", "csp"]}].sum()
            / model._model_data.energy_cap.loc[
                {"techs": ["ccgt", "cold_fusion", "csp"]}
            ].sum()
        )

        assert cap_share == approx(0.2)

    def test_systemwide_equals(self):
        model = calliope.examples.national_scale(
            override_dict={
                "techs.ccgt.constraints.energy_cap_max_systemwide": 10000,
                "techs.ac_transmission.constraints.energy_cap_equals_systemwide": 6000,
            }
        )
        model.build()
        model.solve()
        # Check that setting `_equals` to a finite value leads to forcing
        assert model._model_data.energy_cap.loc[{"techs": "ccgt"}].sum() == 10000
        assert (
            model._model_data.energy_cap.loc[{"techs": "ac_transmission:region1"}].sum()
            == 6000
        )

    @pytest.mark.xfail(reason="no longer a constraint we're creating")
    def test_reserve_margin(self):
        model = calliope.examples.national_scale(scenario="reserve_margin")


@pytest.mark.filterwarnings("ignore:(?s).*Integer:calliope.exceptions.ModelWarning")
class TestUrbanScaleMILP:
    def test_asynchronous_prod_con(self):
        def _get_prod_con(model, prod_con):
            return (
                model._model_data[f"carrier_{prod_con}"]
                .loc[{"techs": "heat_pipes:X1", "carriers": "heat"}]
                .to_pandas()
                .dropna(how="all")
            )

        m = calliope.examples.urban_scale(override_dict={"run.zero_threshold": 1e-6})
        m.build()
        m.solve()
        _prod = _get_prod_con(m, "prod")
        _con = _get_prod_con(m, "con")
        assert any(((_con < 0) & (_prod > 0)).any()) is True

        m_bin = calliope.examples.urban_scale(
            override_dict={
                "techs.heat_pipes.constraints.force_asynchronous_prod_con": True,
                "run.solver_options.mipgap": 0.05,
                "run.zero_threshold": 1e-6,
            }
        )
        m_bin.build()
        m_bin.solve()
        _prod = _get_prod_con(m_bin, "prod")
        _con = _get_prod_con(m_bin, "con")
        assert any(((_con < 0) & (_prod > 0)).any()) is False


class TestModelSettings:
    @pytest.fixture
    def run_model(self):
        def _run_model(feasibility, cap_val):
            override_dict = {
                "nodes.a.techs": {"test_supply_elec": {}, "test_demand_elec": {}},
                "links.a,b.exists": False,
                # pick a time subset where demand is uniformally -10 throughout
                "model.subset_time": ["2005-01-01 06:00:00", "2005-01-01 08:00:00"],
                "run.ensure_feasibility": feasibility,
                "run.bigM": 1e3,
                # Allow setting resource and energy_cap_max/equals to force infeasibility
                "techs.test_supply_elec.constraints": {
                    "resource": cap_val,
                    "energy_eff": 1,
                    "energy_cap_equals": 15,
                },
                "techs.test_supply_elec.switches.force_resource": True,
            }
            model = build_model(
                override_dict=override_dict, scenario="investment_costs"
            )
            model.build()
            model.solve()
            return model

        return _run_model

    @pytest.fixture
    def model_no_unmet(self, run_model):
        return run_model(True, 10)

    @pytest.fixture
    def model_unmet_demand(self, run_model):
        return run_model(True, 5)

    @pytest.fixture
    def model_unused_supply(self, run_model):
        return run_model(True, 15)

    def test_unmet_demand_zero(self, model_no_unmet):
        # Feasible case, but unmet_demand/unused_supply is not deleted
        for i in ["unmet_demand", "unused_supply"]:
            assert i in model_no_unmet.backend.variables
        assert "unmet_demand" in model_no_unmet.results.data_vars.keys()
        assert "unused_supply" not in model_no_unmet.results.data_vars.keys()
        assert (model_no_unmet.results["unmet_demand"] == 0).all()

    def test_unmet_demand_nonzero(self, model_unmet_demand):
        # Infeasible case, unmet_demand is required
        assert "unmet_demand" in model_unmet_demand.backend.variables
        assert "unused_supply" in model_unmet_demand.backend.variables
        assert model_unmet_demand.results["unmet_demand"].sum() == 15
        assert "unused_supply" not in model_unmet_demand.results.data_vars.keys()

    def test_unmet_supply_nonzero(self, model_unused_supply):
        # Infeasible case, unused_supply is required
        assert "unmet_demand" in model_unused_supply.backend.variables
        assert "unused_supply" in model_unused_supply.backend.variables
        assert model_unused_supply.results["unmet_demand"].sum() == -15
        assert "unused_supply" not in model_unused_supply.results.data_vars.keys()

    def test_expected_impact_on_objective_function_value(
        self, model_no_unmet, model_unmet_demand, model_unused_supply
    ):
        assert (
            model_unused_supply.backend.objectives.minmax_cost_optimisation
            - model_no_unmet.backend.objectives.minmax_cost_optimisation
            == approx(1e3 * 15)
        )

        assert (
            model_unmet_demand.backend.objectives.minmax_cost_optimisation
            - model_no_unmet.backend.objectives.minmax_cost_optimisation
            == approx(1e3 * 15)
        )

    @pytest.mark.parametrize("override", (5, 15))
    def test_expected_infeasible_result(self, override, run_model):
        model = run_model(False, override)

        assert "unmet_demand" not in model.backend.variables
        assert "unused_supply" not in model.backend.variables
        assert not model.results.attrs["termination_condition"] == "optimal"


@pytest.mark.xfail(
    reason="Not sure why failing: will need to be checked when turned into LP files"
)
class TestEnergyCapacityPerStorageCapacity:
    @pytest.fixture
    def model_file(self):
        return "energy_cap_per_storage_cap.yaml"

    @pytest.mark.filterwarnings(
        "ignore:(?s).*`energy_cap_per_storage_cap_min/max/equals`:calliope.exceptions.ModelWarning"
    )
    def test_no_constraint_set(self, model_file):
        model = build_model(model_file=model_file)
        model.build()
        model.solve()
        assert model.results.termination_condition == "optimal"
        energy_capacity = (
            model._model_data.energy_cap.loc[{"techs": "my_storage"}].sum().item()
        )
        storage_capacity = (
            model._model_data.storage_cap.loc[{"techs": "my_storage"}].sum().item()
        )
        assert energy_capacity == pytest.approx(10)
        assert storage_capacity == pytest.approx(180)
        assert storage_capacity != pytest.approx(1 / 10 * energy_capacity)

    def test_equals(self, model_file):
        model = build_model(model_file=model_file, scenario="equals")
        model.build()
        model.solve()
        assert model.results.termination_condition == "optimal"
        energy_capacity = (
            model._model_data.energy_cap.loc[{"techs": "my_storage"}].sum().item()
        )
        storage_capacity = (
            model._model_data.storage_cap.loc[{"techs": "my_storage"}].sum().item()
        )
        assert storage_capacity == pytest.approx(1 / 10 * energy_capacity)

    def test_max(self, model_file):
        model = build_model(model_file=model_file, scenario="max")
        model.build()
        model.solve()
        assert model.results.termination_condition == "optimal"
        energy_capacity = (
            model._model_data.energy_cap.loc[{"techs": "my_storage"}].sum().item()
        )
        storage_capacity = (
            model._model_data.storage_cap.loc[{"techs": "my_storage"}].sum().item()
        )
        assert energy_capacity == pytest.approx(10)
        assert storage_capacity == pytest.approx(1000)

    def test_min(self, model_file):
        model = build_model(model_file=model_file, scenario="min")
        model.build()
        model.solve()

        assert model.results.termination_condition == "optimal"
        energy_capacity = (
            model._model_data.energy_cap.loc[{"techs": "my_storage"}].sum().item()
        )
        storage_capacity = (
            model._model_data.storage_cap.loc[{"techs": "my_storage"}].sum().item()
        )
        assert energy_capacity == pytest.approx(180)
        assert storage_capacity == pytest.approx(180)

    @pytest.mark.skip(reason="Not expecting operate mode to work at the moment")
    def test_operate_mode(self, model_file):
        model = build_model(model_file=model_file, scenario="operate_mode_min")
        model.build()
        model.solve()

    @pytest.mark.parametrize(
        "horizon_window", [(24, 24), (48, 48), (72, 48), (144, 24)]
    )
    @pytest.mark.skip(reason="operate mode not yet expected to run")
    def test_operate_mode_horizon_window(self, model_file, horizon_window):
        horizon, window = horizon_window
        override_dict = {
            "model.subset_time": ["2005-01-01", "2005-01-05"],
            "run.operation.horizon": horizon,
            "run.operation.window": window,
        }
        model = build_model(
            model_file=model_file,
            scenario="operate_mode_min",
            override_dict=override_dict,
        )
        model.build()
        model.solve()
