import pytest

import calliope

from .common.util import build_test_model as build_model

approx = pytest.approx


@pytest.mark.skip(reason="to be reimplemented by comparison to LP files")
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

        df_flow_out = (
            model.results.flow_out.loc[dict(carriers="power")]
            .sum("nodes")
            .sum("timesteps")
            .to_pandas()
        )

        prod_share = (
            df_flow_out.loc[["cold_fusion", "csp"]].sum()
            / df_flow_out.loc[["ccgt", "cold_fusion", "csp"]].sum()
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
            model._model_data.flow_cap.loc[{"techs": ["cold_fusion", "csp"]}].sum()
            / model._model_data.flow_cap.loc[
                {"techs": ["ccgt", "cold_fusion", "csp"]}
            ].sum()
        )

        assert cap_share == approx(0.2)

    def test_systemwide_equals(self):
        model = calliope.examples.national_scale(
            override_dict={
                "techs.ccgt.constraints.flow_cap_max_systemwide": 10000,
                "techs.ac_transmission.constraints.flow_cap_equals_systemwide": 6000,
            }
        )
        model.build()
        model.solve()
        # Check that setting `_equals` to a finite value leads to forcing
        assert model._model_data.flow_cap.loc[{"techs": "ccgt"}].sum() == 10000
        assert (
            model._model_data.flow_cap.loc[{"techs": "ac_transmission:region1"}].sum()
            == 6000
        )

    @pytest.mark.xfail(reason="no longer a constraint we're creating")
    def test_reserve_margin(self):
        calliope.examples.national_scale(scenario="reserve_margin")


@pytest.mark.filterwarnings("ignore:(?s).*Integer:calliope.exceptions.ModelWarning")
@pytest.mark.skip(reason="to be reimplemented by comparison to LP files")
class TestUrbanScaleMILP:
    def test_asynchronous_flow(self):
        def _get_flow(model, flow):
            return (
                model._model_data[f"carrier_{flow}"]
                .loc[{"techs": "heat_pipes:X1", "carriers": "heat"}]
                .to_pandas()
                .dropna(how="all")
            )

        m = calliope.examples.urban_scale()
        m.build()
        m.solve(zero_threshold=1e-6)
        _out = _get_flow(m, "out")
        _in = _get_flow(m, "in")
        assert any(((_in < 0) & (_out > 0)).any()) is True

        m_bin = calliope.examples.urban_scale(
            override_dict={"techs.heat_pipes.constraints.force_asynchronous_flow": True}
        )
        m_bin.build(solver_options={"mipgap": 0.05})
        m_bin.solve(zero_threshold=1e-6)
        _out = _get_flow(m_bin, "out")
        _in = _get_flow(m_bin, "in")
        assert any(((_in < 0) & (_out > 0)).any()) is False


@pytest.mark.skip(reason="to be reimplemented by comparison to LP files")
class TestModelSettings:
    @pytest.fixture
    def run_model(self):
        def _run_model(feasibility, cap_val):
            override_dict = {
                "nodes.a.techs": {"test_supply_elec": {}, "test_demand_elec": {}},
                "links.a,b.active": False,
                # pick a time subset where demand is uniformally -10 throughout
                "config.init.time_subset": [
                    "2005-01-01 06:00:00",
                    "2005-01-01 08:00:00",
                ],
                "data_definitions.bigM": 1e3,
                # Allow setting resource and flow_cap_max/equals to force infeasibility
                "techs.test_supply_elec.constraints": {
                    "source_use_equals": cap_val,
                    "flow_out_eff": 1,
                    "flow_cap_equals": 15,
                },
            }
            model = build_model(
                override_dict=override_dict, scenario="investment_costs"
            )
            model.build(ensure_feasibility=feasibility)
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
            model_unused_supply.backend.objectives.min_cost_optimisation.item()()
            - model_no_unmet.backend.objectives.min_cost_optimisation.item()()
            == approx(1e3 * 15)
        )

        assert (
            model_unmet_demand.backend.objectives.min_cost_optimisation.item()()
            - model_no_unmet.backend.objectives.min_cost_optimisation.item()()
            == approx(1e3 * 15)
        )

    @pytest.mark.parametrize("override", [5, 15])
    def test_expected_infeasible_result(self, override, run_model):
        model = run_model(False, override)

        assert "unmet_demand" not in model.backend.variables
        assert "unused_supply" not in model.backend.variables
        assert not model.results.attrs["termination_condition"] == "optimal"


@pytest.mark.skip(reason="to be reimplemented by comparison to LP files")
class TestEnergyCapacityPerStorageCapacity:
    @pytest.fixture
    def model_file(self):
        return "flow_cap_per_storage_cap.yaml"

    @pytest.mark.filterwarnings(
        "ignore:(?s).*`flow_cap_per_storage_cap_min/max/equals`:calliope.exceptions.ModelWarning"
    )
    def test_no_constraint_set(self, model_file):
        model = build_model(model_file=model_file)
        model.build()
        model.solve()
        assert model.results.termination_condition == "optimal"
        flow_capacity = (
            model._model_data.flow_cap.loc[{"techs": "my_storage"}].sum().item()
        )
        storage_capacity = (
            model._model_data.storage_cap.loc[{"techs": "my_storage"}].sum().item()
        )
        assert storage_capacity != pytest.approx(1 / 10 * flow_capacity)

    def test_equals(self, model_file):
        model = build_model(model_file=model_file, scenario="equals")
        model.build()
        model.solve()
        assert model.results.termination_condition == "optimal"
        flow_capacity = (
            model._model_data.flow_cap.loc[{"techs": "my_storage"}].sum().item()
        )
        storage_capacity = (
            model._model_data.storage_cap.loc[{"techs": "my_storage"}].sum().item()
        )
        assert storage_capacity == pytest.approx(1 / 10 * flow_capacity)

    def test_max(self, model_file):
        model = build_model(model_file=model_file, scenario="max")
        model.build()
        model.solve()
        assert model.results.termination_condition == "optimal"
        flow_capacity = (
            model._model_data.flow_cap.loc[{"techs": "my_storage"}].sum().item()
        )
        storage_capacity = (
            model._model_data.storage_cap.loc[{"techs": "my_storage"}].sum().item()
        )
        assert flow_capacity == pytest.approx(5)
        assert storage_capacity == pytest.approx(500)

    def test_min(self, model_file):
        model = build_model(model_file=model_file, scenario="min")
        model.build()
        model.solve()

        assert model.results.termination_condition == "optimal"
        flow_capacity = (
            model._model_data.flow_cap.loc[{"techs": "my_storage"}].sum().item()
        )
        storage_capacity = (
            model._model_data.storage_cap.loc[{"techs": "my_storage"}].sum().item()
        )
        assert flow_capacity == pytest.approx(10)
        assert storage_capacity == pytest.approx(10)
