import pyomo.core as po
import pytest

import calliope

from .common.util import build_test_model as build_model

approx = pytest.approx


@pytest.mark.skip(reason="to be reimplemented by comparison to LP files")
class TestCostMinimisationObjective:
    def test_nationalscale_minimize_emissions(self):
        model = calliope.examples.national_scale(
            scenario="minimize_emissions_costs",
            override_dict={"config.init.time_subset": ["2005-01-01", "2005-01-01"]},
        )
        model.build()
        model.solve()

        assert model.results.flow_cap.sel(nodes="region1_2", techs="csp") == approx(
            10000
        )
        assert model.results.flow_cap.sel(
            nodes="region1", techs="ac_transmission:region2"
        ) == approx(10000)

        assert model.results.flow_out.sum("timesteps").sel(
            carriers="power", nodes="region1", techs="ccgt"
        ) == approx(66530.36492823533)

        assert float(model.results.cost.sel(costs="emissions").sum()) == approx(
            13129619.1
        )

    def test_nationalscale_maximize_utility(self):
        model = calliope.examples.national_scale(
            scenario="maximize_utility_costs",
            override_dict={"config.init.time_subset": ["2005-01-01", "2005-01-01"]},
        )
        model.build()
        model.solve()

        assert model.results.flow_cap.sel(nodes="region1_2", techs="csp") == approx(
            10000.0
        )
        assert model.results.flow_cap.sel(
            nodes="region1", techs="ac_transmission:region2"
        ) == approx(10000.0)

        assert model.results.flow_out.sum("timesteps").sel(
            carriers="power", nodes="region1", techs="ccgt"
        ) == approx(115569.4354)

        assert float(model.results.cost.sum()) > 6.6e7

    @pytest.mark.parametrize(
        ("scenario", "cost_class", "weight"),
        [
            ("monetary_objective", ["monetary"], [1]),
            ("emissions_objective", ["emissions"], [1]),
            ("weighted_objective", ["monetary", "emissions"], [0.9, 0.1]),
        ],
    )
    def test_weighted_objective_results(self, scenario, cost_class, weight):
        model = build_model(model_file="weighted_obj_func.yaml", scenario=scenario)
        model.build()
        model.solve()
        assert sum(
            model.results.cost.loc[{"costs": cost_class[i]}].sum().item() * weight[i]
            for i in range(len(cost_class))
        ) == approx(po.value(model.objectives.min_cost_optimisation.item()))

    @pytest.mark.filterwarnings(
        "ignore:(?s).*The results of rerunning the backend model:calliope.exceptions.ModelWarning"
    )
    @pytest.mark.xfail(reason="Missing update_param functionality")
    def test_update_cost_classes_weights(self):
        model = build_model(
            model_file="weighted_obj_func.yaml", scenario="weighted_objective"
        )

        model.build()
        model.solve()
        obj_value = model.objectives.min_cost_optimisation.item()
        total_cost = model.results.cost.sum()
        model.backend.update_param("objective_cost_weights", {"monetary": 1.8})
        model.backend.update_param("objective_cost_weights", {"emissions": 0.2})

        new_model = model.backend.rerun()
        updated_obj_value = model.objectives.min_cost_optimisation.item()
        updated_total_cost = new_model.results.cost.sum()

        assert updated_obj_value == 2 * obj_value
        assert updated_total_cost == total_cost
