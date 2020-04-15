import pytest
from pytest import approx

import pyomo.core as po

import calliope
from calliope.test.common.util import build_test_model as build_model


class TestCostMinimisationObjective:
    def test_nationalscale_minimize_emissions(self):
        model = calliope.examples.national_scale(
            scenario="minimize_emissions_costs",
            override_dict={"model.subset_time": "2005-01-01"},
        )
        model.run()

        assert model.results.energy_cap.to_pandas()["region1-2::csp"] == approx(10000.0)
        assert model.results.energy_cap.to_pandas()[
            "region1::ac_transmission:region2"
        ] == approx(10000.0)

        assert model.results.carrier_prod.sum("timesteps").to_pandas()[
            "region1::ccgt::power"
        ] == approx(66530.36492823533)

        assert float(model.results.cost.loc[{"costs": "emissions"}].sum()) == approx(
            13129619.1
        )

    def test_nationalscale_maximize_utility(self):
        model = calliope.examples.national_scale(
            scenario="maximize_utility_costs",
            override_dict={"model.subset_time": "2005-01-01"},
        )
        model.run()

        assert model.results.energy_cap.to_pandas()["region1-2::csp"] == approx(10000.0)
        assert model.results.energy_cap.to_pandas()[
            "region1::ac_transmission:region2"
        ] == approx(10000.0)

        assert model.results.carrier_prod.sum("timesteps").to_pandas()[
            "region1::ccgt::power"
        ] == approx(115569.4354)

        assert float(model.results.cost.sum()) > 6.6e7

    @pytest.mark.parametrize(
        "scenario,cost_class,weight",
        [
            ("monetary_objective", ["monetary"], [1]),
            ("emissions_objective", ["emissions"], [1]),
            ("weighted_objective", ["monetary", "emissions"], [0.9, 0.1]),
        ],
    )
    def test_weighted_objective_results(self, scenario, cost_class, weight):
        model = build_model(model_file="weighted_obj_func.yaml", scenario=scenario)
        model.run()
        assert sum(
            model.results.cost.loc[{"costs": cost_class[i]}].sum().item() * weight[i]
            for i in range(len(cost_class))
        ) == approx(po.value(model._backend_model.obj))

    def test_update_cost_classes_weights(self):
        model = build_model(
            model_file="weighted_obj_func.yaml", scenario="weighted_objective"
        )

        model.run()
        obj_value = model._backend_model.obj()
        total_cost = model.results.cost.sum()
        model.backend.update_param("objective_cost_class", {"monetary": 1.8})
        model.backend.update_param("objective_cost_class", {"emissions": 0.2})

        new_model = model.backend.rerun()
        updated_obj_value = model._backend_model.obj()
        updated_total_cost = new_model.results.cost.sum()

        assert updated_obj_value == 2 * obj_value
        assert updated_total_cost == total_cost
