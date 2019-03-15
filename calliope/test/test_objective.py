import pytest
from pytest import approx

import pyomo.core as po

import calliope
from calliope.test.common.util import build_test_model as build_model
from calliope.test.common.util import check_error_or_warning


class TestCostMinimisationObjective:
    def test_nationalscale_minimize_emissions(self):
        model = calliope.examples.national_scale(
            scenario='minimize_emissions_costs',
            override_dict={
                'model.subset_time': '2005-01-01'
            }
        )
        model.run()

        assert model.results.energy_cap.to_pandas()['region1-2::csp'] == approx(10000.0)
        assert model.results.energy_cap.to_pandas()['region1::ac_transmission:region2'] == approx(10000.0)

        assert model.results.carrier_prod.sum('timesteps').to_pandas()['region1::ccgt::power'] == approx(66530.36492823533)

        assert float(model.results.cost.loc[{'costs': 'emissions'}].sum()) == approx(13129619.1)

    def test_nationalscale_maximize_utility(self):
        model = calliope.examples.national_scale(
            scenario='maximize_utility_costs',
            override_dict={
                'model.subset_time': '2005-01-01'
            }
        )
        model.run()

        assert model.results.energy_cap.to_pandas()['region1-2::csp'] == approx(10000.0)
        assert model.results.energy_cap.to_pandas()['region1::ac_transmission:region2'] == approx(10000.0)

        assert model.results.carrier_prod.sum('timesteps').to_pandas()['region1::ccgt::power'] == approx(115569.4354)

        assert float(model.results.cost.sum()) > 6.6e7

    @pytest.mark.parametrize("override", [
        ({'run.objective_options.cost_class': {'monetary': None}}),
        ({'run.objective_options.cost_class': {'monetary': None, 'emissions': None}})
    ])
    def test_warn_on_no_weight(self, override):

        with pytest.warns(calliope.exceptions.ModelWarning) as warn:
            model = build_model(
                model_file='weighted_obj_func.yaml',
                override_dict=override
            )

        assert check_error_or_warning(warn, 'cost class monetary has weight = None, setting weight to 1')
        assert all(
            model.run_config['objective_options']['cost_class'][i] == 1
            for i in override['run.objective_options.cost_class'].keys()
        )

    @pytest.mark.parametrize("scenario,cost_class,weight", [
        ('monetary_objective', ['monetary'], [1]),
        ('emissions_objective', ['emissions'], [1]),
        ('weighted_objective', ['monetary', 'emissions'], [1, 0.1])
    ])
    def test_weighted_objective_results(self, scenario, cost_class, weight):
        model = build_model(
            model_file='weighted_obj_func.yaml',
            scenario=scenario
        )
        model.run()
        assert (
            sum(
                model.results.cost.loc[{'costs': cost_class[i]}].sum().item() * weight[i]
                for i in range(len(cost_class))
            ) == approx(po.value(model._backend_model.obj))
        )
