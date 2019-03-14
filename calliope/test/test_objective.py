from pytest import approx

import calliope


class TestNationalScaleObjectives:
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
