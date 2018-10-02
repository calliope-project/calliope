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
        assert model.results.energy_cap.to_pandas()['region1::ac_transmission:region2'] == approx(3423.336471)

        assert model.results.carrier_prod.sum('timesteps').to_pandas()['region1::ccgt::power'] == approx(272792.633882353)

        assert float(model.results.cost.sum()) == approx(32039968.579169072)

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

        assert model.results.carrier_prod.sum('timesteps').to_pandas()['region1::ccgt::power'] == approx(340154.9722)

        assert float(model.results.cost.sum()) == approx(47857169.51379299)
