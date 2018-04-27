from pytest import approx

import calliope

_OVERRIDE_FILE = calliope.examples._PATHS['national_scale'] + '/overrides.yaml'


class TestNationalScaleObjectives:
    def test_nationalscale_minimize_emissions(self):
        model = calliope.examples.national_scale(
            override_file=_OVERRIDE_FILE + ':minimize_emissions_costs'
        )
        model.run()

        assert model.results.energy_cap.to_pandas()['region1-2::csp'] == approx(10000.0)
        assert model.results.energy_cap.to_pandas()['region1::ac_transmission:region2'] == approx(3423.336471)

        assert model.results.carrier_prod.sum('timesteps').to_pandas()['region1::ccgt::power'] == approx(1099770.5894141179)

        assert float(model.results.cost.sum()) == approx(145746966.86280513)

    def test_nationalscale_maximize_utility(self):
        model = calliope.examples.national_scale(
            override_file=_OVERRIDE_FILE + ':maximize_utility_costs'
        )
        model.run()

        assert model.results.energy_cap.to_pandas()['region1-2::csp'] == approx(10000.0)
        assert model.results.energy_cap.to_pandas()['region1::ac_transmission:region2'] == approx(10000.0)

        assert model.results.carrier_prod.sum('timesteps').to_pandas()['region1::ccgt::power'] == approx(1439820.5542000004)

        assert float(model.results.cost.sum()) == approx(323248560.11846155)
