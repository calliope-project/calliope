import pytest
from pytest import approx

import calliope

_OVERRIDE_FILE = calliope.examples._PATHS['national_scale'] + '/overrides.yaml'


class TestNationalScaleExampleModelSenseChecks:
    def test_group_share_prod_min(self):
        model = calliope.examples.national_scale(
            override_file=_OVERRIDE_FILE + ':cold_fusion,group_share_cold_fusion_prod'
        )
        model.run()

        df_carrier_prod = (
            model.get_formatted_array('carrier_prod')
                 .loc[dict(carriers='power')].sum('locs').sum('timesteps')
                 .to_pandas()
        )

        prod_share = (
            df_carrier_prod.loc[['cold_fusion', 'csp']].sum() /
            df_carrier_prod.loc[['ccgt', 'cold_fusion', 'csp']].sum()
        )

        assert prod_share == approx(0.85)

    def test_group_share_cap_max(self):
        model = calliope.examples.national_scale(
            override_file=_OVERRIDE_FILE + ':cold_fusion,group_share_cold_fusion_cap'
        )
        model.run()

        cap_share = (
            model.get_formatted_array('energy_cap').to_pandas().loc[:, ['cold_fusion', 'csp']].sum().sum() /
            model.get_formatted_array('energy_cap').to_pandas().loc[:, ['ccgt', 'cold_fusion', 'csp']].sum().sum()
        )

        assert cap_share == approx(0.2)

    def test_systemwide_equals(self):
        model = calliope.examples.national_scale(
            override_dict={'techs.ccgt.constraints.energy_cap_max_systemwide': 10000}
        )
        model.run()
        # Check that setting `_equals` to a finite value leads to forcing
        assert (
            model.get_formatted_array('energy_cap').loc[{'techs': 'ccgt'}].sum() == 10000
        )

    def test_reserve_margin(self):
        model = calliope.examples.national_scale(
            override_file=_OVERRIDE_FILE + ':reserve_margin'
        )

        model.run()

        # constraint_string = '-Inf : -1.1 * ( carrier_con[region1::demand_power::power,2005-01-05 16:00:00] + carrier_con[region2::demand_power::power,2005-01-05 16:00:00] ) / timestep_resolution[2005-01-05 16:00:00] - energy_cap[region1::ccgt] - energy_cap[region1-3::csp] - energy_cap[region1-2::csp] - energy_cap[region1-1::csp] :   0.0'

        # FIXME: capture Pyomo's print output...
        # assert constraint_string in model._backend_model.reserve_margin_constraint.pprint()

        assert float(model.results.cost.sum()) == approx(282487.35489)
