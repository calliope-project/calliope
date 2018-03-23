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
