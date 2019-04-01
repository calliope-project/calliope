import pytest

import calliope
from calliope.test.common.util import build_test_model as build_model


@pytest.mark.filterwarnings('always::FutureWarning')
class TestFutureWarnings:

    def test_group_share_warning(self):
        override = {
            'model.group_share.test_supply_elec.carrier_prod_min.electricity': 0.85
        }
        expected_warning_msg = '`group_share` constraints will be removed in v0.7.0'
        with pytest.warns(FutureWarning, match=expected_warning_msg):
            build_model(override_dict=override, scenario='simple_supply,one_day')

    def test_default_cost_class_warning(self):
        expected_warning_msg = 'There will be no default cost class for the objective function in v0.7.0'
        with pytest.warns(FutureWarning, match=expected_warning_msg):
            build_model(scenario='simple_supply,one_day')

    def test_future_warning_for_charge_rate(self):
        with pytest.warns(FutureWarning, match='`charge_rate` is renamed to `energy_cap_per_storage_cap_max`'):
            calliope.examples.national_scale(
                override_dict={'techs.battery.constraints.charge_rate': 5}
            )
