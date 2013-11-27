from __future__ import print_function
from __future__ import division


from calliope import techs

import common


class TestTechnology:
    def test_initialization(self):
        t = techs.Technology(name='TEST!')
        assert t.__repr__() == 'Generic technology (TEST!)'


class TestCspTechnology:
    def test_s_time_active(self):
        model = common.simple_model()
        model.set_option('csp.constraints.use_s_time', True)
        techs.CspTechnology(model)
        # Verify that s_cap_max is as calculated
        assert model.get_option('csp.constraints.s_cap_max') == 500

    def test_s_time_inactive(self):
        model = common.simple_model()
        # Verify the option doesn't exist yet
        assert model.get_option('csp.constraints.use_s_time') is False
        techs.CspTechnology(model)
        # Verify that s_cap_max is as set in techs_minimal.yaml
        assert model.get_option('csp.constraints.s_cap_max') == 1000
