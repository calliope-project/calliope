from __future__ import print_function
from __future__ import division


import pytest

import common


class TestInitialization:
    def test_initialization_attributes(self):
        model = common.simple_model()
        assert hasattr(model, 'data')
        assert hasattr(model, 'config_run')
        assert hasattr(model, 'config_model')

    def test_model_initialization_default(self):
        pass

    def test_model_initialization_simple_model(self):
        pass

    def test_initialize_techs(self):
        pass

    def test_gettimeres(self):
        pass

    def test_scale_to_peak(self):
        pass

    def test_initialize_sets(self):
        pass

    def test_read_data(self):
        pass


class TestOptions:
    def test_get_option(self):
        model = common.simple_model()
        assert model.get_option('ccgt.constraints.e_cap_max') == 50

    def test_get_option_default(self):
        model = common.simple_model()
        assert model.get_option('ccgt.depreciation.plant_life') == 25

    def test_get_option_default_unavailable(self):
        model = common.simple_model()
        with pytest.raises(KeyError):
            model.get_option('ccgt.depreciation.foo')

    def test_get_option_node(self):
        model = common.simple_model()
        assert model.get_option('ccgt.constraints.e_cap_max', 'demand') == 50
        assert model.get_option('ccgt.constraints.e_cap_max', '1') == 100

    def test_get_option_node_default(self):
        model = common.simple_model()
        assert model.get_option('ccgt.depreciation.plant_life', '1') == 25

    def test_get_option_node_default_unavailable(self):
        model = common.simple_model()
        with pytest.raises(KeyError):
            model.get_option('ccgt.depreciation.foo', '1')

    def test_set_option(self):
        model = common.simple_model()
        with pytest.raises(KeyError):  # Ensure that option doesn't exist yet
            model.config_model.techs.test.option
        model.set_option('test.option', True)  # Set option
        assert model.config_model.techs.test.option is True  # Exists now?

    def test_set_get_option(self):
        model = common.simple_model()
        model.set_option('test.option', 'foo')
        assert model.get_option('test.option') == 'foo'

    def test_get_set_get_option(self):
        model = common.simple_model()
        assert model.get_option('ccgt.constraints.e_cap_max') == 50
        model.set_option('ccgt.constraints.e_cap_max', 'foo')
        assert model.get_option('ccgt.constraints.e_cap_max') == 'foo'
