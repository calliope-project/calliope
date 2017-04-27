import numpy as np
import pandas as pd
import pytest
import tempfile

import calliope

from . import common
from .common import assert_almost_equal


class TestInitialization:
    def test_model_initialization_default(self):
        model = calliope.examples.NationalScale()
        assert hasattr(model, 'data')
        assert hasattr(model, 'config_run')
        assert hasattr(model, 'config_model')
        assert model.config_run.mode == 'plan'

    def test_model_initialization_follow_import_statements(self):
        model = calliope.examples.NationalScale()
        assert 'techs' in model.config_model

    def test_model_initialization_follow_nested_import_statements(self):
        model = calliope.examples.NationalScale()
        assert 'links' in model.config_model

    def test_model_initialization_override_dict(self):
        override = {'output.save': True}
        with pytest.raises(AssertionError):
            calliope.examples.NationalScale(override=override)

    def test_model_initialization_override_attrdict(self):
        override = calliope.utils.AttrDict({'output': {'save': True}})
        model = calliope.examples.NationalScale(override=override)
        assert model.config_run.output.save is True

    def test_model_initialization_simple_model(self):
        common.simple_model()

    def test_gettimeres_1hourly(self):
        model = common.simple_model()
        assert model.get_timeres() == 1

    def test_gettimeres_6hourly(self):
        path = common._add_test_path('common/t_6h')
        model = common.simple_model(path=path)
        assert model.get_timeres() == 6

    def test_gettimeres_verify_1hourly(self):
        model = common.simple_model()
        assert model.get_timeres(verify=True) == 1

    def test_gettimeres_verify_erroneous(self):
        path = common._add_test_path('common/t_erroneous')
        model = common.simple_model(path=path)
        with pytest.raises(AssertionError):
            model.get_timeres(verify=True)

    @pytest.fixture
    def sine_wave(self):
        return pd.DataFrame((np.sin(np.arange(0, 10, 0.1)) + 1.0) * 5/2 + 5)

    def test_scale_to_peak_positive(self, sine_wave):
        model = common.simple_model()
        scaled = model.scale_to_peak(sine_wave, 100)
        assert_almost_equal(float(scaled.max()), 100, tolerance=0.01)
        assert_almost_equal(float(scaled.min()), 50, tolerance=0.01)

    def test_scale_to_peak_negative(self, sine_wave):
        model = common.simple_model()
        df = sine_wave * -1
        scaled = model.scale_to_peak(df, -100)
        assert_almost_equal(float(scaled.max()), -50, tolerance=0.01)
        assert_almost_equal(float(scaled.min()), -100, tolerance=0.01)

    def test_scale_to_peak_scale_time_res_true(self, sine_wave):
        path = common._add_test_path('common/t_6h')
        model = common.simple_model(path=path)
        scaled = model.scale_to_peak(sine_wave, 100)
        assert_almost_equal(float(scaled.max()), 600, tolerance=0.1)
        assert_almost_equal(float(scaled.min()), 300, tolerance=0.1)

    def test_scale_to_peak_scale_time_res_false(self, sine_wave):
        path = common._add_test_path('common/t_6h')
        model = common.simple_model(path=path)
        scaled = model.scale_to_peak(sine_wave, 100, scale_time_res=False)
        assert_almost_equal(float(scaled.max()), 100, tolerance=0.1)
        assert_almost_equal(float(scaled.min()), 50, tolerance=0.1)

    def test_scale_to_peak_positive_and_negative(self, sine_wave):
        model = common.simple_model()
        df = sine_wave - 6
        scaled = model.scale_to_peak(df, 10)
        assert_almost_equal(float(scaled.max()), 10, tolerance=0.01)
        assert_almost_equal(float(scaled.min()), -2.5, tolerance=0.01)

    def test_initialize_parents_defaults(self):
        override = """
                    override:
                        techs:
                            bad_tech:
                                parent: defaults
                    """
        override = calliope.utils.AttrDict.from_yaml_string(override)
        with pytest.raises(calliope.exceptions.ModelError):
            model = common.simple_model(override=override)

    def test_initialize_sets_timesteps(self):
        model = common.simple_model()
        daterange = pd.Index(pd.date_range('2005-01-01 00:00', '2005-02-28 23:00', freq='1H'))
        assert (model._sets['t'] == daterange).all()
        assert model._sets['t'][0].minute == 0
        assert model._sets['t'][0].hour == 0
        assert model._sets['t'][0].day == 1
        assert model._sets['t'][0].month == 1
        assert model.data.attrs['time_res'] == 1
        assert model.data['_time_res'].to_series().tolist() == [1] * 1416
        assert model.data.attrs['startup_time_bounds'] == pd.Timestamp('2005-01-01 12:00')

    def test_initialize_sets_timesteps_subset(self):
        config_run = """
                        mode: plan
                        model: ['{techs}', '{locations}']
                        subset_t: ['2005-01-02', '2005-01-03']
                    """
        model = common.simple_model(config_run=config_run)
        daterange = pd.Index(pd.date_range('2005-01-02 00:00', '2005-01-03 23:00', freq='1H'))
        assert (model._sets['t'] == daterange).all()
        assert model._sets['t'][0].minute == 0
        assert model._sets['t'][0].hour == 0
        assert model._sets['t'][0].day == 2
        assert model._sets['t'][0].month == 1
        assert model.data.attrs['time_res'] == 1
        assert model.data['_time_res'].to_series().tolist() == [1] * 48
        assert model.data.attrs['startup_time_bounds'] == pd.Timestamp('2005-01-02 12:00')

    def test_initialize_sets_technologies(self):
        model = common.simple_model()
        y = ['ccgt', 'csp', 'demand_power', 'unmet_demand_power']
        assert sorted(model._sets['y']) == y

    def test_initialize_sets_technologies_loc_invalid_tech(self):
        locations = """
                    override:
                        locations:
                            demand:
                                techs: ['']
                    """
        override = calliope.utils.AttrDict.from_yaml_string(locations)
        with pytest.raises(calliope.exceptions.ModelError):
            model = common.simple_model(override=override)

    def test_initialize_sets_technologies_subset(self):
        config_run = """
                        mode: plan
                        model: ['{techs}', '{locations}']
                        subset_y: ['ccgt', 'demand_power']
                    """
        model = common.simple_model(config_run=config_run)
        assert sorted(model._sets['y']) == ['ccgt', 'demand_power']

    def test_initialize_sets_technologies_too_large_subset(self):
        config_run = """
                        mode: plan
                        model: ['{techs}', '{locations}']
                        subset_y: ['ccgt', 'demand_power', 'foo', 'bar']
                    """
        model = common.simple_model(config_run=config_run)
        assert sorted(model._sets['y']) == ['ccgt', 'demand_power']

    def test_initialize_sets_carriers(self):
        model = common.simple_model()
        assert sorted(model._sets['c']) == ['power']

    # TODO more extensive tests for carriers

    def test_initialize_sets_locations(self):
        model = common.simple_model()
        assert sorted(model._sets['x']) == ['1', '2', 'demand']

    def test_initialize_sets_locations_subset(self):
        config_run = """
                        mode: plan
                        model: ['{techs}', '{locations}']
                        subset_x: ['1', 'demand']
                    """
        model = common.simple_model(config_run=config_run)
        assert sorted(model._sets['x']) == ['1', 'demand']

    def test_initialize_sets_locations_too_large_subset(self):
        config_run = """
                        mode: plan
                        model: ['{techs}', '{locations}']
                        subset_x: ['1', 'demand', 'foo', 'bar']
                    """
        model = common.simple_model(config_run=config_run)
        assert sorted(model._sets['x']) == ['1', 'demand']

    def test_initialize_locations_matrix(self):
        model = common.simple_model()
        cols = ['_level', '_override.ccgt.constraints.e_cap.max',
                '_within', 'ccgt', 'csp', 'demand_power',
                'unmet_demand_power']
        assert sorted(model._locations.columns) == cols
        assert (sorted(model._locations.index.tolist())
                == ['1', '2', 'demand'])

    @pytest.fixture
    def model_transmission(self):
        locations = """
            locations:
                demand:
                    techs: ['demand_power']
                1,2:
                    techs: ['ccgt', 'csp']
            links:
                1,2:
                    hvac:
                        constraints:
                            e_cap.max: 100
        """
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(locations.encode('utf-8'))
            print(f.read())
            model = common.simple_model(config_locations=f.name)
        return model

    def test_initialize_sets_locations_with_transmission(self,
                                                         model_transmission):
        model = model_transmission
        y =['ccgt', 'csp', 'demand_power', 'hvac:1', 'hvac:2']
        assert sorted(model._sets['y']) == y

    def test_initialize_locations_matrix_with_transmission(self,
                                                           model_transmission):
        model = model_transmission
        cols = ['_level',
                '_override.hvac:1.constraints.e_cap.max',
                '_override.hvac:2.constraints.e_cap.max',
                '_within', 'ccgt', 'csp', 'demand_power',
                'hvac:1', 'hvac:2']
        locations = model._locations
        assert sorted(locations.columns) == cols
        assert (sorted(locations.index.tolist())
                == ['1', '2', 'demand'])
        assert locations.at['1', '_override.hvac:2.constraints.e_cap.max'] == 100
        assert np.isnan(locations.at['1', '_override.hvac:1.constraints.e_cap.max'])
        assert locations.at['2', '_override.hvac:1.constraints.e_cap.max'] == 100

    def test_read_data_supply_r_negative_check(self):
        path = common._add_test_path('common/t_positive_demand')
        override = ('override.techs.demand_power.'
                    'constraints.r: file=demand-sin_r.csv')
        override = calliope.utils.AttrDict.from_yaml_string(override)
        with pytest.raises(AssertionError):
            model = common.simple_model(path=path, override=override)

class TestOptions:
    def test_get_option(self):
        model = common.simple_model()
        assert model.get_option('ccgt.constraints.e_cap.max') == 50

    def test_get_option_default(self):
        model = common.simple_model()
        assert model.get_option('ccgt.depreciation.plant_life') == 25

    def test_get_option_default_unavailable(self):
        model = common.simple_model()
        with pytest.raises(calliope.exceptions.OptionNotSetError):
            model.get_option('ccgt.depreciation.foo')

    def test_get_option_specify_default_inexistent(self):
        model = common.simple_model()
        assert model.get_option('ccgt.depreciation.foo',
                                default='ccgt.depreciation.plant_life') == 25

    def test_get_option_specify_default_exists_but_false(self):
        model = common.simple_model()
        assert model.get_option('ccgt.constraints.e_eff_ref',
                                default='ccgt.depreciation.plant_life') is False

    def test_get_option_location(self):
        model = common.simple_model()
        assert model.get_option('ccgt.constraints.e_cap.max', 'demand') == 50
        assert model.get_option('ccgt.constraints.e_cap.max', '1') == 100

    def test_get_option_location_default(self):
        model = common.simple_model()
        assert model.get_option('ccgt.depreciation.plant_life', '1') == 25

    def test_get_option_location_default_unavailable(self):
        model = common.simple_model()
        with pytest.raises(calliope.exceptions.OptionNotSetError):
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
        assert model.get_option('ccgt.constraints.e_cap.max') == 50
        model.set_option('ccgt.constraints.e_cap.max', 'foo')
        assert model.get_option('ccgt.constraints.e_cap.max') == 'foo'
