# import matplotlib
# matplotlib.use('Qt5Agg')  # Prevents `Invalid DISPLAY variable` errors

import pytest
import tempfile

from calliope import examples
from calliope.utils import AttrDict

from calliope import analysis

from . import common
from .common import assert_almost_equal, solver, solver_io

import matplotlib.pyplot as plt
plt.switch_backend('agg')  # Prevents `Invalid DISPLAY variable` errors


class TestModel:
    @pytest.fixture(scope='module')
    def model(self):
        locations = """
            locations:
                1:
                    techs: ['ccgt', 'demand_power']
                    override:
                        ccgt:
                            constraints:
                                e_cap.max: 100
                        demand_power:
                            constraints:
                                r: -50
            metadata:
                map_boundary:
                    lower_left: {lat: 35, lon: -10}
                    upper_right: {lat: 45, lon: 5}
                location_coordinates:
                    1: {lat: 40, lon: -2}
            links:
        """
        config_run = """
            mode: plan
            model: ['{techs}', '{locations}']
            subset_t: ['2005-01-01', '2005-01-02']
        """
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(locations.encode('utf-8'))
            f.read()
            override_dict = AttrDict({
                'solver': solver,
                'solver_io': solver_io,
            })
            model = common.simple_model(config_run=config_run,
                                        config_locations=f.name,
                                        override=override_dict)
        model.run()
        return model

    @pytest.fixture(scope='module')
    def builtin_model(self):
        model = examples.NationalScale()
        model.run()
        return model

    def test_plot_carrier_production(self, model):
        # Just make sure this doesn't raise any exceptions
        analysis.plot_carrier_production(model.solution)

    def test_plot_timeseries(self, model):
        # Just make sure this doesn't raise any exceptions
        analysis.plot_timeseries(model.solution,
                                 model.solution['e'].loc[dict(c='power')].sum(dim='x'),
                                 carrier='power', demand='demand_power')

    def test_plot_installed_capacities(self, model):
        # Just make sure this doesn't raise any exceptions
        analysis.plot_installed_capacities(model.solution)

    def test_plot_transmission(self, model):
        # Just make sure this doesn't raise any exceptions
        analysis.plot_transmission(model.solution, map_resolution='c')

    def test_get_delivered_cost(self, model):
        # TODO this should be tested with a more complex model
        assert_almost_equal(analysis.get_delivered_cost(model.solution), 0.1)

    def test_get_levelized_cost(self, model):
        lcoe = analysis.get_levelized_cost(model.solution)
        assert_almost_equal(lcoe.at['ccgt'], 0.1)

    def test_get_group_share(self, model,):
        # TODO this should be tested with a more complex model
        share = analysis.get_group_share(model.solution, techs=['ccgt'], group='supply')
        assert share == 1.0

    def test_get_unmet_demand_hours(self, builtin_model):
        # TODO this should be tested with a more complex model
        unmet = analysis.get_unmet_demand_hours(builtin_model.solution)
        assert unmet == 1
