# import matplotlib
# matplotlib.use('Qt5Agg')  # Prevents `Invalid DISPLAY variable` errors

import pytest
import tempfile

from calliope import Model
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
                map_boundary: [-10, 35, 5, 45]
                location_coordinates:
                    1: [40, -2]
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
        model = Model()
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

    def test_get_group_share(self, model):
        # TODO this should be tested with a more complex model
        share = analysis.get_group_share(model.solution, techs=['ccgt'])
        assert share == 1.0

    def test_get_unmet_demand_hours(self, builtin_model):
        # TODO this should be tested with a more complex model
        unmet = analysis.get_unmet_demand_hours(builtin_model.solution)
        assert unmet == 1

    def test_recompute_levelized_costs(self, model):
        # Cost in solution
        sol = model.solution
        assert_almost_equal(sol['summary'].to_pandas().loc['ccgt', 'levelized_cost_monetary'], 0.1)
        # Recomputed cost must be the same
        dm = analysis.SolutionModel(model.solution)
        recomputed = dm.recompute_levelized_costs('ccgt')
        assert_almost_equal(recomputed['total'], 0.1)

    def test_recompute_levelized_costs_after_changes(self, model):
        # Make changes
        dm = analysis.SolutionModel(model.solution)
        dm.config_model.techs.ccgt.costs.monetary.e_cap = 50
        dm.config_model.techs.ccgt.costs.monetary.om_fuel = 1.0
        # Recomputed cost
        recomputed = dm.recompute_levelized_costs('ccgt')
        assert_almost_equal(recomputed['total'], 1.0, tolerance=0.001)
