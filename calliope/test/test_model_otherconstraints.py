import tempfile
import numpy as np

from calliope.utils import AttrDict
from . import common
from .common import assert_almost_equal, solver, solver_io

def create_and_run_model(override=""):
    locations = """
        locations:
            1:
                techs: ['ccgt','demand_power', 'unmet_demand_power']
                override:
                    ccgt:
                        constraints:
                            e_cap.max: 100
                    demand_power:
                        x_map: '1: demand'
                        constraints:
                            r: file=demand-sin_r.csv
        links:
    """
    config_run = """
        mode: plan
        model: ['{techs}', '{locations}']
        subset_t: ['2005-01-01', '2005-01-03']
    """

    override = AttrDict.from_yaml_string(override)
    override.set_key('solver', solver)
    override.set_key('solver_io', solver_io)
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(locations.encode('utf-8'))
        f.read()
        model = common.simple_model(config_run=config_run,
                                    config_locations=f.name,
                                    override=override)
    model.run()
    return model


class TestModel:
    def test_system_margin(self):
        override1 = """
            override.system_margin.power: 0
        """
        override2 = """
            override.system_margin.power: 0.5
        """
        model1 = create_and_run_model(override1)
        model2 = create_and_run_model(override2)
        assert str(model1.results.solver.termination_condition) == 'optimal'
        assert str(model2.results.solver.termination_condition) == 'optimal'
        sol1 = model1.solution
        assert_almost_equal(sol1['e'].loc[dict(c='power', y='ccgt')].sum(), 548.79, tolerance=0.01)
        # assert_almost_equal(sol1['e_cap'].loc[dict(y='ccgt')].sum(), 9.5, tolerance=0.01)
        assert_almost_equal(sol1['e_cap'].sum(), 20.00, tolerance=0.01)
        sol2 = model2.solution
        assert_almost_equal(sol2['e'].loc[dict(c='power', y='ccgt')].sum(), 548.79, tolerance=0.01)
        assert_almost_equal(sol2['e_cap'].sum(), 25.00, tolerance=0.01)

    def test_ramping_rate(self):
        override = """
            override.constraints:
                    - constraints.optional.ramping_rate
            override.locations.1.techs: ['ccgt','demand_power',
                'unmet_demand_power', 'test_conversion_plus', 'demand_low_T',
                'demand_V_low_T', 'supply_gas', 'test_conversion', 'demand_heat']
            override.techs.ccgt.constraints.e_ramping: 0.01
            override.techs.test_conversion.constraints.e_ramping: 0.01
            override.locations.1.override.demand_heat:
                                x_map: '1: demand'
                                constraints:
                                    r: file=demand-sin_r.csv
            override.techs.test_conversion_plus.constraints.e_ramping: 0.01
            override.locations.1.override.demand_low_T:
                                x_map: '1: demand'
                                constraints:
                                    r: file=demand-sin_r.csv
            override.locations.1.override.demand_V_low_T:
                                x_map: '1: demand'
                                constraints:
                                    r: file=demand-sin_r.csv
        """
        model = create_and_run_model(override)
        assert str(model.results.solver.termination_condition) == 'optimal'
        sol = model.solution
        tech_carrier_pair = {'ccgt':'power', 'test_conversion':'heat',
            'test_conversion_plus': 'low_T'}
        for k, v in tech_carrier_pair.items():
            e_cap = sol.e_cap.loc[dict(y=k)].sum()
            c_prod = sol.e.loc[dict(c=v, y=k, x='1')]
            assert max(np.absolute(c_prod-c_prod.shift())) <= e_cap/10
