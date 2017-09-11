import tempfile

from calliope.utils import AttrDict, depreciation_getter
from . import common
from .common import assert_almost_equal, solver, solver_io


def create_and_run_model(override=""):
    locations = """
        locations:
            1:
                techs: ['ccgt', 'demand_power', 'unmet_demand_power']
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
    def test_purchase(self):
        override1 = ""
        override2 = """
            override.locations.1.override.ccgt.costs.monetary.purchase: 20
        """
        model1 = create_and_run_model(override1)
        model2 = create_and_run_model(override2)
        assert str(model1.results.solver.termination_condition) == 'optimal'
        assert str(model2.results.solver.termination_condition) == 'optimal'
        sol1 = model1.solution
        sol2 = model2.solution
        time_res = model2.data['_time_res'].to_series()
        weights = model2.data['_weights'].to_series()
        depreciation = depreciation_getter(model2.get_option)('ccgt', '1', 'monetary')
        cost_difference = depreciation * (sum(time_res * weights) / 8760) * 20
        assert_almost_equal(sol1['costs'].loc[dict(k='monetary', y='ccgt')].sum(),
            sol2['costs'].loc[dict(k='monetary', y='ccgt')].sum() - cost_difference,
            tolerance=0.01)
        assert sol2.purchased_units.loc[dict(x='1', y='ccgt')] == 1

    def test_milp(self):
        override1 = """
            override.locations.1.override.ccgt.costs.monetary.purchase: 20
            override.locations.1.override.ccgt.constraints.units.max: 5
            override.locations.1.override.ccgt.constraints.e_cap_per_unit: 4
        """
        override2 = """
            override.locations.1.override.ccgt.constraints.units.max: 2
            override.locations.1.override.ccgt.constraints.e_cap_per_unit: 4
        """
        model1 = create_and_run_model(override1)
        model2 = create_and_run_model(override2)
        assert str(model1.results.solver.termination_condition) == 'optimal'
        assert_almost_equal(model1.get_var('units').loc['1', 'ccgt'],
                            3, tolerance=0.01)
        assert str(model2.results.solver.termination_condition) == 'optimal'
        assert model2.solution.e_cap.loc[dict(y='unmet_demand_power', x='1')] > 0
