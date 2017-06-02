import pytest
import tempfile

from calliope import examples
from calliope.utils import AttrDict
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
                            e_cap.max: 40
                    demand_power:
                        constraints:
                            r: -10
            2:
                techs: ['ccgt', 'demand_power', 'unmet_demand_power']
                override:
                    ccgt:
                        constraints:
                            e_cap.max: 30
                        costs:
                            e_cap: 3000
                    demand_power:
                        constraints:
                            r: -10
        links:
            1,2:
                hvac:
                    constraints:
                        e_cap.max: 100
    """
    config_run = """
        mode: plan
        model: ['{techs}', '{locations}']
        subset_t: ['2005-01-01', '2005-01-01']
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
    @pytest.fixture(scope='module')
    def model(self, override=""):
        locations = """
            locations:
                1:
                    techs: []
                2:
                    techs: ['demand_power']
                    override:
                        demand_power:
                            constraints:
                                r: -90
                sub1,sub2:
                    within: 1
                    techs: ['ccgt']
                    override:
                        ccgt:
                            constraints:
                                e_cap.max: 60
            links:
                1,2:
                    hvac:
                        constraints:
                            e_eff: 0.90
                            e_cap.max: 100
        """
        config_run = """
            mode: plan
            model: ['{techs}', '{locations}']
            subset_t: ['2005-01-01', '2005-01-02']
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

    def test_model_solves(self, model):
         assert str(model.results.solver.termination_condition) == 'optimal'

    def test_model_balanced(self, model):
        sol = model.solution
        assert sol['e'].loc[dict(c='power', y='ccgt')].sum(dim='x').mean() == 100
        assert (sol['e'].loc[dict(c='power', y='hvac:1')].sum(dim='x') ==
                -1 * sol['e'].loc[dict(c='power', y='demand_power')].sum(dim='x')).all()

    def test_model_costs(self, model):
        sol = model.solution
        assert_almost_equal(sol['summary'].to_pandas().loc['ccgt', 'levelized_cost_monetary'], 0.1)

    def test_one_way(self):
        """
        Check that one_way transmission can be forced using 'one_way' in model
        formulation.
        """
        override="""
            override.links:
                    X1,N1:
                        heat_pipes:
                            constraints:
                                one_way: true
                    N1,X2:
                        heat_pipes:
                            constraints:
                                one_way: true
                    N1,X3:
                        heat_pipes:
                            constraints:
                                one_way: true
        """
        model = examples.UrbanScale(override=AttrDict.from_yaml_string(override))

        model.run()
        sol = model.solution

        # Usual urban scale model has non-zero transmission along each of these
        # links, one_way forces them to zero
        assert_almost_equal(sol.c_prod.loc[
            dict(y='heat_pipes:N1', c='heat', x='X2')], 0, 0.1)
        assert_almost_equal(sol.c_prod.loc[
            dict(y='heat_pipes:N1', c='heat', x='X3')], 0, 0.1)
        assert_almost_equal(sol.c_prod.loc[
            dict(y='heat_pipes:X1', c='heat', x='N1')], 0, 0.1)

        # Usual urban scale model has zero transmission along this link,
        # one_way forces a change in optimal solution, making it non-zero
        assert_almost_equal(sol.c_prod.loc[
            dict(y='heat_pipes:X2', c='heat', x='N1')], 2.6, 0.1)
