import pytest
import tempfile

from calliope.utils import AttrDict
from . import common
from .common import assert_almost_equal, solver, solver_io


class TestModel:
    @pytest.fixture(scope='module')
    def model(self):
        locations = """
            locations:
                1:
                    techs: ['demand_power']
                    override:
                        demand_power:
                            constraints:
                                r: -10
                sub1:
                    within: 1
                    techs: ['ccgt']
                    override:
                        ccgt:
                            constraints:
                                e_cap.max: 9
                sub2:
                    within: 1
                    techs: ['pv']
                    override:
                        pv:
                            x_map: 'sub2: demand'
                            constraints:
                                e_cap.max: 9
                                r: file=demand-sin_r.csv
                                r_scale_to_peak: 5

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

    def test_model_solves(self, model):
         assert str(model.results.solver.termination_condition) == 'optimal'

    def test_model_balanced(self, model):
        sol = model.solution
        assert_almost_equal(sol['e'].loc[dict(c='power', y='pv')].sum(dim='x')[dict(t=0)], 7.5,
                            tolerance=0.01)
        assert (sol['e'].loc[dict(c='power', y='pv')].sum(dim='x') +
                sol['e'].loc[dict(c='power', y='ccgt')].sum(dim='x') ==
                -1 * sol['e'].loc[dict(c='power', y='demand_power')].sum(dim='x')).all()

    def test_model_costs(self, model):
        sol = model.solution
        assert_almost_equal(sol['summary'].to_pandas().loc['ccgt', 'levelized_cost_monetary'], 0.1,
                            tolerance=0.001)
