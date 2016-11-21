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
                    techs: ['ccgt', 'test_storage', 'demand_power',
                            'unmet_demand_power']
                    override:
                        test_storage:
                            constraints:
                                e_cap.max: 0.5
                                s_init: 0
                        ccgt:
                            constraints:
                                e_cap.max: 9.5
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
        assert sol['e'].loc[dict(c='power', y='ccgt')].sum(dim='x')[dict(t=0)].mean() == 8.0
        assert_almost_equal(sol['e'].loc[dict(c='power', y='ccgt')].sum(dim='x').mean(),
                            7.62, tolerance=0.01)
