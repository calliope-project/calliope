import pytest
import tempfile

from calliope.utils import AttrDict
from . import common
from .common import assert_almost_equal, solver


class TestModel:
    @pytest.fixture(scope='module')
    def model(self):
        locations = """
            locations:
                1:
                    level: 1
                    within:
                    techs: ['demand_electricity']
                    override:
                        demand_electricity:
                            constraints:
                                r: -10
                sub1:
                    level: 0
                    within: 1
                    techs: ['ccgt']
                    override:
                        ccgt:
                            constraints:
                                e_cap_max: 9
                sub2:
                    level: 0
                    within: 1
                    techs: ['pv']
                    override:
                        pv:
                            x_map: 'sub2: demand'
                            constraints:
                                e_cap_max: 9
                                r: file=demand-sin_r.csv
                                r_scale_to_peak: 5

            links:
        """
        config_run = """
            mode: plan
            model: [{techs}, {locations}]
            subset_t: ['2005-01-01', '2005-01-02']
        """
        with tempfile.NamedTemporaryFile() as f:
            f.write(locations.encode('utf-8'))
            f.read()
            model = common.simple_model(config_run=config_run,
                                        config_locations=f.name,
                                        override=AttrDict({'solver': solver}))
        model.run()
        return model

    def test_model_solves(self, model):
         assert str(model.results.solver.termination_condition) == 'optimal'

    def test_model_balanced(self, model):
        df = model.solution.node
        assert_almost_equal(df.loc['e:power', 'pv', :, :].iloc[0, :].sum(), 7.5,
                            tolerance=0.01)
        assert (df.loc['e:power', 'pv', :, :].sum(1) +
                df.loc['e:power', 'ccgt', :, :].sum(1) ==
                -1 * df.loc['e:power', 'demand_electricity', :, :].sum(1)).all()

    def test_model_costs(self, model):
        df = model.solution.levelized_cost
        assert_almost_equal(df.at['monetary', 'power', 'total', 'ccgt'], 0.1,
                            tolerance=0.001)
