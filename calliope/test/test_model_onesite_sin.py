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
                    techs: ['ccgt', 'demand_electricity']
                    override:
                        ccgt:
                            constraints:
                                e_cap_max: 100
                        demand_electricity:
                            x_map: '1: demand'
                            constraints:
                                r: file=demand-sin_r.csv
            links:
        """
        config_run = """
            mode: plan
            model: [{techs}, {locations}]
            subset_t: ['2005-01-01', '2005-01-03']
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
        assert df.loc['e:power', 'ccgt', :, :].iloc[0, :].sum() == 7.5
        assert_almost_equal(df.loc['e:power', 'ccgt', :, :].sum(1).mean(),
                            7.62, tolerance=0.01)
        assert (df.loc['e:power', 'ccgt', :, :].sum(1) ==
                -1 * df.loc['e:power', 'demand_electricity', :, :].sum(1)).all()
