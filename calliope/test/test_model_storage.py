from __future__ import print_function
from __future__ import division

import pytest
import tempfile

from calliope.utils import AttrDict
import common
from common import assert_almost_equal, solver


class TestModel:
    @pytest.fixture(scope='module')
    def model(self):
        locations = """
            locations:
                1:
                    level: 1
                    within:
                    techs: ['ccgt', 'test_storage', 'demand_electricity',
                            'unmet_demand_electricity']
                    override:
                        test_storage:
                            constraints:
                                e_cap_max: 0.5
                                s_init: 0
                        ccgt:
                            constraints:
                                e_cap_max: 9.5
                        demand_electricity:
                            x_map: 'demand: 1'
                            constraints:
                                r: file=demand-sin_r.csv
            links:
        """
        config_run = """
            mode: plan
            input:
                model: [{techs}, {locations}]
                path: '{path}'
            output:
                save: false
            subset_t: ['2005-01-01', '2005-01-03']
        """
        with tempfile.NamedTemporaryFile() as f:
            f.write(locations)
            f.read()
            model = common.simple_model(config_run=config_run,
                                        config_locations=f.name,
                                        override=AttrDict({'solver': solver}))
        model.run()
        return model

    def test_model_solves(self, model):
        assert str(model.results.Solution.Status) == 'optimal'

    def test_model_balanced(self, model):
        df = model.get_system_variables()
        assert df.ix['power', 0, 'ccgt'] == 8.0
        assert_almost_equal(df.loc['power', :, 'ccgt'].mean(),
                            7.62, tolerance=0.01)
