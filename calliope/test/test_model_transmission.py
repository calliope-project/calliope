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
                    techs: []
                2:
                    level: 1
                    within:
                    techs: ['demand_electricity']
                    override:
                        demand_electricity:
                            constraints:
                                r: -90
                sub1,sub2:
                    level: 0
                    within: 1
                    techs: ['ccgt']
                    override:
                        ccgt:
                            constraints:
                                e_cap_max: 60
            links:
                1,2:
                    hvac:
                        constraints:
                            e_eff: 0.90
                            e_cap_max: 100
        """
        config_run = """
            mode: plan
            input:
                techs: {techs}
                locations: {locations}
                path: '{path}'
            output:
                save: false
            subset_t: ['2005-01-01', '2005-01-02']
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
        assert df.loc['power', :, 'ccgt'].mean() == 100
        assert (df.loc['power', :, 'hvac:1'] ==
                -1 * df.loc['power', :, 'demand_electricity']).all()

    def test_model_costs(self, model):
        df = model.get_costs()
        assert_almost_equal(df.at['lcoe', 'total', 'ccgt'], 0.1)
