from __future__ import print_function
from __future__ import division

import pytest
import tempfile

import common
from common import assert_almost_equal


class TestModel:
    @pytest.fixture(scope='module')
    def model_one_tech_one_site(self):
        nodes = """
            nodes:
                1:
                    level: 1
                    within:
                    techs: ['ccgt', 'storage', 'demand', 'unmet_demand']
                    override:
                        storage:
                            constraints:
                                e_cap_max: 0.5
                                s_init: 0
                        ccgt:
                            constraints:
                                e_cap_max: 9.5
                        demand:
                            x_map: 'demand: 1'
                            constraints:
                                r: file=demand-sin_r.csv
            links:
        """
        config_run = """
            input:
                techs: {techs}
                nodes: {nodes}
                path: '{path}'
            output:
                save: false
            subset_t: ['2005-01-01', '2005-01-03']
        """
        with tempfile.NamedTemporaryFile() as f:
            f.write(nodes)
            f.read()
            model = common.simple_model(config_run=config_run,
                                        config_nodes=f.name)
        model.run()
        return model

    def test_one_tech_one_site_solves(self, model_one_tech_one_site):
        model = model_one_tech_one_site
        assert str(model.results.Solution.Status) == 'optimal'

    def test_one_tech_one_site_balanced(self, model_one_tech_one_site):
        model = model_one_tech_one_site
        df = model.get_system_variables()
        assert df.ix[0, 'ccgt'] == 8.0
        assert_almost_equal(df['ccgt'].mean(), 7.62, tolerance=0.01)
        # assert (df['ccgt'] == -1 * df['demand']).all()
