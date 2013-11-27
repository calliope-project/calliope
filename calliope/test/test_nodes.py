from __future__ import print_function
from __future__ import division

import cStringIO as StringIO

import pandas as pd
import pytest

from calliope import nodes, utils


class TestNodes:
    @pytest.fixture
    def sample_nodes(self):
        setup = StringIO.StringIO("""
        test:
            level: 1
            within:
            techs: ['demand', 'unmet_demand', 'ccgt']
            override:
                demand:
                    x_map: 'a: test'
                    constraints:
                        r: file=demand.csv
                        r_scale_to_peak: -10
                ccgt:
                    constraints:
                        e_cap_max: 10
        test1:
            level: 0
            within: test
            techs: ['csp']
        """)
        return utils.AttrDict.from_yaml(setup)

    @pytest.fixture
    def sample_unexploded_nodes(self):
        setup = StringIO.StringIO("""
        1,2,3:
        a,b,c:
        10--15:
        10-20:
        21--23,25,z:
        x:
        y:
        """)
        return utils.AttrDict.from_yaml(setup)

    def test_generate_node(self, sample_nodes):
        node = 'test'
        items = sample_nodes[node]
        techs = ['demand', 'unmet_demand', 'ccgt']
        result = nodes._generate_node(node, items, techs)
        wanted_cols = ['_level', '_node',
                       '_override.ccgt.constraints.e_cap_max',
                       '_override.demand.constraints.r',
                       '_override.demand.constraints.r_scale_to_peak',
                       '_override.demand.x_map',
                       '_within', 'ccgt', 'demand', 'unmet_demand']
        assert sorted(result.keys()) == wanted_cols

    def test_generate_node_lacking_techs(self, sample_nodes):
        node = 'test'
        items = sample_nodes[node]
        techs = ['unmet_demand', 'ccgt']
        result = nodes._generate_node(node, items, techs)
        wanted_cols = ['_level', '_node',
                       '_override.ccgt.constraints.e_cap_max',
                       '_override.demand.constraints.r',
                       '_override.demand.constraints.r_scale_to_peak',
                       '_override.demand.x_map',
                       '_within', 'ccgt', 'unmet_demand']
        assert sorted(result.keys()) == wanted_cols

    def test_explode_node_single(self):
        assert nodes.explode_node('a') == ['a']

    def test_explode_node_range(self):
        assert nodes.explode_node('1--3') == ['1', '2', '3']

    def test_explode_node_range_backwards(self):
        with pytest.raises(KeyError):
            nodes.explode_node('3--1')

    def test_explode_node_range_nonnumeric(self):
        with pytest.raises(ValueError):
            nodes.explode_node('a--c')

    def test_explore_node_list(self):
        assert nodes.explode_node('1,2,3') == ['1', '2', '3']

    def test_explode_node_mixed(self):
        assert nodes.explode_node('a,b,1--3,c') == ['a', 'b', '1', '2', '3', 'c']

    def test_explode_node_empty(self):
        with pytest.raises(KeyError):
            assert nodes.explode_node('')

    def test_explode_node_invalid(self):
        with pytest.raises(AssertionError):
            assert nodes.explode_node(['a', 'b'])

    def test_get_nodes(self, sample_unexploded_nodes):
        results = nodes.get_nodes(sample_unexploded_nodes)
        print(sorted(results))
        assert sorted(results) == ['1', '10', '10-20', '11', '12', '13', '14',
                                   '15', '2', '21', '22', '23', '25', '3',
                                   'a', 'b', 'c', 'x', 'y', 'z']

    def test_generate_node_matrix_cols(self, sample_nodes):
        techs = ['demand', 'unmet_demand', 'ccgt', 'csp']
        df = nodes.generate_node_matrix(sample_nodes, techs)
        wanted_cols = ['_level',
                       '_override.ccgt.constraints.e_cap_max',
                       '_override.demand.constraints.r',
                       '_override.demand.constraints.r_scale_to_peak',
                       '_override.demand.x_map',
                       '_within', 'ccgt', 'csp', 'demand', 'unmet_demand']
        assert sorted(df.columns) == wanted_cols

    def test_generate_node_matrix_index(self, sample_nodes):
        techs = ['demand', 'unmet_demand', 'ccgt', 'csp']
        df = nodes.generate_node_matrix(sample_nodes, techs)
        assert df.index.tolist() == ['test', 'test1']

    def test_generate_node_matrix_values(self, sample_nodes):
        techs = ['demand', 'unmet_demand', 'ccgt', 'csp']
        df = nodes.generate_node_matrix(sample_nodes, techs)
        assert df.at['test', 'demand'] == 1
        assert df.at['test1', 'demand'] == 0
        assert (df.at['test', '_override.demand.constraints.r']
                == 'file=demand.csv')
        assert pd.isnull(df.at['test1', '_override.demand.constraints.r'])

    def test_generate_node_matrix_additional_techs(self, sample_nodes):
        techs = ['demand', 'unmet_demand', 'ccgt', 'csp', 'foo']
        df = nodes.generate_node_matrix(sample_nodes, techs)
        assert sum(df['foo']) == 0

    def test_generate_node_matrix_missing_techs_cols(self, sample_nodes):
        techs = ['ccgt']
        df = nodes.generate_node_matrix(sample_nodes, techs)
        wanted_cols = ['_level',
                       '_override.ccgt.constraints.e_cap_max',
                       '_override.demand.constraints.r',
                       '_override.demand.constraints.r_scale_to_peak',
                       '_override.demand.x_map',
                       '_within', 'ccgt']
        assert sorted(df.columns) == wanted_cols
