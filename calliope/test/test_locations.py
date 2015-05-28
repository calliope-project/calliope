from io import StringIO

import pandas as pd
import pytest

from calliope import locations, utils


class TestLocations:
    @pytest.fixture
    def sample_locations(self):
        setup = StringIO("""
        test:
            techs: ['demand', 'unmet_demand', 'ccgt']
            override:
                demand:
                    x_map: 'a: test'
                    constraints:
                        r: file=demand.csv
                        r_scale_to_peak: -10
                ccgt:
                    constraints:
                        e_cap.max: 10
        test1:
            within: test
            techs: ['csp']
        """)
        return utils.AttrDict.from_yaml(setup)

    @pytest.fixture
    def sample_unexploded_locations(self):
        setup = StringIO("""
        1,2,3:
            foo:
        a,b,c:
            foo:
        10--15:
            foo:
        10-20:
            foo:
        21--23,25,z:
            foo:
        x:
            foo:
        y:
            foo:
        """)
        return utils.AttrDict.from_yaml(setup)

    @pytest.fixture
    def sample_nested_locations(self):
        setup = StringIO("""
        1,2,3:
            techs: ['foo']
        foo:
            techs: ['foo']
        10,11,12:
            within: 1
            techs: ['foo']
        20,21,22:
            within: 2
            techs: ['foo']
        bar,baz:
            within: foo
            techs: ['foo']
        qux:
            within: bar
            techs: ['foo']
        """)
        return utils.AttrDict.from_yaml(setup)

    @pytest.fixture
    def sample_overlapping_locations(self):
        setup = StringIO("""
        1,2,3:
            techs: ['foo']
        1:
            override:
                bar: baz
        """)
        return utils.AttrDict.from_yaml(setup)

    def test_generate_location(self, sample_locations):
        location = 'test'
        items = sample_locations[location]
        techs = ['demand', 'unmet_demand', 'ccgt']
        result = locations._generate_location(location, items, techs)
        wanted_cols = ['_level', '_location',
                       '_override.ccgt.constraints.e_cap.max',
                       '_override.demand.constraints.r',
                       '_override.demand.constraints.r_scale_to_peak',
                       '_override.demand.x_map',
                       '_within', 'ccgt', 'demand', 'unmet_demand']
        assert sorted(result.keys()) == wanted_cols

    def test_generate_location_lacking_techs(self, sample_locations):
        location = 'test'
        items = sample_locations[location]
        techs = ['unmet_demand', 'ccgt']
        result = locations._generate_location(location, items, techs)
        wanted_cols = ['_level', '_location',
                       '_override.ccgt.constraints.e_cap.max',
                       '_override.demand.constraints.r',
                       '_override.demand.constraints.r_scale_to_peak',
                       '_override.demand.x_map',
                       '_within', 'ccgt', 'unmet_demand']
        assert sorted(result.keys()) == wanted_cols

    def test_explode_location_single(self):
        assert locations.explode_locations('a') == ['a']

    def test_explode_location_range(self):
        assert locations.explode_locations('1--3') == ['1', '2', '3']

    def test_explode_location_range_backwards(self):
        with pytest.raises(KeyError):
            locations.explode_locations('3--1')

    def test_explode_location_range_nonnumeric(self):
        with pytest.raises(ValueError):
            locations.explode_locations('a--c')

    def test_explore_location_list(self):
        assert locations.explode_locations('1,2,3') == ['1', '2', '3']

    def test_explode_location_mixed(self):
        assert (locations.explode_locations('a,b,1--3,c')
                == ['a', 'b', '1', '2', '3', 'c'])

    def test_explode_location_empty(self):
        with pytest.raises(KeyError):
            assert locations.explode_locations('')

    def test_explode_location_invalid(self):
        with pytest.raises(AssertionError):
            assert locations.explode_locations(['a', 'b'])

    def test_process_locations(self, sample_unexploded_locations):
        fixture = sample_unexploded_locations
        o = locations.process_locations(fixture)
        assert '10' in o
        assert 'x' in o
        assert len(list(o.keys())) == 20

    def test_process_locations_overlap(self, sample_overlapping_locations):
        fixture = sample_overlapping_locations
        o = locations.process_locations(fixture)
        assert o['1'].level == 0
        assert o['1'].override.bar == 'baz'

    def test_process_locations_levels(self, sample_nested_locations):
        fixture = sample_nested_locations
        o = locations.process_locations(fixture)
        assert o['1'].level == 0
        assert o['bar'].level == 1
        assert o['qux'].level == 2

    def test_generate_location_matrix_cols(self, sample_locations):
        techs = ['demand', 'unmet_demand', 'ccgt', 'csp']
        df = locations.generate_location_matrix(sample_locations, techs)
        wanted_cols = ['_level',
                       '_override.ccgt.constraints.e_cap.max',
                       '_override.demand.constraints.r',
                       '_override.demand.constraints.r_scale_to_peak',
                       '_override.demand.x_map',
                       '_within', 'ccgt', 'csp', 'demand', 'unmet_demand']
        assert sorted(df.columns) == wanted_cols

    def test_generate_location_matrix_index(self, sample_locations):
        techs = ['demand', 'unmet_demand', 'ccgt', 'csp']
        df = locations.generate_location_matrix(sample_locations, techs)
        assert df.index.tolist() == ['test', 'test1']

    def test_generate_location_matrix_values(self, sample_locations):
        techs = ['demand', 'unmet_demand', 'ccgt', 'csp']
        df = locations.generate_location_matrix(sample_locations, techs)
        assert df.at['test', 'demand'] == 1
        assert df.at['test1', 'demand'] == 0
        assert (df.at['test', '_override.demand.constraints.r']
                == 'file=demand.csv')
        assert pd.isnull(df.at['test1', '_override.demand.constraints.r'])

    def test_generate_location_matrix_additional_techs(self, sample_locations):
        techs = ['demand', 'unmet_demand', 'ccgt', 'csp', 'foo']
        df = locations.generate_location_matrix(sample_locations, techs)
        assert sum(df['foo']) == 0

    def test_generate_location_matrix_missing_techs_cols(self, sample_locations):
        techs = ['ccgt']
        df = locations.generate_location_matrix(sample_locations, techs)
        wanted_cols = ['_level',
                       '_override.ccgt.constraints.e_cap.max',
                       '_override.demand.constraints.r',
                       '_override.demand.constraints.r_scale_to_peak',
                       '_override.demand.x_map',
                       '_within', 'ccgt']
        assert sorted(df.columns) == wanted_cols

    def test_generate_location_matrix_within_only_strings(self,
                                                      sample_nested_locations):
        techs = ['foo']
        df = locations.generate_location_matrix(sample_nested_locations, techs)
        for i in df['_within'].tolist():
            assert (i is None or isinstance(i, str))
