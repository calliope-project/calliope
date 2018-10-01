import pytest  # pylint: disable=unused-import

import datetime
import os
import logging
import tempfile

import calliope

from calliope.test.common.util import check_error_or_warning

from calliope.core.util import dataset
from calliope.core.util.tools import \
    memoize, \
    memoize_instancemethod
from calliope.core.util.logging import log_time
from calliope.core.util.generate_runs import generate_runs


F_MODEL = os.path.join(os.path.dirname(__file__), 'common', 'test_model', 'model_minimal.yaml')
F_OVERRIDE = os.path.join(os.path.dirname(__file__), 'common', 'test_model', 'overrides.yaml')


class TestDataset:
    @pytest.fixture()
    def loc_techs(self):
        loc_techs = [
            'region1-3::csp', 'region1::demand_power', 'region1-1::csp',
            'region1-2::csp', 'region2::demand_power', 'region1-1::demand_power'
        ]
        return loc_techs

    def test_get_loc_techs_tech(self, loc_techs):
        loc_techs = dataset.get_loc_techs(loc_techs, tech='csp')
        assert loc_techs == [
            'region1-3::csp', 'region1-1::csp', 'region1-2::csp'
        ]

    def test_get_loc_techs_loc(self, loc_techs):
        loc_techs = dataset.get_loc_techs(loc_techs, loc='region1-1')
        assert loc_techs == [
            'region1-1::csp', 'region1-1::demand_power'
        ]

    def test_get_loc_techs_loc_and_tech(self, loc_techs):
        loc_techs = dataset.get_loc_techs(
            loc_techs, tech='demand_power', loc='region1')
        assert loc_techs == ['region1::demand_power']


class TestMemoization:
    @memoize_instancemethod
    def instance_method(self, a, b):
        return a + b

    def test_memoize_one_arg(self):
        @memoize
        def test(a):
            return a + 1
        assert test(1) == 2
        assert test(1) == 2

    def test_memoize_two_args(self):
        @memoize
        def test(a, b):
            return a + b
        assert test(1, 2) == 3
        assert test(1, 2) == 3

    def test_memoize_instancemethod(self):
        assert self.instance_method(1, 2) == 3
        assert self.instance_method(1, 2) == 3


class TestLogging:
    def test_set_log_level(self):

        # We assign a handler to the Calliope logger on loading calliope
        assert calliope._logger.hasHandlers() is True
        assert len(calliope._logger.handlers) == 1

        for level in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']:
            calliope.set_log_level(level)
            assert calliope._logger.level == getattr(logging, level)

        # We have a custom level 'SOLVER' at level 19
        calliope.set_log_level('SOLVER')
        assert calliope._logger.level == 19

    def test_timing_log(self):
        timings = {'model_creation': datetime.datetime.now()}

        # TODO: capture logging output and check that comment is in string
        log_time(timings, 'test', comment='test_comment', level='info')
        assert isinstance(timings['test'], datetime.datetime)

        log_time(timings, 'test2', comment=None, level='info')
        assert isinstance(timings['test2'], datetime.datetime)

        # TODO: capture logging output and check that time_since_start is in the string
        log_time(timings, 'test', comment=None, level='info', time_since_start=True)


class TestGenerateRuns:
    def test_generate_runs_groups_file_combinations(self):
        groups_str = """
        combinations: [
            ['a', 'b', 'c'],
            ['1', '2']
        ]
        """

        with tempfile.TemporaryDirectory() as tempdir:
            groups_file = os.path.join(tempdir, 'groups.yaml')
            with open(groups_file, 'w') as f:
                f.write(groups_str)
            runs = generate_runs(
                F_MODEL, F_OVERRIDE, groups_file=groups_file
            )

        run_results = [
            'overrides.yaml:a,1 --save_netcdf out_1_a,1.nc --save_plots plots_1_a,1.html',
            'overrides.yaml:a,2 --save_netcdf out_2_a,2.nc --save_plots plots_2_a,2.html',
            'overrides.yaml:b,1 --save_netcdf out_3_b,1.nc --save_plots plots_3_b,1.html',
            'overrides.yaml:b,2 --save_netcdf out_4_b,2.nc --save_plots plots_4_b,2.html',
            'overrides.yaml:c,1 --save_netcdf out_5_c,1.nc --save_plots plots_5_c,1.html',
            'overrides.yaml:c,2 --save_netcdf out_6_c,2.nc --save_plots plots_6_c,2.html'
        ]

        for i in range(len(runs)):
            assert run_results[i] in runs[i]

    def test_generate_runs_groups_file_groups(self):
        groups_str = """
        groups: [
            'a,1',
            'b,1',
        ]
        """

        with tempfile.TemporaryDirectory() as tempdir:
            groups_file = os.path.join(tempdir, 'groups.yaml')
            with open(groups_file, 'w') as f:
                f.write(groups_str)
            runs = generate_runs(
                F_MODEL, F_OVERRIDE, groups_file=groups_file
            )

        run_results = [
            'overrides.yaml:a,1 --save_netcdf out_1_a,1.nc --save_plots plots_1_a,1.html',
            'overrides.yaml:b,1 --save_netcdf out_2_b,1.nc --save_plots plots_2_b,1.html'
        ]

        for i in range(len(runs)):
            assert run_results[i] in runs[i]

    def test_generate_runs_groups_file_combinations_and_groups(self):
        groups_str = """
        combinations: [['a', 'b', 'c'], ['1', '2']]
        groups: ['a,1', 'a,2', 'b,1']
        """

        with tempfile.TemporaryDirectory() as tempdir:
            groups_file = os.path.join(tempdir, 'groups.yaml')
            with open(groups_file, 'w') as f:
                f.write(groups_str)
            with pytest.raises(ValueError) as excinfo:
                generate_runs(
                    F_MODEL, F_OVERRIDE, groups_file=groups_file
                )

        assert check_error_or_warning(excinfo, 'Only one of `combinations` or `groups` may be defined in the groups_file.')
