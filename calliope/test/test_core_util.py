import pytest  # pylint: disable=unused-import
import calliope
import logging
import datetime
import pyomo

from calliope.core.util import dataset

from calliope.core.util.tools import \
    memoize, \
    memoize_instancemethod

from calliope.core.util.logging import log_time

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
