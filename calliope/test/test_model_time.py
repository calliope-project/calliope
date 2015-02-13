import pytest
import tempfile

from calliope.utils import AttrDict
from . import common
from .common import assert_almost_equal, solver, _add_test_path


def create_and_run_model(override, iterative_warmstart=False):
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
                            r: file=demand-blocky_r.csv
        links:
    """
    config_run = """
        mode: plan
        model: [{techs}, {locations}]
    """
    override = AttrDict.from_yaml_string(override)
    override.set_key('solver', solver)
    with tempfile.NamedTemporaryFile() as f:
        f.write(locations.encode('utf-8'))
        f.read()
        model = common.simple_model(config_run=config_run,
                                    config_locations=f.name,
                                    override=override,
                                    path=_add_test_path('common/t_time'))
    model.run(iterative_warmstart)
    return model


class TestModel:
    def test_model_subset_t_from_middata(self):
        override = """
            subset_t: ['2005-01-02', '2005-01-03']
        """
        model = create_and_run_model(override)
        assert str(model.results.solver.termination_condition) == 'optimal'
        # Make sure the result is valid
        df = model.solution.node
        assert df.loc['e:power', 'ccgt', :, :].sum().sum() == 720

    def test_model_time_res_uniform(self):
        override = """
            time: {function: resolution_series_uniform, function_options: {resolution: 24}}
        """
        model = create_and_run_model(override)
        assert str(model.results.solver.termination_condition) == 'optimal'
        # Make sure the result is valid
        df = model.solution.node
        assert df.loc['e:power', 'ccgt', :, :].sum().sum() == 1320

    def test_model_time_res_uniform_subset_t_from_start(self):
        override = """
            time: {function: resolution_series_uniform, function_options: {resolution: 24}}
            subset_t: ['2005-01-01', '2005-01-02']
        """
        model = create_and_run_model(override)
        assert str(model.results.solver.termination_condition) == 'optimal'
        # Make sure the result is valid
        df = model.solution.node
        assert df.loc['e:power', 'ccgt', :, :].sum().sum() == 600

    def test_model_time_res_uniform_subset_t_from_middata(self):
        override = """
            time: {function: resolution_series_uniform, function_options: {resolution: 24}}
            subset_t: ['2005-01-02', '2005-01-03']
        """
        model = create_and_run_model(override)
        assert str(model.results.solver.termination_condition) == 'optimal'
        # Make sure the result is valid
        df = model.solution.node
        assert df.loc['e:power', 'ccgt', :, :].sum().sum() == 720

    def test_model_subset_t_operational(self):
        override = """
            mode: operate
            override:
                opmode:
                    horizon: 24  # Optimization period length (hours)
                    window: 12  # Operation period length (hours)
            subset_t: ['2005-01-02', '2005-01-03']
        """
        model = create_and_run_model(override)
        assert str(model.results.solver.termination_condition) == 'optimal'
        # Make sure the result is valid
        df = model.solution.node
        assert df.loc['e:power', 'ccgt', :, :].sum().sum() == 720

    def test_model_time_res_uniform_subset_t_operational(self):
        override = """
            mode: operate
            override:
                opmode:
                    horizon: 48  # Optimization period length (hours)
                    window: 24  # Operation period length (hours)
            time: {function: resolution_series_uniform, function_options: {resolution: 12}}
            subset_t: ['2005-01-02', '2005-01-03']
        """
        model = create_and_run_model(override)
        assert str(model.results.solver.termination_condition) == 'optimal'
        # Make sure the result is valid
        df = model.solution.node
        assert df.loc['e:power', 'ccgt', :, :].sum().sum() == 720
