import tempfile

from calliope.utils import AttrDict
from . import common
from .common import assert_almost_equal, solver, solver_io


def create_and_run_model(override=""):
    locations = """
        locations:
            1:
                techs: ['ccgt', 'test_storage', 'demand_power',
                        'unmet_demand_power']
                override:
                    ccgt:
                        constraints:
                            e_cap.max: 9.5
                    demand_power:
                        x_map: '1: demand'
                        constraints:
                            r: file=demand-sin_r.csv
        links:
    """
    config_run = """
        mode: plan
        model: ['{techs}', '{locations}']
        subset_t: ['2005-01-01', '2005-01-03']
    """

    override = AttrDict.from_yaml_string(override)
    override.set_key('solver', solver)
    override.set_key('solver_io', solver_io)
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(locations.encode('utf-8'))
        f.read()
        model = common.simple_model(config_run=config_run,
                                    config_locations=f.name,
                                    override=override)
    model.run()
    return model


class TestModel:
    def test_model_e_cap_max(self):
        override = """
            override.techs.test_storage.constraints.e_cap.max: 0.6
        """
        model = create_and_run_model(override)
        assert str(model.results.solver.termination_condition) == 'optimal'
        sol = model.solution
        assert_almost_equal(sol['e'].loc[dict(c='power', y='ccgt')].sum(dim='x')[dict(t=0)].mean(),
                            8.03, tolerance=0.01)
        assert_almost_equal(sol['e'].loc[dict(c='power', y='ccgt')].sum(dim='x').mean(),
                            7.62, tolerance=0.01)

    def test_model_c_rate(self):
        override = """
            override.techs.test_storage.constraints.c_rate: 0.006
        """
        model = create_and_run_model(override)
        assert str(model.results.solver.termination_condition) == 'optimal'
        sol = model.solution
        assert_almost_equal(sol['e'].loc[dict(c='power', y='ccgt')].sum(dim='x')[dict(t=0)].mean(),
                            8.03, tolerance=0.01)
        assert_almost_equal(sol['e'].loc[dict(c='power', y='ccgt')].sum(dim='x').mean(),
                            7.62, tolerance=0.01)

    def test_model_compare_c_rate_e_cap(self):
        override1 = """
            override.techs.test_storage.constraints:
                            c_rate: 0.006
                            e_cap.max: 0.6
        """
        override2 = """
            override.techs.test_storage.constraints:
                            c_rate: 0.008
                            e_cap.max: 0.6
        """
        model1 = create_and_run_model(override1)
        model2 = create_and_run_model(override2)
        sol1 = model1.solution
        sol2 = model2.solution
        assert_almost_equal(sol1.e_cap.loc[dict(y='test_storage')].sum(dim='x'),
                            sol2.e_cap.loc[dict(y='test_storage')].sum(dim='x'),
                            tolerance=0.01)
        assert (sol1.s_cap.loc[dict(y='test_storage')].sum(dim='x') >
                sol2.s_cap.loc[dict(y='test_storage')].sum(dim='x'))

    def test_model_s_time(self):
        override1 = """
            override.techs.test_storage.constraints:
                            s_time.max: 12
                            c_rate: 0.01
        """
        override2 = """
            override.techs.test_storage.constraints:
                            s_time.max: 12
                            e_cap.max: 1
        """
        model1 = create_and_run_model(override1)
        model2 = create_and_run_model(override2)
        assert str(model1.results.solver.termination_condition) == 'optimal'
        assert str(model2.results.solver.termination_condition) == 'optimal'
        sol1 = model1.solution
        sol2 = model2.solution
        assert (sol1.e_cap.loc[dict(y='test_storage')].sum(dim='x') <
                sol2.e_cap.loc[dict(y='test_storage')].sum(dim='x'))
        assert (sol1.s_cap.loc[dict(y='test_storage')].sum(dim='x') >
                sol2.s_cap.loc[dict(y='test_storage')].sum(dim='x'))
