import tempfile

from calliope.utils import AttrDict
from . import common
from .common import assert_almost_equal, solver, solver_io


def create_and_run_model(override=""):
    locations = """
        locations:
            1:
                techs: ['ccgt', 'test_conversion', 'test_conversion_plus',
                        'demand_power', 'unmet_demand_power', 'demand_heat',
                        'demand_low_T', 'demand_V_low_T', 'supply_gas',
                        'unmet_demand_heat', 'unmet_demand_low_T',
                        'unmet_demand_V_low_T',]
                override:
                    ccgt:
                        constraints:
                            e_cap.max: 30
                    test_conversion:
                        constraints:
                            e_cap.max: 20
                    demand_power:
                        constraints:
                            r: -10
                    demand_heat:
                        constraints:
                            r: -6
        links:
    """
    config_run = """
        mode: plan
        model: ['{techs}', '{locations}']
        subset_t: ['2005-01-01', '2005-01-01']
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
    def test_model_conversion(self):
        model = create_and_run_model()
        assert str(model.results.solver.termination_condition) == 'optimal'

    def test_model_conversion_plus(self):
        override = """
            override.techs.demand_low_T.constraints.r: -5
            override.techs.demand_V_low_T.constraints.r: -5
        """
        model = create_and_run_model(override)
        assert str(model.results.solver.termination_condition) == 'optimal'
        sol = model.solution
        assert_almost_equal(sol.costs.sum(), 41.47, tolerance=0.001)
        assert_almost_equal(sol.c_prod.loc[
            dict(y='test_conversion_plus', c='power')], 13.5, tolerance=0.01)
