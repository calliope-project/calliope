import pytest
import tempfile

from calliope.utils import AttrDict
from . import common
from .common import solver, solver_io, assert_almost_equal

def create_and_run_model(override=""):
    locations = """
        locations:
            1:
                techs: ['ccgt', 'test_conversion', 'test_conversion_plus',
                        'demand_power', 'unmet_demand_power',
                        'demand_heat', 'unmet_demand_heat',
                        'demand_low_T', 'unmet_demand_low_T',
                        'demand_V_low_T', 'unmet_demand_V_low_T']
                override:
                    demand_power:
                            x_map: '1: demand'
                            constraints:
                                r_scale_to_peak: -10
                                r: file=demand-sin_r.csv
                    demand_heat:
                            x_map: '1: demand'
                            constraints:
                                r_scale_to_peak: -5
                                r: file=demand-sin_r.csv
                    demand_low_T:
                            constraints:
                                r: -12
                    demand_V_low_T:
                            constraints:
                                r: -12
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
    def test_export_from_supply(self):
        # giving export revenue *should* lead to there being an export of power
        # and a higher consumption of resource for the ccgt
        override = """
                    override.techs.ccgt:
                                export: true
                                costs.monetary.export: -10
                    """
        model1 = create_and_run_model()
        model2 = create_and_run_model(override)
        sol1 = model1.solution
        sol2 = model2.solution
        with pytest.raises(AttributeError):
            sol1.export # there should be no export here
        assert_almost_equal(sol1.c_con.sum(), sol2.c_con.sum())
        assert_almost_equal(sol2.export.sum(), 687.5, tolerance=0.01)

    def test_export_from_conversion(self):
        # conversion technology exporting heat
        override = """
                    override.techs.test_conversion:
                                export: true
                                costs.monetary.export: -10
                    """
        model = create_and_run_model(override)
        assert_almost_equal(model.solution.export.sum(), 1375, tolerance=0.01)

    def test_export_from_conversion_plus(self):
        # conversion_plus technology exporting low_T (primary carrier), V_low_T, and power
        override1 = """
                    override.techs.test_conversion_plus:
                                export: true # will default to primary carrier
                                costs.monetary.export: -10
                    """
        override2 = """
                    override.techs.test_conversion_plus:
                                export: V_low_T
                                costs.monetary.export: -10
                    """
        override3 = """
                    override.techs.test_conversion_plus:
                                export: power
                                costs.monetary.export: -100
                    """
        model1 = create_and_run_model(override1)
        assert_almost_equal(model1.solution.export.sum(), 1901.6, tolerance=0.01)
        assert_almost_equal(model1.solution.c_con.loc[dict(y='demand_low_T', c='low_T')].sum()
                             - model1.solution.export.sum(),
                             -model1.solution.c_prod.loc[dict(y='test_conversion_plus', c='low_T')].sum())

        model2 = create_and_run_model(override2)
        assert_almost_equal(model2.solution.export.sum(), 1521.28, tolerance=0.01)
        assert_almost_equal(model2.solution.c_con.loc[dict(c='V_low_T')].sum()
                             - model2.solution.export.sum(),
                             -model2.solution.c_prod.loc[dict(y='test_conversion_plus', c='V_low_T')].sum())

        model3 = create_and_run_model(override3)
        assert_almost_equal(model3.solution.export.sum(), 32.4, tolerance=0.01)
        assert_almost_equal(model3.solution.c_con.loc[dict(c='power')].sum()
                             - model3.solution.export.sum(),
                             -model3.solution.c_prod.loc[dict(c='power')].sum())
