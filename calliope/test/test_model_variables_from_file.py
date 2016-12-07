import pytest
import tempfile

from calliope.utils import AttrDict
from . import common
from .common import assert_almost_equal, solver, solver_io

def create_and_run_model(override, iterative_warmstart=False,
                         demand_file='demand-static_r.csv'):
    locations = """
        locations:
            1:
                techs: ['ccgt', 'demand_power']
                override:
                    ccgt:
                        constraints:
                            e_cap.max: 100
                    demand_power:
                        x_map: '1: demand'
                        constraints:
                            r: file={demand_file}
        links:
    """
    config_run = """
        mode: operate
        model: ['{techs}', '{locations}']
         subset_t: ['2005-01-01', '2005-01-02']
    """
    override = AttrDict.from_yaml_string(override)
    override.set_key('solver', solver)
    override.set_key('solver_io', solver_io)
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(locations.format(demand_file=demand_file).encode('utf-8'))
        f.read()
        model = common.simple_model(config_run=config_run,
                                    config_locations=f.name,
                                    override=override,
                                    path=_add_test_path('common/t_time'))
    model.run(iterative_warmstart)
    return model

class TestModel:
 
    # all constraints are fixed values
    def test_model_fixed(self, model):
        override = None
        assert str(model.results.solver.termination_condition) == 'optimal'
    
    # Demand is a timeseries variable
    def test_model_var_demand(self, model):
        override = None
        model1 = create_and_run_model(demand_file=demand)
        assert str(model.results.solver.termination_condition) == 'optimal'

    # e_eff is a timeseries variable
    def test_model_var_e_eff(self, model):
        override = """
            override:
                techs:
                    ccgt:
                        constraints:
                            e_eff: file=eff.csv
                    """
        model1 = create_and_run_model(override)
        assert str(model.results.solver.termination_condition) == 'optimal'

    # r_eff is a timeseries variable
    def test_model_var_r_eff(self, model):
        override = """
            override:
                techs:
                    ccgt:
                        constraints:
                            r_eff: file=eff.csv
                    """
        model1 = create_and_run_model(override)
        assert str(model.results.solver.termination_condition) == 'optimal'

    # costs are a timeseries variables
    def test_model_var_om_var(self, model):
        override = """
            override:
                techs:
                    ccgt:
                        costs:
                            monetary:
                                om_fuel: file=cost_rev_var.csv
                                om_var: file=cost_rev_var.csv
                    """
        model1 = create_and_run_model(override)
        assert str(model.results.solver.termination_condition) == 'optimal'

    # costs are a timeseries variables
    def test_model_var_sub_var(self, model):
        override = """
            override:
                techs:
                    ccgt:
                        revenue:
                            monetary:
                                sub_var: file=cost_rev_var.csv
                    """
        model1 = create_and_run_model(override)
        assert str(model.results.solver.termination_condition) == 'optimal'
