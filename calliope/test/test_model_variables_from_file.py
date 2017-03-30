import tempfile

from calliope.utils import AttrDict
from . import common
from .common import solver, solver_io, _add_test_path

def create_and_run_model(override, iterative_warmstart=False):
    locations = """
        locations:
            1:
                techs: ['ccgt', 'demand_power']
                override:
                    ccgt:
                        constraints:
                            e_cap.max: 100
                    demand_power:
                        constraints:
                            r: -10
        links:
    """
    config_run = """
        mode: plan
        model: ['{techs}', '{locations}']
        subset_t: ['2005-01-01', '2005-01-02']
    """
    override = AttrDict.from_yaml_string(override)
    override.set_key('solver', solver)
    override.set_key('solver_io', solver_io)
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(locations.encode('utf-8'))
        f.read()
        model = common.simple_model(config_run=config_run,
                                    config_locations=f.name,
                                    override=override,
                                    path=_add_test_path('common/t_constraints_from_file'))
    model.run(iterative_warmstart)
    return model

class TestModel:

    # all constraints are fixed values
    def test_model_fixed(self):
        override = """override:"""
        model = create_and_run_model(override)
        assert str(model.results.solver.termination_condition) == 'optimal'

    # e_eff is a timeseries variable
    def test_model_var_e_eff(self):
        override = """
            override:
                techs:
                    ccgt:
                        constraints:
                            e_eff: file=eff_var_sin.csv
                    """
        model = create_and_run_model(override)
        assert str(model.results.solver.termination_condition) == 'optimal'

    # r_eff is a timeseries variable
    def test_model_var_r_eff(self):
        override = """
            override:
                techs:
                    ccgt:
                        constraints:
                            r_eff: file=eff_var_sin.csv
                    """
        model = create_and_run_model(override)
        assert str(model.results.solver.termination_condition) == 'optimal'

    # costs are timeseries variables
    def test_model_var_om_var(self):
        override = """
            override:
                techs:
                    ccgt:
                        costs:
                            monetary:
                                om_fuel: file=cost_var.csv
                                om_var: file=cost_var.csv
                    """
        model = create_and_run_model(override)
        assert str(model.results.solver.termination_condition) == 'optimal'


    # revenue is a timeseries variable
    def test_model_var_sub_var(self):
        override = """
            override.techs.ccgt.costs.monetary.om_var: file=rev_var.csv
                    """
        model = create_and_run_model(override)
        assert str(model.results.solver.termination_condition) == 'optimal'
