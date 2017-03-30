import pytest
import tempfile

import calliope
from calliope.utils import AttrDict
from . import common
from .common import solver, solver_io

def create_and_run_model(override):
    locations = """
        locations:
            1:
                techs: ['ccgt', 'demand_power',
                        'unmet_demand_power', 'csp']
                override:
                    demand_power:
                        x_map: '1: demand'
                        constraints:
                            r: file=demand-sin_r.csv
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
                                    override=override)
    model.run()
    return model

class TestModel:

    def test_model_export(self):
        override = """
            override.techs.ccgt:
                        export: True
                        costs:
                            monetary:
                                export: -0.2
        """
        model = create_and_run_model(override)
        assert str(model.results.solver.termination_condition) == 'optimal'

    def test_model_fixed_rev(self):
        override = """
            override.techs.ccgt.costs.monetary.e_cap: -5
        """
        model = create_and_run_model(override)
        assert str(model.results.solver.termination_condition) == 'optimal'

    def test_model_var_rev(self):
        override = """
            override.techs.ccgt.costs.monetary.om_var: -0.1
        """
        model = create_and_run_model(override)
        assert str(model.results.solver.termination_condition) == 'optimal'

    def test_model_all_rev(self):
        override = """
            override.techs.ccgt:
                        export: True
                        costs.monetary:
                                om_var: -0.1
                                e_cap: -5
                                export: -0.2
        """
        model = create_and_run_model(override)
        assert str(model.results.solver.termination_condition) == 'optimal'

    def test_model_invalid_r_cap(self):
        override = """
            override.techs.csp.costs.monetary.r_cap: -5
        """
        with pytest.raises(calliope.exceptions.OptionNotSetError):
            model = create_and_run_model(override)

    def test_model_valid_r_cap(self):
        override = """
            override.techs.csp:
                        constraints.r_cap.max: 10
                        costs.monetary.r_cap: -5
        """
        model = create_and_run_model(override)
        assert str(model.results.solver.termination_condition) == 'optimal'
