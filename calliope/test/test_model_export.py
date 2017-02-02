import pytest
import tempfile

from calliope.utils import AttrDict
from . import common
from .common import solver, solver_io


class TestModel:
    @pytest.fixture(scope='module')
    def model(self):
        locations = """
            locations:
                1:
                    techs: ['ccgt', 'demand_power',
                            'unmet_demand_power']
                    override:
                        ccgt:
                            export: True
                        demand_power:
                            x_map: '1: demand'
                            constraints:
                                r: file=demand-sin_r.csv
                        revenue:
                            sub_export: 0.12
            links:
        """
        config_run = """
            mode: plan
            model: ['{techs}', '{locations}']
            subset_t: ['2005-01-01', '2005-01-02']
        """
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(locations.encode('utf-8'))
            f.read()
            override_dict = AttrDict({
                'solver': solver,
                'solver_io': solver_io,
            })
            model = common.simple_model(config_run=config_run,
                                        config_locations=f.name,
                                        override=override_dict)
        model.run()
        return model

    def test_model_solves(self, model):
        assert str(model.results.solver.termination_condition) == 'optimal'
