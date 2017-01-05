import pytest  # pylint: disable=unused-import
import tempfile

from calliope.utils import AttrDict
from . import common
from .common import assert_almost_equal, solver, solver_io, _add_test_path


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
    def test_model_fixed_costs_op(self):
        override = """
            override:
                techs:
                    ccgt:
                        costs:
                            monetary:
                                e_cap: 5
                                om_fuel: 0
            subset_t: ['2005-01-01', '2005-01-0{}']
        """
        model1 = create_and_run_model(override.format(2))
        cost1 = model1.solution['costs'].loc[dict(kc='monetary', x='1', y='ccgt')]
        lcoe1 = model1.solution['summary'].loc[dict(techs='ccgt', cols_summary='levelized_cost_monetary')]
        model2 = create_and_run_model(override.format(4))
        cost2 = model2.solution['costs'].loc[dict(kc='monetary', x='1', y='ccgt')]
        lcoe2 = model2.solution['summary'].loc[dict(techs='ccgt', cols_summary='levelized_cost_monetary')]
        # LCOE should be the same, as the output is constant throughout
        assert_almost_equal(lcoe1, lcoe2, tolerance=0.0000001)
        # Cost should be double in the second case as it's twice the time
        assert_almost_equal(2 * cost1, cost2, tolerance=0.0000001)

    def test_model_var_costs_op(self):
        override = """
            override:
                techs:
                    ccgt:
                        costs:
                            monetary:
                                e_cap: 0
                                om_fuel: 0.1
            subset_t: ['2005-01-01', '2005-01-0{}']
        """
        model1 = create_and_run_model(override.format(2))
        cost1 = model1.solution['costs'].loc[dict(kc='monetary', x='1', y='ccgt')]
        lcoe1 = model1.solution['summary'].loc[dict(techs='ccgt', cols_summary='levelized_cost_monetary')]
        model2 = create_and_run_model(override.format(4))
        cost2 = model2.solution['costs'].loc[dict(kc='monetary', x='1', y='ccgt')]
        lcoe2 = model2.solution['summary'].loc[dict(techs='ccgt', cols_summary='levelized_cost_monetary')]
        # LCOE should be the same, as the output is constant throughout
        assert_almost_equal(lcoe1, lcoe2, tolerance=0.0000001)
        # Cost should be double in the second case as it's twice the time
        assert_almost_equal(2 * cost1, cost2, tolerance=0.0000001)

    def test_model_fixed_costs_op_with_varying_demand(self):
        override = """
            override:
                techs:
                    ccgt:
                        costs:
                            monetary:
                                e_cap: 5
                                om_fuel: 0
            subset_t: ['2005-01-01', '2005-01-0{}']
        """
        demand = 'demand-blocky_r.csv'
        model1 = create_and_run_model(override.format(2), demand_file=demand)
        cost1 = model1.solution['costs'].loc[dict(kc='monetary', x='1', y='ccgt')]
        lcoe1 = model1.solution['summary'].loc[dict(techs='ccgt', cols_summary='levelized_cost_monetary')]
        model2 = create_and_run_model(override.format(4), demand_file=demand)
        cost2 = model2.solution['costs'].loc[dict(kc='monetary', x='1', y='ccgt')]
        lcoe2 = model2.solution['summary'].loc[dict(techs='ccgt', cols_summary='levelized_cost_monetary')]
        # LCOE should be different
        # TODO should check for exact values
        assert lcoe1 > lcoe2  # lcoe1 has lower output, so is higher
        # Cost should be double in the second case as it's twice the time
        assert_almost_equal(2 * cost1, cost2, tolerance=0.0000001)

    def test_model_var_costs_op_with_varying_demand(self):
        override = """
            override:
                techs:
                    ccgt:
                        costs:
                            monetary:
                                e_cap: 0
                                om_fuel: 0.1
            subset_t: ['2005-01-01', '2005-01-0{}']
        """
        demand = 'demand-blocky_r.csv'
        model1 = create_and_run_model(override.format(2), demand_file=demand)
        cost1 = model1.solution['costs'].loc[dict(kc='monetary', x='1', y='ccgt')]
        lcoe1 = model1.solution['summary'].loc[dict(techs='ccgt', cols_summary='levelized_cost_monetary')]
        model2 = create_and_run_model(override.format(4), demand_file=demand)
        cost2 = model2.solution['costs'].loc[dict(kc='monetary', x='1', y='ccgt')]
        lcoe2 = model2.solution['summary'].loc[dict(techs='ccgt', cols_summary='levelized_cost_monetary')]
        # LCOE should be the same since only fuel costs matter
        assert_almost_equal(lcoe1, lcoe2, tolerance=0.0000001)
        # Cost in second case are more than 2 * first case,
        # because higher output in second case
        assert cost1 == 60
        assert cost2 == 132

    ## duplicate of cost tests, but now for revenue generation ##
    def test_model_fixed_rev_op(self):
        override = """
            override:
                techs:
                    ccgt:
                        revenue:
                            monetary:
                                sub_cap: 5
                                sub_var: 0
            subset_t: ['2005-01-01', '2005-01-0{}']
        """
        model1 = create_and_run_model(override.format(2))
        rev1 = model1.solution['revenue'].loc[dict(kr='monetary', x='1', y='ccgt')]
        model2 = create_and_run_model(override.format(4))
        rev2 = model2.solution['revenue'].loc[dict(kr='monetary', x='1', y='ccgt')]
        # Cost should be double in the second case as it's twice the time
        assert_almost_equal(2 * rev1, rev2, tolerance=0.0000001)
    
    def test_model_var_rev_op(self):
        override = """
            override:
                techs:
                    ccgt:
                        revenue:
                            monetary:
                                sub_cap: 0
                                sub_var: 0.1
            subset_t: ['2005-01-01', '2005-01-0{}']
        """
        model1 = create_and_run_model(override.format(2))
        rev1 = model1.solution['revenue'].loc[dict(kr='monetary', x='1', y='ccgt')]
        model2 = create_and_run_model(override.format(4))
        rev2 = model2.solution['revenue'].loc[dict(kr='monetary', x='1', y='ccgt')]
        # Cost should be double in the second case as it's twice the time
        assert_almost_equal(2 * rev1, rev2, tolerance=0.0000001)
    
    def test_model_fixed_costs_op_with_varying_demand(self):
        override = """
            override:
                techs:
                    ccgt:
                        revenue:
                            monetary:
                                sub_cap: 5
                                sub_var: 0
            subset_t: ['2005-01-01', '2005-01-0{}']
        """
        demand = 'demand-blocky_r.csv'
        model1 = create_and_run_model(override.format(2), demand_file=demand)
        rev1 = model1.solution['revenue'].loc[dict(kr='monetary', x='1', y='ccgt')]
        model2 = create_and_run_model(override.format(4), demand_file=demand)
        rev2 = model2.solution['revenue'].loc[dict(kr='monetary', x='1', y='ccgt')]
        # Cost should be double in the second case as it's twice the time
        assert_almost_equal(2 * rev1, rev2, tolerance=0.0000001)
    
    def test_model_var_rev_op_with_varying_demand(self):
        override = """
            override:
                techs:
                    ccgt:
                        revenue:
                            monetary:
                                sub_cap: 0
                                sub_var: 0.1
            subset_t: ['2005-01-01', '2005-01-0{}']
        """
        demand = 'demand-blocky_r.csv'
        model1 = create_and_run_model(override.format(2), demand_file=demand)
        rev1 = model1.solution['revenue'].loc[dict(kr='monetary', x='1', y='ccgt')]
        model2 = create_and_run_model(override.format(4), demand_file=demand)
        rev2 = model2.solution['revenue'].loc[dict(kr='monetary', x='1', y='ccgt')]
        # Cost in second case are more than 2 * first case,
        # because higher output in second case
        assert rev1 == 60
        assert rev2 == 132

    ## test both cost and revenue simultaneously ##
    def test_model_fixed_rev_op_fixed_cost_op(self):
        override = """
            override:
                techs:
                    ccgt:
                        cost:
                            monetary:
                                e_cap: 5
                                om_fuel: 0
                        revenue:
                            monetary:
                                sub_cap: 5
                                sub_var: 0
            subset_t: ['2005-01-01', '2005-01-0{}']
        """
        model1 = create_and_run_model(override.format(2))
        rev1 = model1.solution['revenue'].loc[dict(kr='monetary', x='1', y='ccgt')]
        cost1 = model1.solution['costs'].loc[dict(kc='monetary', x='1', y='ccgt')]
        model2 = create_and_run_model(override.format(4))
        rev2 = model2.solution['revenue'].loc[dict(kr='monetary', x='1', y='ccgt')]
        cost2 = model2.solution['costs'].loc[dict(kc='monetary', x='1', y='ccgt')]
        # Cost should be double in the second case as it's twice the time
        assert_almost_equal(2 * rev1, rev2, tolerance=0.0000001)
        assert_almost_equal(2 * cost1, cost2, tolerance=0.0000001)
        # Revenue and cost should be equal ##CURRENTLY FAILS
        assert_almost_equal(cost1, rev1, tolerance=0.0000001)
        assert_almost_equal(cost2, rev2, tolerance=0.0000001)