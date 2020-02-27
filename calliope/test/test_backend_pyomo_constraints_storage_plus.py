import pytest  # pylint: disable=unused-import

from calliope.test.common.util import build_test_model as build_model
from calliope.test.common.util import check_variable_exists


class TestBuildStoragePlusConstraints:
    # storage_plus.py
    def test_loc_techs_storage_plus_max_constraint(self):
        
        """
        sets.loc_techs_storage_plus_cap_per_time,
        """
        m = build_model({}, 'simple_supply,two_hours,investment_costs')
        m.run(build_only=True)
        assert not hasattr(m._backend_model, 'loc_techs_storage_plus_max_constraint')

        m = build_model({'techs.test_storage_plus.constraints': {'storage_cap_equals_per_timestep': 'file=storage_plus_cap_equals_per_time.csv'}}, 'simple_storage_plus,two_hours,investment_costs')
        m.run(build_only=True)
        assert hasattr(m._backend_model, 'loc_techs_storage_plus_max_constraint')

        m = build_model(
            {'techs.test_storage_plus.constraints': {'storage_cap_max': 20, 
            'storage_cap_equals_per_timestep': 'file=storage_plus_cap_equals_per_time.csv'}},
            'simple_storage_plus,two_hours,investment_costs'
        )
        m.run(build_only=True)
        assert hasattr(m._backend_model, 'loc_techs_storage_plus_max_constraint')

        m = build_model(
            {'techs.test_storage_plus.constraints': {
                'storage_cap_equals': 10
            }},
            'simple_storage_plus,two_hours,investment_costs'
        )
        m.run(build_only=True)
        assert not hasattr(m._backend_model, 'loc_techs_storage_plus_max_constraint')

    def test_loc_techs_loc_techs_storage_plus_discharge_depth_constraint(self):
        """
        i for i in sets.loc_techs_storage_plus
        if any([
            constraint_exists(model_run, i, 'constraints.storage_discharge_depth'),
            constraint_exists(model_run, i, 'constraints.storage_discharge_depth_per_timestep')
            ]
        """

        m = build_model({}, 'simple_storage_plus,two_hours,investment_costs')
        m.run(build_only=True)
        assert not hasattr(m._backend_model, 'loc_techs_storage_plus_discharge_depth_constraint')

        m = build_model({'techs.test_storage_plus.constraints':
            {'storage_discharge_depth_per_timestep': 'file=storage_plus_discharge_depth.csv'}},
             'simple_storage_plus,two_hours,investment_costs')
        m.run(build_only=True)
        assert hasattr(m._backend_model, 'loc_techs_storage_plus_discharge_depth_constraint')


        m = build_model({'techs.test_storage_plus.constraints':
            {'storage_discharge_depth': 0.1}},
             'simple_storage_plus,two_hours,investment_costs')
        m.run(build_only=True)
        assert hasattr(m._backend_model, 'loc_techs_storage_plus_discharge_depth_constraint')

    def test_loc_techs_loc_techs_storage_plus_balance_constraint(self):
        """
        sets.loc_techs_storage_plus
        """

        m = build_model({}, 'simple_storage,two_hours,investment_costs')
        m.run(build_only=True)
        assert not hasattr(m._backend_model, 'loc_techs_storage_plus_balance_constraint')


        m = build_model({}, 'simple_storage_plus,two_hours,investment_costs')
        m.run(build_only=True)
        assert hasattr(m._backend_model, 'loc_techs_storage_plus_balance_constraint')


        m = build_model({},'simple_conversion_plus,two_hours,investment_costs')
        m.run(build_only=True)
        assert not hasattr(m._backend_model, 'loc_techs_storage_plus_balance_constraint')


    def test_loc_techs_loc_techs_storage_plus_storage_time_constraint(self):
        """
        sets.loc_techs_storage_plus_storage_time, 
        """
        
        m = build_model({},'simple_supply,two_hours,investment_costs')
        m.run(build_only=True)
        assert not hasattr(m._backend_model, 'loc_techs_storage_plus_storage_time_constraint')

        
        m = build_model({}, 'simple_storage_plus,two_hours,investment_costs')
        m.run(build_only=True)
        assert not hasattr(m._backend_model, 'loc_techs_storage_plus_storage_time_constraint')

        m = build_model({}, 'storage_plus_shared_storage,two_hours,investment_costs')
        m.run(build_only=True)
        assert not hasattr(m._backend_model, 'loc_techs_storage_plus_storage_time_constraint')

        m = build_model(
            {'techs.test_storage_plus.constraints':{'storage_time':3}},
            'simple_storage_plus,two_hours,investment_costs'
            )
        m.run(build_only=True)
        assert hasattr(m._backend_model, 'loc_techs_storage_plus_storage_time_constraint')

        m = build_model(
            {'techs.test_storage_plus.constraints':
                {'storage_time_per_timestep':'file=storage_plus_storage_time.csv'}}, 
                'simple_storage_plus,two_hours,investment_costs'
            )
        m.run(build_only=True)
        assert hasattr(m._backend_model, 'loc_techs_storage_plus_storage_time_constraint')

    def test_loc_techs_loc_techs_storage_plus_shared_storage_constraint(self):
        """
        sets.loc_techs_storage_plus_shared_storage
        """

        m = build_model({}, 'simple_supply,two_hours,investment_costs')
        m.run(build_only=True)
        assert not hasattr(m._backend_model, 'loc_techs_storage_plus_shared_storage_constraint')

        m = build_model(
            {'techs.test_storage_plus.constraints':
            {'shared_storage_tech':'1::test_storage'}},
            'storage_plus_shared_storage,two_hours,investment_costs')
        m.run(build_only=True)
        assert hasattr(m._backend_model, 'loc_techs_storage_plus_shared_storage_constraint')

        m = build_model({}, 'simple_storage_plus,two_hours,investment_costs')
        m.run(build_only=True)
        assert not hasattr(m._backend_model, 'loc_techs_storage_plus_shared_storage_constraint')

class TestStoragePlusConstraintResults:

    def test_storage_plus_energy_balance(self):
        m = build_model(
            {'techs.test_storage_plus':{'storage_cap_equals_per_timestep': 'file=storage_plus_cap_equals_per_time.csv'}}, 'simple_storage_plus,two_hours,investment_costs')
        )
        m.run()

        assert #energy balance is obeyed

#     def test_shared_storage_link(self):

#         return
    
#     def test_storage_time(self):

#         return

#     def test_storage_time_per_time(self):

#         return

#     def test_storage_discharge_depth_per_time_from_file(self):

#         return


