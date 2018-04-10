import pytest  # pylint: disable=unused-import
import numpy as np
import xarray as xr
import pyomo.core as po
from pyomo.core.base.expr import identify_variables

from calliope.backend.pyomo.util import get_param
import calliope.exceptions as exceptions

from calliope.test.common.util import build_test_model as build_model
from calliope.test.common.util import check_error_or_warning


def check_variable_exists(backend_model, constraint, variable):
    """
    Search for existence of a decision variable in a Pyomo constraint.

    Parameters
    ----------
    backend_model : Pyomo ConcreteModel
    constraint : str, name of constraint which could exist in the backend
    variable : str, string to search in the list of variables to check if existing
    """
    exists = []
    for v in getattr(backend_model, constraint).values():
        variables = identify_variables(v.body)
        exists.append(any(variable in j.getname() for j in list(variables)))
    return any(exists)


def check_standard_warning(info, warning):

    if warning == 'transmission':
        return check_error_or_warning(
            info,
            'dimension loc_techs_transmission and associated variables distance, '
            'lookup_remotes were empty, so have been deleted'
        )


class TestUtil:
    def test_get_param_with_timestep_existing(self):
        """
        """
        m = build_model({}, 'simple_supply,two_hours,investment_costs')
        m.run()
        param = get_param(
            m._backend_model,
            'resource',
            ('1::test_demand_elec', m._backend_model.timesteps[1])
        )
        assert po.value(param) == -5  # see demand_elec.csv

    def test_get_param_no_timestep_existing(self):
        """
        """
        m = build_model({}, 'simple_supply,two_hours,investment_costs')
        m.run()
        param = get_param(
            m._backend_model,
            'energy_eff',
            ('1::test_supply_elec', m._backend_model.timesteps[1])
        )
        assert po.value(param) == 0.9  # see test model.yaml

    def test_get_param_no_timestep_possible(self):
        """
        """
        m = build_model({}, 'simple_supply,two_hours,investment_costs')
        m.run()
        param = get_param(
            m._backend_model,
            'energy_cap_max',
            ('1::test_supply_elec')
        )
        assert po.value(param) == 10  # see test model.yaml

        param = get_param(
            m._backend_model,
            'cost_energy_cap',
            ('monetary', '0::test_supply_elec')
        )
        assert po.value(param) == 10

    def test_get_param_from_default(self):
        """
        """
        m = build_model({}, 'simple_supply_and_supply_plus,two_hours,investment_costs')
        m.run()

        param = get_param(
            m._backend_model,
            'parasitic_eff',
            ('1::test_supply_plus', m._backend_model.timesteps[1])
        )
        assert po.value(param) == 1  # see defaults.yaml

        param = get_param(
            m._backend_model,
            'resource_cap_min',
            ('0::test_supply_plus')
        )
        assert po.value(param) == 0  # see defaults.

        param = get_param(
            m._backend_model,
            'cost_resource_cap',
            ('monetary', '1::test_supply_plus')
        )
        assert po.value(param) == 0  # see defaults.yaml

    def test_get_param_no_default_defined(self):
        """
        If a default is not defined, raise KeyError
        """
        m = build_model({}, 'simple_supply,two_hours,investment_costs')
        m.run()
        with pytest.raises(KeyError):
            get_param(
                m._backend_model,
                'random_param',
                ('1::test_demand_elec', m._backend_model.timesteps[1])
            )
            get_param(
                m._backend_model,
                'random_param',
                ('1::test_supply_elec')
            )


class TestInterface:
    def test_get_input_params(self):
        """
        Test that the function access_model_inputs works
        """
        m = build_model({}, 'simple_supply,two_hours,investment_costs')
        m.run()
        m.backend.access_model_inputs()

    def test_update_param(self):
        """
        test that the function update_param works
        """
        m = build_model({}, 'simple_supply,two_hours,investment_costs')
        m.run()
        m.backend.update_param('energy_cap_max', '1::test_supply_elec', 20)
        assert (
            m._backend_model.energy_cap_max.extract_values()['1::test_supply_elec'] == 20
        )

    def test_activate_constraint(self):
        """
        test that the function activate_constraint works
        """
        m = build_model({}, 'simple_supply,two_hours,investment_costs')
        m.run()
        m.backend.activate_constraint('system_balance_constraint', active=False)
        assert not m._backend_model.system_balance_constraint.active

    def test_rerun(self):
        """
        test that the function rerun works
        """
        m = build_model({}, 'simple_supply,two_hours,investment_costs')
        m.run()
        returned_dataset = m.backend.rerun()
        assert isinstance(returned_dataset, xr.Dataset)

        # should fail if the run mode is not 'plan'
        with pytest.raises(exceptions.ModelError) as error:
            m._model_data.attrs['run.mode'] = 'operate'
            m.backend.rerun()
        assert check_error_or_warning(error, 'Cannot rerun the backend in operate run mode')


class TestConstraints:

    def test_loc_carriers_system_balance_constraint(self):
        """
        sets.loc_carriers
        """
        m = build_model({}, 'simple_supply,two_hours,investment_costs')
        m.run(build_only=True)
        assert hasattr(m._backend_model, 'system_balance_constraint')

    def test_loc_techs_balance_supply_constraint(self):
        """
        sets.loc_techs_finite_resource_supply,
        """
        m = build_model({'techs.test_supply_elec.constraints.resource': 20},
                        'simple_supply,two_hours,investment_costs')
        m.run(build_only=True)
        assert hasattr(m._backend_model, 'balance_supply_constraint')

    def test_loc_techs_balance_demand_constraint(self):
        """
        sets.loc_techs_finite_resource_demand,
        """
        m = build_model({}, 'simple_supply,two_hours,investment_costs')
        m.run(build_only=True)
        assert hasattr(m._backend_model, 'balance_demand_constraint')

    def test_loc_techs_resource_availability_supply_plus_constraint(self):
        """
        sets.loc_techs_finite_resource_supply_plus,
        """
        m = build_model({}, 'simple_supply_and_supply_plus,two_hours,investment_costs')
        m.run(build_only=True)
        assert hasattr(
            m._backend_model, 'resource_availability_supply_plus_constraint'
        )

    def test_loc_techs_balance_transmission_constraint(self):
        """
        sets.loc_techs_transmission,
        """
        m = build_model({}, 'simple_supply,two_hours,investment_costs')
        m.run(build_only=True)
        assert hasattr(m._backend_model, 'balance_transmission_constraint')

    def test_loc_techs_balance_supply_plus_constraint(self):
        """
        sets.loc_techs_supply_plus,
        """
        m = build_model({}, 'simple_supply_and_supply_plus,two_hours,investment_costs')
        m.run(build_only=True)
        assert hasattr(m._backend_model, 'balance_supply_plus_constraint')

    def test_loc_techs_balance_storage_constraint(self):
        """
        sets.loc_techs_storage,
        """
        m = build_model({}, 'simple_storage,two_hours,investment_costs')
        m.run(build_only=True)
        assert hasattr(m._backend_model, 'balance_storage_constraint')

    def test_carriers_reserve_margin_constraint(self):
        """
        i for i in sets.carriers if i in model_run.model.get_key('reserve_margin', {}).keys()
        """
        m = build_model({'model.reserve_margin.electricity': 0.01}, 'simple_supply,two_hours,investment_costs')
        m.run(build_only=True)
        assert hasattr(m._backend_model, 'reserve_margin_constraint')

    # costs.py
    def test_loc_techs_cost_constraint(self):
        """
        sets.loc_techs_cost,
        """
        m = build_model({}, 'simple_supply,two_hours,investment_costs')
        m.run(build_only=True)
        assert hasattr(m._backend_model, 'cost_constraint')

    def test_loc_techs_cost_investment_constraint(self):
        """
        sets.loc_techs_investment_cost,
        """
        m = build_model({}, 'simple_conversion,two_hours,investment_costs')
        m.run(build_only=True)
        assert hasattr(m._backend_model, 'cost_investment_constraint')

        m = build_model(
            {'techs.test_supply_elec.constraints.lifetime': 10,
                'techs.test_supply_elec.costs.monetary.interest_rate': 0.1},
            'supply_purchase,two_hours'
        )
        m.run(build_only=True)

        assert hasattr(m._backend_model, 'cost_investment_constraint')

    def test_loc_techs_cost_var_constraint(self):
        """
        i for i in sets.loc_techs_om_cost if i not in sets.loc_techs_conversion_plus + sets.loc_techs_conversion

        """
        m = build_model({}, 'simple_conversion,two_hours,investment_costs')
        m.run(build_only=True)
        assert not hasattr(m._backend_model, 'cost_var_constraint')

        m = build_model(
            {'techs.test_conversion.costs.monetary.om_con': 1},
            'simple_conversion,two_hours'
        )
        m.run(build_only=True)
        assert hasattr(m._backend_model, 'cost_var_rhs')
        assert not hasattr(m._backend_model, 'cost_var_constraint')

        m = build_model(
            {'techs.test_conversion_plus.costs.monetary.om_prod': 1},
            'simple_conversion_plus,two_hours'
        )

        m.run(build_only=True)
        assert hasattr(m._backend_model, 'cost_var_rhs')
        assert not hasattr(m._backend_model, 'cost_var_constraint')

        m = build_model(
            {'techs.test_supply_elec.costs.monetary.om_prod': 1},
            'simple_supply,two_hours'
        )
        m.run(build_only=True)
        assert hasattr(m._backend_model, 'cost_var_constraint')

        m = build_model(
            {'techs.test_supply_elec.costs.monetary.om_con': 1},
            'simple_supply,two_hours'
        )
        m.run(build_only=True)
        assert hasattr(m._backend_model, 'cost_var_constraint')

        m = build_model(
            {'techs.test_supply_plus.costs.monetary.om_con': 1},
            'simple_supply_and_supply_plus,two_hours'
        )
        m.run(build_only=True)
        assert hasattr(m._backend_model, 'cost_var_constraint')

    # export.py
    def test_loc_carriers_update_system_balance_constraint(self):
        """
        i for i in sets.loc_carriers if sets.loc_techs_export
        and any(['{0}::{2}'.format(*j.split('::')) == i
        for j in sets.loc_tech_carriers_export])
        """

        m = build_model({}, 'simple_supply,two_hours,investment_costs')
        m.run(build_only=True)
        export_exists = check_variable_exists(
            m._backend_model, 'system_balance_constraint', 'carrier_export'
        )
        assert not export_exists

        m = build_model({}, 'supply_export,two_hours,investment_costs')
        m.run(build_only=True)

        export_exists = check_variable_exists(
            m._backend_model, 'system_balance_constraint', 'carrier_export'
        )
        assert export_exists

    def test_loc_tech_carriers_export_balance_constraint(self):
        """
        sets.loc_tech_carriers_export,
        """

        m = build_model({}, 'supply_export,two_hours,investment_costs')
        m.run(build_only=True)

        assert hasattr(m._backend_model, 'export_balance_constraint')

    def test_loc_techs_update_costs_var_constraint(self):
        """
        i for i in sets.loc_techs_om_cost if i in sets.loc_techs_export
        """

        m = build_model({}, 'supply_export,two_hours,investment_costs')
        m.run(build_only=True)
        assert hasattr(m._backend_model, 'cost_var_rhs')
        assert hasattr(m._backend_model, 'cost_var_constraint')

        m = build_model(
            {'techs.test_supply_elec.costs.monetary.om_prod': 0.1},
            'supply_export,two_hours,investment_costs'
        )
        m.run(build_only=True)
        assert hasattr(m._backend_model, 'cost_var_rhs')
        assert hasattr(m._backend_model, 'cost_var_constraint')

        export_exists = check_variable_exists(
            m._backend_model, 'cost_var_constraint', 'carrier_export'
        )
        assert export_exists

    def test_loc_tech_carriers_export_max_constraint(self):
        """
        i for i in sets.loc_tech_carriers_export
        if constraint_exists(model_run, i.rsplit('::', 1)[0], 'constraints.export_cap')
        """

        m = build_model(
            {'techs.test_supply_elec.constraints.export_cap': 5},
            'supply_export,two_hours,investment_costs'
        )
        m.run(build_only=True)
        assert hasattr(m._backend_model, 'export_max_constraint')

    # capacity.py
    def test_loc_techs_storage_capacity_constraint(self):
        """
        i for i in sets.loc_techs_store if i not in sets.loc_techs_milp
        """
        m = build_model({}, 'simple_storage,two_hours,investment_costs')
        m.run(build_only=True)
        assert hasattr(m._backend_model, 'storage_capacity_constraint')

        m = build_model({}, 'simple_supply_and_supply_plus,two_hours,investment_costs')
        m.run(build_only=True)
        assert hasattr(m._backend_model, 'storage_capacity_constraint')

        m = build_model(
            {'techs.test_storage.constraints':
                {'units_max': 1, 'energy_cap_per_unit': 20,
                 'storage_cap_per_unit': 20}},
            'simple_storage,two_hours,investment_costs'
        )
        m.run(build_only=True)
        assert not hasattr(m._backend_model, 'storage_capacity_constraint')

        m = build_model(
            {'techs.test_storage.constraints.storage_cap_equals': 20},
            'simple_storage,two_hours,investment_costs'
        )
        m.run(build_only=True)
        assert m._backend_model.storage_capacity_constraint['0::test_storage'].upper() == 20
        assert m._backend_model.storage_capacity_constraint['0::test_storage'].lower() == 20

    def test_loc_techs_energy_capacity_storage_constraint(self):
        """
        i for i in sets.loc_techs_store if constraint_exists(model_run, i, 'constraints.charge_rate')
        """
        m = build_model({}, 'simple_storage,two_hours,investment_costs')
        m.run(build_only=True)
        assert hasattr(m._backend_model, 'energy_capacity_storage_constraint')

        m = build_model({}, 'simple_supply_and_supply_plus,two_hours,investment_costs')
        m.run(build_only=True)
        assert hasattr(m._backend_model, 'energy_capacity_storage_constraint')

        # constraint should exist in the MILP case too
        m = build_model({}, 'supply_and_supply_plus_milp,two_hours,investment_costs')
        m.run(build_only=True)
        assert hasattr(m._backend_model, 'energy_capacity_storage_constraint')

    def test_loc_techs_resource_capacity_constraint(self):
        """
        i for i in sets.loc_techs_finite_resource_supply_plus
        if any([constraint_exists(model_run, i, 'constraints.resource_cap_equals'),
                constraint_exists(model_run, i, 'constraints.resource_cap_max'),
                constraint_exists(model_run, i, 'constraints.resource_cap_min')])
        """

        m = build_model({}, 'simple_supply_and_supply_plus,two_hours,investment_costs')
        m.run(build_only=True)
        assert not hasattr(m._backend_model, 'resource_capacity_constraint')

        m = build_model(
            {'techs.test_supply_plus.constraints.resource_cap_max': 10},
            'simple_supply_and_supply_plus,two_hours,investment_costs'
        )
        m.run(build_only=True)
        assert hasattr(m._backend_model, 'energy_capacity_storage_constraint')

        m = build_model(
            {'techs.test_supply_plus.constraints.resource_cap_min': 10},
            'simple_supply_and_supply_plus,two_hours,investment_costs'
        )
        m.run(build_only=True)
        assert hasattr(m._backend_model, 'energy_capacity_storage_constraint')

        m = build_model(
            {'techs.test_supply_plus.constraints.resource_cap_equals': 10},
            'simple_supply_and_supply_plus,two_hours,investment_costs'
        )
        m.run(build_only=True)
        assert hasattr(m._backend_model, 'energy_capacity_storage_constraint')

    def test_loc_techs_resource_capacity_equals_energy_capacity_constraint(self):
        """
        i for i in sets.loc_techs_finite_resource_supply_plus
        if constraint_exists(model_run, i, 'constraints.resource_cap_equals_energy_cap')
        """

        m = build_model({}, 'simple_supply_and_supply_plus,two_hours,investment_costs')
        m.run(build_only=True)
        assert not hasattr(
            m._backend_model, 'resource_capacity_equals_energy_capacity_constraint'
        )

        m = build_model(
            {'techs.test_supply_plus.constraints.resource_cap_equals_energy_cap': True},
            'simple_supply_and_supply_plus,two_hours,investment_costs'
        )
        m.run(build_only=True)
        assert hasattr(
            m._backend_model, 'resource_capacity_equals_energy_capacity_constraint'
        )

    def test_loc_techs_resource_area_constraint(self):
        """
        i for i in sets.loc_techs_area if i in sets.loc_techs_supply_plus
        """

        m = build_model({}, 'simple_supply_and_supply_plus,two_hours,investment_costs')
        m.run(build_only=True)
        assert not hasattr(m._backend_model, 'resource_area_constraint')

        m = build_model(
            {'techs.test_supply_plus.constraints.resource_area_max': 10},
            'simple_supply_and_supply_plus,two_hours,investment_costs'
        )
        m.run(build_only=True)
        assert hasattr(m._backend_model, 'resource_area_constraint')

        m = build_model(
            {'techs.test_supply_elec.constraints.resource_area_max': 10},
            'simple_supply_and_supply_plus,two_hours,investment_costs'
        )
        m.run(build_only=True)
        assert hasattr(m._backend_model, 'resource_area_constraint')

        # Check that setting energy_cap_max to 0 also forces this constraint to 0
        m = build_model(
            {'techs.test_supply_plus.constraints': {'resource_area_max': 10, 'energy_cap_max': 0}},
            'simple_supply_and_supply_plus,two_hours,investment_costs'
        )
        m.run(build_only=True)
        assert m._backend_model.resource_area_constraint['0::test_supply_plus'].upper() == 0

    def test_loc_techs_resource_area_per_energy_capacity_constraint(self):
        """
        i for i in sets.loc_techs_area if i in sets.loc_techs_supply_plus
        and constraint_exists(model_run, i, 'constraints.resource_area_per_energy_cap')
        """
        m = build_model({}, 'simple_supply_and_supply_plus,two_hours,investment_costs')
        m.run(build_only=True)
        assert not hasattr(m._backend_model, 'resource_area_per_energy_capacity_constraint')

        m = build_model(
            {'techs.test_supply_plus.constraints.resource_area_max': 10},
            'simple_supply_and_supply_plus,two_hours,investment_costs'
        )
        m.run(build_only=True)
        assert not hasattr(m._backend_model, 'resource_area_per_energy_capacity_constraint')

        m = build_model(
            {'techs.test_supply_elec.constraints.resource_area_per_energy_cap': 10},
            'simple_supply_and_supply_plus,two_hours,investment_costs'
        )
        m.run(build_only=True)
        assert hasattr(m._backend_model, 'resource_area_per_energy_capacity_constraint')

        m = build_model(
            {'techs.test_supply_elec.constraints': {'resource_area_per_energy_cap': 10,
                                                    'resource_area_max': 10}},
            'simple_supply_and_supply_plus,two_hours,investment_costs'
        )
        m.run(build_only=True)
        assert hasattr(m._backend_model, 'resource_area_per_energy_capacity_constraint')

    def test_locs_resource_area_capacity_per_loc_constraint(self):
        """
        i for i in sets.locs
        if model_run.locations[i].get_key('available_area', None) is not None
        """
        m = build_model({}, 'simple_supply_and_supply_plus,two_hours,investment_costs')
        m.run(build_only=True)
        assert not hasattr(m._backend_model, 'resource_area_capacity_per_loc_constraint')

        m = build_model({'locations.0.available_area': 1}, 'simple_supply_and_supply_plus,two_hours,investment_costs')
        m.run(build_only=True)
        assert not hasattr(m._backend_model, 'resource_area_capacity_per_loc_constraint')

        m = build_model(
            {'locations.0.available_area': 1,
             'techs.test_supply_plus.constraints.resource_area_max': 10},
            'simple_supply_and_supply_plus,two_hours,investment_costs'
        )
        m.run(build_only=True)
        assert hasattr(m._backend_model, 'resource_area_capacity_per_loc_constraint')

    def test_loc_techs_energy_capacity_constraint(self):
        """
        i for i in sets.loc_techs
        if i not in sets.loc_techs_milp + sets.loc_techs_purchase
        """
        m = build_model({}, 'simple_supply_and_supply_plus,two_hours,investment_costs')
        m.run(build_only=True)
        assert hasattr(m._backend_model, 'energy_capacity_constraint')

        m2 = build_model(
            {'techs.test_supply_elec.constraints.energy_cap_scale': 5},
            'simple_supply_and_supply_plus,two_hours,investment_costs'
        )
        m2.run(build_only=True)
        assert (
            m2._backend_model.energy_capacity_constraint['0::test_supply_elec'].upper() ==
            m._backend_model.energy_capacity_constraint['0::test_supply_elec'].upper() * 5
        )

        m = build_model({}, 'supply_milp,two_hours,investment_costs')  # demand still is in loc_techs
        m.run(build_only=True)
        assert hasattr(m._backend_model, 'energy_capacity_constraint')

        # Check that setting `_equals` to infinity is caught:
        override = {'locations.0.techs.test_supply_elec.constraints.energy_cap_equals': np.inf}
        with pytest.raises(exceptions.ModelError) as error:
            m = build_model(override, 'simple_supply,two_hours,investment_costs')
            m.run(build_only=True)

        assert check_error_or_warning(
            error,
            'Cannot use inf for energy_cap_equals for loc:tech `0::test_supply_elec`'
        )

    def test_techs_energy_capacity_systemwide_constraint(self):
        """
        i for i in sets.techs
        if model_run.get_key('techs.{}.constraints.energy_cap_max_systemwide'.format(i), None)
        """
        m = build_model({}, 'simple_supply,two_hours,investment_costs')
        m.run(build_only=True)
        assert not hasattr(m._backend_model, 'energy_capacity_systemwide_constraint')

        m = build_model(
            {'techs.test_supply_elec.constraints.energy_cap_max_systemwide': 20},
            'simple_supply,two_hours,investment_costs'
        )
        m.run(build_only=True)
        assert hasattr(m._backend_model, 'energy_capacity_systemwide_constraint')
        assert (
            'test_supply_elec' in
            m._backend_model.energy_capacity_systemwide_constraint.keys()
        )

        # setting the constraint to infinity leads to Pyomo creating NoConstraint
        m = build_model(
            {'techs.test_supply_elec.constraints.energy_cap_max_systemwide': np.inf},
            'simple_supply,two_hours,investment_costs'
        )
        m.run(build_only=True)
        assert hasattr(m._backend_model, 'energy_capacity_systemwide_constraint')
        assert (
            'test_supply_elec' not in
            m._backend_model.energy_capacity_systemwide_constraint.keys()
        )

        # Check that setting `_equals` to infinity is caught:
        with pytest.raises(exceptions.ModelError) as error:
            m = build_model(
                {'techs.test_supply_elec.constraints.energy_cap_equals_systemwide': np.inf},
                'simple_supply,two_hours,investment_costs'
            )
            m.run(build_only=True)

        assert check_error_or_warning(
            error,
            'Cannot use inf for energy_cap_equals_systemwide for tech `test_supply_elec`'
        )

        m = build_model(
            {'techs.test_supply_elec.constraints.energy_cap_equals_systemwide': 20},
            'simple_supply,two_hours,investment_costs'
        )
        m.run(build_only=True)
        assert hasattr(m._backend_model, 'energy_capacity_systemwide_constraint')

    # dispatch.py
    def test_loc_tech_carriers_carrier_production_max_constraint(self):
        """
        i for i in sets.loc_tech_carriers_prod
        if i not in sets.loc_tech_carriers_conversion_plus
        and i.rsplit('::', 1)[0] not in sets.loc_techs_milp
        """
        m = build_model({}, 'simple_supply,two_hours,investment_costs')
        m.run(build_only=True)
        assert hasattr(m._backend_model, 'carrier_production_max_constraint')

        m = build_model({}, 'supply_milp,two_hours,investment_costs')
        m.run(build_only=True)
        assert not hasattr(m._backend_model, 'carrier_production_max_constraint')


    def test_loc_tech_carriers_carrier_production_min_constraint(self):
        """
        i for i in sets.loc_tech_carriers_prod
        if i not in sets.loc_tech_carriers_conversion_plus
        and constraint_exists(model_run, i, 'constraints.energy_cap_min_use')
        and i.rsplit('::', 1)[0] not in sets.loc_techs_milp
        """
        m = build_model({}, 'simple_supply,two_hours,investment_costs')
        m.run(build_only=True)
        assert not hasattr(m._backend_model, 'carrier_production_min_constraint')

        m = build_model(
            {'techs.test_supply_elec.constraints.energy_cap_min_use': 0.1},
            'simple_supply,two_hours,investment_costs'
        )
        m.run(build_only=True)
        assert hasattr(m._backend_model, 'carrier_production_min_constraint')

        m = build_model({}, 'supply_milp,two_hours,investment_costs')
        m.run(build_only=True)
        assert not hasattr(m._backend_model, 'carrier_production_min_constraint')

        m = build_model(
            {'techs.test_supply_elec.constraints.energy_cap_min_use': 0.1},
            'supply_milp,two_hours,investment_costs'
        )
        m.run(build_only=True)
        assert not hasattr(m._backend_model, 'carrier_production_min_constraint')


    def test_loc_tech_carriers_carrier_consumption_max_constraint(self):
        """
        i for i in sets.loc_tech_carriers_con
        if i.rsplit('::', 1)[0] in sets.loc_techs_demand +
            sets.loc_techs_storage + sets.loc_techs_transmission
        and i.rsplit('::', 1)[0] not in sets.loc_techs_milp
        """

        m = build_model({}, 'simple_supply,two_hours,investment_costs')
        m.run(build_only=True)
        assert hasattr(m._backend_model, 'carrier_consumption_max_constraint')

        m = build_model({}, 'supply_milp,two_hours,investment_costs')
        m.run(build_only=True)
        assert hasattr(m._backend_model, 'carrier_consumption_max_constraint')


    def test_loc_techs_resource_max_constraint(self):
        """
        sets.loc_techs_finite_resource_supply_plus,
        """
        m = build_model({}, 'simple_supply,two_hours,investment_costs')
        m.run(build_only=True)
        assert not hasattr(m._backend_model, 'resource_max_constraint')

        m = build_model({}, 'simple_supply_and_supply_plus,two_hours,investment_costs')
        m.run(build_only=True)
        assert hasattr(m._backend_model, 'resource_max_constraint')

        m = build_model(
            {'techs.test_supply_plus.constraints.resource': np.inf},
            'simple_supply_and_supply_plus,two_hours,investment_costs'
        )
        m.run(build_only=True)
        assert hasattr(m._backend_model, 'resource_max_constraint')

    def test_loc_techs_storage_max_constraint(self):
        """
        sets.loc_techs_store
        """
        m = build_model({}, 'simple_supply,two_hours,investment_costs')
        m.run(build_only=True)
        assert not hasattr(m._backend_model, 'storage_max_constraint')

        m = build_model({}, 'simple_supply_and_supply_plus,two_hours,investment_costs')
        m.run(build_only=True)
        assert hasattr(m._backend_model, 'storage_max_constraint')

        m = build_model({}, 'simple_storage,two_hours,investment_costs')
        m.run(build_only=True)
        assert hasattr(m._backend_model, 'storage_max_constraint')

    def test_loc_tech_carriers_ramping_constraint(self):
        """
        i for i in sets.loc_tech_carriers_prod
        if i.rsplit('::', 1)[0] in sets.loc_techs_ramping
        """
        m = build_model({}, 'simple_supply,two_hours,investment_costs')
        m.run(build_only=True)
        assert not hasattr(m._backend_model, 'ramping_up_constraint')
        assert not hasattr(m._backend_model, 'ramping_down_constraint')

        m = build_model(
            {'techs.test_supply_elec.constraints.energy_ramping': 0.1},
            'simple_supply,two_hours,investment_costs'
        )
        m.run(build_only=True)
        assert hasattr(m._backend_model, 'ramping_up_constraint')
        assert hasattr(m._backend_model, 'ramping_down_constraint')

        m = build_model(
            {'techs.test_conversion.constraints.energy_ramping': 0.1},
            'simple_conversion,two_hours,investment_costs'
        )
        m.run(build_only=True)
        assert hasattr(m._backend_model, 'ramping_up_constraint')
        assert hasattr(m._backend_model, 'ramping_down_constraint')

    # milp.py
    def test_loc_techs_unit_commitment_constraint(self):
        """
        sets.loc_techs_milp,
        """
        m = build_model({}, 'simple_supply,two_hours,investment_costs')
        m.run(build_only=True)
        assert not hasattr(m._backend_model, 'unit_commitment_constraint')

        m = build_model({}, 'supply_milp,two_hours,investment_costs')
        m.run(build_only=True)
        assert hasattr(m._backend_model, 'unit_commitment_constraint')

        m = build_model({}, 'supply_purchase,two_hours,investment_costs')
        m.run(build_only=True)
        assert not hasattr(m._backend_model, 'unit_commitment_constraint')

    def test_loc_techs_unit_capacity_constraint(self):
        """
        sets.loc_techs_milp,
        """
        m = build_model({}, 'simple_supply,two_hours,investment_costs')
        m.run(build_only=True)
        assert not hasattr(m._backend_model, 'unit_capacity_constraint')

        m = build_model({}, 'supply_milp,two_hours,investment_costs')
        m.run(build_only=True)
        assert hasattr(m._backend_model, 'unit_capacity_constraint')

        m = build_model({}, 'supply_purchase,two_hours,investment_costs')
        m.run(build_only=True)
        assert not hasattr(m._backend_model, 'unit_capacity_constraint')

    def test_loc_tech_carriers_carrier_production_max_milp_constraint(self):
        """
        i for i in sets.loc_tech_carriers_prod
        if i not in sets.loc_tech_carriers_conversion_plus
        and i.rsplit('::', 1)[0] in sets.loc_techs_milp
        """
        m = build_model({}, 'simple_supply,two_hours,investment_costs')
        m.run(build_only=True)
        assert not hasattr(m._backend_model, 'carrier_production_max_milp_constraint')

        m = build_model({}, 'supply_milp,two_hours,investment_costs')
        m.run(build_only=True)
        assert hasattr(m._backend_model, 'carrier_production_max_milp_constraint')

        m = build_model({}, 'supply_purchase,two_hours,investment_costs')
        m.run(build_only=True)
        assert not hasattr(m._backend_model, 'carrier_production_max_milp_constraint')

        m = build_model({}, 'conversion_plus_milp,two_hours,investment_costs')
        m.run(build_only=True)
        assert not hasattr(m._backend_model, 'carrier_production_max_milp_constraint')

    def test_loc_techs_carrier_production_max_conversion_plus_milp_constraint(self):
        """
        i for i in sets.loc_techs_conversion_plus
        if i in sets.loc_techs_milp
        """
        m = build_model({}, 'simple_supply,two_hours,investment_costs')
        m.run(build_only=True)
        assert not hasattr(
            m._backend_model,
            'carrier_production_max_conversion_plus_milp_constraint'
        )

        m = build_model({}, 'supply_milp,two_hours,investment_costs')
        m.run(build_only=True)
        assert not hasattr(
            m._backend_model,
            'carrier_production_max_conversion_plus_milp_constraint'
        )

        m = build_model({}, 'supply_purchase,two_hours,investment_costs')
        m.run(build_only=True)
        assert not hasattr(
            m._backend_model,
            'carrier_production_max_conversion_plus_milp_constraint'
        )

        m = build_model({}, 'conversion_plus_milp,two_hours,investment_costs')
        m.run(build_only=True)
        assert hasattr(
            m._backend_model,
            'carrier_production_max_conversion_plus_milp_constraint'
        )

        m = build_model({}, 'conversion_plus_purchase,two_hours,investment_costs')
        m.run(build_only=True)
        assert not hasattr(
            m._backend_model,
            'carrier_production_max_conversion_plus_milp_constraint'
        )

    def test_loc_tech_carriers_carrier_production_min_milp_constraint(self):
        """
        i for i in sets.loc_tech_carriers_prod
        if i not in sets.loc_tech_carriers_conversion_plus
        and constraint_exists(model_run, i.rsplit('::', 1)[0], 'constraints.energy_cap_min_use')
        and i.rsplit('::', 1)[0] in sets.loc_techs_milp
        """
        m = build_model(
            {'techs.test_supply_elec.constraints.energy_cap_min_use': 0.1},
            'simple_supply,two_hours,investment_costs'
        )
        m.run(build_only=True)
        assert not hasattr(m._backend_model, 'carrier_production_min_milp_constraint')

        m = build_model(
            {'techs.test_supply_elec.constraints.energy_cap_min_use': 0.1},
            'supply_milp,two_hours,investment_costs'
        )
        m.run(build_only=True)
        assert hasattr(m._backend_model, 'carrier_production_min_milp_constraint')

        m = build_model(
            {'techs.test_supply_elec.constraints.energy_cap_min_use': 0.1},
            'supply_purchase,two_hours,investment_costs'
        )
        m.run(build_only=True)
        assert not hasattr(m._backend_model, 'carrier_production_min_milp_constraint')

        m = build_model(
            {'techs.test_supply_elec.constraints.energy_cap_min_use': 0.1},
            'conversion_plus_milp,two_hours,investment_costs'
        )
        m.run(build_only=True)
        assert not hasattr(m._backend_model, 'carrier_production_min_milp_constraint')

        m = build_model(
            {'techs.test_conversion_plus.constraints.energy_cap_min_use': 0.1},
            'conversion_plus_milp,two_hours,investment_costs'
        )
        m.run(build_only=True)
        assert not hasattr(m._backend_model, 'carrier_production_min_milp_constraint')

    def test_loc_techs_carrier_production_min_conversion_plus_milp_constraint(self):
        """
        i for i in sets.loc_techs_conversion_plus
        if constraint_exists(model_run, i, 'constraints.energy_cap_min_use')
        and i in sets.loc_techs_milp
        """
        m = build_model(
            {'techs.test_supply_elec.constraints.energy_cap_min_use': 0.1},
            'simple_supply,two_hours,investment_costs'
        )
        m.run(build_only=True)
        assert not hasattr(
            m._backend_model,
            'carrier_production_min_conversion_plus_milp_constraint'
        )

        m = build_model(
            {'techs.test_supply_elec.constraints.energy_cap_min_use': 0.1},
            'supply_milp,two_hours,investment_costs'
        )
        m.run(build_only=True)
        assert not hasattr(
            m._backend_model,
            'carrier_production_min_conversion_plus_milp_constraint'
        )

        m = build_model(
            {'techs.test_supply_elec.constraints.energy_cap_min_use': 0.1},
            'conversion_plus_milp,two_hours,investment_costs'
        )
        m.run(build_only=True)
        assert not hasattr(
            m._backend_model,
            'carrier_production_min_conversion_plus_milp_constraint'
        )

        m = build_model(
            {'techs.test_conversion_plus.constraints.energy_cap_min_use': 0.1},
            'conversion_plus_milp,two_hours,investment_costs'
        )
        m.run(build_only=True)
        assert hasattr(
            m._backend_model,
            'carrier_production_min_conversion_plus_milp_constraint'
        )

        m = build_model(
            {'techs.test_conversion_plus.constraints.energy_cap_min_use': 0.1},
            'conversion_plus_purchase,two_hours,investment_costs'
        )
        m.run(build_only=True)
        assert not hasattr(
            m._backend_model,
            'carrier_production_min_conversion_plus_milp_constraint'
        )

    def test_loc_tech_carriers_carrier_consumption_max_milp_constraint(self):
        """
        i for i in sets.loc_tech_carriers_con
        if i.rsplit('::', 1)[0] in sets.loc_techs_demand +
            sets.loc_techs_storage + sets.loc_techs_transmission
        and i.rsplit('::', 1)[0] in sets.loc_techs_milp
        """
        m = build_model({}, 'simple_supply,two_hours,investment_costs')
        m.run(build_only=True)
        assert not hasattr(m._backend_model, 'carrier_consumption_max_milp_constraint')

        m = build_model({}, 'supply_milp,two_hours,investment_costs')
        m.run(build_only=True)
        assert not hasattr(m._backend_model, 'carrier_consumption_max_milp_constraint')

        m = build_model({}, 'storage_milp,two_hours,investment_costs')
        m.run(build_only=True)
        assert hasattr(m._backend_model, 'carrier_consumption_max_milp_constraint')

        m = build_model({}, 'conversion_plus_milp,two_hours,investment_costs')
        m.run(build_only=True)
        assert not hasattr(m._backend_model, 'carrier_consumption_max_milp_constraint')

    def test_loc_techs_energy_capacity_units_constraint(self):
        """
        i for i in sets.loc_techs_milp
        if constraint_exists(model_run, i, 'constraints.energy_cap_per_unit')
        is not None
        """
        m = build_model({}, 'simple_supply,two_hours,investment_costs')
        m.run(build_only=True)
        assert not hasattr(m._backend_model, 'energy_capacity_units_constraint')

        m = build_model({}, 'supply_milp,two_hours,investment_costs')
        m.run(build_only=True)
        assert hasattr(m._backend_model, 'energy_capacity_units_constraint')

        m = build_model({}, 'storage_milp,two_hours,investment_costs')
        m.run(build_only=True)
        assert hasattr(m._backend_model, 'energy_capacity_units_constraint')

        m = build_model({}, 'conversion_plus_milp,two_hours,investment_costs')
        m.run(build_only=True)
        assert hasattr(m._backend_model, 'energy_capacity_units_constraint')

    def test_loc_techs_storage_capacity_units_constraint(self):
        """
        i for i in sets.loc_techs_milp if i in sets.loc_techs_store
        """
        m = build_model({}, 'simple_supply,two_hours,investment_costs')
        m.run(build_only=True)
        assert not hasattr(m._backend_model, 'storage_capacity_units_constraint')

        m = build_model({}, 'supply_milp,two_hours,investment_costs')
        m.run(build_only=True)
        assert not hasattr(m._backend_model, 'storage_capacity_units_constraint')

        m = build_model({}, 'storage_milp,two_hours,investment_costs')
        m.run(build_only=True)
        assert hasattr(m._backend_model, 'storage_capacity_units_constraint')

        m = build_model({}, 'conversion_plus_milp,two_hours,investment_costs')
        m.run(build_only=True)
        assert not hasattr(m._backend_model, 'storage_capacity_units_constraint')

        m = build_model({}, 'supply_and_supply_plus_milp,two_hours,investment_costs')
        m.run(build_only=True)
        assert hasattr(m._backend_model, 'storage_capacity_units_constraint')

    def test_loc_techs_energy_capacity_max_purchase_constraint(self):
        """
        i for i in sets.loc_techs_purchase
        if (constraint_exists(model_run, i, 'constraints.energy_cap_equals') is not None
            or constraint_exists(model_run, i, 'constraints.energy_cap_max') is not None)
        """
        m = build_model({}, 'simple_supply,two_hours,investment_costs')
        m.run(build_only=True)
        assert not hasattr(m._backend_model, 'energy_capacity_max_purchase_constraint')

        m = build_model({}, 'supply_milp,two_hours,investment_costs')
        m.run(build_only=True)
        assert not hasattr(m._backend_model, 'energy_capacity_max_purchase_constraint')

        m = build_model({}, 'supply_purchase,two_hours,investment_costs')
        m.run(build_only=True)
        assert hasattr(m._backend_model, 'energy_capacity_max_purchase_constraint')

        m = build_model(
            {'techs.test_supply_elec.constraints': {'energy_cap_max': None, 'energy_cap_equals': 15}},
            'supply_purchase,two_hours,investment_costs'
        )
        m.run(build_only=True)
        assert hasattr(m._backend_model, 'energy_capacity_max_purchase_constraint')

    def test_loc_techs_energy_capacity_min_purchase_constraint(self):
        """
        i for i in sets.loc_techs_purchase
        if (not constraint_exists(model_run, i, 'constraints.energy_cap_equals')
            and constraint_exists(model_run, i, 'constraints.energy_cap_min'))
        """
        m = build_model({}, 'simple_supply,two_hours,investment_costs')
        m.run(build_only=True)
        assert not hasattr(m._backend_model, 'energy_capacity_min_purchase_constraint')

        m = build_model({}, 'supply_milp,two_hours,investment_costs')
        m.run(build_only=True)
        assert not hasattr(m._backend_model, 'energy_capacity_min_purchase_constraint')

        m = build_model({}, 'supply_purchase,two_hours,investment_costs')
        m.run(build_only=True)
        assert not hasattr(m._backend_model, 'energy_capacity_min_purchase_constraint')

        m = build_model(
            {'techs.test_supply_elec.constraints': {'energy_cap_max': None, 'energy_cap_equals': 15}},
            'supply_purchase,two_hours,investment_costs'
        )
        m.run(build_only=True)
        assert not hasattr(m._backend_model, 'energy_capacity_min_purchase_constraint')

        m = build_model(
            {'techs.test_supply_elec.constraints.energy_cap_min': 10},
            'supply_purchase,two_hours,investment_costs'
        )
        m.run(build_only=True)
        assert hasattr(m._backend_model, 'energy_capacity_min_purchase_constraint')

    def test_loc_techs_storage_capacity_max_purchase_constraint(self):
        """
        i for i in set(sets.loc_techs_purchase).intersection(sets.loc_techs_store)
        """
        m = build_model({}, 'simple_storage,two_hours,investment_costs')
        m.run(build_only=True)
        assert not hasattr(m._backend_model, 'storage_capacity_max_purchase_constraint')

        m = build_model({}, 'storage_milp,two_hours,investment_costs')
        m.run(build_only=True)
        assert not hasattr(m._backend_model, 'storage_capacity_max_purchase_constraint')

        m = build_model({}, 'storage_purchase,two_hours,investment_costs')
        m.run(build_only=True)
        assert hasattr(m._backend_model, 'storage_capacity_max_purchase_constraint')

        m = build_model({}, 'supply_purchase,two_hours,investment_costs')
        m.run(build_only=True)
        assert not hasattr(m._backend_model, 'storage_capacity_max_purchase_constraint')

    def test_loc_techs_storage_capacity_min_purchase_constraint(self):
        """
        i for i in set(sets.loc_techs_purchase).intersection(sets.loc_techs_store)
        if (not constraint_exists(model_run, i, 'constraints.storage_cap_equals')
            and (constraint_exists(model_run, i, 'constraints.storage_cap_min')
                or constraint_exists(model_run, i, 'constraints.energy_cap_min')))
        """
        m = build_model(
            {'techs.test_storage.constraints.storage_cap_min': 10},
            'simple_storage,two_hours,investment_costs'
        )
        m.run(build_only=True)
        assert not hasattr(m._backend_model, 'storage_capacity_min_purchase_constraint')

        m = build_model(
            {'techs.test_storage.constraints.storage_cap_min': 10},
            'storage_milp,two_hours,investment_costs'
        )
        m.run(build_only=True)
        assert not hasattr(m._backend_model, 'storage_capacity_min_purchase_constraint')

        m = build_model({}, 'storage_purchase,two_hours,investment_costs')
        m.run(build_only=True)
        assert not hasattr(m._backend_model, 'storage_capacity_min_purchase_constraint')

        m = build_model(
            {'techs.test_storage.constraints.storage_cap_min': 10},
            'storage_purchase,two_hours,investment_costs'
        )
        m.run(build_only=True)
        assert hasattr(m._backend_model, 'storage_capacity_min_purchase_constraint')

        m = build_model(
            {'techs.test_storage.constraints': {'storage_cap_equals': 10, 'storage_cap_min': 10}},
            'storage_purchase,two_hours,investment_costs'
        )
        m.run(build_only=True)
        assert not hasattr(m._backend_model, 'storage_capacity_min_purchase_constraint')

    def test_loc_techs_update_costs_investment_units_constraint(self):
        """
        i for i in sets.loc_techs_milp
        if i in sets.loc_techs_investment_cost and
        any(constraint_exists(model_run, i, 'costs.{}.purchase'.format(j))
               for j in model_run.sets.costs)
        """
        m = build_model({}, 'simple_supply,two_hours,investment_costs')
        m.run(build_only=True)
        assert not check_variable_exists(m._backend_model, 'cost_investment_constraint', 'purchased')
        assert not check_variable_exists(m._backend_model, 'cost_investment_constraint', 'units')

        m = build_model({}, 'supply_milp,two_hours,investment_costs')
        m.run(build_only=True)
        assert not check_variable_exists(m._backend_model, 'cost_investment_constraint', 'purchased')
        assert not check_variable_exists(m._backend_model, 'cost_investment_constraint', 'units')

        m = build_model(
            {'techs.test_supply_elec.costs.monetary.purchase': 1},
            'supply_milp,two_hours,investment_costs'
        )
        m.run(build_only=True)
        assert not check_variable_exists(m._backend_model, 'cost_investment_constraint', 'purchased')
        assert check_variable_exists(m._backend_model, 'cost_investment_constraint', 'units')

    def test_loc_techs_update_costs_investment_purchase_constraint(self):
        """
        sets.loc_techs_purchase,
        """

        m = build_model({}, 'supply_purchase,two_hours,investment_costs')
        m.run(build_only=True)
        assert check_variable_exists(m._backend_model, 'cost_investment_constraint', 'purchased')
        assert not check_variable_exists(m._backend_model, 'cost_investment_constraint', 'units')

    # conversion.py
    def test_loc_techs_balance_conversion_constraint(self):
        """
        sets.loc_techs_conversion,
        """
        m = build_model({}, 'simple_supply,two_hours,investment_costs')
        m.run(build_only=True)
        assert not hasattr(m._backend_model, 'balance_conversion_constraint')

        m = build_model({}, 'simple_conversion,two_hours,investment_costs')
        m.run(build_only=True)
        assert hasattr(m._backend_model, 'balance_conversion_constraint')

        m = build_model({}, 'simple_conversion_plus,two_hours,investment_costs')
        m.run(build_only=True)
        assert not hasattr(m._backend_model, 'balance_conversion_constraint')

    def test_loc_techs_cost_var_conversion_constraint(self):
        """
        sets.loc_techs_om_cost_conversion,
        """
        m = build_model(
            {'techs.test_supply_elec.costs.monetary.om_prod': 0.1},
            'simple_supply,two_hours,investment_costs'
        )
        m.run(build_only=True)
        assert not hasattr(m._backend_model, 'cost_var_conversion_constraint')

        m = build_model({}, 'simple_conversion,two_hours,investment_costs')
        m.run(build_only=True)
        assert not hasattr(m._backend_model, 'cost_var_conversion_constraint')

        m = build_model(
            {'techs.test_conversion.costs.monetary.om_prod': 0.1},
            'simple_conversion,two_hours,investment_costs'
        )
        m.run(build_only=True)
        assert hasattr(m._backend_model, 'cost_var_conversion_constraint')

        m = build_model(
            {'techs.test_conversion.costs.monetary.om_con': 0.1},
            'simple_conversion,two_hours,investment_costs'
        )
        m.run(build_only=True)
        assert hasattr(m._backend_model, 'cost_var_conversion_constraint')

    # conversion_plus.py
    def test_loc_techs_balance_conversion_plus_primary_constraint(self):
        """
        sets.loc_techs_conversion_plus,
        """
        m = build_model({}, 'simple_supply,two_hours,investment_costs')
        m.run(build_only=True)
        assert not hasattr(m._backend_model, 'balance_conversion_plus_primary_constraint')

        m = build_model({}, 'simple_conversion_plus,two_hours,investment_costs')
        m.run(build_only=True)
        assert hasattr(m._backend_model, 'balance_conversion_plus_primary_constraint')

        m = build_model(
            {'techs.test_conversion_plus.essentials.carrier_out': ['electricity', 'heat']},
            'simple_conversion_plus,two_hours,investment_costs'
        )
        m.run(build_only=True)
        assert hasattr(m._backend_model, 'balance_conversion_plus_primary_constraint')

        m = build_model(
            {'techs.test_conversion_plus.essentials.carrier_in': ['coal', 'gas']},
            'simple_conversion_plus,two_hours,investment_costs'
        )
        m.run(build_only=True)
        assert hasattr(m._backend_model, 'balance_conversion_plus_primary_constraint')

    def test_loc_techs_carrier_production_max_conversion_plus_constraint(self):
        """
        i for i in sets.loc_techs_conversion_plus
        if i not in sets.loc_techs_milp
        """

        m = build_model({}, 'simple_conversion_plus,two_hours,investment_costs')
        m.run(build_only=True)
        assert hasattr(m._backend_model, 'carrier_production_max_conversion_plus_constraint')

        m = build_model({}, 'conversion_plus_milp,two_hours,investment_costs')
        m.run(build_only=True)
        assert not hasattr(m._backend_model, 'carrier_production_max_conversion_plus_constraint')

    def test_loc_techs_carrier_production_min_conversion_plus_constraint(self):
        """
        i for i in sets.loc_techs_conversion_plus
        if constraint_exists(model_run, i, 'constraints.energy_cap_min_use')
        and i not in sets.loc_techs_milp
        """

        m = build_model({}, 'simple_conversion_plus,two_hours,investment_costs')
        m.run(build_only=True)
        assert not hasattr(m._backend_model, 'carrier_production_min_conversion_plus_constraint')

        m = build_model({}, 'conversion_plus_milp,two_hours,investment_costs')
        m.run(build_only=True)
        assert not hasattr(m._backend_model, 'carrier_production_min_conversion_plus_constraint')

        m = build_model(
            {'techs.test_conversion_plus.constraints.energy_cap_min_use': 0.1},
            'conversion_plus_milp,two_hours,investment_costs'
        )
        m.run(build_only=True)
        assert not hasattr(m._backend_model, 'carrier_production_min_conversion_plus_constraint')

        m = build_model(
            {'techs.test_conversion_plus.constraints.energy_cap_min_use': 0.1},
            'simple_conversion_plus,two_hours,investment_costs'
        )
        m.run(build_only=True)
        assert hasattr(m._backend_model, 'carrier_production_min_conversion_plus_constraint')

    def test_loc_techs_cost_var_conversion_plus_constraint(self):
        """
        sets.loc_techs_om_cost_conversion_plus,
        """

        m = build_model(
            {'techs.test_supply_elec.costs.monetary.om_prod': 0.1},
            'simple_supply,two_hours,investment_costs'
        )
        m.run(build_only=True)
        assert not hasattr(m._backend_model, 'cost_var_conversion_plus_constraint')

        m = build_model(
            {'techs.test_conversion.costs.monetary.om_prod': 0.1},
            'simple_conversion,two_hours,investment_costs'
        )
        m.run(build_only=True)
        assert not hasattr(m._backend_model, 'cost_var_conversion_plus_constraint')


        m = build_model({}, 'simple_conversion_plus,two_hours,investment_costs')
        m.run(build_only=True)
        assert not hasattr(m._backend_model, 'cost_var_conversion_plus_constraint')

        m = build_model(
            {'techs.test_conversion_plus.costs.monetary.om_prod': 0.1},
            'simple_conversion_plus,two_hours,investment_costs'
        )
        m.run(build_only=True)
        assert hasattr(m._backend_model, 'cost_var_conversion_plus_constraint')

        m = build_model(
            {'techs.test_conversion_plus.costs.monetary.om_con': 0.1},
            'simple_conversion_plus,two_hours,investment_costs'
        )
        m.run(build_only=True)
        assert hasattr(m._backend_model, 'cost_var_conversion_plus_constraint')

    def test_loc_techs_balance_conversion_plus_in_2_constraint(self):
        """
        sets.loc_techs_in_2,
        """


        m = build_model({}, 'simple_conversion_plus,two_hours,investment_costs')
        m.run(build_only=True)
        assert not hasattr(m._backend_model, 'balance_conversion_plus_in_2_constraint')

        m = build_model(
            {'techs.test_conversion_plus.essentials.carrier_in_2': 'coal'},
            'simple_conversion_plus,two_hours,investment_costs'
        )
        m.run(build_only=True)
        assert hasattr(m._backend_model, 'balance_conversion_plus_in_2_constraint')

        m = build_model(
            {'techs.test_conversion_plus.essentials.carrier_in_2': ['coal', 'heat']},
            'simple_conversion_plus,two_hours,investment_costs'
        )
        m.run(build_only=True)
        assert hasattr(m._backend_model, 'balance_conversion_plus_in_2_constraint')

    def test_loc_techs_balance_conversion_plus_in_3_constraint(self):
        """
        sets.loc_techs_in_3,
        """

        m = build_model({}, 'simple_conversion_plus,two_hours,investment_costs')
        m.run(build_only=True)
        assert not hasattr(m._backend_model, 'balance_conversion_plus_in_3_constraint')

        m = build_model(
            {'techs.test_conversion_plus.essentials.carrier_in_3': 'coal'},
            'simple_conversion_plus,two_hours,investment_costs'
        )
        m.run(build_only=True)
        assert hasattr(m._backend_model, 'balance_conversion_plus_in_3_constraint')

        m = build_model(
            {'techs.test_conversion_plus.essentials.carrier_in_3': ['coal', 'heat']},
            'simple_conversion_plus,two_hours,investment_costs'
        )
        m.run(build_only=True)
        assert hasattr(m._backend_model, 'balance_conversion_plus_in_3_constraint')

    def test_loc_techs_balance_conversion_plus_out_2_constraint(self):
        """
        sets.loc_techs_out_2,
        """

        m = build_model(
            {'techs.test_conversion_plus.essentials.carrier_out_2': ['coal', 'heat']},
            'simple_conversion_plus,two_hours,investment_costs'
        )
        m.run(build_only=True)
        assert hasattr(m._backend_model, 'balance_conversion_plus_out_2_constraint')

    def test_loc_techs_balance_conversion_plus_out_3_constraint(self):
        """
        sets.loc_techs_out_3,
        """

        m = build_model({}, 'simple_conversion_plus,two_hours,investment_costs')
        m.run(build_only=True)
        assert not hasattr(m._backend_model, 'balance_conversion_plus_out_3_constraint')

        m = build_model(
            {'techs.test_conversion_plus.essentials.carrier_out_3': 'coal'},
            'simple_conversion_plus,two_hours,investment_costs'
        )
        m.run(build_only=True)
        assert hasattr(m._backend_model, 'balance_conversion_plus_out_3_constraint')

        m = build_model(
            {'techs.test_conversion_plus.essentials.carrier_out_3': ['coal', 'heat']},
            'simple_conversion_plus,two_hours,investment_costs'
        )
        m.run(build_only=True)
        assert hasattr(m._backend_model, 'balance_conversion_plus_out_3_constraint')

    # network.py
    def test_loc_techs_symmetric_transmission_constraint(self):
        """
        sets.loc_techs_transmission,
        """
        m = build_model({}, 'simple_supply,two_hours,investment_costs')
        m.run(build_only=True)
        assert hasattr(m._backend_model, 'symmetric_transmission_constraint')


        m = build_model({}, 'simple_conversion_plus,two_hours,investment_costs')
        m.run(build_only=True)
        assert not hasattr(m._backend_model, 'symmetric_transmission_constraint')

    # policy.py
    def test_techlists_group_share_energy_cap_min_constraint(self):
        """
        i for i in sets.techlists
        if 'energy_cap_min' in model_run.model.get_key('group_share.{}'.format(i), {}).keys()
        """
        m = build_model({}, 'simple_supply,group_share_energy_cap_min,two_hours,investment_costs')
        m.run(build_only=True)
        assert hasattr(m._backend_model, 'group_share_energy_cap_min_constraint')
        assert not hasattr(m._backend_model, 'group_share_energy_cap_max_constraint')
        assert not hasattr(m._backend_model, 'group_share_energy_cap_equals_constraint')

    def test_techlists_group_share_energy_cap_max_constraint(self):
        """
        i for i in sets.techlists
        if 'energy_cap_max' in model_run.model.get_key('group_share.{}'.format(i), {}).keys()
        """
        m = build_model({}, 'simple_supply,group_share_energy_cap_max,two_hours,investment_costs')
        m.run(build_only=True)
        assert not hasattr(m._backend_model, 'group_share_energy_cap_min_constraint')
        assert hasattr(m._backend_model, 'group_share_energy_cap_max_constraint')
        assert not hasattr(m._backend_model, 'group_share_energy_cap_equals_constraint')

    def test_techlists_group_share_energy_cap_equals_constraint(self):
        """
        i for i in sets.techlists
        if 'energy_cap_equals' in model_run.model.get_key('group_share.{}'.format(i), {}).keys()
        """
        m = build_model({}, 'simple_supply,group_share_energy_cap_equals,two_hours,investment_costs')
        m.run(build_only=True)
        assert not hasattr(m._backend_model, 'group_share_energy_cap_min_constraint')
        assert not hasattr(m._backend_model, 'group_share_energy_cap_max_constraint')
        assert hasattr(m._backend_model, 'group_share_energy_cap_equals_constraint')

    def test_techlists_carrier_group_share_carrier_prod_min_constraint(self):
        """
        i + '::' + carrier
        for i in sets.techlists
        if 'carrier_prod_min' in model_run.model.get_key('group_share.{}'.format(i), {}).keys()
        for carrier in sets.carriers
        if carrier in model_run.model.get_key('group_share.{}.carrier_prod_min'.format(i), {}).keys()
        """

        m = build_model({}, 'conversion_and_conversion_plus,group_share_carrier_prod_min,two_hours,investment_costs')
        m.run(build_only=True)
        assert hasattr(m._backend_model, 'group_share_carrier_prod_min_constraint')
        assert not hasattr(m._backend_model, 'group_share_carrier_prod_max_constraint')
        assert not hasattr(m._backend_model, 'group_share_carrier_prod_equals_constraint')

    def test_techlists_carrier_group_share_carrier_prod_max_constraint(self):
        """
        i + '::' + carrier
        for i in sets.techlists
        if 'carrier_prod_max' in model_run.model.get_key('group_share.{}'.format(i), {}).keys()
        for carrier in sets.carriers
        if carrier in model_run.model.get_key('group_share.{}.carrier_prod_max'.format(i), {}).keys()
        """

        m = build_model({}, 'conversion_and_conversion_plus,group_share_carrier_prod_max,two_hours,investment_costs')
        m.run(build_only=True)
        assert not hasattr(m._backend_model, 'group_share_carrier_prod_min_constraint')
        assert hasattr(m._backend_model, 'group_share_carrier_prod_max_constraint')
        assert not hasattr(m._backend_model, 'group_share_carrier_prod_equals_constraint')

    def test_techlists_carrier_group_share_carrier_prod_equals_constraint(self):
        """
        i + '::' + carrier
        for i in sets.techlists
        if 'carrier_prod_equals' in model_run.model.get_key('group_share.{}'.format(i), {}).keys()
        for carrier in sets.carriers
        if carrier in model_run.model.get_key('group_share.{}.carrier_prod_equals'.format(i), {}).keys()
        """

        m = build_model({}, 'conversion_and_conversion_plus,group_share_carrier_prod_equals,two_hours,investment_costs')
        m.run(build_only=True)
        assert not hasattr(m._backend_model, 'group_share_carrier_prod_min_constraint')
        assert not hasattr(m._backend_model, 'group_share_carrier_prod_max_constraint')
        assert hasattr(m._backend_model, 'group_share_carrier_prod_equals_constraint')
