import pytest  # pylint: disable=unused-import
import pandas as pd

import calliope
import calliope.exceptions as exceptions

from calliope.test.common.util import build_test_model as build_model
from calliope.test.common.util import check_error_or_warning


@pytest.fixture(scope='class')
def model():
    m = build_model({}, 'simple_supply,two_hours,investment_costs')
    m.run()
    return m


class TestGetInputParams:
    def test_get_input_params(self, model):
        """
        Test that the function access_model_inputs works
        """

        inputs = model.backend.access_model_inputs()
        assert (
            set(model.inputs.data_vars).symmetric_difference(inputs.data_vars)
            == {'objective_cost_class'}
        )


class TestUpdateParam:
    def test_update_param_single_dim(self, model):
        """
        test that the function update_param works with a single dimension
        """

        model.backend.update_param('energy_cap_max', {'1::test_supply_elec': 20})

        assert (
            model._backend_model.energy_cap_max.extract_values()['1::test_supply_elec'] == 20
        )

    def test_update_param_multiple_vals(self, model):
        """
        test that the function update_param works with a single dimension
        """

        model.backend.update_param(
            'energy_cap_max', {'1::test_supply_elec': 20, '0::test_supply_elec': 30}
        )

        assert (
            model._backend_model.energy_cap_max.extract_values()['1::test_supply_elec'] == 20
        )
        assert (
            model._backend_model.energy_cap_max.extract_values()['0::test_supply_elec'] == 30
        )

    def test_update_param_multiple_dim(self, model):
        """
        test that the function update_param works with multiple dimensions
        """

        model.backend.update_param(
            'resource', {('0::test_demand_elec', pd.Timestamp('2005-01-01 01:00:00')): -10}
        )

        assert (
            model._backend_model.resource.extract_values()
            [('0::test_demand_elec', pd.Timestamp('2005-01-01 01:00:00'))] == -10
        )

    def test_unknown_param(self, model):
        """
        Raise error on unknown param
        """

        with pytest.raises(exceptions.ModelError) as excinfo:
            model.backend.update_param(
                'unknown_param', {('1::test_supply_elec'): 20}
            )

        assert check_error_or_warning(excinfo, 'Parameter `unknown_param` not in the Pyomo Backend.')

    def test_not_a_param(self, model):
        """
        Raise error when trying to update a non-Param Pyomo object
        """
        with pytest.raises(exceptions.ModelError) as excinfo:
            model.backend.update_param(
                'energy_cap', {'1::test_supply_elec': 20}
            )

        assert check_error_or_warning(excinfo, '`energy_cap` not a Parameter in the Pyomo Backend.')

        with pytest.raises(exceptions.ModelError) as excinfo:
            model.backend.update_param(
                'loc_techs', {'1::test_supply_elec': 20}
            )

        assert check_error_or_warning(excinfo, '`loc_techs` not a Parameter in the Pyomo Backend.')

    def index_not_in_param(self, model):
        """
        Raise error when accessing unknown index
        """

        with pytest.raises(KeyError, match=r"Index 'region1-xc1::csp'"):
            model.backend.update_param(
                'energy_cap_max', {'2::test_supply_elec': 20}
            )

        with pytest.raises(KeyError, match=r"Index 'region1-xc1::csp'"):
            model.backend.update_param(
                'energy_cap_max', {'1::test_supply_elec': 20, '2::test_supply_elec': 20}
            )


class TestActivateConstraint:

    def test_activate_constraint(self, model):
        """
        test that the function activate_constraint works
        """
        model.backend.activate_constraint('system_balance_constraint', active=False)
        assert not model._backend_model.system_balance_constraint.active

        model.backend.activate_constraint('system_balance_constraint', active=True)
        assert model._backend_model.system_balance_constraint.active

    def test_fail_on_activate_unknown_constraint(self, model):
        """
        test that the function activate_constraint fails if unknown constraint
        """
        with pytest.raises(exceptions.ModelError) as excinfo:
            model.backend.activate_constraint('unknown_constraint', active=False)

        assert check_error_or_warning(
            excinfo, 'constraint/objective `unknown_constraint` not in the Pyomo Backend.'
        )

    def test_fail_on_parameter_activate(self, model):
        """
        test that the function activate_constraint fails if trying to activate a
        non-constraint Pyomo object.
        """
        with pytest.raises(exceptions.ModelError) as excinfo:
            model.backend.activate_constraint('resource', active=False)

        assert check_error_or_warning(
            excinfo, '`resource` not a constraint in the Pyomo Backend.'
        )

    def test_non_boolean_parameter_activate(self, model):
        """
        test that the function activate_constraint fails when setting active to
        non-boolean
        """
        with pytest.raises(ValueError) as excinfo:
            model.backend.activate_constraint('system_balance_constraint', active=None)

        assert check_error_or_warning(excinfo, 'Argument `active` must be True or False')


class TestBackendRerun:

    def test_rerun(self, model):
        """
        test that the function rerun works
        """
        with pytest.warns(exceptions.ModelWarning) as excinfo:
            new_model = model.backend.rerun()

        assert isinstance(new_model, calliope.Model)
        for i in ['_timings', 'inputs', 'results']:
            assert hasattr(new_model, i)

        # In new_model.inputs all NaN values have been replace by their default,
        # so we can't directly compare new_model.inputs and model.inputs
        assert new_model.inputs.equals(model.backend.access_model_inputs().reindex(new_model.inputs.coords))

        assert new_model.results.equals(model.results)

        assert check_error_or_warning(
            excinfo,
            'The results of rerunning the backend model are only available'
        )

    def test_update_and_rerun(self, model):
        """
        test that the function rerun works
        """
        model.backend.update_param('energy_cap_max', {'1::test_supply_elec': 20})
        with pytest.warns(exceptions.ModelWarning) as excinfo:
            new_model = model.backend.rerun()

        assert new_model.inputs.energy_cap_max.loc[{'loc_techs': '1::test_supply_elec'}] == 20

        assert check_error_or_warning(
            excinfo,
            'The results of rerunning the backend model are only available'
        )

    def test_rerun_fail_on_operate(self, model):
        # should fail if the run mode is not 'plan'
        model.run_config['mode'] = 'operate'
        with pytest.raises(exceptions.ModelError) as excinfo:
            model.backend.rerun()
        assert check_error_or_warning(
            excinfo, 'Cannot rerun the backend in operate run mode'
        )
