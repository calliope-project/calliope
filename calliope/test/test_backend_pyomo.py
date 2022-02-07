import pytest  # pylint: disable=unused-import
import numpy as np
import pyomo.core as po
import pandas as pd
import os
import collections
import logging
from itertools import product

from calliope.backend.pyomo.util import get_param
import calliope.exceptions as exceptions
from calliope.core.attrdict import AttrDict

from calliope.test.common.util import build_test_model as build_model
from calliope.test.common.util import (
    check_error_or_warning,
    check_variable_exists,
    get_indexed_constraint_body,
)


def check_standard_warning(info, warning):

    if warning == "transmission":
        return check_error_or_warning(
            info,
            "dimension loc_techs_transmission and associated variables distance, "
            "lookup_remotes were empty, so have been deleted",
        )


class TestModel:
    @pytest.mark.skip("Buggy")
    @pytest.mark.serial  # Cannot run in parallel with other tests
    def test_load_constraints_no_order(self):
        temp_file = os.path.join(
            os.path.dirname(__file__),
            "..",
            "backend",
            "pyomo",
            "constraints",
            "temp_constraint_file_for_testing.py",
        )

        # Write an empty file
        with open(temp_file, "w") as f:
            f.write("")

        # Should fail, since the empty .py file is included in the list, but
        # has no attribute 'ORDER'.
        with pytest.raises(AttributeError) as excinfo:
            m = build_model({}, "simple_supply,two_hours,investment_costs")
            m.run(build_only=True)

        # We can't use `with` because reasons related to Windows,
        # so we manually remove the temp file in the end
        os.remove(temp_file)

        assert check_error_or_warning(
            excinfo,
            "module 'calliope.backend.pyomo.constraints.temp_constraint_file_for_testing' "
            "has no attribute 'ORDER'",
        )

    @pytest.mark.parametrize(
        "var, domain",
        (
            ("energy_cap_max", "NonNegativeReals"),
            ("resource", "Reals"),
            ("cost_energy_cap", "Reals"),
            ("force_resource", "Boolean"),
            ("names", "Any"),
        ),
    )
    def test_domains(self, var, domain):
        m = build_model({}, "simple_supply,two_hours,investment_costs")
        m.run(build_only=True)
        assert getattr(m._backend_model, var).domain.name == domain

    def test_first_timestep(self):
        """
        Pyomo likes to 1-index its Sets, which we now expect.
        This test will fail if they ever decide to move to the more pythonic zero-indexing.
        """
        m = build_model({}, "simple_supply,two_hours,investment_costs")
        m.run(build_only=True)
        timestep_0 = "2005-01-01 00:00"
        assert m._backend_model.timesteps.ord(timestep_0) == 1


class TestChecks:
    @pytest.mark.parametrize("on", (True, False))
    def test_operate_cyclic_storage(self, on):
        """Cannot have cyclic storage in operate mode"""
        if on is True:
            override = {}  # cyclic storage is True by default
            m = build_model(
                override, "simple_supply_and_supply_plus,operate,investment_costs"
            )
            assert m.run_config["cyclic_storage"] is True
        elif on is False:
            override = {"run.cyclic_storage": False}
            m = build_model(
                override, "simple_supply_and_supply_plus,operate,investment_costs"
            )
            assert m.run_config["cyclic_storage"] is False
        with pytest.warns(exceptions.ModelWarning) as warning:
            m.run(build_only=True)
        check_warn = check_error_or_warning(
            warning, "Storage cannot be cyclic in operate run mode"
        )
        if on is True:
            assert check_warn
        elif on is True:
            assert not check_warn
        assert (
            AttrDict.from_yaml_string(m._model_data.attrs["run_config"]).cyclic_storage
            is False
        )

    @pytest.mark.parametrize(
        "param", [("energy_eff"), ("resource_eff"), ("parasitic_eff")]
    )
    def test_loading_timeseries_operate_efficiencies(self, param):
        m = build_model(
            {
                "techs.test_supply_plus.constraints."
                + param: "file=supply_plus_resource.csv:1"
            },
            "simple_supply_and_supply_plus,operate,investment_costs",
        )
        assert "timesteps" in m._model_data[param].dims

        with pytest.warns(exceptions.ModelWarning) as warning:
            m.run(build_only=True)  # will fail to complete run if there's a problem

    def test_operate_group_demand_share_per_timestep_decision(self):
        """Cannot have group_demand_share_per_timestep_decision in operate mode"""
        m = build_model(
            {},
            "simple_supply,investment_costs,operate,enable_group_demand_share_per_timestep_decision",
        )
        with pytest.warns(exceptions.ModelWarning) as warning:
            m.run(build_only=True)
        assert check_error_or_warning(
            warning, "`demand_share_per_timestep_decision` group constraints cannot be"
        )
        assert "group_demand_share_per_timestep_decision" not in m._model_data

    @pytest.mark.parametrize("force", (True, False))
    def test_operate_energy_cap_min_use(self, force):
        """If we depend on a finite energy_cap, we have to error on a user failing to define it"""
        m = build_model(
            {
                "techs.test_supply_elec.constraints": {
                    "force_resource": force,
                    "energy_cap_min_use": 0.1,
                    "resource": "file=supply_plus_resource.csv:1",
                    "energy_cap_equals": np.inf,
                }
            },
            "simple_supply_and_supply_plus,operate,investment_costs",
        )

        with pytest.raises(exceptions.ModelError) as error:
            with pytest.warns(exceptions.ModelWarning):
                m.run(build_only=True)

        assert check_error_or_warning(
            error, ["Operate mode: User must define a finite energy_cap"]
        )

    @pytest.mark.parametrize("force", (True, False))
    def test_operate_energy_cap_resource_unit(self, force):
        """If we depend on a finite energy_cap, we have to error on a user failing to define it"""
        m = build_model(
            {
                "techs.test_supply_elec.constraints": {
                    "force_resource": force,
                    "resource_unit": "energy_per_cap",
                    "resource": "file=supply_plus_resource.csv:1",
                    "energy_cap_equals": np.inf,
                    "energy_cap_max": np.inf,
                }
            },
            "simple_supply_and_supply_plus,operate,investment_costs",
        )

        if force is True:
            with pytest.raises(exceptions.ModelError) as error:
                with pytest.warns(exceptions.ModelWarning) as warning:
                    m.run(build_only=True)
            assert check_error_or_warning(
                error, ["Operate mode: User must define a finite energy_cap"]
            )
        elif force is False:
            with pytest.warns(exceptions.ModelWarning) as warning:
                m.run(build_only=True)

    @pytest.mark.parametrize(
        "resource_unit,force",
        list(product(("energy", "energy_per_cap", "energy_per_area"), (True, False))),
    )
    def test_operate_resource_unit_with_resource_area(self, resource_unit, force):
        """Different resource unit affects the capacities which are set to infinite"""
        m = build_model(
            {
                "techs.test_supply_elec.constraints": {
                    "resource_unit": resource_unit,
                    "resource_area_max": 10,
                    "energy_cap_max": 15,
                    "resource": "file=supply_plus_resource.csv:1",
                    "force_resource": force,
                }
            },
            "simple_supply_and_supply_plus,operate,investment_costs",
        )
        with pytest.warns(exceptions.ModelWarning) as warning:
            m.run(build_only=True)

        if resource_unit == "energy":
            _warnings = [
                "Energy capacity constraint removed from 0::test_supply_elec as force_resource is applied and resource is not linked to energy flow (resource_unit = `energy`)",
                "Resource area constraint removed from 0::test_supply_elec as force_resource is applied and resource is not linked to energy flow (resource_unit = `energy`)",
            ]
        elif resource_unit == "energy_per_area":
            _warnings = [
                "Energy capacity constraint removed from 0::test_supply_elec as force_resource is applied and resource is linked to energy flow using `energy_per_area`"
            ]
        elif resource_unit == "energy_per_cap":
            _warnings = [
                "Resource area constraint removed from 0::test_supply_elec as force_resource is applied and resource is linked to energy flow using `energy_per_cap`"
            ]

        if force is True:
            assert check_error_or_warning(warning, _warnings)
        elif force is False:
            assert ~check_error_or_warning(warning, _warnings)

    @pytest.mark.parametrize(
        "resource_unit", [("energy"), ("energy_per_cap"), ("energy_per_area")]
    )
    def test_operate_resource_unit_without_resource_area(self, resource_unit):
        """Different resource unit affects the capacities which are set to infinite"""
        m = build_model(
            {
                "techs.test_supply_elec.constraints": {
                    "resource_unit": resource_unit,
                    "force_resource": True,
                    "resource": "file=supply_plus_resource.csv:1",
                    "energy_cap_max": 15,
                }
            },
            "simple_supply_and_supply_plus,operate,investment_costs",
        )

        with pytest.warns(exceptions.ModelWarning) as warning:
            # energy_per_area without a resource_cap will cause an error, which we have to catch here
            if resource_unit == "energy_per_area":
                with pytest.raises(exceptions.ModelError) as error:
                    m.run(build_only=True)
            else:
                m.run(build_only=True)

        if resource_unit == "energy":
            _warnings = [
                "Energy capacity constraint removed from 0::test_supply_elec as force_resource is applied and resource is not linked to energy flow (resource_unit = `energy`)"
            ]
            not_warnings = [
                "Resource area constraint removed from 0::test_supply_elec as force_resource is applied and resource is not linked to energy flow (resource_unit = `energy`)",
                "Energy capacity constraint removed from 0::test_demand_elec as force_resource is applied and resource is not linked to energy flow (resource_unit = `energy`)",
                "Energy capacity constraint removed from 1::test_demand_elec as force_resource is applied and resource is not linked to energy flow (resource_unit = `energy`)",
            ]
        elif resource_unit == "energy_per_area":
            _warnings = [
                "Energy capacity constraint removed from 0::test_supply_elec as force_resource is applied and resource is linked to energy flow using `energy_per_area`"
            ]
            not_warnings = [
                "Resource area constraint removed from 0::test_supply_elec as force_resource is applied and resource is linked to energy flow using `energy_per_cap`",
                "Energy capacity constraint removed from 0::test_demand_elec as force_resource is applied and resource is not linked to energy flow (resource_unit = `energy`)",
                "Energy capacity constraint removed from 1::test_demand_elec as force_resource is applied and resource is not linked to energy flow (resource_unit = `energy`)",
            ]
            # energy_per_area without a resource_cap will cause an error
            check_error_or_warning(
                error,
                "Operate mode: User must define a finite resource_area "
                "(via resource_area_equals or resource_area_max) for 0::test_supply_elec",
            )
        elif resource_unit == "energy_per_cap":
            _warnings = []
            not_warnings = [
                "Resource area constraint removed from 0::test_supply_elec as force_resource is applied and resource is linked to energy flow using `energy_per_cap`",
                "Energy capacity constraint removed from 0::test_supply_elec as force_resource is applied and resource is not linked to energy flow (resource_unit = `energy`)",
                "Energy capacity constraint removed from 0::test_demand_elec as force_resource is applied and resource is not linked to energy flow (resource_unit = `energy`)",
                "Energy capacity constraint removed from 1::test_demand_elec as force_resource is applied and resource is not linked to energy flow (resource_unit = `energy`)",
            ]
        assert check_error_or_warning(warning, _warnings)
        assert not check_error_or_warning(warning, not_warnings)

    @pytest.mark.parametrize("param", ("charge_rate", "energy_cap_per_storage_cap_max"))
    def test_operate_storage(self, param):
        """Can't violate storage capacity constraints in the definition of a technology"""
        m = build_model(
            {"techs.test_supply_plus.constraints": {param: 0.1}},
            "simple_supply_and_supply_plus,operate,investment_costs",
        )

        with pytest.warns(exceptions.ModelWarning) as warning:
            with pytest.raises(exceptions.ModelError) as error:
                m.run(build_only=True)

        assert check_error_or_warning(
            error,
            "fixed storage capacity * {} is not larger than fixed energy "
            "capacity for loc::tech {}".format(param, "0::test_supply_plus"),
        )
        assert check_error_or_warning(
            warning,
            [
                "Initial stored energy not defined",
                "Resource capacity constraint defined and set to infinity",
                "Storage cannot be cyclic in operate run mode",
            ],
        )

    @pytest.mark.parametrize("on", (True, False))
    def test_operate_resource_cap_max(self, on):
        """Some constraints, if not defined, will throw a warning and possibly change values in model_data"""

        if on is False:
            override = {}
        else:
            override = {"techs.test_supply_plus.constraints.resource_cap_max": 1e6}
        m = build_model(
            override, "simple_supply_and_supply_plus,operate,investment_costs"
        )

        with pytest.warns(exceptions.ModelWarning) as warning:
            m.run(build_only=True)
        if on is False:
            assert check_error_or_warning(
                warning, "Resource capacity constraint defined and set to infinity"
            )
            assert np.isinf(
                m._model_data.resource_cap.loc["0::test_supply_plus"].item()
            )
        elif on is True:
            assert not check_error_or_warning(
                warning, "Resource capacity constraint defined and set to infinity"
            )
            assert m._model_data.resource_cap.loc["0::test_supply_plus"].item() == 1e6

    @pytest.mark.parametrize("on", (True, False))
    def test_operate_storage_initial(self, on):
        """Some constraints, if not defined, will throw a warning and possibly change values in model_data"""

        if on is False:
            override = {}
        else:
            override = {"techs.test_supply_plus.constraints.storage_initial": 0.5}
        m = build_model(
            override, "simple_supply_and_supply_plus,operate,investment_costs"
        )

        with pytest.warns(exceptions.ModelWarning) as warning:
            m.run(build_only=True)
        if on is False:
            assert check_error_or_warning(warning, "Initial stored energy not defined")
            assert m._model_data.storage_initial.loc["0::test_supply_plus"].item() == 0
        elif on is True:
            assert not check_error_or_warning(
                warning, "Initial stored energy not defined"
            )
            assert (
                m._model_data.storage_initial.loc["0::test_supply_plus"].item() == 0.5
            )


class TestBalanceConstraints:
    def test_loc_carriers_system_balance_constraint(self):
        """
        sets.loc_carriers
        """
        m = build_model({}, "simple_supply,two_hours,investment_costs")
        m.run(build_only=True)
        assert hasattr(m._backend_model, "system_balance_constraint")

    def test_loc_techs_balance_supply_constraint(self):
        """
        sets.loc_techs_finite_resource_supply,
        """
        m = build_model(
            {"techs.test_supply_elec.constraints.resource": 20},
            "simple_supply,two_hours,investment_costs",
        )
        m.run(build_only=True)
        assert hasattr(m._backend_model, "balance_supply_constraint")

        m = build_model(
            {
                "techs.test_supply_elec.constraints.resource": 20,
                "techs.test_supply_elec.constraints.resource_unit": "energy_per_cap",
            },
            "simple_supply,two_hours,investment_costs",
        )
        m.run(build_only=True)
        assert check_variable_exists(
            m._backend_model, "balance_supply_constraint", "energy_cap"
        )

        m = build_model(
            {
                "techs.test_supply_elec.constraints.resource": 20,
                "techs.test_supply_elec.constraints.resource_unit": "energy_per_area",
            },
            "simple_supply,two_hours,investment_costs",
        )
        m.run(build_only=True)
        assert check_variable_exists(
            m._backend_model, "balance_supply_constraint", "resource_area"
        )

    def test_loc_techs_balance_demand_constraint(self):
        """
        sets.loc_techs_finite_resource_demand,
        """
        m = build_model({}, "simple_supply,two_hours,investment_costs")
        m.run(build_only=True)
        assert hasattr(m._backend_model, "balance_demand_constraint")

        m = build_model(
            {"techs.test_demand_elec.constraints.resource_unit": "energy_per_cap"},
            "simple_supply,two_hours,investment_costs",
        )
        m.run(build_only=True)
        assert check_variable_exists(
            m._backend_model, "balance_demand_constraint", "energy_cap"
        )

        m = build_model(
            {"techs.test_demand_elec.constraints.resource_unit": "energy_per_area"},
            "simple_supply,two_hours,investment_costs",
        )
        m.run(build_only=True)
        assert check_variable_exists(
            m._backend_model, "balance_demand_constraint", "resource_area"
        )

    def test_loc_techs_resource_availability_supply_plus_constraint(self):
        """
        sets.loc_techs_finite_resource_supply_plus,
        """
        m = build_model({}, "simple_supply_and_supply_plus,two_hours,investment_costs")
        m.run(build_only=True)
        assert hasattr(m._backend_model, "resource_availability_supply_plus_constraint")

        m = build_model(
            {"techs.test_supply_plus.constraints.resource_unit": "energy_per_cap"},
            "simple_supply_and_supply_plus,two_hours,investment_costs",
        )
        m.run(build_only=True)
        assert check_variable_exists(
            m._backend_model,
            "resource_availability_supply_plus_constraint",
            "energy_cap",
        )

        m = build_model(
            {"techs.test_supply_plus.constraints.resource_unit": "energy_per_area"},
            "simple_supply_and_supply_plus,two_hours,investment_costs",
        )
        m.run(build_only=True)
        assert check_variable_exists(
            m._backend_model,
            "resource_availability_supply_plus_constraint",
            "resource_area",
        )

    def test_loc_techs_balance_transmission_constraint(self):
        """
        sets.loc_techs_transmission,
        """
        m = build_model({}, "simple_supply,two_hours,investment_costs")
        m.run(build_only=True)
        assert hasattr(m._backend_model, "balance_transmission_constraint")

    def test_loc_techs_balance_supply_plus_constraint(self):
        """
        sets.loc_techs_supply_plus,
        """
        m = build_model({}, "simple_supply_and_supply_plus,two_hours,investment_costs")
        m.run(build_only=True)
        assert hasattr(m._backend_model, "balance_supply_plus_constraint")

    def test_loc_techs_balance_storage_constraint(self):
        """
        sets.loc_techs_storage,
        """
        m = build_model({}, "simple_storage,two_hours,investment_costs")
        m.run(build_only=True)
        assert hasattr(m._backend_model, "balance_storage_constraint")
        assert not hasattr(m._backend_model, "storage_initial_constraint")

    def test_loc_techs_balance_storage_discharge_depth_constraint(self):
        """
        sets.loc_techs_storage,
        """
        m = build_model(
            {}, "simple_storage,two_hours,investment_costs,storage_discharge_depth"
        )
        m.run(build_only=True)
        assert hasattr(m._backend_model, "storage_discharge_depth_constraint")
        assert not hasattr(m._backend_model, "storage_initial_constraint")

        with pytest.raises(exceptions.ModelError) as error:
            m2 = build_model(
                {"techs.test_storage.constraints.storage_initial": 0.0},
                "simple_storage,one_day,investment_costs,storage_discharge_depth",
            )
            m2.run(build_only=True)

        assert check_error_or_warning(
            error, "storage_initial is smaller than storage_discharge_depth."
        )
        m3 = build_model(
            {"techs.test_storage.constraints.storage_initial": 1},
            "simple_storage,one_day,investment_costs,storage_discharge_depth",
        )
        m3.run(build_only=True)
        assert (
            m3._model_data.storage_initial.values
            > m3._model_data.storage_discharge_depth.values
        ).all()

    def test_storage_initial_constraint(self):
        """
        sets.loc_techs_store,
        """
        m = build_model({}, "simple_storage,one_day,investment_costs")
        m.run(build_only=True)
        assert hasattr(m._backend_model, "balance_storage_constraint")
        assert not hasattr(m._backend_model, "storage_initial_constraint")

        m2 = build_model(
            {"techs.test_storage.constraints.storage_initial": 0},
            "simple_storage,one_day,investment_costs",
        )
        m2.run(build_only=True)
        assert hasattr(m2._backend_model, "balance_storage_constraint")
        assert hasattr(m2._backend_model, "storage_initial_constraint")

    def test_carriers_reserve_margin_constraint(self):
        """
        i for i in sets.carriers if i in model_run.model.get_key('reserve_margin', {}).keys()
        """
        m = build_model(
            {"model.reserve_margin.electricity": 0.01},
            "simple_supply,two_hours,investment_costs",
        )
        m.run(build_only=True)
        assert hasattr(m._backend_model, "reserve_margin_constraint")


class TestCostConstraints:
    # costs.py
    def test_loc_techs_cost_constraint(self):
        """
        sets.loc_techs_cost,
        """
        m = build_model({}, "simple_supply,two_hours,investment_costs")
        m.run(build_only=True)
        assert hasattr(m._backend_model, "cost_constraint")

    def test_loc_techs_cost_investment_constraint(self):
        """
        sets.loc_techs_investment_cost,
        """
        m = build_model({}, "simple_conversion,two_hours,investment_costs")
        m.run(build_only=True)
        assert hasattr(m._backend_model, "cost_investment_constraint")

    @pytest.mark.filterwarnings("ignore:(?s).*Integer:calliope.exceptions.ModelWarning")
    def test_loc_techs_cost_investment_milp_constraint(self):
        m = build_model(
            {
                "techs.test_supply_elec.constraints.lifetime": 10,
                "techs.test_supply_elec.costs.monetary.interest_rate": 0.1,
            },
            "supply_purchase,two_hours",
        )
        m.run(build_only=True)

        assert hasattr(m._backend_model, "cost_investment_constraint")

    def test_loc_techs_not_cost_var_constraint(self):
        """
        i for i in sets.loc_techs_om_cost if i not in sets.loc_techs_conversion_plus + sets.loc_techs_conversion

        """
        m = build_model({}, "simple_conversion,two_hours,investment_costs")
        m.run(build_only=True)
        assert not hasattr(m._backend_model, "cost_var_constraint")

    @pytest.mark.parametrize(
        "tech,scenario,cost",
        (
            ("test_conversion", "simple_conversion", "om_con"),
            ("test_conversion_plus", "simple_conversion_plus", "om_prod"),
        ),
    )
    def test_loc_techs_cost_var_rhs(self, tech, scenario, cost):
        m = build_model(
            {"techs.{}.costs.monetary.{}".format(tech, cost): 1},
            "{},two_hours".format(scenario),
        )
        m.run(build_only=True)
        assert hasattr(m._backend_model, "cost_var_rhs")
        assert not hasattr(m._backend_model, "cost_var_constraint")

    @pytest.mark.parametrize(
        "tech,scenario,cost",
        (
            ("test_supply_elec", "simple_supply", "om_prod"),
            ("test_supply_elec", "simple_supply", "om_con"),
            ("test_supply_plus", "simple_supply_and_supply_plus", "om_con"),
            ("test_demand_elec", "simple_supply", "om_con"),
            ("test_transmission_elec", "simple_supply", "om_prod"),
        ),
    )
    def test_loc_techs_cost_var_constraint(self, tech, scenario, cost):
        """
        i for i in sets.loc_techs_om_cost if i not in sets.loc_techs_conversion_plus + sets.loc_techs_conversion

        """
        m = build_model(
            {"techs.{}.costs.monetary.{}".format(tech, cost): 1},
            "{},two_hours".format(scenario),
        )
        m.run(build_only=True)
        assert hasattr(m._backend_model, "cost_var_constraint")

    def test_one_way_om_cost(self):
        """
        With one_way transmission, it should still be possible to set an om_prod cost.
        """
        m = build_model(
            {
                "techs.test_transmission_elec.costs.monetary.om_prod": 1,
                "links.0,1.techs.test_transmission_elec.constraints.one_way": True,
            },
            "simple_supply,two_hours",
        )
        m.run(build_only=True)
        arg1 = m._backend_model
        arg2 = "cost_var_constraint"
        arg3 = [
            "monetary",
            "1::test_transmission_elec:0",
            m._backend_model.timesteps.at(1),
        ]
        has_cost = get_indexed_constraint_body(arg1, arg2, tuple(arg3)).to_string()

        arg3[1] = "0::test_transmission_elec:1"
        has_no_cost = get_indexed_constraint_body(arg1, arg2, tuple(arg3)).to_string()
        assert "cost_om_prod" in has_cost and "carrier_prod" in has_cost
        assert "cost_om_prod" not in has_no_cost and "carrier_prod" not in has_no_cost


class TestExportConstraints:
    # export.py
    def test_loc_carriers_update_system_balance_constraint(self):
        """
        i for i in sets.loc_carriers if sets.loc_techs_export
        and any(['{0}::{2}'.format(*j.split('::')) == i
        for j in sets.loc_tech_carriers_export])
        """

        m = build_model({}, "simple_supply,two_hours,investment_costs")
        m.run(build_only=True)
        export_exists = check_variable_exists(
            m._backend_model, "system_balance_constraint", "carrier_export"
        )
        assert not export_exists

        m = build_model({}, "supply_export,two_hours,investment_costs")
        m.run(build_only=True)

        export_exists = check_variable_exists(
            m._backend_model, "system_balance_constraint", "carrier_export"
        )
        assert export_exists

    def test_loc_tech_carriers_export_balance_constraint(self):
        """
        sets.loc_tech_carriers_export,
        """

        m = build_model({}, "supply_export,two_hours,investment_costs")
        m.run(build_only=True)

        assert hasattr(m._backend_model, "export_balance_constraint")

    def test_loc_techs_update_costs_var_constraint(self):
        """
        i for i in sets.loc_techs_om_cost if i in sets.loc_techs_export
        """

        m = build_model({}, "supply_export,two_hours,investment_costs")
        m.run(build_only=True)
        assert hasattr(m._backend_model, "cost_var_rhs")
        assert hasattr(m._backend_model, "cost_var_constraint")

        m = build_model(
            {"techs.test_supply_elec.costs.monetary.om_prod": 0.1},
            "supply_export,two_hours,investment_costs",
        )
        m.run(build_only=True)
        assert hasattr(m._backend_model, "cost_var_rhs")
        assert hasattr(m._backend_model, "cost_var_constraint")

        export_exists = check_variable_exists(
            m._backend_model, "cost_var_constraint", "carrier_export"
        )
        assert export_exists

    def test_loc_tech_carriers_export_max_constraint(self):
        """
        i for i in sets.loc_tech_carriers_export
        if constraint_exists(model_run, i.rsplit('::', 1)[0], 'constraints.export_cap')
        """

        m = build_model(
            {"techs.test_supply_elec.constraints.export_cap": 5},
            "supply_export,two_hours,investment_costs",
        )
        m.run(build_only=True)
        assert hasattr(m._backend_model, "export_max_constraint")


class TestCapacityConstraints:
    # capacity.py
    def test_loc_techs_storage_capacity_constraint(self):
        """
        i for i in sets.loc_techs_store if i not in sets.loc_techs_milp
        """
        m = build_model({}, "simple_storage,two_hours,investment_costs")
        m.run(build_only=True)
        assert hasattr(m._backend_model, "storage_capacity_constraint")

        m = build_model({}, "simple_supply_and_supply_plus,two_hours,investment_costs")
        m.run(build_only=True)
        assert hasattr(m._backend_model, "storage_capacity_constraint")

        m = build_model(
            {"techs.test_storage.constraints.storage_cap_equals": 20},
            "simple_storage,two_hours,investment_costs",
        )
        m.run(build_only=True)
        assert (
            m._backend_model.storage_capacity_constraint["0::test_storage"].upper()
            == 20
        )
        assert (
            m._backend_model.storage_capacity_constraint["0::test_storage"].lower()
            == 20
        )

    @pytest.mark.filterwarnings("ignore:(?s).*Integer:calliope.exceptions.ModelWarning")
    def test_loc_techs_storage_capacity_milp_constraint(self):
        m = build_model(
            {
                "techs.test_storage.constraints": {
                    "units_max": 1,
                    "energy_cap_per_unit": 20,
                    "storage_cap_per_unit": 20,
                }
            },
            "simple_storage,two_hours,investment_costs",
        )
        m.run(build_only=True)
        assert not hasattr(m._backend_model, "storage_capacity_constraint")

    @pytest.mark.parametrize(
        "scenario,tech,override",
        [
            i + (j,)
            for i in [
                ("simple_supply_and_supply_plus", "test_supply_plus"),
                ("simple_storage", "test_storage"),
            ]
            for j in ["max", "equals", "min"]
        ],
    )
    def test_loc_techs_energy_capacity_storage_constraint(
        self, scenario, tech, override
    ):
        """
        i for i in sets.loc_techs_store if constraint_exists(model_run, i, 'constraints.energy_cap_per_storage_cap_max')
        """
        m = build_model(
            {
                "techs.{}.constraints.energy_cap_per_storage_cap_{}".format(
                    tech, override
                ): 0.5
            },
            "{},two_hours,investment_costs".format(scenario),
        )
        m.run(build_only=True)
        assert hasattr(
            m._backend_model, "energy_capacity_storage_{}_constraint".format(override)
        )
        if override == "equals":
            assert not any(
                [
                    hasattr(
                        m._backend_model,
                        "energy_capacity_storage_{}_constraint".format(i),
                    )
                    for i in set(["max", "min"])
                ]
            )

    @pytest.mark.filterwarnings("ignore:(?s).*Integer:calliope.exceptions.ModelWarning")
    @pytest.mark.parametrize("override", (("max", "equals", "min")))
    def test_loc_techs_energy_capacity_milp_storage_constraint(self, override):
        """
        i for i in sets.loc_techs_store if constraint_exists(model_run, i, 'constraints.energy_cap_per_storage_cap_max')
        """

        m = build_model(
            {
                "techs.test_supply_plus.constraints.energy_cap_per_storage_cap_{}".format(
                    override
                ): 0.5
            },
            "supply_and_supply_plus_milp,two_hours,investment_costs",
        )
        m.run(build_only=True)
        assert hasattr(
            m._backend_model, "energy_capacity_storage_{}_constraint".format(override)
        )
        if override == "equals":
            assert not any(
                [
                    hasattr(
                        m._backend_model,
                        "energy_capacity_storage_{}_constraint".format(i),
                    )
                    for i in set(["max", "min"])
                ]
            )

    def test_no_loc_techs_energy_capacity_storage_constraint(self, caplog):
        """
        i for i in sets.loc_techs_store if constraint_exists(model_run, i, 'constraints.energy_cap_per_storage_cap_max')
        """
        with caplog.at_level(logging.INFO):
            m = build_model(model_file="energy_cap_per_storage_cap.yaml")

        assert (
            "consider defining a `energy_cap_per_storage_cap_min/max/equals` constraint"
            in caplog.text
        )

        m.run(build_only=True)
        assert not any(
            [
                hasattr(
                    m._backend_model, "energy_capacity_storage_{}_constraint".format(i)
                )
                for i in ["max", "equals", "min"]
            ]
        )

    @pytest.mark.parametrize("override", ((None, "max", "equals", "min")))
    def test_loc_techs_resource_capacity_constraint(self, override):
        """
        i for i in sets.loc_techs_finite_resource_supply_plus
        if any([constraint_exists(model_run, i, 'constraints.resource_cap_equals'),
                constraint_exists(model_run, i, 'constraints.resource_cap_max'),
                constraint_exists(model_run, i, 'constraints.resource_cap_min')])
        """

        if override is None:
            m = build_model(
                {}, "simple_supply_and_supply_plus,two_hours,investment_costs"
            )
            m.run(build_only=True)
            assert not hasattr(m._backend_model, "resource_capacity_constraint")

        else:
            m = build_model(
                {
                    "techs.test_supply_plus.constraints.resource_cap_{}".format(
                        override
                    ): 10
                },
                "simple_supply_and_supply_plus,two_hours,investment_costs",
            )
            m.run(build_only=True)
            assert hasattr(m._backend_model, "resource_capacity_constraint")

    def test_loc_techs_resource_capacity_equals_energy_capacity_constraint(self):
        """
        i for i in sets.loc_techs_finite_resource_supply_plus
        if constraint_exists(model_run, i, 'constraints.resource_cap_equals_energy_cap')
        """

        m = build_model({}, "simple_supply_and_supply_plus,two_hours,investment_costs")
        m.run(build_only=True)
        assert not hasattr(
            m._backend_model, "resource_capacity_equals_energy_capacity_constraint"
        )

        m = build_model(
            {"techs.test_supply_plus.constraints.resource_cap_equals_energy_cap": True},
            "simple_supply_and_supply_plus,two_hours,investment_costs",
        )
        m.run(build_only=True)
        assert hasattr(
            m._backend_model, "resource_capacity_equals_energy_capacity_constraint"
        )

    def test_loc_techs_resource_area_constraint(self):
        """
        i for i in sets.loc_techs_area if i in sets.loc_techs_supply_plus
        """

        m = build_model({}, "simple_supply_and_supply_plus,two_hours,investment_costs")
        m.run(build_only=True)
        assert not hasattr(m._backend_model, "resource_area_constraint")

        m = build_model(
            {"techs.test_supply_plus.constraints.resource_area_max": 10},
            "simple_supply_and_supply_plus,two_hours,investment_costs",
        )
        m.run(build_only=True)
        assert hasattr(m._backend_model, "resource_area_constraint")

        m = build_model(
            {"techs.test_supply_elec.constraints.resource_area_max": 10},
            "simple_supply_and_supply_plus,two_hours,investment_costs",
        )
        m.run(build_only=True)
        assert hasattr(m._backend_model, "resource_area_constraint")

        # Check that setting energy_cap_max to 0 also forces this constraint to 0
        m = build_model(
            {
                "techs.test_supply_plus.constraints": {
                    "resource_area_max": 10,
                    "energy_cap_max": 0,
                }
            },
            "simple_supply_and_supply_plus,two_hours,investment_costs",
        )
        m.run(build_only=True)
        assert (
            m._backend_model.resource_area_constraint["0::test_supply_plus"].upper()
            == 0
        )

    def test_loc_techs_resource_area_per_energy_capacity_constraint(self):
        """
        i for i in sets.loc_techs_area if i in sets.loc_techs_supply_plus
        and constraint_exists(model_run, i, 'constraints.resource_area_per_energy_cap')
        """
        m = build_model({}, "simple_supply_and_supply_plus,two_hours,investment_costs")
        m.run(build_only=True)
        assert not hasattr(
            m._backend_model, "resource_area_per_energy_capacity_constraint"
        )

        m = build_model(
            {"techs.test_supply_plus.constraints.resource_area_max": 10},
            "simple_supply_and_supply_plus,two_hours,investment_costs",
        )
        m.run(build_only=True)
        assert not hasattr(
            m._backend_model, "resource_area_per_energy_capacity_constraint"
        )

        m = build_model(
            {"techs.test_supply_elec.constraints.resource_area_per_energy_cap": 10},
            "simple_supply_and_supply_plus,two_hours,investment_costs",
        )
        m.run(build_only=True)
        assert hasattr(m._backend_model, "resource_area_per_energy_capacity_constraint")

        m = build_model(
            {
                "techs.test_supply_elec.constraints": {
                    "resource_area_per_energy_cap": 10,
                    "resource_area_max": 10,
                }
            },
            "simple_supply_and_supply_plus,two_hours,investment_costs",
        )
        m.run(build_only=True)
        assert hasattr(m._backend_model, "resource_area_per_energy_capacity_constraint")

    def test_locs_resource_area_capacity_per_loc_constraint(self):
        """
        i for i in sets.locs
        if model_run.locations[i].get_key('available_area', None) is not None
        """
        m = build_model({}, "simple_supply_and_supply_plus,two_hours,investment_costs")
        m.run(build_only=True)
        assert not hasattr(
            m._backend_model, "resource_area_capacity_per_loc_constraint"
        )

        m = build_model(
            {"locations.0.available_area": 1},
            "simple_supply_and_supply_plus,two_hours,investment_costs",
        )
        m.run(build_only=True)
        assert not hasattr(
            m._backend_model, "resource_area_capacity_per_loc_constraint"
        )

        m = build_model(
            {
                "locations.0.available_area": 1,
                "techs.test_supply_plus.constraints.resource_area_max": 10,
            },
            "simple_supply_and_supply_plus,two_hours,investment_costs",
        )
        m.run(build_only=True)
        assert hasattr(m._backend_model, "resource_area_capacity_per_loc_constraint")

    def test_loc_techs_energy_capacity_constraint(self):
        """
        i for i in sets.loc_techs
        if i not in sets.loc_techs_milp + sets.loc_techs_purchase
        """
        m = build_model({}, "simple_supply_and_supply_plus,two_hours,investment_costs")
        m.run(build_only=True)
        assert hasattr(m._backend_model, "energy_capacity_constraint")

        m2 = build_model(
            {"techs.test_supply_elec.constraints.energy_cap_scale": 5},
            "simple_supply_and_supply_plus,two_hours,investment_costs",
        )
        m2.run(build_only=True)
        assert (
            m2._backend_model.energy_capacity_constraint["0::test_supply_elec"].upper()
            == m._backend_model.energy_capacity_constraint[
                "0::test_supply_elec"
            ].upper()
            * 5
        )

    @pytest.mark.filterwarnings("ignore:(?s).*Integer:calliope.exceptions.ModelWarning")
    def test_loc_techs_energy_capacity_milp_constraint(self):
        m = build_model(
            {}, "supply_milp,two_hours,investment_costs"
        )  # demand still is in loc_techs
        m.run(build_only=True)
        assert hasattr(m._backend_model, "energy_capacity_constraint")

    def test_loc_techs_energy_capacity_constraint_warning_on_infinite_equals(self):
        # Check that setting `_equals` to infinity is caught:
        override = {
            "locations.0.techs.test_supply_elec.constraints.energy_cap_equals": np.inf
        }
        with pytest.raises(ValueError) as error:
            m = build_model(override, "simple_supply,two_hours,investment_costs")
            m.run(build_only=True)

        assert check_error_or_warning(
            error,
            "Cannot use inf for parameter energy_cap_equals['0::test_supply_elec']",
        )

    def test_techs_energy_capacity_systemwide_constraint(self):
        """
        i for i in sets.techs
        if model_run.get_key('techs.{}.constraints.energy_cap_max_systemwide'.format(i), None)
        """
        m = build_model({}, "simple_supply,two_hours,investment_costs")
        m.run(build_only=True)
        assert not hasattr(m._backend_model, "energy_capacity_systemwide_constraint")

        m = build_model(
            {"techs.test_supply_elec.constraints.energy_cap_max_systemwide": 20},
            "simple_supply,two_hours,investment_costs",
        )
        m.run(build_only=True)
        assert hasattr(m._backend_model, "energy_capacity_systemwide_constraint")
        assert (
            "test_supply_elec"
            in m._backend_model.energy_capacity_systemwide_constraint.keys()
        )

        # setting the constraint to infinity leads to Pyomo creating NoConstraint
        m = build_model(
            {"techs.test_supply_elec.constraints.energy_cap_max_systemwide": np.inf},
            "simple_supply,two_hours,investment_costs",
        )
        m.run(build_only=True)
        assert hasattr(m._backend_model, "energy_capacity_systemwide_constraint")
        assert (
            "test_supply_elec"
            not in m._backend_model.energy_capacity_systemwide_constraint.keys()
        )

        # Check that setting `_equals` to infinity is caught:
        with pytest.raises(ValueError) as error:
            m = build_model(
                {
                    "techs.test_supply_elec.constraints.energy_cap_equals_systemwide": np.inf
                },
                "simple_supply,two_hours,investment_costs",
            )
            m.run(build_only=True)

        assert check_error_or_warning(
            error,
            "Cannot use inf for parameter energy_cap_equals_systemwide[test_supply_elec]",
        )

        m = build_model(
            {"techs.test_supply_elec.constraints.energy_cap_equals_systemwide": 20},
            "simple_supply,two_hours,investment_costs",
        )
        m.run(build_only=True)
        assert hasattr(m._backend_model, "energy_capacity_systemwide_constraint")

        # Check that a model without transmission techs doesn't cause an error
        m = build_model(
            {"techs.test_supply_elec.constraints.energy_cap_equals_systemwide": 20},
            "simple_supply,two_hours,investment_costs",
            model_file="model_minimal.yaml",
        )
        m.run(build_only=True)
        assert hasattr(m._backend_model, "energy_capacity_systemwide_constraint")


class TestDispatchConstraints:
    # dispatch.py
    def test_loc_tech_carriers_carrier_production_max_constraint(self):
        """
        i for i in sets.loc_tech_carriers_prod
        if i not in sets.loc_tech_carriers_conversion_plus
        and i.rsplit('::', 1)[0] not in sets.loc_techs_milp
        """
        m = build_model({}, "simple_supply,two_hours,investment_costs")
        m.run(build_only=True)
        assert hasattr(m._backend_model, "carrier_production_max_constraint")

    @pytest.mark.filterwarnings("ignore:(?s).*Integer:calliope.exceptions.ModelWarning")
    def test_loc_tech_carriers_carrier_production_max_milp_constraint(self):
        m = build_model({}, "supply_milp,two_hours,investment_costs")
        m.run(build_only=True)
        assert not hasattr(m._backend_model, "carrier_production_max_constraint")

    def test_loc_tech_carriers_carrier_production_min_constraint(self):
        """
        i for i in sets.loc_tech_carriers_prod
        if i not in sets.loc_tech_carriers_conversion_plus
        and constraint_exists(model_run, i, 'constraints.energy_cap_min_use')
        and i.rsplit('::', 1)[0] not in sets.loc_techs_milp
        """
        m = build_model({}, "simple_supply,two_hours,investment_costs")
        m.run(build_only=True)
        assert not hasattr(m._backend_model, "carrier_production_min_constraint")

        m = build_model(
            {"techs.test_supply_elec.constraints.energy_cap_min_use": 0.1},
            "simple_supply,two_hours,investment_costs",
        )
        m.run(build_only=True)
        assert hasattr(m._backend_model, "carrier_production_min_constraint")

    @pytest.mark.filterwarnings("ignore:(?s).*Integer:calliope.exceptions.ModelWarning")
    def test_loc_tech_carriers_carrier_production_min_milp_constraint(self):
        m = build_model({}, "supply_milp,two_hours,investment_costs")
        m.run(build_only=True)
        assert not hasattr(m._backend_model, "carrier_production_min_constraint")

        m = build_model(
            {"techs.test_supply_elec.constraints.energy_cap_min_use": 0.1},
            "supply_milp,two_hours,investment_costs",
        )
        m.run(build_only=True)
        assert not hasattr(m._backend_model, "carrier_production_min_constraint")

    def test_loc_tech_carriers_carrier_consumption_max_constraint(self):
        """
        i for i in sets.loc_tech_carriers_con
        if i.rsplit('::', 1)[0] in sets.loc_techs_demand +
            sets.loc_techs_storage + sets.loc_techs_transmission
        and i.rsplit('::', 1)[0] not in sets.loc_techs_milp
        """

        m = build_model({}, "simple_supply,two_hours,investment_costs")
        m.run(build_only=True)
        assert hasattr(m._backend_model, "carrier_consumption_max_constraint")

    @pytest.mark.filterwarnings("ignore:(?s).*Integer:calliope.exceptions.ModelWarning")
    def test_loc_tech_carriers_carrier_consumption_max_milp_constraint(self):
        m = build_model({}, "supply_milp,two_hours,investment_costs")
        m.run(build_only=True)
        assert hasattr(m._backend_model, "carrier_consumption_max_constraint")

    def test_loc_techs_resource_max_constraint(self):
        """
        sets.loc_techs_finite_resource_supply_plus,
        """
        m = build_model({}, "simple_supply,two_hours,investment_costs")
        m.run(build_only=True)
        assert not hasattr(m._backend_model, "resource_max_constraint")

        m = build_model({}, "simple_supply_and_supply_plus,two_hours,investment_costs")
        m.run(build_only=True)
        assert hasattr(m._backend_model, "resource_max_constraint")

        m = build_model(
            {"techs.test_supply_plus.constraints.resource": np.inf},
            "simple_supply_and_supply_plus,two_hours,investment_costs",
        )
        m.run(build_only=True)
        assert hasattr(m._backend_model, "resource_max_constraint")

    def test_loc_techs_storage_max_constraint(self):
        """
        sets.loc_techs_store
        """
        m = build_model({}, "simple_supply,two_hours,investment_costs")
        m.run(build_only=True)
        assert not hasattr(m._backend_model, "storage_max_constraint")

        m = build_model({}, "simple_supply_and_supply_plus,two_hours,investment_costs")
        m.run(build_only=True)
        assert hasattr(m._backend_model, "storage_max_constraint")

        m = build_model({}, "simple_storage,two_hours,investment_costs")
        m.run(build_only=True)
        assert hasattr(m._backend_model, "storage_max_constraint")

    def test_loc_tech_carriers_ramping_constraint(self):
        """
        i for i in sets.loc_tech_carriers_prod
        if i.rsplit('::', 1)[0] in sets.loc_techs_ramping
        """
        m = build_model({}, "simple_supply,two_hours,investment_costs")
        m.run(build_only=True)
        assert not hasattr(m._backend_model, "ramping_up_constraint")
        assert not hasattr(m._backend_model, "ramping_down_constraint")

        m = build_model(
            {"techs.test_supply_elec.constraints.energy_ramping": 0.1},
            "simple_supply,two_hours,investment_costs",
        )
        m.run(build_only=True)
        assert hasattr(m._backend_model, "ramping_up_constraint")
        assert hasattr(m._backend_model, "ramping_down_constraint")

        m = build_model(
            {"techs.test_conversion.constraints.energy_ramping": 0.1},
            "simple_conversion,two_hours,investment_costs",
        )
        m.run(build_only=True)
        assert hasattr(m._backend_model, "ramping_up_constraint")
        assert hasattr(m._backend_model, "ramping_down_constraint")


@pytest.mark.filterwarnings("ignore:(?s).*Integer:calliope.exceptions.ModelWarning")
class TestMILPConstraints:
    # milp.py
    def test_loc_techs_unit_commitment_milp_constraint(self):
        """
        sets.loc_techs_milp,
        """
        m = build_model({}, "simple_supply,two_hours,investment_costs")
        m.run(build_only=True)
        assert not hasattr(m._backend_model, "unit_commitment_milp_constraint")

        m = build_model({}, "supply_milp,two_hours,investment_costs")
        m.run(build_only=True)
        assert hasattr(m._backend_model, "unit_commitment_milp_constraint")

        m = build_model({}, "supply_purchase,two_hours,investment_costs")
        m.run(build_only=True)
        assert not hasattr(m._backend_model, "unit_commitment_milp_constraint")

    def test_loc_techs_unit_capacity_milp_constraint(self):
        """
        sets.loc_techs_milp,
        """
        m = build_model({}, "simple_supply,two_hours,investment_costs")
        m.run(build_only=True)
        assert not hasattr(m._backend_model, "unit_capacity_milp_constraint")

        m = build_model({}, "supply_milp,two_hours,investment_costs")
        m.run(build_only=True)
        assert hasattr(m._backend_model, "unit_capacity_milp_constraint")

        m = build_model({}, "supply_purchase,two_hours,investment_costs")
        m.run(build_only=True)
        assert not hasattr(m._backend_model, "unit_capacity_milp_constraint")

    def test_loc_tech_carriers_carrier_production_max_milp_constraint(self):
        """
        i for i in sets.loc_tech_carriers_prod
        if i not in sets.loc_tech_carriers_conversion_plus
        and i.rsplit('::', 1)[0] in sets.loc_techs_milp
        """
        m = build_model({}, "simple_supply,two_hours,investment_costs")
        m.run(build_only=True)
        assert not hasattr(m._backend_model, "carrier_production_max_milp_constraint")

        m = build_model({}, "supply_milp,two_hours,investment_costs")
        m.run(build_only=True)
        assert hasattr(m._backend_model, "carrier_production_max_milp_constraint")

        m = build_model({}, "supply_purchase,two_hours,investment_costs")
        m.run(build_only=True)
        assert not hasattr(m._backend_model, "carrier_production_max_milp_constraint")

        m = build_model({}, "conversion_plus_milp,two_hours,investment_costs")
        m.run(build_only=True)
        assert not hasattr(m._backend_model, "carrier_production_max_milp_constraint")

    def test_loc_techs_carrier_production_max_conversion_plus_milp_constraint(self):
        """
        i for i in sets.loc_techs_conversion_plus
        if i in sets.loc_techs_milp
        """
        m = build_model({}, "simple_supply,two_hours,investment_costs")
        m.run(build_only=True)
        assert not hasattr(
            m._backend_model, "carrier_production_max_conversion_plus_milp_constraint"
        )

        m = build_model({}, "supply_milp,two_hours,investment_costs")
        m.run(build_only=True)
        assert not hasattr(
            m._backend_model, "carrier_production_max_conversion_plus_milp_constraint"
        )

        m = build_model({}, "supply_purchase,two_hours,investment_costs")
        m.run(build_only=True)
        assert not hasattr(
            m._backend_model, "carrier_production_max_conversion_plus_milp_constraint"
        )

        m = build_model({}, "conversion_plus_milp,two_hours,investment_costs")
        m.run(build_only=True)
        assert hasattr(
            m._backend_model, "carrier_production_max_conversion_plus_milp_constraint"
        )

        m = build_model({}, "conversion_plus_purchase,two_hours,investment_costs")
        m.run(build_only=True)
        assert not hasattr(
            m._backend_model, "carrier_production_max_conversion_plus_milp_constraint"
        )

    def test_loc_tech_carriers_carrier_production_min_milp_constraint(self):
        """
        i for i in sets.loc_tech_carriers_prod
        if i not in sets.loc_tech_carriers_conversion_plus
        and constraint_exists(model_run, i.rsplit('::', 1)[0], 'constraints.energy_cap_min_use')
        and i.rsplit('::', 1)[0] in sets.loc_techs_milp
        """
        m = build_model(
            {"techs.test_supply_elec.constraints.energy_cap_min_use": 0.1},
            "simple_supply,two_hours,investment_costs",
        )
        m.run(build_only=True)
        assert not hasattr(m._backend_model, "carrier_production_min_milp_constraint")

        m = build_model(
            {"techs.test_supply_elec.constraints.energy_cap_min_use": 0.1},
            "supply_milp,two_hours,investment_costs",
        )
        m.run(build_only=True)
        assert hasattr(m._backend_model, "carrier_production_min_milp_constraint")

        m = build_model(
            {"techs.test_supply_elec.constraints.energy_cap_min_use": 0.1},
            "supply_purchase,two_hours,investment_costs",
        )
        m.run(build_only=True)
        assert not hasattr(m._backend_model, "carrier_production_min_milp_constraint")

        m = build_model(
            {"techs.test_supply_elec.constraints.energy_cap_min_use": 0.1},
            "conversion_plus_milp,two_hours,investment_costs",
        )
        m.run(build_only=True)
        assert not hasattr(m._backend_model, "carrier_production_min_milp_constraint")

        m = build_model(
            {"techs.test_conversion_plus.constraints.energy_cap_min_use": 0.1},
            "conversion_plus_milp,two_hours,investment_costs",
        )
        m.run(build_only=True)
        assert not hasattr(m._backend_model, "carrier_production_min_milp_constraint")

    def test_loc_techs_carrier_production_min_conversion_plus_milp_constraint(self):
        """
        i for i in sets.loc_techs_conversion_plus
        if constraint_exists(model_run, i, 'constraints.energy_cap_min_use')
        and i in sets.loc_techs_milp
        """
        m = build_model(
            {"techs.test_supply_elec.constraints.energy_cap_min_use": 0.1},
            "simple_supply,two_hours,investment_costs",
        )
        m.run(build_only=True)
        assert not hasattr(
            m._backend_model, "carrier_production_min_conversion_plus_milp_constraint"
        )

        m = build_model(
            {"techs.test_supply_elec.constraints.energy_cap_min_use": 0.1},
            "supply_milp,two_hours,investment_costs",
        )
        m.run(build_only=True)
        assert not hasattr(
            m._backend_model, "carrier_production_min_conversion_plus_milp_constraint"
        )

        m = build_model(
            {"techs.test_supply_elec.constraints.energy_cap_min_use": 0.1},
            "conversion_plus_milp,two_hours,investment_costs",
        )
        m.run(build_only=True)
        assert not hasattr(
            m._backend_model, "carrier_production_min_conversion_plus_milp_constraint"
        )

        m = build_model(
            {"techs.test_conversion_plus.constraints.energy_cap_min_use": 0.1},
            "conversion_plus_milp,two_hours,investment_costs",
        )
        m.run(build_only=True)
        assert hasattr(
            m._backend_model, "carrier_production_min_conversion_plus_milp_constraint"
        )

        m = build_model(
            {"techs.test_conversion_plus.constraints.energy_cap_min_use": 0.1},
            "conversion_plus_purchase,two_hours,investment_costs",
        )
        m.run(build_only=True)
        assert not hasattr(
            m._backend_model, "carrier_production_min_conversion_plus_milp_constraint"
        )

    def test_loc_tech_carriers_carrier_consumption_max_milp_constraint(self):
        """
        i for i in sets.loc_tech_carriers_con
        if i.rsplit('::', 1)[0] in sets.loc_techs_demand +
            sets.loc_techs_storage + sets.loc_techs_transmission
        and i.rsplit('::', 1)[0] in sets.loc_techs_milp
        """
        m = build_model({}, "simple_supply,two_hours,investment_costs")
        m.run(build_only=True)
        assert not hasattr(m._backend_model, "carrier_consumption_max_milp_constraint")

        m = build_model({}, "supply_milp,two_hours,investment_costs")
        m.run(build_only=True)
        assert not hasattr(m._backend_model, "carrier_consumption_max_milp_constraint")

        m = build_model({}, "storage_milp,two_hours,investment_costs")
        m.run(build_only=True)
        assert hasattr(m._backend_model, "carrier_consumption_max_milp_constraint")

        m = build_model({}, "conversion_plus_milp,two_hours,investment_costs")
        m.run(build_only=True)
        assert not hasattr(m._backend_model, "carrier_consumption_max_milp_constraint")

    def test_loc_techs_energy_capacity_units_milp_constraint(self):
        """
        i for i in sets.loc_techs_milp
        if constraint_exists(model_run, i, 'constraints.energy_cap_per_unit')
        is not None
        """
        m = build_model({}, "simple_supply,two_hours,investment_costs")
        m.run(build_only=True)
        assert not hasattr(m._backend_model, "energy_capacity_units_milp_constraint")

        m = build_model({}, "supply_milp,two_hours,investment_costs")
        m.run(build_only=True)
        assert hasattr(m._backend_model, "energy_capacity_units_milp_constraint")

        m = build_model({}, "storage_milp,two_hours,investment_costs")
        m.run(build_only=True)
        assert hasattr(m._backend_model, "energy_capacity_units_milp_constraint")

        m = build_model({}, "conversion_plus_milp,two_hours,investment_costs")
        m.run(build_only=True)
        assert hasattr(m._backend_model, "energy_capacity_units_milp_constraint")

    def test_loc_techs_storage_capacity_units_milp_constraint(self):
        """
        i for i in sets.loc_techs_milp if i in sets.loc_techs_store
        """
        m = build_model({}, "simple_supply,two_hours,investment_costs")
        m.run(build_only=True)
        assert not hasattr(m._backend_model, "storage_capacity_units_milp_constraint")

        m = build_model({}, "supply_milp,two_hours,investment_costs")
        m.run(build_only=True)
        assert not hasattr(m._backend_model, "storage_capacity_units_milp_constraint")

        m = build_model({}, "storage_milp,two_hours,investment_costs")
        m.run(build_only=True)
        assert hasattr(m._backend_model, "storage_capacity_units_milp_constraint")

        m = build_model({}, "conversion_plus_milp,two_hours,investment_costs")
        m.run(build_only=True)
        assert not hasattr(m._backend_model, "storage_capacity_units_milp_constraint")

        m = build_model({}, "supply_and_supply_plus_milp,two_hours,investment_costs")
        m.run(build_only=True)
        assert hasattr(m._backend_model, "storage_capacity_units_milp_constraint")

    def test_loc_techs_energy_capacity_max_purchase_milp_constraint(self):
        """
        i for i in sets.loc_techs_purchase
        if (constraint_exists(model_run, i, 'constraints.energy_cap_equals') is not None
            or constraint_exists(model_run, i, 'constraints.energy_cap_max') is not None)
        """
        m = build_model({}, "simple_supply,two_hours,investment_costs")
        m.run(build_only=True)
        assert not hasattr(
            m._backend_model, "energy_capacity_max_purchase_milp_constraint"
        )

        m = build_model({}, "supply_milp,two_hours,investment_costs")
        m.run(build_only=True)
        assert not hasattr(
            m._backend_model, "energy_capacity_max_purchase_milp_constraint"
        )

        m = build_model({}, "supply_purchase,two_hours,investment_costs")
        m.run(build_only=True)
        assert hasattr(m._backend_model, "energy_capacity_max_purchase_milp_constraint")

        m = build_model(
            {
                "techs.test_supply_elec.constraints": {
                    "energy_cap_max": None,
                    "energy_cap_equals": 15,
                }
            },
            "supply_purchase,two_hours,investment_costs",
        )
        m.run(build_only=True)
        assert hasattr(m._backend_model, "energy_capacity_max_purchase_milp_constraint")

    def test_loc_techs_energy_capacity_min_purchase_milp_constraint(self):
        """
        i for i in sets.loc_techs_purchase
        if (not constraint_exists(model_run, i, 'constraints.energy_cap_equals')
            and constraint_exists(model_run, i, 'constraints.energy_cap_min'))
        """
        m = build_model({}, "simple_supply,two_hours,investment_costs")
        m.run(build_only=True)
        assert not hasattr(
            m._backend_model, "energy_capacity_min_purchase_milp_constraint"
        )

        m = build_model({}, "supply_milp,two_hours,investment_costs")
        m.run(build_only=True)
        assert not hasattr(
            m._backend_model, "energy_capacity_min_purchase_milp_constraint"
        )

        m = build_model({}, "supply_purchase,two_hours,investment_costs")
        m.run(build_only=True)
        assert not hasattr(
            m._backend_model, "energy_capacity_min_purchase_milp_constraint"
        )

        m = build_model(
            {
                "techs.test_supply_elec.constraints": {
                    "energy_cap_max": None,
                    "energy_cap_equals": 15,
                }
            },
            "supply_purchase,two_hours,investment_costs",
        )
        m.run(build_only=True)
        assert not hasattr(
            m._backend_model, "energy_capacity_min_purchase_milp_constraint"
        )

        m = build_model(
            {"techs.test_supply_elec.constraints.energy_cap_min": 10},
            "supply_purchase,two_hours,investment_costs",
        )
        m.run(build_only=True)
        assert hasattr(m._backend_model, "energy_capacity_min_purchase_milp_constraint")

    def test_loc_techs_storage_capacity_max_purchase_milp_constraint(self):
        """
        i for i in set(sets.loc_techs_purchase).intersection(sets.loc_techs_store)
        """
        m = build_model({}, "simple_storage,two_hours,investment_costs")
        m.run(build_only=True)
        assert not hasattr(
            m._backend_model, "storage_capacity_max_purchase_milp_constraint"
        )

        m = build_model({}, "storage_milp,two_hours,investment_costs")
        m.run(build_only=True)
        assert not hasattr(
            m._backend_model, "storage_capacity_max_purchase_milp_constraint"
        )

        m = build_model({}, "storage_purchase,two_hours,investment_costs")
        m.run(build_only=True)
        assert hasattr(
            m._backend_model, "storage_capacity_max_purchase_milp_constraint"
        )

        m = build_model({}, "supply_purchase,two_hours,investment_costs")
        m.run(build_only=True)
        assert not hasattr(
            m._backend_model, "storage_capacity_max_purchase_milp_constraint"
        )

    def test_loc_techs_storage_capacity_min_purchase_milp_constraint(self):
        """
        i for i in set(sets.loc_techs_purchase).intersection(sets.loc_techs_store)
        if (not constraint_exists(model_run, i, 'constraints.storage_cap_equals')
            and (constraint_exists(model_run, i, 'constraints.storage_cap_min')
                or constraint_exists(model_run, i, 'constraints.energy_cap_min')))
        """
        m = build_model(
            {"techs.test_storage.constraints.storage_cap_min": 10},
            "simple_storage,two_hours,investment_costs",
        )
        m.run(build_only=True)
        assert not hasattr(
            m._backend_model, "storage_capacity_min_purchase_milp_constraint"
        )

        m = build_model(
            {"techs.test_storage.constraints.storage_cap_min": 10},
            "storage_milp,two_hours,investment_costs",
        )
        m.run(build_only=True)
        assert not hasattr(
            m._backend_model, "storage_capacity_min_purchase_milp_constraint"
        )

        m = build_model({}, "storage_purchase,two_hours,investment_costs")
        m.run(build_only=True)
        assert not hasattr(
            m._backend_model, "storage_capacity_min_purchase_milp_constraint"
        )

        m = build_model(
            {"techs.test_storage.constraints.storage_cap_min": 10},
            "storage_purchase,two_hours,investment_costs",
        )
        m.run(build_only=True)
        assert hasattr(
            m._backend_model, "storage_capacity_min_purchase_milp_constraint"
        )

        m = build_model(
            {
                "techs.test_storage.constraints": {
                    "storage_cap_equals": 10,
                    "storage_cap_min": 10,
                }
            },
            "storage_purchase,two_hours,investment_costs",
        )
        m.run(build_only=True)
        assert not hasattr(
            m._backend_model, "storage_capacity_min_purchase_milp_constraint"
        )

    def test_loc_techs_update_costs_investment_units_milp_constraint(self):
        """
        i for i in sets.loc_techs_milp
        if i in sets.loc_techs_investment_cost and
        any(constraint_exists(model_run, i, 'costs.{}.purchase'.format(j))
               for j in model_run.sets.costs)
        """
        m = build_model({}, "simple_supply,two_hours,investment_costs")
        m.run(build_only=True)
        assert not check_variable_exists(
            m._backend_model, "cost_investment_constraint", "purchased"
        )
        assert not check_variable_exists(
            m._backend_model, "cost_investment_constraint", "units"
        )

        m = build_model({}, "supply_milp,two_hours,investment_costs")
        m.run(build_only=True)
        assert not check_variable_exists(
            m._backend_model, "cost_investment_constraint", "purchased"
        )
        assert not check_variable_exists(
            m._backend_model, "cost_investment_constraint", "units"
        )

        m = build_model(
            {"techs.test_supply_elec.costs.monetary.purchase": 1},
            "supply_milp,two_hours,investment_costs",
        )
        m.run(build_only=True)
        assert not check_variable_exists(
            m._backend_model, "cost_investment_constraint", "purchased"
        )
        assert check_variable_exists(
            m._backend_model, "cost_investment_constraint", "units"
        )

    def test_loc_techs_update_costs_investment_purchase_milp_constraint(self):
        """
        sets.loc_techs_purchase,
        """

        m = build_model({}, "supply_purchase,two_hours,investment_costs")
        m.run(build_only=True)
        assert check_variable_exists(
            m._backend_model, "cost_investment_constraint", "purchased"
        )
        assert not check_variable_exists(
            m._backend_model, "cost_investment_constraint", "units"
        )

    def test_techs_unit_capacity_systemwide_milp_constraint(self):
        """
        sets.techs if unit_cap_max_systemwide or unit_cap_equals_systemwide
        """

        override_max = {
            "links.0,1.exists": True,
            "techs.test_conversion_plus.constraints.units_max_systemwide": 2,
            "locations.1.techs.test_conversion_plus.constraints": {
                "units_max": 2,
                "energy_cap_per_unit": 5,
            },
        }
        override_equals = {
            "links.0,1.exists": True,
            "techs.test_conversion_plus.constraints.units_equals_systemwide": 1,
            "locations.1.techs.test_conversion_plus.costs.monetary.purchase": 1,
        }
        override_equals_inf = {
            "links.0,1.exists": True,
            "techs.test_conversion_plus.constraints.units_equals_systemwide": np.inf,
            "locations.1.techs.test_conversion_plus.costs.monetary.purchase": 1,
        }
        override_transmission = {
            "links.0,1.exists": True,
            "techs.test_transmission_elec.constraints": {
                "units_max_systemwide": 1,
                "lifetime": 25,
            },
            "techs.test_transmission_elec.costs.monetary": {
                "purchase": 1,
                "interest_rate": 0.1,
            },
        }
        override_no_transmission = {
            "techs.test_supply_elec.constraints.units_equals_systemwide": 1,
            "locations.1.techs.test_supply_elec.costs.monetary.purchase": 1,
        }

        m = build_model(override_max, "conversion_plus_milp,two_hours,investment_costs")
        m.run(build_only=True)
        assert hasattr(m._backend_model, "unit_capacity_systemwide_milp_constraint")
        assert (
            m._backend_model.unit_capacity_systemwide_milp_constraint[
                "test_conversion_plus"
            ].upper()
            == 2
        )

        m = build_model(
            override_equals, "conversion_plus_milp,two_hours,investment_costs"
        )
        m.run(build_only=True)
        assert hasattr(m._backend_model, "unit_capacity_systemwide_milp_constraint")
        assert (
            m._backend_model.unit_capacity_systemwide_milp_constraint[
                "test_conversion_plus"
            ].lower()
            == 1
        )
        assert (
            m._backend_model.unit_capacity_systemwide_milp_constraint[
                "test_conversion_plus"
            ].upper()
            == 1
        )

        with pytest.raises(ValueError) as error:
            m = build_model(
                override_equals_inf, "conversion_plus_milp,two_hours,investment_costs"
            )
            m.run(build_only=True)
        assert check_error_or_warning(
            error,
            "Cannot use inf for parameter units_equals_systemwide[test_conversion_plus]",
        )

        m = build_model(
            override_transmission, "simple_supply,two_hours,investment_costs"
        )
        m.run(build_only=True)
        assert hasattr(m._backend_model, "unit_capacity_systemwide_milp_constraint")
        assert (
            m._backend_model.unit_capacity_systemwide_milp_constraint[
                "test_transmission_elec"
            ].upper()
            == 2
        )

        m = build_model(
            override_no_transmission,
            "simple_supply,two_hours,investment_costs",
            model_file="model_minimal.yaml",
        )
        m.run(build_only=True)
        assert hasattr(m._backend_model, "unit_capacity_systemwide_milp_constraint")

    def test_asynchronous_prod_con_constraint(self):
        """
        Binary switch for prod/con can be activated using the option
        'asynchronous_prod_con'
        """
        m_store = build_model(
            {"techs.test_storage.constraints.force_asynchronous_prod_con": True},
            "simple_storage,investment_costs",
        )
        m_store.run(build_only=True)
        assert hasattr(m_store._backend_model, "prod_con_switch")
        assert hasattr(m_store._backend_model, "asynchronous_con_milp_constraint")
        assert hasattr(m_store._backend_model, "asynchronous_prod_milp_constraint")

        m_trans = build_model(
            {
                "techs.test_transmission_elec.constraints.force_asynchronous_prod_con": True
            },
            "simple_storage,investment_costs",
        )
        m_trans.run(build_only=True)
        assert hasattr(m_trans._backend_model, "prod_con_switch")
        assert hasattr(m_trans._backend_model, "asynchronous_con_milp_constraint")
        assert hasattr(m_trans._backend_model, "asynchronous_prod_milp_constraint")


class TestConversionConstraints:

    # conversion.py
    def test_loc_techs_balance_conversion_constraint(self):
        """
        sets.loc_techs_conversion,
        """
        m = build_model({}, "simple_supply,two_hours,investment_costs")
        m.run(build_only=True)
        assert not hasattr(m._backend_model, "balance_conversion_constraint")

        m = build_model({}, "simple_conversion,two_hours,investment_costs")
        m.run(build_only=True)
        assert hasattr(m._backend_model, "balance_conversion_constraint")

        m = build_model({}, "simple_conversion_plus,two_hours,investment_costs")
        m.run(build_only=True)
        assert not hasattr(m._backend_model, "balance_conversion_constraint")

    def test_loc_techs_cost_var_conversion_constraint(self):
        """
        sets.loc_techs_om_cost_conversion,
        """
        m = build_model(
            {"techs.test_supply_elec.costs.monetary.om_prod": 0.1},
            "simple_supply,two_hours,investment_costs",
        )
        m.run(build_only=True)
        assert not hasattr(m._backend_model, "cost_var_conversion_constraint")

        m = build_model({}, "simple_conversion,two_hours,investment_costs")
        m.run(build_only=True)
        assert not hasattr(m._backend_model, "cost_var_conversion_constraint")

        m = build_model(
            {"techs.test_conversion.costs.monetary.om_prod": 0.1},
            "simple_conversion,two_hours,investment_costs",
        )
        m.run(build_only=True)
        assert hasattr(m._backend_model, "cost_var_conversion_constraint")

        assert check_variable_exists(
            m._backend_model, "cost_var_conversion_constraint", "carrier_prod"
        )
        assert not check_variable_exists(
            m._backend_model, "cost_var_conversion_constraint", "carrier_con"
        )

        m = build_model(
            {"techs.test_conversion.costs.monetary.om_con": 0.1},
            "simple_conversion,two_hours,investment_costs",
        )
        m.run(build_only=True)
        assert hasattr(m._backend_model, "cost_var_conversion_constraint")
        assert check_variable_exists(
            m._backend_model, "cost_var_conversion_constraint", "carrier_con"
        )
        assert not check_variable_exists(
            m._backend_model, "cost_var_conversion_constraint", "carrier_prod"
        )


class TestNetworkConstraints:
    # network.py
    def test_loc_techs_symmetric_transmission_constraint(self):
        """
        sets.loc_techs_transmission,
        """
        m = build_model({}, "simple_supply,two_hours,investment_costs")
        m.run(build_only=True)
        assert hasattr(m._backend_model, "symmetric_transmission_constraint")

        m = build_model({}, "simple_conversion_plus,two_hours,investment_costs")
        m.run(build_only=True)
        assert not hasattr(m._backend_model, "symmetric_transmission_constraint")


class TestPolicyConstraints:
    # policy.py
    def test_techlists_group_share_energy_cap_min_constraint(self):
        """
        i for i in sets.techlists
        if 'energy_cap_min' in model_run.model.get_key('group_share.{}'.format(i), {}).keys()
        """
        m = build_model(
            {}, "simple_supply,group_share_energy_cap_min,two_hours,investment_costs"
        )
        m.run(build_only=True)
        assert hasattr(m._backend_model, "group_share_energy_cap_min_constraint")
        assert not hasattr(m._backend_model, "group_share_energy_cap_max_constraint")
        assert not hasattr(m._backend_model, "group_share_energy_cap_equals_constraint")

    def test_techlists_group_share_energy_cap_max_constraint(self):
        """
        i for i in sets.techlists
        if 'energy_cap_max' in model_run.model.get_key('group_share.{}'.format(i), {}).keys()
        """
        m = build_model(
            {}, "simple_supply,group_share_energy_cap_max,two_hours,investment_costs"
        )
        m.run(build_only=True)
        assert not hasattr(m._backend_model, "group_share_energy_cap_min_constraint")
        assert hasattr(m._backend_model, "group_share_energy_cap_max_constraint")
        assert not hasattr(m._backend_model, "group_share_energy_cap_equals_constraint")

    def test_techlists_group_share_energy_cap_equals_constraint(self):
        """
        i for i in sets.techlists
        if 'energy_cap_equals' in model_run.model.get_key('group_share.{}'.format(i), {}).keys()
        """
        m = build_model(
            {}, "simple_supply,group_share_energy_cap_equals,two_hours,investment_costs"
        )
        m.run(build_only=True)
        assert not hasattr(m._backend_model, "group_share_energy_cap_min_constraint")
        assert not hasattr(m._backend_model, "group_share_energy_cap_max_constraint")
        assert hasattr(m._backend_model, "group_share_energy_cap_equals_constraint")

    def test_techlists_carrier_group_share_carrier_prod_min_constraint(self):
        """
        i + '::' + carrier
        for i in sets.techlists
        if 'carrier_prod_min' in model_run.model.get_key('group_share.{}'.format(i), {}).keys()
        for carrier in sets.carriers
        if carrier in model_run.model.get_key('group_share.{}.carrier_prod_min'.format(i), {}).keys()
        """

        m = build_model(
            {},
            "conversion_and_conversion_plus,group_share_carrier_prod_min,two_hours,investment_costs",
        )
        m.run(build_only=True)
        assert hasattr(m._backend_model, "group_share_carrier_prod_min_constraint")
        assert not hasattr(m._backend_model, "group_share_carrier_prod_max_constraint")
        assert not hasattr(
            m._backend_model, "group_share_carrier_prod_equals_constraint"
        )

    def test_techlists_carrier_group_share_carrier_prod_max_constraint(self):
        """
        i + '::' + carrier
        for i in sets.techlists
        if 'carrier_prod_max' in model_run.model.get_key('group_share.{}'.format(i), {}).keys()
        for carrier in sets.carriers
        if carrier in model_run.model.get_key('group_share.{}.carrier_prod_max'.format(i), {}).keys()
        """

        m = build_model(
            {},
            "conversion_and_conversion_plus,group_share_carrier_prod_max,two_hours,investment_costs",
        )
        m.run(build_only=True)
        assert not hasattr(m._backend_model, "group_share_carrier_prod_min_constraint")
        assert hasattr(m._backend_model, "group_share_carrier_prod_max_constraint")
        assert not hasattr(
            m._backend_model, "group_share_carrier_prod_equals_constraint"
        )

    def test_techlists_carrier_group_share_carrier_prod_equals_constraint(self):
        """
        i + '::' + carrier
        for i in sets.techlists
        if 'carrier_prod_equals' in model_run.model.get_key('group_share.{}'.format(i), {}).keys()
        for carrier in sets.carriers
        if carrier in model_run.model.get_key('group_share.{}.carrier_prod_equals'.format(i), {}).keys()
        """

        m = build_model(
            {},
            "conversion_and_conversion_plus,group_share_carrier_prod_equals,two_hours,investment_costs",
        )
        m.run(build_only=True)
        assert not hasattr(m._backend_model, "group_share_carrier_prod_min_constraint")
        assert not hasattr(m._backend_model, "group_share_carrier_prod_max_constraint")
        assert hasattr(m._backend_model, "group_share_carrier_prod_equals_constraint")


# clustering constraints
class TestClusteringConstraints:
    def constraints(self):
        return [
            "balance_storage_inter_cluster_constraint",
            "storage_intra_max_constraint",
            "storage_intra_min_constraint",
            "storage_inter_max_constraint",
            "storage_inter_min_constraint",
        ]

    def decision_variables(self):
        return [
            "storage_inter_cluster",
            "storage_intra_cluster_max",
            "storage_intra_cluster_min",
        ]

    def cluster_model(
        self,
        how="mean",
        storage_inter_cluster=True,
        cyclic=False,
        storage_initial=False,
    ):
        override = {
            "model.subset_time": ["2005-01-01", "2005-01-04"],
            "model.time": {
                "function": "apply_clustering",
                "function_options": {
                    "clustering_func": "file=cluster_days.csv:0",
                    "how": how,
                    "storage_inter_cluster": storage_inter_cluster,
                },
            },
            "run.cyclic_storage": cyclic,
        }
        if storage_initial:
            override.update({"techs.test_storage.constraints.storage_initial": 0})
        return build_model(override, "simple_storage,investment_costs")

    def test_cluster_storage_constraints(self):
        m = self.cluster_model()
        m.run(build_only=True)

        for variable in self.decision_variables():
            assert hasattr(m._backend_model, variable)

        for constraint in self.constraints():
            assert hasattr(m._backend_model, constraint)

        assert not hasattr(m._backend_model, "storage_max_constraint")
        assert not hasattr(m._backend_model, "storage_initial_constraint")

    def test_cluster_cyclic_storage_constraints(self):
        m = self.cluster_model(cyclic=True)
        m.run(build_only=True)

        for variable in self.decision_variables():
            assert hasattr(m._backend_model, variable)

        for constraint in self.constraints():
            assert hasattr(m._backend_model, constraint)

        assert not hasattr(m._backend_model, "storage_max_constraint")
        assert not hasattr(m._backend_model, "storage_initial_constraint")

    def test_no_cluster_storage_constraints(self):
        m = self.cluster_model(storage_inter_cluster=False)
        m.run(build_only=True)

        for variable in self.decision_variables():
            assert not hasattr(m._backend_model, variable)

        for constraint in self.constraints():
            assert not hasattr(m._backend_model, constraint)

        assert hasattr(m._backend_model, "storage_max_constraint")


class TestLogging:
    @pytest.fixture(scope="module")
    def gurobi_model(self):
        pytest.importorskip("gurobipy")
        model_file = os.path.join("model_config_group", "base_model.yaml")
        return build_model(
            model_file=model_file,
            override_dict={"run": {"solver": "gurobi", "solver_io": "python"}},
        )

    def test_no_duplicate_log_message(self, caplog, gurobi_model):
        caplog.set_level(logging.DEBUG)
        gurobi_model.run()
        all_log_messages = [r.msg for r in caplog.records]
        duplicates = [
            item
            for item, count in collections.Counter(all_log_messages).items()
            if count > 1 and item != "" and not item.startswith("Constructing")
        ]
        assert duplicates == []
