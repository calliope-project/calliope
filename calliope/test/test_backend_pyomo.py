import logging
from itertools import product

import numpy as np
import pyomo.kernel as pmo
import pytest  # noqa: F401
import xarray as xr

import calliope.exceptions as exceptions
from calliope.test.common.util import build_test_model as build_model
from calliope.test.common.util import check_error_or_warning, check_variable_exists


@pytest.mark.xfail(reason="Not expecting operate mode to work at the moment")
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
            m.build()
        check_warn = check_error_or_warning(
            warning, "Storage cannot be cyclic in operate run mode"
        )
        if on is True:
            assert check_warn
        elif on is True:
            assert not check_warn
        assert m._model_data.attrs["run_config"].cyclic_storage is False

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

        with pytest.warns(exceptions.ModelWarning):
            m.build()  # will fail to complete run if there's a problem

    @pytest.mark.parametrize("force", (True, False))
    def test_operate_energy_cap_min_use(self, force):
        """If we depend on a finite energy_cap, we have to error on a user failing to define it"""
        m = build_model(
            {
                "techs.test_supply_elec": {
                    "switches.force_resource": force,
                    "constraints": {
                        "energy_cap_min_use": 0.1,
                        "resource": "file=supply_plus_resource.csv:1",
                        "energy_cap_equals": np.inf,
                    },
                }
            },
            "simple_supply_and_supply_plus,operate,investment_costs",
        )

        with pytest.raises(exceptions.ModelError) as error:
            with pytest.warns(exceptions.ModelWarning):
                m.build()

        assert check_error_or_warning(
            error, ["Operate mode: User must define a finite energy_cap"]
        )

    @pytest.mark.parametrize("force", (True, False))
    def test_operate_energy_cap_resource_unit(self, force):
        """If we depend on a finite energy_cap, we have to error on a user failing to define it"""
        m = build_model(
            {
                "techs.test_supply_elec": {
                    "constraints": {
                        "resource": "file=supply_plus_resource.csv:1",
                        "energy_cap_equals": np.inf,
                        "energy_cap_max": np.inf,
                    },
                    "switches": {
                        "force_resource": force,
                        "resource_unit": "energy_per_cap",
                    },
                }
            },
            "simple_supply_and_supply_plus,operate,investment_costs",
        )

        if force is True:
            with pytest.raises(exceptions.ModelError) as error:
                with pytest.warns(exceptions.ModelWarning):
                    m.build()
            assert check_error_or_warning(
                error, ["Operate mode: User must define a finite energy_cap"]
            )
        elif force is False:
            with pytest.warns(exceptions.ModelWarning):
                m.build()

    @pytest.mark.parametrize(
        "resource_unit,force",
        list(product(("energy", "energy_per_cap", "energy_per_area"), (True, False))),
    )
    def test_operate_resource_unit_with_resource_area(self, resource_unit, force):
        """Different resource unit affects the capacities which are set to infinite"""
        m = build_model(
            {
                "techs.test_supply_elec": {
                    "constraints": {
                        "resource_area_max": 10,
                        "energy_cap_max": 15,
                        "resource": "file=supply_plus_resource.csv:1",
                    },
                    "switches": {
                        "force_resource": force,
                        "resource_unit": resource_unit,
                    },
                }
            },
            "simple_supply_and_supply_plus,operate,investment_costs",
        )
        with pytest.warns(exceptions.ModelWarning) as warning:
            m.build()

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
                "techs.test_supply_elec": {
                    "constraints": {
                        "resource": "file=supply_plus_resource.csv:1",
                        "energy_cap_max": 15,
                    },
                    "switches": {
                        "force_resource": True,
                        "resource_unit": resource_unit,
                    },
                }
            },
            "simple_supply_and_supply_plus,operate,investment_costs",
        )

        with pytest.warns(exceptions.ModelWarning) as warning:
            # energy_per_area without a resource_cap will cause an error, which we have to catch here
            if resource_unit == "energy_per_area":
                with pytest.raises(exceptions.ModelError) as error:
                    m.build()
            else:
                m.build()

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

    def test_operate_storage(self, param):
        """Can't violate storage capacity constraints in the definition of a technology"""
        param = "energy_cap_per_storage_cap_max"
        m = build_model(
            {f"techs.test_supply_plus.constraints.{param}": 0.1},
            "simple_supply_and_supply_plus,operate,investment_costs",
        )

        with pytest.warns(exceptions.ModelWarning) as warning:
            with pytest.raises(exceptions.ModelError) as error:
                m.build()

        assert check_error_or_warning(
            error,
            "fixed storage capacity * {} is not larger than fixed energy "
            "capacity for loc, tech {}".format(param, ("a", "test_supply_plus")),
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
            m.build()
        if on is False:
            assert check_error_or_warning(
                warning, "Resource capacity constraint defined and set to infinity"
            )
            assert np.isinf(
                m._model_data.resource_cap.loc["a", "test_supply_plus"].item()
            )
        elif on is True:
            assert not check_error_or_warning(
                warning, "Resource capacity constraint defined and set to infinity"
            )
            assert m._model_data.resource_cap.loc["a", "test_supply_plus"].item() == 1e6

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
            m.build()
        if on is False:
            assert check_error_or_warning(warning, "Initial stored energy not defined")
            assert (
                m._model_data.storage_initial.loc["a", "test_supply_plus"].item() == 0
            )
        elif on is True:
            assert not check_error_or_warning(
                warning, "Initial stored energy not defined"
            )
            assert (
                m._model_data.storage_initial.loc["a", "test_supply_plus"].item() == 0.5
            )


@pytest.mark.skip(reason="to be reimplemented by comparison to LP files")
class TestBalanceConstraints:
    def test_loc_carriers_system_balance_constraint(self, simple_supply):
        """
        sets.loc_carriers
        """

        assert "system_balance" in simple_supply.backend.constraints

    def test_loc_techs_balance_supply_constraint(self):
        """
        sets.loc_techs_finite_resource_supply,
        """
        m = build_model(
            {"techs.test_supply_elec.constraints.resource": 20},
            "simple_supply,two_hours,investment_costs",
        )
        m.build()
        assert "balance_supply" in m.backend.constraints

        m = build_model(
            {
                "techs.test_supply_elec.constraints.resource": 20,
                "techs.test_supply_elec.switches.resource_unit": "energy_per_cap",
            },
            "simple_supply,two_hours,investment_costs",
        )
        m.build()
        assert check_variable_exists(
            m.backend.get_constraint("balance_supply", as_backend_objs=False),
            "energy_cap",
        )

        m = build_model(
            {
                "techs.test_supply_elec.constraints.resource": 20,
                "techs.test_supply_elec.switches.resource_unit": "energy_per_area",
            },
            "simple_supply,two_hours,investment_costs",
        )
        m.build()
        assert check_variable_exists(
            m.backend.get_constraint("balance_supply", as_backend_objs=False),
            "resource_area",
        )

    def test_loc_techs_balance_demand_constraint(self, simple_supply):
        """
        sets.loc_techs_finite_resource_demand,
        """
        assert "balance_demand" in simple_supply.backend.constraints

        m = build_model(
            {"techs.test_demand_elec.switches.resource_unit": "energy_per_cap"},
            "simple_supply,two_hours,investment_costs",
        )
        m.build()
        assert check_variable_exists(
            m.backend.get_constraint("balance_demand", as_backend_objs=False),
            "energy_cap",
        )

        m = build_model(
            {"techs.test_demand_elec.switches.resource_unit": "energy_per_area"},
            "simple_supply,two_hours,investment_costs",
        )
        m.build()
        assert check_variable_exists(
            m.backend.get_constraint("balance_demand", as_backend_objs=False),
            "resource_area",
        )

    def test_loc_techs_resource_availability_supply_plus_constraint(
        self, simple_supply_and_supply_plus
    ):
        """
        sets.loc_techs_finite_resource_supply_plus,
        """
        assert (
            "resource_availability_supply_plus"
            in simple_supply_and_supply_plus.backend.constraints
        )

        m = build_model(
            {"techs.test_supply_plus.switches.resource_unit": "energy_per_cap"},
            "simple_supply_and_supply_plus,two_hours,investment_costs",
        )
        m.build()
        assert check_variable_exists(
            m.backend.get_constraint(
                "resource_availability_supply_plus", as_backend_objs=False
            ),
            "energy_cap",
        )

        m = build_model(
            {"techs.test_supply_plus.switches.resource_unit": "energy_per_area"},
            "simple_supply_and_supply_plus,two_hours,investment_costs",
        )
        m.build()
        assert check_variable_exists(
            m.backend.get_constraint(
                "resource_availability_supply_plus", as_backend_objs=False
            ),
            "resource_area",
        )

    def test_loc_techs_balance_transmission_constraint(self, simple_supply):
        """
        sets.loc_techs_transmission,
        """
        assert "balance_transmission" in simple_supply.backend.constraints

    def test_loc_techs_balance_supply_plus_constraint(
        self, simple_supply_and_supply_plus
    ):
        """
        sets.loc_techs_supply_plus,
        """

        assert (
            "balance_supply_plus_with_storage"
            in simple_supply_and_supply_plus.backend.constraints
        )
        assert (
            "balance_supply_plus_no_storage"
            not in simple_supply_and_supply_plus.backend.constraints
        )

    def test_loc_techs_balance_storage_constraint(self, simple_storage):
        """
        sets.loc_techs_storage,
        """
        assert "balance_storage" in simple_storage.backend.constraints
        assert "set_storage_initial" not in simple_storage.backend.constraints

    def test_loc_techs_balance_storage_discharge_depth_constraint(self):
        """
        sets.loc_techs_storage,
        """
        m = build_model(
            {}, "simple_storage,two_hours,investment_costs,storage_discharge_depth"
        )
        m.build()
        assert "storage_discharge_depth_limit" in m.backend.constraints
        assert "set_storage_initial" not in m.backend.constraints

        m3 = build_model(
            {"techs.test_storage.constraints.storage_initial": 1},
            "simple_storage,one_day,investment_costs,storage_discharge_depth",
        )
        m3.build()
        assert (
            m3._model_data.storage_initial.to_series().dropna()
            > m3._model_data.storage_discharge_depth.to_series().dropna()
        ).all()

    def test_storage_initial_constraint(self, simple_storage):
        """
        sets.loc_techs_store,
        """
        assert "balance_storage" in simple_storage.backend.constraints
        assert "set_storage_initial" not in simple_storage.backend.constraints

        m2 = build_model(
            {"techs.test_storage.constraints.storage_initial": 0},
            "simple_storage,one_day,investment_costs",
        )
        m2.build()
        assert "balance_storage" in m2.backend.constraints
        assert "set_storage_initial" in m2.backend.constraints

    @pytest.mark.xfail(reason="no longer a constraint we're creating")
    def test_carriers_reserve_margin_constraint(self):
        """
        i for i in sets.carriers if i in model_run.model.get_key('reserve_margin', {}).keys()
        """
        m = build_model(
            {"model.reserve_margin.electricity": 0.01},
            "simple_supply,two_hours,investment_costs",
        )
        m.build()
        assert "reserve_margin" in m.backend.constraints


@pytest.mark.skip(reason="to be reimplemented by comparison to LP files")
class TestCostConstraints:
    # costs.py
    def test_loc_techs_cost_constraint(self, simple_supply):
        """
        sets.loc_techs_cost,
        """
        assert "cost" in simple_supply.backend.expressions

    def test_loc_techs_cost_investment_constraint(self, simple_conversion):
        """
        sets.loc_techs_investment_cost,
        """
        assert "cost_investment" in simple_conversion.backend.expressions

    @pytest.mark.filterwarnings("ignore:(?s).*Integer:calliope.exceptions.ModelWarning")
    def test_loc_techs_cost_investment_milp_constraint(self):
        m = build_model(
            {
                "techs.test_supply_elec.constraints.lifetime": 10,
                "techs.test_supply_elec.costs.monetary.interest_rate": 0.1,
            },
            "supply_purchase,two_hours",
        )
        m.build()

        assert "cost_investment" in m.backend.expressions

    def test_loc_techs_not_cost_var_constraint(self, simple_conversion):
        """
        i for i in sets.loc_techs_om_cost if i not in sets.loc_techs_conversion_plus + sets.loc_techs_conversion

        """
        assert "cost_var" not in simple_conversion.backend.expressions

    @pytest.mark.parametrize(
        "tech,scenario,cost",
        (
            ("test_supply_elec", "simple_supply", "om_prod"),
            ("test_supply_elec", "simple_supply", "om_con"),
            ("test_supply_plus", "simple_supply_and_supply_plus", "om_con"),
            ("test_demand_elec", "simple_supply", "om_con"),
            ("test_transmission_elec", "simple_supply", "om_prod"),
            ("test_conversion", "simple_conversion", "om_con"),
            ("test_conversion_plus", "simple_conversion_plus", "om_prod"),
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
        m.build()
        assert "cost_var" in m.backend.expressions

    def test_one_way_om_cost(self):
        """
        With one_way transmission, it should still be possible to set an om_prod cost.
        """
        m = build_model(
            {
                "techs.test_transmission_elec.costs.monetary.om_prod": 1,
                "links.a,b.techs.test_transmission_elec.switches.one_way": True,
            },
            "simple_supply,two_hours",
        )
        m.build()
        idx = {
            "costs": "monetary",
            "nodes": "b",
            "techs": "test_transmission_elec:a",
            "timesteps": m.backend._dataset.timesteps[1],
        }
        assert check_variable_exists(
            m.backend.get_expression("cost_var", as_backend_objs=False),
            "carrier_prod",
            idx,
        )

        idx["nodes"] = "a"
        idx["techs"] = "test_transmission_elec:b"
        assert not check_variable_exists(
            m.backend.get_expression("cost_var", as_backend_objs=False),
            "carrier_prod",
            idx,
        )


@pytest.mark.skip(reason="to be reimplemented by comparison to LP files")
class TestExportConstraints:
    # export.py
    def test_loc_carriers_system_balance_no_export(self, simple_supply):
        """
        i for i in sets.loc_carriers if sets.loc_techs_export
        and any(['{0}::{2}'.format(*j.split('::')) == i
        for j in sets.loc_tech_carriers_export])
        """

        assert not check_variable_exists(
            simple_supply.backend.get_constraint(
                "system_balance", as_backend_objs=False
            ),
            "carrier_export",
        )

    def test_loc_carriers_system_balance_export(self, supply_export):
        assert check_variable_exists(
            supply_export.backend.get_constraint(
                "system_balance", as_backend_objs=False
            ),
            "carrier_export",
        )

    def test_loc_tech_carriers_export_balance_constraint(self, supply_export):
        """
        sets.loc_tech_carriers_export,
        """
        assert "export_balance" in supply_export.backend.constraints

    def test_loc_techs_update_costs_var_constraint(self, supply_export):
        """
        i for i in sets.loc_techs_om_cost if i in sets.loc_techs_export
        """
        assert "cost_var" in supply_export.backend.expressions

        m = build_model(
            {"techs.test_supply_elec.costs.monetary.om_prod": 0.1},
            "supply_export,two_hours,investment_costs",
        )
        m.build()
        assert "cost_var" in m.backend.expressions

        assert check_variable_exists(
            m.backend.get_expression("cost_var", as_backend_objs=False),
            "carrier_export",
        )

    def test_loc_tech_carriers_export_max_constraint(self):
        """
        i for i in sets.loc_tech_carriers_export
        if constraint_exists(model_run, i.rsplit('::', 1)[0], 'constraints.export_max')
        """

        m = build_model(
            {"techs.test_supply_elec.constraints.export_max": 5},
            "supply_export,two_hours,investment_costs",
        )
        m.build()
        assert "carrier_export_max" in m.backend.constraints


@pytest.mark.skip(reason="to be reimplemented by comparison to LP files")
class TestCapacityConstraints:
    # capacity.py
    def test_loc_techs_storage_capacity_constraint(
        self, simple_storage, simple_supply_and_supply_plus
    ):
        """
        i for i in sets.loc_techs_store if i not in sets.loc_techs_milp
        """
        assert "storage_max" in simple_storage.backend.constraints
        assert "storage_max" in simple_supply_and_supply_plus.backend.constraints

        m = build_model(
            {"techs.test_storage.constraints.storage_cap_equals": 20},
            "simple_storage,two_hours,investment_costs",
        )
        m.build()
        assert (
            m.backend.variables.storage_cap.sel(nodes="a", techs="test_storage")
            .item()
            .ub
            == 20
        )
        assert (
            m.backend.variables.storage_cap.sel(nodes="a", techs="test_storage")
            .item()
            .lb
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
        m.build()
        assert "storage_capacity" not in m.backend.constraints

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
            {f"techs.{tech}.constraints.energy_cap_per_storage_cap_{override}": 0.5},
            f"{scenario},two_hours,investment_costs",
        )
        m.build()
        assert (
            f"energy_capacity_per_storage_capacity_{override}" in m.backend.constraints
        )

        if override == "equals":
            assert not any(
                "energy_capacity_per_storage_capacity_{i}" in m.backend.constraints
                for i in ["max", "min"]
            )

    @pytest.mark.filterwarnings("ignore:(?s).*Integer:calliope.exceptions.ModelWarning")
    @pytest.mark.parametrize("override", (("max", "equals", "min")))
    def test_loc_techs_energy_capacity_milp_storage_constraint(self, override):
        """
        i for i in sets.loc_techs_store if constraint_exists(model_run, i, 'constraints.energy_cap_per_storage_cap_max')
        """

        m = build_model(
            {
                f"techs.test_supply_plus.constraints.energy_cap_per_storage_cap_{override}": 0.5
            },
            "supply_and_supply_plus_milp,two_hours,investment_costs",
        )
        m.build()
        assert (
            f"energy_capacity_per_storage_capacity_{override}" in m.backend.constraints
        )
        if override == "equals":
            assert not any(
                "energy_capacity_per_storage_capacity_{i}" in m.backend.constraints
                for i in ["max", "min"]
            )

    def test_no_loc_techs_energy_capacity_storage_constraint(self, caplog):
        """
        i for i in sets.loc_techs_store if constraint_exists(model_run, i, 'constraints.energy_cap_per_storage_cap_max')
        """
        with caplog.at_level(logging.INFO):
            m = build_model(model_file="energy_cap_per_storage_cap.yaml")

        m.build()
        assert not any(
            "energy_capacity_storage_{i}" in m.backend.constraints
            for i in ["max", "min", "equals"]
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
            m.build()
            expr = m.backend.variables.resource_cap.sel(
                nodes="b", techs="test_supply_plus"
            ).item()
            assert expr.lb == 0
            assert np.isinf(expr.ub)

        else:
            m = build_model(
                {
                    "techs.test_supply_plus.constraints.resource_cap_{}".format(
                        override
                    ): 10
                },
                "simple_supply_and_supply_plus,two_hours,investment_costs",
            )
            m.build()
            expr = m.backend.variables.resource_cap.sel(
                nodes="b", techs="test_supply_plus"
            ).item()
            if override == "max":
                assert expr.ub == 10
                assert expr.lb == 0
            elif override == "equals":
                assert expr.ub == 10
                assert expr.lb == 10
            if override == "min":
                assert expr.lb == 10
                assert np.isinf(expr.ub)

    def test_loc_techs_resource_capacity_equals_energy_capacity_constraint(
        self, simple_supply_and_supply_plus
    ):
        """
        i for i in sets.loc_techs_finite_resource_supply_plus
        if constraint_exists(model_run, i, 'constraints.resource_cap_equals_energy_cap')
        """
        assert (
            "resource_capacity_equals_energy_capacity"
            not in simple_supply_and_supply_plus.backend.constraints
        )

        m = build_model(
            {"techs.test_supply_plus.switches.resource_cap_equals_energy_cap": True},
            "simple_supply_and_supply_plus,two_hours,investment_costs",
        )
        m.build()
        assert "resource_capacity_equals_energy_capacity" in m.backend.constraints

    def test_loc_techs_resource_area_constraint(self, simple_supply_and_supply_plus):
        """
        i for i in sets.loc_techs_area if i in sets.loc_techs_supply_plus
        """
        assert "resource_area" not in simple_supply_and_supply_plus.backend.variables

        m = build_model(
            {"techs.test_supply_plus.constraints.resource_area_max": 10},
            "simple_supply_and_supply_plus,two_hours,investment_costs",
        )
        m.build()
        assert "resource_area" in m.backend.variables

        m = build_model(
            {"techs.test_supply_elec.constraints.resource_area_max": 10},
            "simple_supply_and_supply_plus,two_hours,investment_costs",
        )
        m.build()
        assert "resource_area" in m.backend.variables

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
        m.build()
        ub = m.backend.get_constraint(
            "force_zero_resource_area", as_backend_objs=False
        ).ub

        assert (ub.to_series().dropna() == 0).all()

    def test_loc_techs_resource_area_per_energy_capacity_constraint(
        self, simple_supply_and_supply_plus
    ):
        """
        i for i in sets.loc_techs_area if i in sets.loc_techs_supply_plus
        and constraint_exists(model_run, i, 'constraints.resource_area_per_energy_cap')
        """
        assert (
            "resource_area_per_energy_capacity"
            not in simple_supply_and_supply_plus.backend.constraints
        )

        m = build_model(
            {"techs.test_supply_plus.constraints.resource_area_max": 10},
            "simple_supply_and_supply_plus,two_hours,investment_costs",
        )
        m.build()
        assert "resource_area_per_energy_capacity" not in m.backend.constraints

        m = build_model(
            {"techs.test_supply_elec.constraints.resource_area_per_energy_cap": 10},
            "simple_supply_and_supply_plus,two_hours,investment_costs",
        )
        m.build()
        assert "resource_area_per_energy_capacity" in m.backend.constraints

        m = build_model(
            {
                "techs.test_supply_elec.constraints": {
                    "resource_area_per_energy_cap": 10,
                    "resource_area_max": 10,
                }
            },
            "simple_supply_and_supply_plus,two_hours,investment_costs",
        )
        m.build()
        assert "resource_area_per_energy_capacity" in m.backend.constraints

    def test_locs_resource_area_capacity_per_loc_constraint(
        self, simple_supply_and_supply_plus
    ):
        """
        i for i in sets.locs
        if model_run.nodes[i].get_key('available_area', None) is not None
        """
        assert (
            "resource_area_capacity_per_loc"
            not in simple_supply_and_supply_plus.backend.constraints
        )

        m = build_model(
            {"nodes.a.available_area": 1},
            "simple_supply_and_supply_plus,two_hours,investment_costs",
        )
        m.build()
        assert "resource_area_capacity_per_loc" not in m.backend.constraints

        m = build_model(
            {
                "nodes.a.available_area": 1,
                "techs.test_supply_plus.constraints.resource_area_max": 10,
            },
            "simple_supply_and_supply_plus,two_hours,investment_costs",
        )
        m.build()
        assert "resource_area_capacity_per_loc" in m.backend.constraints

    def test_loc_techs_energy_capacity_constraint(self, simple_supply_and_supply_plus):
        """
        i for i in sets.loc_techs
        if i not in sets.loc_techs_milp + sets.loc_techs_purchase
        """
        m2 = build_model(
            {"techs.test_supply_elec.constraints.energy_cap_scale": 5},
            "simple_supply_and_supply_plus,two_hours,investment_costs",
        )
        m2.build()
        assert (
            m2.backend.variables.energy_cap.sel(nodes="a", techs="test_supply_elec")
            .item()
            .ub
            == simple_supply_and_supply_plus.backend.variables.energy_cap.sel(
                nodes="a", techs="test_supply_elec"
            )
            .item()
            .ub
            * 5
        )

    @pytest.mark.filterwarnings("ignore:(?s).*Integer:calliope.exceptions.ModelWarning")
    def test_loc_techs_energy_capacity_milp_constraint(self):
        m = build_model(
            {}, "supply_milp,two_hours,investment_costs"
        )  # demand still is in loc_techs
        m.build()
        assert np.isinf(
            m.backend.variables.energy_cap.sel(nodes="a", techs="test_demand_elec")
            .item()
            .ub
        )
        assert (
            m.backend.variables.energy_cap.sel(nodes="a", techs="test_supply_elec")
            .item()
            .ub
            == 10
        )

    @pytest.mark.xfail(reason="This will be caught by typedconfig")
    def test_loc_techs_energy_capacity_constraint_warning_on_infinite_equals(self):
        # Check that setting `_equals` to infinity is caught:
        override = {
            "nodes.a.techs.test_supply_elec.constraints.energy_cap_equals": np.inf
        }
        with pytest.raises(exceptions.ModelError) as error:
            m = build_model(override, "simple_supply,two_hours,investment_costs")
            m.build()

        assert check_error_or_warning(
            error,
            "Cannot use inf for energy_cap_equals for node, tech `('a', 'test_supply_elec')`",
        )

    @pytest.mark.parametrize("bound", (("equals", "max")))
    def test_techs_energy_capacity_systemwide_constraint(self, bound):
        """
        i for i in sets.techs
        if model_run.get_key('techs.{}.constraints.energy_cap_max_systemwide'.format(i), None)
        """

        def check_bounds(constraint):
            assert constraint.ub.item() == 20
            if bound == "equals":
                assert constraint.lb.item() == 20
            if bound == "max":
                assert constraint.lb.item() is None

        m = build_model(
            {
                f"techs.test_supply_elec.constraints.energy_cap_{bound}_systemwide": 20
            },  # foo
            "simple_supply,two_hours,investment_costs",
        )
        m.build()
        assert "energy_capacity_systemwide" in m.backend.constraints
        assert isinstance(
            m.backend.constraints.energy_capacity_systemwide.sel(
                techs="test_supply_elec"
            ).item(),
            pmo.constraint,
        )

        check_bounds(
            m.backend.get_constraint(
                "energy_capacity_systemwide", as_backend_objs=False
            ).sel(techs="test_supply_elec")
        )

        # Check that a model without transmission techs doesn't cause an error
        m = build_model(
            {f"techs.test_supply_elec.constraints.energy_cap_{bound}_systemwide": 20},
            "simple_supply,two_hours,investment_costs",
            model_file="model_minimal.yaml",
        )
        m.build()
        assert "energy_capacity_systemwide" in m.backend.constraints
        check_bounds(
            m.backend.get_constraint(
                "energy_capacity_systemwide", as_backend_objs=False
            ).sel(techs="test_supply_elec")
        )

    @pytest.mark.parametrize("bound", (("equals", "max")))
    def test_techs_energy_capacity_systemwide_no_constraint(self, simple_supply, bound):
        assert "energy_capacity_systemwide" not in simple_supply.backend.constraints

        # setting the constraint to infinity leads to no constraint being built
        m = build_model(
            {
                f"techs.test_supply_elec.constraints.energy_cap_{bound}_systemwide": np.inf
            },
            "simple_supply,two_hours,investment_costs",
        )
        m.build()
        assert "energy_capacity_systemwide" not in m.backend.constraints


@pytest.mark.skip(reason="to be reimplemented by comparison to LP files")
class TestDispatchConstraints:
    # dispatch.py
    def test_loc_tech_carriers_carrier_production_max_constraint(self, simple_supply):
        """
        i for i in sets.loc_tech_carriers_prod
        if i not in sets.loc_tech_carriers_conversion_plus
        and i.rsplit('::', 1)[0] not in sets.loc_techs_milp
        """
        assert "carrier_production_max" in simple_supply.backend.constraints

    @pytest.mark.filterwarnings("ignore:(?s).*Integer:calliope.exceptions.ModelWarning")
    def test_loc_tech_carriers_carrier_production_max_milp_constraint(
        self, supply_milp
    ):
        assert "carrier_production_max" not in supply_milp.backend.constraints

    def test_loc_tech_carriers_carrier_production_min_constraint(self, simple_supply):
        """
        i for i in sets.loc_tech_carriers_prod
        if i not in sets.loc_tech_carriers_conversion_plus
        and constraint_exists(model_run, i, 'constraints.energy_cap_min_use')
        and i.rsplit('::', 1)[0] not in sets.loc_techs_milp
        """
        assert "carrier_production_min" not in simple_supply.backend.constraints

        m = build_model(
            {"techs.test_supply_elec.constraints.energy_cap_min_use": 0.1},
            "simple_supply,two_hours,investment_costs",
        )
        m.build()
        assert "carrier_production_min" in m.backend.constraints

    @pytest.mark.filterwarnings("ignore:(?s).*Integer:calliope.exceptions.ModelWarning")
    def test_loc_tech_carriers_carrier_production_min_milp_constraint(
        self, supply_milp
    ):
        assert "carrier_production_min" not in supply_milp.backend.constraints

        m = build_model(
            {"techs.test_supply_elec.constraints.energy_cap_min_use": 0.1},
            "supply_milp,two_hours,investment_costs",
        )
        m.build()
        assert "carrier_production_min" not in m.backend.constraints

    def test_loc_tech_carriers_carrier_consumption_max_constraint(self, simple_supply):
        """
        i for i in sets.loc_tech_carriers_con
        if i.rsplit('::', 1)[0] in sets.loc_techs_demand +
            sets.loc_techs_storage + sets.loc_techs_transmission
        and i.rsplit('::', 1)[0] not in sets.loc_techs_milp
        """
        assert "carrier_consumption_max" in simple_supply.backend.constraints

    @pytest.mark.filterwarnings("ignore:(?s).*Integer:calliope.exceptions.ModelWarning")
    def test_loc_tech_carriers_carrier_consumption_max_milp_constraint(
        self, supply_milp
    ):
        assert "carrier_consumption_max" in supply_milp.backend.constraints

    def test_loc_techs_resource_max_constraint(
        self, simple_supply, simple_supply_and_supply_plus
    ):
        """
        sets.loc_techs_finite_resource_supply_plus,
        """
        assert "resource_max" not in simple_supply.backend.constraints
        assert "resource_max" in simple_supply_and_supply_plus.backend.constraints

        m = build_model(
            {"techs.test_supply_plus.constraints.resource": np.inf},
            "simple_supply_and_supply_plus,two_hours,investment_costs",
        )
        m.build()
        assert "resource_max" in m.backend.constraints

    def test_loc_techs_storage_max_constraint(
        self, simple_supply, simple_supply_and_supply_plus, simple_storage
    ):
        """
        sets.loc_techs_store
        """
        assert "storage_max" not in simple_supply.backend.constraints
        assert "storage_max" in simple_supply_and_supply_plus.backend.constraints
        assert "storage_max" in simple_storage.backend.constraints

    def test_loc_tech_carriers_ramping_constraint(self, simple_supply):
        """
        i for i in sets.loc_tech_carriers_prod
        if i.rsplit('::', 1)[0] in sets.loc_techs_ramping
        """
        assert "ramping_up" not in simple_supply.backend.constraints
        assert "ramping_down" not in simple_supply.backend.constraints

        m = build_model(
            {"techs.test_supply_elec.constraints.energy_ramping": 0.1},
            "simple_supply,two_hours,investment_costs",
        )
        m.build()
        assert "ramping_up" in m.backend.constraints
        assert "ramping_down" in m.backend.constraints

        m = build_model(
            {"techs.test_conversion.constraints.energy_ramping": 0.1},
            "simple_conversion,two_hours,investment_costs",
        )
        m.build()
        assert "ramping_up" in m.backend.constraints
        assert "ramping_down" in m.backend.constraints


@pytest.mark.filterwarnings("ignore:(?s).*Integer:calliope.exceptions.ModelWarning")
@pytest.mark.skip(reason="to be reimplemented by comparison to LP files")
class TestMILPConstraints:
    # milp.py
    def test_loc_techs_unit_commitment_milp_constraint(
        self, simple_supply, supply_milp, supply_purchase
    ):
        """
        sets.loc_techs_milp,
        """
        assert "unit_commitment_milp" not in simple_supply.backend.constraints
        assert "unit_commitment_milp" in supply_milp.backend.constraints
        assert "unit_commitment_milp" not in supply_purchase.backend.constraints

    def test_loc_techs_unit_capacity_milp_constraint(
        self, simple_supply, supply_milp, supply_purchase
    ):
        """
        sets.loc_techs_milp,
        """
        assert "units" not in simple_supply.backend.variables
        assert "units" in supply_milp.backend.variables
        assert "units" not in supply_purchase.backend.variables

    def test_loc_tech_carriers_carrier_production_max_milp_constraint(
        self, simple_supply, supply_milp, supply_purchase, conversion_plus_milp
    ):
        """
        i for i in sets.loc_tech_carriers_prod
        if i not in sets.loc_tech_carriers_conversion_plus
        and i.rsplit('::', 1)[0] in sets.loc_techs_milp
        """
        assert "carrier_production_max_milp" not in simple_supply.backend.constraints
        assert "carrier_production_max_milp" in supply_milp.backend.constraints

        assert "carrier_production_max_milp" not in supply_purchase.backend.constraints
        assert (
            "carrier_production_max_milp"
            not in conversion_plus_milp.backend.constraints
        )

    def test_loc_techs_carrier_production_max_conversion_plus_milp_constraint(
        self,
        simple_supply,
        supply_milp,
        supply_purchase,
        conversion_plus_milp,
        conversion_plus_purchase,
    ):
        """
        i for i in sets.loc_techs_conversion_plus
        if i in sets.loc_techs_milp
        """

        assert (
            "carrier_production_max_conversion_plus_milp"
            not in simple_supply.backend.constraints
        )
        assert (
            "carrier_production_max_conversion_plus_milp"
            not in supply_milp.backend.constraints
        )
        assert (
            "carrier_production_max_conversion_plus_milp"
            not in supply_purchase.backend.constraints
        )
        assert (
            "carrier_production_max_conversion_plus_milp"
            in conversion_plus_milp.backend.constraints
        )
        assert (
            "carrier_production_max_conversion_plus_milp"
            not in conversion_plus_purchase.backend.constraints
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
        m.build()
        assert "carrier_production_min_milp" not in m.backend.constraints

        m = build_model(
            {"techs.test_supply_elec.constraints.energy_cap_min_use": 0.1},
            "supply_milp,two_hours,investment_costs",
        )
        m.build()
        assert "carrier_production_min_milp" in m.backend.constraints

        m = build_model(
            {"techs.test_supply_elec.constraints.energy_cap_min_use": 0.1},
            "supply_purchase,two_hours,investment_costs",
        )
        m.build()
        assert "carrier_production_min_milp" not in m.backend.constraints

        m = build_model(
            {"techs.test_supply_elec.constraints.energy_cap_min_use": 0.1},
            "conversion_plus_milp,two_hours,investment_costs",
        )
        m.build()
        assert "carrier_production_min_milp" not in m.backend.constraints

        m = build_model(
            {"techs.test_conversion_plus.constraints.energy_cap_min_use": 0.1},
            "conversion_plus_milp,two_hours,investment_costs",
        )
        m.build()
        assert "carrier_production_min_milp" not in m.backend.constraints

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
        m.build()
        assert (
            "carrier_production_min_conversion_plus_milp" not in m.backend.constraints
        )

        m = build_model(
            {"techs.test_supply_elec.constraints.energy_cap_min_use": 0.1},
            "supply_milp,two_hours,investment_costs",
        )
        m.build()
        assert (
            "carrier_production_min_conversion_plus_milp" not in m.backend.constraints
        )

        m = build_model(
            {"techs.test_supply_elec.constraints.energy_cap_min_use": 0.1},
            "conversion_plus_milp,two_hours,investment_costs",
        )
        m.build()
        assert (
            "carrier_production_min_conversion_plus_milp" not in m.backend.constraints
        )

        m = build_model(
            {"techs.test_conversion_plus.constraints.energy_cap_min_use": 0.1},
            "conversion_plus_milp,two_hours,investment_costs",
        )
        m.build()
        assert "carrier_production_min_conversion_plus_milp" in m.backend.constraints

        m = build_model(
            {"techs.test_conversion_plus.constraints.energy_cap_min_use": 0.1},
            "conversion_plus_purchase,two_hours,investment_costs",
        )
        m.build()
        assert (
            "carrier_production_min_conversion_plus_milp" not in m.backend.constraints
        )

    def test_loc_tech_carriers_carrier_consumption_max_milp_constraint(
        self, simple_supply, supply_milp, storage_milp, conversion_plus_milp
    ):
        """
        i for i in sets.loc_tech_carriers_con
        if i.rsplit('::', 1)[0] in sets.loc_techs_demand +
            sets.loc_techs_storage + sets.loc_techs_transmission
        and i.rsplit('::', 1)[0] in sets.loc_techs_milp
        """
        assert "carrier_consumption_max_milp" not in simple_supply.backend.constraints
        assert "carrier_consumption_max_milp" not in supply_milp.backend.constraints
        assert "carrier_consumption_max_milp" in storage_milp.backend.constraints
        assert (
            "carrier_consumption_max_milp"
            not in conversion_plus_milp.backend.constraints
        )

    def test_loc_techs_energy_capacity_units_milp_constraint(
        self, simple_supply, supply_milp, storage_milp, conversion_plus_milp
    ):
        """
        i for i in sets.loc_techs_milp
        if constraint_exists(model_run, i, 'constraints.energy_cap_per_unit')
        is not None
        """
        assert "energy_capacity_units_milp" not in simple_supply.backend.constraints
        assert "energy_capacity_units_milp" in supply_milp.backend.constraints
        assert "energy_capacity_units_milp" in storage_milp.backend.constraints
        assert "energy_capacity_units_milp" in conversion_plus_milp.backend.constraints

    def test_loc_techs_storage_capacity_units_milp_constraint(
        self,
        simple_supply,
        supply_milp,
        storage_milp,
        conversion_plus_milp,
        supply_and_supply_plus_milp,
    ):
        """
        i for i in sets.loc_techs_milp if i in sets.loc_techs_store
        """
        assert "storage_capacity_units_milp" not in simple_supply.backend.constraints
        assert "storage_capacity_units_milp" not in supply_milp.backend.constraints
        assert "storage_capacity_units_milp" in storage_milp.backend.constraints
        assert (
            "storage_capacity_units_milp"
            not in conversion_plus_milp.backend.constraints
        )
        assert (
            "storage_capacity_units_milp"
            in supply_and_supply_plus_milp.backend.constraints
        )

    def test_loc_techs_energy_capacity_max_purchase_milp_constraint(
        self, simple_supply, supply_milp, supply_purchase
    ):
        """
        i for i in sets.loc_techs_purchase
        if (constraint_exists(model_run, i, 'constraints.energy_cap_equals') is not None
            or constraint_exists(model_run, i, 'constraints.energy_cap_max') is not None)
        """
        assert (
            "energy_capacity_max_purchase_milp" not in simple_supply.backend.constraints
        )
        assert (
            "energy_capacity_max_purchase_milp" not in supply_milp.backend.constraints
        )
        assert (
            "energy_capacity_max_purchase_milp" in supply_purchase.backend.constraints
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
        m.build()
        assert "energy_capacity_max_purchase_milp" in m.backend.constraints

    def test_loc_techs_energy_capacity_min_purchase_milp_constraint(
        self, simple_supply, supply_milp, supply_purchase
    ):
        """
        i for i in sets.loc_techs_purchase
        if (not constraint_exists(model_run, i, 'constraints.energy_cap_equals')
            and constraint_exists(model_run, i, 'constraints.energy_cap_min'))
        """
        assert (
            "energy_capacity_min_purchase_milp" not in simple_supply.backend.constraints
        )
        assert (
            "energy_capacity_min_purchase_milp" not in supply_milp.backend.constraints
        )
        assert (
            "energy_capacity_min_purchase_milp"
            not in supply_purchase.backend.constraints
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
        m.build()
        assert "energy_capacity_min_purchase_milp" not in m.backend.constraints

        m = build_model(
            {"techs.test_supply_elec.constraints.energy_cap_min": 10},
            "supply_purchase,two_hours,investment_costs",
        )
        m.build()
        assert "energy_capacity_min_purchase_milp" in m.backend.constraints

    def test_loc_techs_storage_capacity_max_purchase_milp_constraint(
        self, simple_storage, storage_milp, storage_purchase, supply_purchase
    ):
        """
        i for i in set(sets.loc_techs_purchase).intersection(sets.loc_techs_store)
        """
        assert (
            "storage_capacity_max_purchase_milp"
            not in simple_storage.backend.constraints
        )
        assert (
            "storage_capacity_max_purchase_milp" not in storage_milp.backend.constraints
        )
        assert (
            "storage_capacity_max_purchase_milp" in storage_purchase.backend.constraints
        )
        assert (
            "storage_capacity_max_purchase_milp"
            not in supply_purchase.backend.constraints
        )

    def test_loc_techs_storage_capacity_min_purchase_milp_constraint(
        self, storage_purchase
    ):
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
        m.build()
        assert "storage_capacity_min_purchase_milp" not in m.backend.constraints

        m = build_model(
            {"techs.test_storage.constraints.storage_cap_min": 10},
            "storage_milp,two_hours,investment_costs",
        )
        m.build()
        assert "storage_capacity_min_purchase_milp" not in m.backend.constraints

        assert (
            "storage_capacity_min_purchase_milp"
            not in storage_purchase.backend.constraints
        )

        m = build_model(
            {"techs.test_storage.constraints.storage_cap_min": 10},
            "storage_purchase,two_hours,investment_costs",
        )
        m.build()
        assert "storage_capacity_min_purchase_milp" in m.backend.constraints

        m = build_model(
            {
                "techs.test_storage.constraints": {
                    "storage_cap_equals": 10,
                    "storage_cap_min": 10,
                }
            },
            "storage_purchase,two_hours,investment_costs",
        )
        m.build()
        assert "storage_capacity_min_purchase_milp" not in m.backend.constraints

    @pytest.mark.parametrize(
        ("scenario", "exists", "override_dict"),
        (
            ("simple_supply", ("not", "not"), {}),
            ("supply_milp", ("not", "not"), {}),
            (
                "supply_milp",
                ("not", "is"),
                {"techs.test_supply_elec.costs.monetary.purchase": 1},
            ),
            ("supply_purchase", ("is", "not"), {}),
        ),
    )
    def test_loc_techs_update_costs_investment_units_milp_constraint(
        self, scenario, exists, override_dict
    ):
        """
        i for i in sets.loc_techs_milp
        if i in sets.loc_techs_investment_cost and
        any(constraint_exists(model_run, i, 'costs.{}.purchase'.format(j))
               for j in model_run.sets.costs)
        """

        m = build_model(override_dict, f"{scenario},two_hours,investment_costs")
        m.build()
        if exists[0] == "not":
            assert not check_variable_exists(
                m.backend.get_expression("cost_investment", as_backend_objs=False),
                "purchased",
            )
        else:
            assert check_variable_exists(
                m.backend.get_expression("cost_investment", as_backend_objs=False),
                "purchased",
            )
        if exists[1] == "not":
            assert not check_variable_exists(
                m.backend.get_expression("cost_investment", as_backend_objs=False),
                "units",
            )
        else:
            assert check_variable_exists(
                m.backend.get_expression("cost_investment", as_backend_objs=False),
                "units",
            )

    def test_techs_unit_capacity_max_systemwide_milp_constraint(self):
        """
        sets.techs if unit_cap_max_systemwide or unit_cap_equals_systemwide
        """

        override_max = {
            "links.a,b.exists": True,
            "techs.test_conversion_plus.constraints.units_max_systemwide": 2,
            "nodes.b.techs.test_conversion_plus.constraints": {
                "units_max": 2,
                "energy_cap_per_unit": 5,
            },
        }
        m = build_model(override_max, "conversion_plus_milp,two_hours,investment_costs")
        m.build()
        assert "unit_capacity_systemwide_milp" in m.backend.constraints
        assert (
            m.backend.get_constraint(
                "unit_capacity_systemwide_milp", as_backend_objs=False
            )
            .sel(techs="test_conversion_plus")
            .ub.item()
            == 2
        )

    def test_techs_unit_capacity_equals_systemwide_milp_constraint(self):
        """
        sets.techs if unit_cap_max_systemwide or unit_cap_equals_systemwide
        """
        override_equals = {
            "links.a,b.exists": True,
            "techs.test_conversion_plus.constraints.units_equals_systemwide": 1,
            "nodes.b.techs.test_conversion_plus.costs.monetary.purchase": 1,
        }
        m = build_model(
            override_equals, "conversion_plus_milp,two_hours,investment_costs"
        )
        m.build()
        assert "unit_capacity_systemwide_milp" in m.backend.constraints
        assert (
            m.backend.get_constraint(
                "unit_capacity_systemwide_milp", as_backend_objs=False
            )
            .sel(techs="test_conversion_plus")
            .lb.item()
            == 1
        )
        assert (
            m.backend.get_constraint(
                "unit_capacity_systemwide_milp", as_backend_objs=False
            )
            .sel(techs="test_conversion_plus")
            .ub.item()
            == 1
        )

    # TODO: always have transmission techs be independent of node names
    @pytest.mark.xfail(
        reason="systemwide constraints now don't work with transmission techs, since transmission tech names are now never independent of a node"
    )
    def test_techs_unit_capacity_max_systemwide_transmission_milp_constraint(self):
        """
        sets.techs if unit_cap_max_systemwide or unit_cap_equals_systemwide
        """
        override_transmission = {
            "links.a,b.exists": True,
            "techs.test_transmission_elec.constraints": {
                "units_max_systemwide": 1,
                "lifetime": 25,
            },
            "techs.test_transmission_elec.costs.monetary": {
                "purchase": 1,
                "interest_rate": 0.1,
            },
        }
        m = build_model(
            override_transmission, "simple_supply,two_hours,investment_costs"
        )
        m.build()
        assert "unit_capacity_systemwide_milp" in m.backend.constraints
        assert (
            m.backend.get_constraint(
                "unit_capacity_systemwide_milp", as_backend_objs=False
            )
            .sel(techs="test_transmission_elec")
            .item()
            .ub
            == 2
        )

    def test_techs_unit_capacity_max_systemwide_no_transmission_milp_constraint(self):
        override_no_transmission = {
            "techs.test_supply_elec.constraints.units_equals_systemwide": 1,
            "nodes.b.techs.test_supply_elec.costs.monetary.purchase": 1,
        }
        m = build_model(
            override_no_transmission,
            "supply_milp,two_hours,investment_costs",
            model_file="model_minimal.yaml",
        )
        m.build()
        assert "unit_capacity_systemwide_milp" in m.backend.constraints

    @pytest.mark.parametrize("tech", (("test_storage"), ("test_transmission_elec")))
    def test_asynchronous_prod_con_constraint(self, tech):
        """
        Binary switch for prod/con can be activated using the option
        'asynchronous_prod_con'
        """
        m = build_model(
            {f"techs.{tech}.constraints.force_asynchronous_prod_con": True},
            "simple_storage,investment_costs",
        )
        m.build()
        assert "prod_con_switch" in m.backend.variables
        assert "asynchronous_con_milp" in m.backend.constraints
        assert "asynchronous_prod_milp" in m.backend.constraints


@pytest.mark.skip(reason="to be reimplemented by comparison to LP files")
class TestConversionConstraints:
    # conversion.py
    def test_loc_techs_balance_conversion_constraint(
        self, simple_supply, simple_conversion, simple_conversion_plus
    ):
        """
        sets.loc_techs_conversion,
        """
        assert "balance_conversion" not in simple_supply.backend.constraints
        assert "balance_conversion" in simple_conversion.backend.constraints
        assert "balance_conversion" not in simple_conversion_plus.backend.constraints


@pytest.mark.skip(reason="to be reimplemented by comparison to LP files")
class TestNetworkConstraints:
    # network.py
    def test_loc_techs_symmetric_transmission_constraint(
        self, simple_supply, simple_conversion_plus
    ):
        """
        sets.loc_techs_transmission,
        """
        assert "symmetric_transmission" in simple_supply.backend.constraints
        assert (
            "symmetric_transmission" not in simple_conversion_plus.backend.constraints
        )


@pytest.mark.skip(reason="to be reimplemented by comparison to LP files")
class TestClusteringConstraints:
    def constraints(self):
        return [
            "balance_storage_inter",
            "storage_intra_max",
            "storage_intra_min",
            "storage_inter_max",
            "storage_inter_min",
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
                    "clustering_func": "file=cluster_days.csv:a",
                    "how": how,
                },
            },
            "model.custom_math": ["storage_inter_cluster"]
            if storage_inter_cluster
            else [],
            "run.cyclic_storage": cyclic,
        }
        if storage_initial:
            override.update({"techs.test_storage.constraints.storage_initial": 0})
        return build_model(override, "simple_storage,investment_costs")

    def test_cluster_storage_constraints(self):
        m = self.cluster_model()
        m.build()

        for variable in self.decision_variables():
            assert variable in m.backend.variables

        for constraint in self.constraints():
            assert constraint in m.backend.constraints

        assert "storage_max" not in m.backend.constraints
        assert "set_storage_initial" not in m.backend.constraints

    def test_cluster_cyclic_storage_constraints(self):
        m = self.cluster_model(cyclic=True)
        m.build()

        for variable in self.decision_variables():
            assert variable in m.backend.variables

        for constraint in self.constraints():
            assert constraint in m.backend.constraints

        assert "storage_max" not in m.backend.constraints
        assert "set_storage_initial" not in m.backend.constraints

    def test_no_cluster_storage_constraints(self):
        m = self.cluster_model(storage_inter_cluster=False)
        m.build()

        for variable in self.decision_variables():
            assert variable not in m.backend.variables

        for constraint in self.constraints():
            assert constraint not in m.backend.constraints

        assert "storage_max" in m.backend.constraints


class TestLogging:
    @pytest.fixture(scope="module")
    def gurobi_model(self):
        pytest.importorskip("gurobipy")
        model_file = "model.yaml"
        model = build_model(
            model_file=model_file,
            scenario="simple_supply,investment_costs",
            override_dict={"run": {"solver": "gurobi", "solver_io": "python"}},
        )
        model.build()
        return model

    def test_no_duplicate_log_message(self, caplog, gurobi_model):
        caplog.set_level(logging.DEBUG)
        gurobi_model.solve()
        all_log_messages = [r.msg for r in caplog.records]
        assert sum([i.find("Gurobi Optimizer") > -1 for i in all_log_messages]) == 1


class TestNewBackend:
    @pytest.fixture(scope="class")
    def simple_supply_longnames(self):
        m = build_model({}, "simple_supply,two_hours,investment_costs")
        m.build()
        m.backend.verbose_strings()
        return m

    def test_new_build_has_backend(self, simple_supply_new_build):
        assert hasattr(simple_supply_new_build, "backend")

    def test_new_build_optimal(self, simple_supply_new_build):
        assert hasattr(simple_supply_new_build, "results")
        assert (
            simple_supply_new_build._model_data.attrs["termination_condition"]
            == "optimal"
        )

    @pytest.mark.parametrize(
        "component_type", ["variable", "global_expression", "parameter", "constraint"]
    )
    def test_new_build_get_missing_component(
        self, simple_supply_new_build, component_type
    ):
        returned_ = getattr(simple_supply_new_build.backend, f"get_{component_type}")(
            "foo"
        )
        assert returned_ is None

    def test_new_build_get_variable(self, simple_supply_new_build):
        var = simple_supply_new_build.backend.get_variable("energy_cap")
        assert (
            var.to_series().dropna().apply(lambda x: isinstance(x, pmo.variable)).all()
        )
        assert var.attrs == {
            "variables": 1,
            "references": {
                "carrier_consumption_max",
                "carrier_production_max",
                "cost_investment",
                "symmetric_transmission",
            },
            "description": "A technology's energy capacity, also known as its nominal or nameplate capacity.",
            "unit": "power",
            "coords_in_name": False,
        }

    def test_new_build_get_variable_as_vals(self, simple_supply_new_build):
        var = simple_supply_new_build.backend.get_variable(
            "energy_cap", as_backend_objs=False
        )
        assert (
            not var.to_series()
            .dropna()
            .apply(lambda x: isinstance(x, pmo.variable))
            .any()
        )

    def test_new_build_get_parameter(self, simple_supply_new_build):
        param = simple_supply_new_build.backend.get_parameter("energy_eff")
        assert (
            param.to_series()
            .dropna()
            .apply(lambda x: isinstance(x, pmo.parameter))
            .all()
        )
        assert param.attrs == {
            "parameters": 1,
            "is_result": 0,
            "original_dtype": np.dtype("float64"),
            "references": {"balance_demand", "balance_transmission"},
            "coords_in_name": False,
        }

    def test_new_build_get_parameter_as_vals(self, simple_supply_new_build):
        param = simple_supply_new_build.backend.get_parameter(
            "energy_eff", as_backend_objs=False
        )
        assert param.dtype == np.dtype("float64")

    def test_new_build_get_global_expression(self, simple_supply_new_build):
        expr = simple_supply_new_build.backend.get_global_expression("cost_investment")
        assert (
            expr.to_series()
            .dropna()
            .apply(lambda x: isinstance(x, pmo.expression))
            .all()
        )
        assert expr.attrs == {
            "global_expressions": 1,
            "references": {"cost"},
            "description": "The installation costs of a technology, including annualised investment costs and annual maintenance costs.",
            "unit": "cost",
            "coords_in_name": False,
        }

    def test_new_build_get_global_expression_as_str(self, simple_supply_new_build):
        expr = simple_supply_new_build.backend.get_global_expression(
            "cost", as_backend_objs=False
        )
        assert expr.to_series().dropna().apply(lambda x: isinstance(x, str)).all()

    def test_new_build_get_global_expression_as_vals(self, simple_supply_new_build):
        expr = simple_supply_new_build.backend.get_global_expression(
            "cost", as_backend_objs=False, eval_body=True
        )
        assert (
            expr.to_series().dropna().apply(lambda x: isinstance(x, (float, int))).all()
        )

    def test_new_build_get_constraint(self, simple_supply_new_build):
        constr = simple_supply_new_build.backend.get_constraint("system_balance")
        assert (
            constr.to_series()
            .dropna()
            .apply(lambda x: isinstance(x, pmo.constraint))
            .all()
        )
        assert constr.attrs == {
            "constraints": 1,
            "references": set(),
            "description": "Set the global energy balance of the optimisation problem by fixing the total production of a given energy carrier to equal the total consumption of that carrier at every node in every timestep.",
            "coords_in_name": False,
        }

    def test_new_build_get_constraint_as_str(self, simple_supply_new_build):
        constr = simple_supply_new_build.backend.get_constraint(
            "system_balance", as_backend_objs=False
        )
        assert isinstance(constr, xr.Dataset)
        assert set(constr.data_vars.keys()) == {"ub", "body", "lb"}
        assert (
            constr["body"]
            .to_series()
            .dropna()
            .apply(lambda x: isinstance(x, str))
            .all()
        )

    def test_new_build_get_constraint_as_vals(self, simple_supply_new_build):
        constr = simple_supply_new_build.backend.get_constraint(
            "system_balance", as_backend_objs=False, eval_body=True
        )
        assert (
            constr["body"]
            .to_series()
            .dropna()
            .apply(lambda x: isinstance(x, (float, int)))
            .all()
        )

    @pytest.mark.parametrize("bound", ["lb", "ub"])
    def test_new_build_get_constraint_bounds(self, simple_supply_new_build, bound):
        constr = simple_supply_new_build.backend.get_constraint(
            "system_balance", as_backend_objs=False
        )
        assert (constr[bound].to_series().dropna() == 0).all()

    def test_solve_before_build(self):
        m = build_model({}, "simple_supply,two_hours,investment_costs")
        with pytest.raises(exceptions.ModelError) as excinfo:
            m.solve()
        assert check_error_or_warning(excinfo, "You must build the optimisation")

    def test_solve_after_solve(self, simple_supply_new_build):
        with pytest.raises(exceptions.ModelError) as excinfo:
            simple_supply_new_build.solve()
        assert check_error_or_warning(excinfo, "This model object already has results.")

    def test_solve_operate_not_allowed(self, simple_supply_new_build):
        simple_supply_new_build.run_config["mode"] = "operate"
        simple_supply_new_build._model_data.attrs["allow_operate_mode"] = False

        try:
            with pytest.raises(exceptions.ModelError) as excinfo:
                simple_supply_new_build.solve(force=True)
            assert check_error_or_warning(excinfo, "Unable to run this model in op")
        except AssertionError as e:
            simple_supply_new_build.run_config["mode"] = "plan"
            simple_supply_new_build._model_data.attrs["allow_operate_mode"] = True
            raise e
        else:
            simple_supply_new_build.run_config["mode"] = "plan"
            simple_supply_new_build._model_data.attrs["allow_operate_mode"] = True

    def test_solve_warmstart_not_possible(self, simple_supply_new_build):
        with pytest.warns(exceptions.ModelWarning) as excinfo:
            simple_supply_new_build.solve(force=True, warmstart=True)
        assert check_error_or_warning(excinfo, "cbc, does not support warmstart")

    def test_solve_non_optimal(self, simple_supply_new_build):
        def _update_param(param):
            param.value = param.value * 1000

        simple_supply_new_build.backend.apply_func(
            _update_param,
            simple_supply_new_build.backend.parameters.resource.loc[
                {"techs": "test_demand_elec"}
            ],
        )
        with pytest.warns(exceptions.BackendWarning) as excinfo:
            simple_supply_new_build.solve(force=True)

        assert check_error_or_warning(excinfo, "Model solution was non-optimal")
        assert (
            simple_supply_new_build._model_data.attrs["termination_condition"]
            == "infeasible"
        )
        assert not simple_supply_new_build.results
        assert "energy_cap" not in simple_supply_new_build._model_data.data_vars

    def test_raise_error_on_preexistence_same_type(self, simple_supply_new_build):
        with pytest.raises(exceptions.BackendError) as excinfo:
            simple_supply_new_build.backend.add_parameter("energy_eff", xr.DataArray(1))

        assert check_error_or_warning(
            excinfo,
            "Trying to add already existing `energy_eff` to backend model parameters.",
        )

    def test_raise_error_on_preexistence_diff_type(self, simple_supply_new_build):
        with pytest.raises(exceptions.BackendError) as excinfo:
            simple_supply_new_build.backend.add_parameter(
                "carrier_prod", xr.DataArray(1)
            )

        assert check_error_or_warning(
            excinfo,
            "Trying to add already existing *variable* `carrier_prod` as a backend model *parameter*.",
        )

    def test_raise_error_on_constraint_with_nan(self, simple_supply_new_build):
        """
        A very simple constraint: For each tech, let the annual and regional sum of `carrier_prod` be larger than 100.
        However, not every tech has the variable `carrier_prod`.
        How to solve it? Let the constraint be active only where carrier_prod exists by setting 'where' accordingly.
        """
        # add constraint without nan
        constraint_dict = {
            "foreach": ["techs", "carriers"],
            "equation": "sum(carrier_prod, over=[nodes, timesteps]) >= 100",
            "where": "carrier AND allowed_carrier_prod=True AND [out, out_2, out_3] in carrier_tiers",  # <- no error is raised because of this
        }
        constraint_name = "constraint-without-nan"

        simple_supply_new_build.backend.add_constraint(
            simple_supply_new_build.inputs,
            constraint_name,
            constraint_dict,
        )

        assert (
            simple_supply_new_build.backend.get_constraint(constraint_name).name
            == constraint_name
        )

        # add constraint with nan
        constraint_dict = {
            "foreach": ["techs", "carriers"],
            "equation": "sum(carrier_prod, over=[nodes, timesteps]) >= 100",
            # "where": "carrier AND allowed_carrier_prod=True AND [out, out_2, out_3] in carrier_tiers",  # <- no error would be raised with this uncommented
        }
        constraint_name = "constraint-with-nan"

        with pytest.raises(exceptions.BackendError) as error:
            simple_supply_new_build.backend.add_constraint(
                simple_supply_new_build.inputs,
                constraint_name,
                constraint_dict,
            )

        assert check_error_or_warning(
            error,
            f"(constraints, {constraint_name}): Missing a linear expression for some coordinates selected by 'where'. Adapting 'where' might help.",
        )

    def test_raise_error_on_expression_with_nan(self, simple_supply_new_build):
        """
        A very simple expression: The annual and regional sum of `carrier_prod` for each tech.
        However, not every tech has the variable `carrier_prod`.
        How to solve it? Let the constraint be active only where carrier_prod exists by setting 'where' accordingly.
        """
        # add expression without nan
        expression_dict = {
            "foreach": ["techs", "carriers"],
            "equation": "sum(carrier_prod, over=[nodes, timesteps])",
            "where": "carrier AND allowed_carrier_prod=True AND [out, out_2, out_3] in carrier_tiers",  # <- no error is raised because of this
        }
        expression_name = "expression-without-nan"

        # add expression with nan
        simple_supply_new_build.backend.add_global_expression(
            simple_supply_new_build.inputs,
            expression_name,
            expression_dict,
        )

        assert (
            simple_supply_new_build.backend.get_global_expression(expression_name).name
            == expression_name
        )

        expression_dict = {
            "foreach": ["techs", "carriers"],
            "equation": "sum(carrier_prod, over=[nodes, timesteps])",
            # "where": "carrier AND allowed_carrier_prod=True AND [out, out_2, out_3] in carrier_tiers",  # <- no error would be raised with this uncommented
        }
        expression_name = "expression-with-nan"

        with pytest.raises(exceptions.BackendError) as error:
            simple_supply_new_build.backend.add_global_expression(
                simple_supply_new_build.inputs,
                expression_name,
                expression_dict,
            )

        assert check_error_or_warning(
            error,
            f"(expressions, {expression_name}): Missing a linear expression for some coordinates selected by 'where'. Adapting 'where' might help.",
        )

    def test_raise_error_on_excess_dimensions(self, simple_supply_new_build):
        """
        A very simple constraint: For each tech, let the `energy_cap` be larger than 100.
        However, we forgot to include `nodes` in `foreach`.
        With `nodes` included, this constraint should build.
        """
        # add constraint without excess dimensions
        constraint_dict = {
            "foreach": [
                "techs",
                "nodes",
            ],  # as 'nodes' is listed here, the constraint will have no excess dimensions
            "equation": "energy_cap >= 100",
        }
        constraint_name = "constraint-without-excess-dimensions"

        simple_supply_new_build.backend.add_constraint(
            simple_supply_new_build.inputs,
            constraint_name,
            constraint_dict,
        )

        assert (
            simple_supply_new_build.backend.get_constraint(constraint_name).name
            == constraint_name
        )

        # add constraint with excess dimensions
        constraint_dict = {
            "foreach": [
                "techs"
            ],  # as 'nodes' is not listed here, the constraint will have excess dimensions
            "equation": "energy_cap >= 100",
        }
        constraint_name = "constraint-with-excess-dimensions"

        with pytest.raises(exceptions.BackendError) as error:
            simple_supply_new_build.backend.add_constraint(
                simple_supply_new_build.inputs,
                constraint_name,
                constraint_dict,
            )

        assert check_error_or_warning(
            error,
            f"(constraints, {constraint_name}): The linear expression array is indexed over dimensions not present in `foreach`: {{'nodes'}}",
        )

    @pytest.mark.parametrize(
        "component", ["parameters", "variables", "global_expressions", "constraints"]
    )
    def test_create_and_delete_pyomo_list(self, simple_supply_new_build, component):
        backend_instance = simple_supply_new_build.backend._instance
        simple_supply_new_build.backend._create_pyomo_list("foo", component)
        assert "foo" in getattr(backend_instance, component).keys()

        simple_supply_new_build.backend._delete_pyomo_list("foo", component)
        assert "foo" not in getattr(backend_instance, component).keys()

    @pytest.mark.parametrize(
        "component", ["parameters", "variables", "global_expressions", "constraints"]
    )
    def test_delete_inexistent_pyomo_list(self, simple_supply_new_build, component):
        backend_instance = simple_supply_new_build.backend._instance
        assert "bar" not in getattr(backend_instance, component).keys()
        simple_supply_new_build.backend._delete_pyomo_list("bar", component)
        assert "bar" not in getattr(backend_instance, component).keys()

    @pytest.mark.parametrize(
        ["component", "eq"],
        [("global_expressions", "energy_cap + 1"), ("constraints", "energy_cap >= 1")],
    )
    def test_add_allnull_expr_or_constr(self, simple_supply_new_build, component, eq):
        adder = getattr(
            simple_supply_new_build.backend, "add_" + component.removesuffix("s")
        )
        constr_dict = {
            "foreach": ["nodes", "techs"],
            "where": "True",
            "equations": [{"expression": eq, "where": "False"}],
        }
        adder(simple_supply_new_build._model_data, "foo", constr_dict)

        assert (
            "foo"
            not in getattr(simple_supply_new_build.backend._instance, component).keys()
        )
        assert "foo" not in simple_supply_new_build.backend._dataset.data_vars.keys()

    def test_add_allnull_param_no_shape(self, simple_supply_new_build):
        simple_supply_new_build.backend.add_parameter("foo", xr.DataArray(np.nan))

        assert "foo" not in simple_supply_new_build.backend._instance.parameters.keys()
        # We keep it in the dataset since it might be fillna'd by another param later.
        assert "foo" in simple_supply_new_build.backend._dataset.data_vars.keys()
        del simple_supply_new_build.backend._dataset["foo"]

    def test_add_allnull_param_with_shape(self, simple_supply_new_build):
        nan_array = simple_supply_new_build._model_data.energy_cap_max.where(
            lambda x: x < 0
        )
        simple_supply_new_build.backend.add_parameter("foo", nan_array)

        assert "foo" not in simple_supply_new_build.backend._instance.parameters.keys()
        # We keep it in the dataset since it might be fillna'd by another param later.
        assert "foo" in simple_supply_new_build.backend._dataset.data_vars.keys()
        del simple_supply_new_build.backend._dataset["foo"]

    def test_add_allnull_var(self, simple_supply_new_build):
        simple_supply_new_build.backend.add_variable(
            simple_supply_new_build._model_data,
            "foo",
            {"foreach": ["nodes"], "where": "False"},
        )
        assert "foo" not in simple_supply_new_build.backend._instance.variables.keys()
        assert "foo" not in simple_supply_new_build.backend._dataset.data_vars.keys()

    def test_add_allnull_obj(self, simple_supply_new_build):
        eq = {"expression": "bigM", "where": "False"}
        simple_supply_new_build.backend.add_objective(
            simple_supply_new_build._model_data,
            "foo",
            {"equations": [eq, eq], "sense": "minimise"},
        )
        assert len(simple_supply_new_build.backend._instance.objectives) == 1
        assert "foo" not in simple_supply_new_build.backend._dataset.data_vars.keys()

    def test_add_two_same_obj(self, simple_supply_new_build):
        eq = {"expression": "bigM", "where": "True"}
        with pytest.raises(exceptions.BackendError) as excinfo:
            simple_supply_new_build.backend.add_objective(
                simple_supply_new_build._model_data,
                "foo",
                {"equations": [eq, eq], "sense": "minimise"},
            )
        assert check_error_or_warning(
            excinfo,
            "More than one foo objective is valid for this optimisation problem; only one is allowed.",
        )

    def test_add_valid_obj(self, simple_supply_new_build):
        eq = {"expression": "bigM", "where": "True"}
        simple_supply_new_build.backend.add_objective(
            simple_supply_new_build._model_data,
            "foo",
            {"equations": [eq], "sense": "minimise"},
        )
        assert "foo" in simple_supply_new_build.backend.objectives
        assert not simple_supply_new_build.backend.objectives.foo.item().active

    def test_object_string_representation(self, simple_supply_new_build):
        assert (
            simple_supply_new_build.backend.variables.carrier_prod.sel(
                nodes="a",
                techs="test_supply_elec",
                carriers="electricity",
                timesteps="2005-01-01 00:00",
            )
            .item()
            .name
            == "variables[carrier_prod][0]"
        )
        assert not simple_supply_new_build.backend.variables.carrier_prod.coords_in_name

    @pytest.mark.parametrize(
        ["objname", "dims", "objtype"],
        [
            (
                "carrier_prod",
                {
                    "nodes": "a",
                    "techs": "test_supply_elec",
                    "carriers": "electricity",
                    "timesteps": "2005-01-01 00:00",
                },
                "variables",
            ),
            ("energy_eff", {"nodes": "a", "techs": "test_supply_elec"}, "parameters"),
        ],
    )
    def test_verbose_strings(self, simple_supply_longnames, objname, dims, objtype):
        obj = simple_supply_longnames.backend._dataset[objname]
        assert (
            obj.sel(dims).item().name
            == f"{objtype}[{objname}][{', '.join(dims[i] for i in obj.dims)}]"
        )
        assert obj.coords_in_name

    def test_verbose_strings_constraint(self, simple_supply_longnames):
        dims = {
            "nodes": "a",
            "techs": "test_demand_elec",
            "carriers": "electricity",
            "timesteps": "2005-01-01 00:00",
        }

        obj = simple_supply_longnames.backend.get_constraint(
            "balance_demand", as_backend_objs=False
        )
        energy_eff_dims = ", ".join(
            dims[i] for i in simple_supply_longnames.backend.parameters.energy_eff.dims
        )
        assert (
            obj.sel(dims).body.item()
            == f"parameters[energy_eff][{energy_eff_dims}]*variables[carrier_con][{', '.join(dims[i] for i in obj.dims)}]"
        )
        assert not obj.coords_in_name

    def test_verbose_strings_expression(self, simple_supply_longnames):
        dims = {
            "nodes": "a",
            "techs": "test_supply_elec",
            "costs": "monetary",
        }

        obj = simple_supply_longnames.backend.get_global_expression(
            "cost_investment", as_backend_objs=False
        )

        assert "variables[energy_cap][test_supply_elec, a]" in obj.sel(dims).item()
        assert "parameters[annualisation_weight]" in obj.sel(dims).item()

        assert not obj.coords_in_name

    def test_verbose_strings_no_len(self, simple_supply_longnames):
        obj = simple_supply_longnames.backend.parameters.annualisation_weight

        assert obj.item().name == "parameters[annualisation_weight]"
        assert obj.coords_in_name
