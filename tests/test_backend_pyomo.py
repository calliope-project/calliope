import importlib
import logging
from copy import deepcopy
from itertools import product

import calliope.exceptions as exceptions
import numpy as np
import pandas as pd
import pyomo.kernel as pmo
import pytest  # noqa: F401
import xarray as xr
from calliope.attrdict import AttrDict
from calliope.backend.pyomo_backend_model import PyomoBackendModel

from .common.util import build_test_model as build_model
from .common.util import check_error_or_warning, check_variable_exists


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
            assert m.config.build["cyclic_storage"] is True
        elif on is False:
            override = {"config.build.cyclic_storage": False}
            m = build_model(
                override, "simple_supply_and_supply_plus,operate,investment_costs"
            )
            assert m.config.build["cyclic_storage"] is False
        with pytest.warns(exceptions.ModelWarning) as warning:
            m.build()
        check_warn = check_error_or_warning(
            warning, "Storage cannot be cyclic in operate run mode"
        )
        if on is True:
            assert check_warn
        elif on is True:
            assert not check_warn
        assert m._model_data.attrs["config"].build.cyclic_storage is False

    @pytest.mark.parametrize(
        "param", [("flow_eff"), ("source_eff"), ("flow_out_parasitic_eff")]
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

    @pytest.mark.parametrize("constr", ("max", "equals"))
    def test_operate_flow_out_min_relative(self, constr):
        """If we depend on a finite flow_cap, we have to error on a user failing to define it"""
        m = build_model(
            {
                "techs.test_supply_elec": {
                    "constraints": {
                        "flow_out_min_relative": 0.1,
                        f"source_{constr}": "file=supply_plus_resource.csv:1",
                        "flow_cap_max": np.inf,
                    }
                }
            },
            "simple_supply_and_supply_plus,operate,investment_costs",
        )

        with pytest.raises(exceptions.ModelError) as error:
            with pytest.warns(exceptions.ModelWarning):
                m.build()

        assert check_error_or_warning(
            error, ["Operate mode: User must define a finite flow_cap"]
        )

    @pytest.mark.parametrize("constr", ("max", "equals"))
    def test_operate_flow_cap_source_unit(self, constr):
        """If we depend on a finite flow_cap, we have to error on a user failing to define it"""
        m = build_model(
            {
                "techs.test_supply_elec": {
                    "constraints": {
                        f"source_{constr}": "file=supply_plus_resource.csv:1",
                        "flow_cap_max": np.inf,
                    },
                    "switches": {"source_unit": "per_cap"},
                }
            },
            "simple_supply_and_supply_plus,operate,investment_costs",
        )

        if constr == "equals":
            with pytest.raises(exceptions.ModelError) as error:
                with pytest.warns(exceptions.ModelWarning):
                    m.build()
            assert check_error_or_warning(
                error, ["Operate mode: User must define a finite flow_cap"]
            )
        else:
            with pytest.warns(exceptions.ModelWarning):
                m.build()

    @pytest.mark.parametrize(
        ["source_unit", "constr"],
        list(product(("absolute", "per_cap", "per_area"), ("max", "equals"))),
    )
    def test_operate_source_unit_with_area_use(self, source_unit, constr):
        """Different source unit affects the capacities which are set to infinite"""
        m = build_model(
            {
                "techs.test_supply_elec": {
                    "constraints": {
                        "area_use_max": 10,
                        "flow_cap_max": 15,
                        f"source_{constr}": "file=supply_plus_resource.csv:1",
                    },
                    "switches": {"source_unit": source_unit},
                }
            },
            "simple_supply_and_supply_plus,operate,investment_costs",
        )
        with pytest.warns(exceptions.ModelWarning) as warning:
            m.build()

        if source_unit == "absolute":
            _warnings = [
                "Flow capacity constraint removed from 0::test_supply_elec as force_source is applied and source is not linked to inflow (source_unit = `absolute`)",
                "Source area constraint removed from 0::test_supply_elec as force_source is applied and source is not linked to inflow (source_unit = `absolute`)",
            ]
        elif source_unit == "per_area":
            _warnings = [
                "Flow capacity constraint removed from 0::test_supply_elec as force_source is applied and source is linked to flow using `per_area`"
            ]
        elif source_unit == "per_cap":
            _warnings = [
                "Source area constraint removed from 0::test_supply_elec as force_source is applied and source is linked to flow using `per_cap`"
            ]

        if constr == "equals":
            assert check_error_or_warning(warning, _warnings)
        else:
            assert ~check_error_or_warning(warning, _warnings)

    @pytest.mark.parametrize("source_unit", [("absolute"), ("per_cap"), ("per_area")])
    def test_operate_source_unit_without_area_use(self, source_unit):
        """Different source unit affects the capacities which are set to infinite"""
        m = build_model(
            {
                "techs.test_supply_elec": {
                    "constraints": {
                        "source_max": "file=supply_plus_resource.csv:1",
                        "flow_cap_max": 15,
                    },
                    "switches": {"force_source": True, "source_unit": source_unit},
                }
            },
            "simple_supply_and_supply_plus,operate,investment_costs",
        )

        with pytest.warns(exceptions.ModelWarning) as warning:
            # per_area without a source_cap will cause an error, which we have to catch here
            if source_unit == "per_area":
                with pytest.raises(exceptions.ModelError) as error:
                    m.build()
            else:
                m.build()

        if source_unit == "absolute":
            _warnings = [
                "Flow capacity constraint removed from 0::test_supply_elec as force_source is applied and source is not linked to flow (source_unit = `absolute`)"
            ]
            not_warnings = [
                "Area use constraint removed from 0::test_supply_elec as force_source is applied and source is not linked to flow (source_unit = `absolute`)",
                "Flow capacity constraint removed from 0::test_demand_elec as force_source is applied and source is not linked to flow (source_unit = `absolute`)",
                "Flow capacity constraint removed from 1::test_demand_elec as force_source is applied and source is not linked to flow (source_unit = `absolute`)",
            ]
        elif source_unit == "per_area":
            _warnings = [
                "Flow capacity constraint removed from 0::test_supply_elec as force_source is applied and source is linked to flow using `per_area`"
            ]
            not_warnings = [
                "Area use constraint removed from 0::test_supply_elec as force_source is applied and source is linked to flow using `per_cap`",
                "Flow capacity constraint removed from 0::test_demand_elec as force_source is applied and source is not linked to flow (source_unit = `absolute`)",
                "Flow capacity constraint removed from 1::test_demand_elec as force_source is applied and source is not linked to flow (source_unit = `absolute`)",
            ]
            # per_area without a source_cap will cause an error
            check_error_or_warning(
                error,
                "Operate mode: User must define a finite area_use "
                "(via area_use_equals or area_use_max) for 0::test_supply_elec",
            )
        elif source_unit == "per_cap":
            _warnings = []
            not_warnings = [
                "Area use constraint removed from 0::test_supply_elec as force_source is applied and source is linked to flow using `per_cap`",
                "Flow capacity constraint removed from 0::test_supply_elec as force_source is applied and source is not linked to flow (source_unit = `absolute`)",
                "Flow capacity constraint removed from 0::test_demand_elec as force_source is applied and source is not linked to flow (source_unit = `absolute`)",
                "Flow capacity constraint removed from 1::test_demand_elec as force_source is applied and source is not linked to flow (source_unit = `absolute`)",
            ]
        assert check_error_or_warning(warning, _warnings)
        assert not check_error_or_warning(warning, not_warnings)

    def test_operate_storage(self, param):
        """Can't violate storage capacity constraints in the definition of a technology"""
        param = "flow_cap_per_storage_cap_max"
        m = build_model(
            {f"techs.test_supply_plus.constraints.{param}": 0.1},
            "simple_supply_and_supply_plus,operate,investment_costs",
        )

        with pytest.warns(exceptions.ModelWarning) as warning:
            with pytest.raises(exceptions.ModelError) as error:
                m.build()

        assert check_error_or_warning(
            error,
            "fixed storage capacity * {} is not larger than fixed flow "
            "capacity for loc, tech {}".format(param, ("a", "test_supply_plus")),
        )
        assert check_error_or_warning(
            warning,
            [
                "Initial stored carrier not defined",
                "Source capacity constraint defined and set to infinity",
                "Storage cannot be cyclic in operate run mode",
            ],
        )

    @pytest.mark.parametrize("on", (True, False))
    def test_operate_source_cap_max(self, on):
        """Some constraints, if not defined, will throw a warning and possibly change values in model_data"""

        if on is False:
            override = {}
        else:
            override = {"techs.test_supply_plus.constraints.source_cap_max": 1e6}
        m = build_model(
            override, "simple_supply_and_supply_plus,operate,investment_costs"
        )

        with pytest.warns(exceptions.ModelWarning) as warning:
            m.build()
        if on is False:
            assert check_error_or_warning(
                warning, "Source capacity constraint defined and set to infinity"
            )
            assert np.isinf(
                m._model_data.source_cap.loc["a", "test_supply_plus"].item()
            )
        elif on is True:
            assert not check_error_or_warning(
                warning, "Source capacity constraint defined and set to infinity"
            )
            assert m._model_data.source_cap.loc["a", "test_supply_plus"].item() == 1e6

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
            assert check_error_or_warning(warning, "Initial stored carrier not defined")
            assert (
                m._model_data.storage_initial.loc["a", "test_supply_plus"].item() == 0
            )
        elif on is True:
            assert not check_error_or_warning(
                warning, "Initial stored carrier not defined"
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
                "techs.test_supply_elec.switches.source_unit": "per_cap",
            },
            "simple_supply,two_hours,investment_costs",
        )
        m.build()
        assert check_variable_exists(
            m.backend.get_constraint("balance_supply", as_backend_objs=False),
            "flow_cap",
        )

        m = build_model(
            {
                "techs.test_supply_elec.constraints.resource": 20,
                "techs.test_supply_elec.switches.source_unit": "per_area",
            },
            "simple_supply,two_hours,investment_costs",
        )
        m.build()
        assert check_variable_exists(
            m.backend.get_constraint("balance_supply", as_backend_objs=False),
            "area_use",
        )

    def test_loc_techs_balance_demand_constraint(self, simple_supply):
        """
        sets.loc_techs_finite_resource_demand,
        """
        assert "balance_demand" in simple_supply.backend.constraints

        m = build_model(
            {"techs.test_demand_elec.switches.source_unit": "per_cap"},
            "simple_supply,two_hours,investment_costs",
        )
        m.build()
        assert check_variable_exists(
            m.backend.get_constraint("balance_demand", as_backend_objs=False),
            "flow_cap",
        )

        m = build_model(
            {"techs.test_demand_elec.switches.source_unit": "per_area"},
            "simple_supply,two_hours,investment_costs",
        )
        m.build()
        assert check_variable_exists(
            m.backend.get_constraint("balance_demand", as_backend_objs=False),
            "area_use",
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
            {"techs.test_supply_plus.switches.source_unit": "per_cap"},
            "simple_supply_and_supply_plus,two_hours,investment_costs",
        )
        m.build()
        assert check_variable_exists(
            m.backend.get_constraint(
                "resource_availability_supply_plus", as_backend_objs=False
            ),
            "flow_cap",
        )

        m = build_model(
            {"techs.test_supply_plus.switches.source_unit": "per_area"},
            "simple_supply_and_supply_plus,two_hours,investment_costs",
        )
        m.build()
        assert check_variable_exists(
            m.backend.get_constraint(
                "resource_availability_supply_plus", as_backend_objs=False
            ),
            "area_use",
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
            ("test_supply_elec", "simple_supply", "flow_out"),
            ("test_supply_elec", "simple_supply", "flow_in"),
            ("test_supply_plus", "simple_supply_and_supply_plus", "flow_in"),
            ("test_demand_elec", "simple_supply", "flow_in"),
            ("test_transmission_elec", "simple_supply", "flow_out"),
            ("test_conversion", "simple_conversion", "flow_in"),
            ("test_conversion_plus", "simple_conversion_plus", "flow_out"),
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
        With one_way transmission, it should still be possible to set an flow_out cost.
        """
        m = build_model(
            {
                "techs.test_transmission_elec.costs.monetary.flow_out": 1,
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
            m.backend.get_expression("cost_var", as_backend_objs=False), "flow_out", idx
        )

        idx["nodes"] = "a"
        idx["techs"] = "test_transmission_elec:b"
        assert not check_variable_exists(
            m.backend.get_expression("cost_var", as_backend_objs=False), "flow_out", idx
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
            "flow_export",
        )

    def test_loc_carriers_system_balance_export(self, supply_export):
        assert check_variable_exists(
            supply_export.backend.get_constraint(
                "system_balance", as_backend_objs=False
            ),
            "flow_export",
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
            {"techs.test_supply_elec.costs.monetary.flow_out": 0.1},
            "supply_export,two_hours,investment_costs",
        )
        m.build()
        assert "cost_var" in m.backend.expressions

        assert check_variable_exists(
            m.backend.get_expression("cost_var", as_backend_objs=False), "flow_export"
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
        assert "flow_export_max" in m.backend.constraints


@pytest.mark.skip(reason="to be reimplemented by comparison to LP files")
class TestCapacityConstraints:
    # capacity.py
    @pytest.mark.xfail(reason="storage_cap_equals is no more")
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

    def test_loc_techs_storage_capacity_milp_constraint(self):
        m = build_model(
            {
                "techs.test_storage.constraints": {
                    "units_max": 1,
                    "flow_cap_per_unit": 20,
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
            for j in ["max", "min"]
        ],
    )
    def test_loc_techs_flow_capacity_storage_constraint(self, scenario, tech, override):
        """
        i for i in sets.loc_techs_store if constraint_exists(model_run, i, 'constraints.flow_cap_per_storage_cap_max')
        """
        m = build_model(
            {f"techs.{tech}.constraints.flow_cap_per_storage_cap_{override}": 0.5},
            f"{scenario},two_hours,investment_costs",
        )
        m.build()
        assert hasattr(
            m._backend_model,
            "flow_capacity_per_storage_capacity_{}_constraint".format(override),
        )

    @pytest.mark.filterwarnings("ignore:(?s).*Integer:calliope.exceptions.ModelWarning")
    @pytest.mark.parametrize("override", (("max", "min")))
    def test_loc_techs_flow_capacity_milp_storage_constraint(self, override):
        """
        i for i in sets.loc_techs_store if constraint_exists(model_run, i, 'constraints.flow_cap_per_storage_cap_max')
        """

        m = build_model(
            {
                f"techs.test_supply_plus.constraints.flow_cap_per_storage_cap_{override}": 0.5
            },
            "supply_and_supply_plus_milp,two_hours,investment_costs",
        )
        m.build()
        assert hasattr(
            m._backend_model,
            f"flow_capacity_per_storage_capacity_{override}_constraint",
        )

    def test_no_loc_techs_flow_capacity_storage_constraint(self, caplog):
        """
        i for i in sets.loc_techs_store if constraint_exists(model_run, i, 'constraints.flow_cap_per_storage_cap_max')
        """
        with caplog.at_level(logging.INFO):
            m = build_model(model_file="flow_cap_per_storage_cap.yaml")

        m.build()
        assert not any(
            [
                hasattr(
                    m._backend_model, "flow_capacity_storage_{}_constraint".format(i)
                )
                for i in ["max", "min"]
            ]
        )

    @pytest.mark.parametrize("override", ((None, "max", "min")))
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
            if override == "min":
                assert expr.lb == 10
                assert np.isinf(expr.ub)

    def test_loc_techs_resource_capacity_equals_flow_capacity_constraint(
        self, simple_supply_and_supply_plus
    ):
        """
        i for i in sets.loc_techs_finite_resource_supply_plus
        if constraint_exists(model_run, i, 'constraints.resource_cap_equals_flow_cap')
        """
        assert (
            "resource_capacity_equals_flow_capacity"
            not in simple_supply_and_supply_plus.backend.constraints
        )

        m = build_model(
            {"techs.test_supply_plus.switches.resource_cap_equals_flow_cap": True},
            "simple_supply_and_supply_plus,two_hours,investment_costs",
        )
        m.build()
        assert "resource_capacity_equals_flow_capacity" in m.backend.constraints

    def test_loc_techs_area_use_constraint(self, simple_supply_and_supply_plus):
        """
        i for i in sets.loc_techs_area if i in sets.loc_techs_supply_plus
        """
        assert "area_use" not in simple_supply_and_supply_plus.backend.variables

        m = build_model(
            {"techs.test_supply_plus.constraints.area_use_max": 10},
            "simple_supply_and_supply_plus,two_hours,investment_costs",
        )
        m.build()
        assert "area_use" in m.backend.variables

        m = build_model(
            {"techs.test_supply_elec.constraints.area_use_max": 10},
            "simple_supply_and_supply_plus,two_hours,investment_costs",
        )
        m.build()
        assert "area_use" in m.backend.variables

        # Check that setting flow_cap_max to 0 also forces this constraint to 0
        m = build_model(
            {
                "techs.test_supply_plus.constraints": {
                    "area_use_max": 10,
                    "flow_cap_max": 0,
                }
            },
            "simple_supply_and_supply_plus,two_hours,investment_costs",
        )
        m.build()
        ub = m.backend.get_constraint("force_zero_area_use", as_backend_objs=False).ub

        assert (ub.to_series().dropna() == 0).all()

    def test_loc_techs_area_use_per_flow_capacity_constraint(
        self, simple_supply_and_supply_plus
    ):
        """
        i for i in sets.loc_techs_area if i in sets.loc_techs_supply_plus
        and constraint_exists(model_run, i, 'constraints.area_use_per_flow_cap')
        """
        assert (
            "area_use_per_flow_capacity"
            not in simple_supply_and_supply_plus.backend.constraints
        )

        m = build_model(
            {"techs.test_supply_plus.constraints.area_use_max": 10},
            "simple_supply_and_supply_plus,two_hours,investment_costs",
        )
        m.build()
        assert "area_use_per_flow_capacity" not in m.backend.constraints

        m = build_model(
            {"techs.test_supply_elec.constraints.area_use_per_flow_cap": 10},
            "simple_supply_and_supply_plus,two_hours,investment_costs",
        )
        m.build()
        assert "area_use_per_flow_capacity" in m.backend.constraints

        m = build_model(
            {
                "techs.test_supply_elec.constraints": {
                    "area_use_per_flow_cap": 10,
                    "area_use_max": 10,
                }
            },
            "simple_supply_and_supply_plus,two_hours,investment_costs",
        )
        m.build()
        assert "area_use_per_flow_capacity" in m.backend.constraints

    def test_locs_area_use_capacity_per_loc_constraint(
        self, simple_supply_and_supply_plus
    ):
        """
        i for i in sets.locs
        if model_run.nodes[i].get_key('available_area', None) is not None
        """
        assert (
            "area_use_capacity_per_loc"
            not in simple_supply_and_supply_plus.backend.constraints
        )

        m = build_model(
            {"nodes.a.available_area": 1},
            "simple_supply_and_supply_plus,two_hours,investment_costs",
        )
        m.build()
        assert "area_use_capacity_per_loc" not in m.backend.constraints

        m = build_model(
            {
                "nodes.a.available_area": 1,
                "techs.test_supply_plus.constraints.area_use_max": 10,
            },
            "simple_supply_and_supply_plus,two_hours,investment_costs",
        )
        m.build()
        assert "area_use_capacity_per_loc" in m.backend.constraints

    @pytest.mark.xfail(reason="flow_cap_scale is no more")
    def test_loc_techs_flow_capacity_constraint(self, simple_supply_and_supply_plus):
        """
        i for i in sets.loc_techs
        if i not in sets.loc_techs_milp + sets.loc_techs_purchase
        """
        m2 = build_model(
            {"techs.test_supply_elec.constraints.flow_cap_scale": 5},
            "simple_supply_and_supply_plus,two_hours,investment_costs",
        )
        m2.build()
        assert (
            m2.backend.variables.flow_cap.sel(nodes="a", techs="test_supply_elec")
            .item()
            .ub
            == simple_supply_and_supply_plus.backend.variables.flow_cap.sel(
                nodes="a", techs="test_supply_elec"
            )
            .item()
            .ub
            * 5
        )

    def test_loc_techs_flow_capacity_milp_constraint(self):
        m = build_model(
            {}, "supply_milp,two_hours,investment_costs"
        )  # demand still is in loc_techs
        m.build()
        assert np.isinf(
            m.backend.variables.flow_cap.sel(nodes="a", techs="test_demand_elec")
            .item()
            .ub
        )
        assert (
            m.backend.variables.flow_cap.sel(nodes="a", techs="test_supply_elec")
            .item()
            .ub
            == 10
        )

    @pytest.mark.xfail(reason="flow_cap_equals is no more")
    def test_loc_techs_flow_capacity_constraint_warning_on_infinite_equals(self):
        # Check that setting `_equals` to infinity is caught:
        override = {
            "nodes.a.techs.test_supply_elec.constraints.flow_cap_equals": np.inf
        }
        with pytest.raises(exceptions.ModelError) as error:
            m = build_model(override, "simple_supply,two_hours,investment_costs")
            m.build()

        assert check_error_or_warning(
            error,
            "Cannot use inf for flow_cap_equals for node, tech `('a', 'test_supply_elec')`",
        )

    @pytest.mark.parametrize("bound", (("max")))
    def test_techs_flow_capacity_systemwide_constraint(self, bound):
        """
        i for i in sets.techs
        if model_run.get_key('techs.{}.constraints.flow_cap_max_systemwide'.format(i), None)
        """

        def check_bounds(constraint):
            assert constraint.ub.item() == 20
            if bound == "max":
                assert constraint.lb.item() is None

        m = build_model(
            {
                f"techs.test_supply_elec.constraints.flow_cap_{bound}_systemwide": 20
            },  # foo
            "simple_supply,two_hours,investment_costs",
        )
        m.build()
        assert "flow_capacity_systemwide" in m.backend.constraints
        assert isinstance(
            m.backend.constraints.flow_capacity_systemwide.sel(
                techs="test_supply_elec"
            ).item(),
            pmo.constraint,
        )

        check_bounds(
            m.backend.get_constraint(
                "flow_capacity_systemwide", as_backend_objs=False
            ).sel(techs="test_supply_elec")
        )

        # Check that a model without transmission techs doesn't cause an error
        m = build_model(
            {f"techs.test_supply_elec.constraints.flow_cap_{bound}_systemwide": 20},
            "simple_supply,two_hours,investment_costs",
            model_file="model_minimal.yaml",
        )
        m.build()
        assert "flow_capacity_systemwide" in m.backend.constraints
        check_bounds(
            m.backend.get_constraint(
                "flow_capacity_systemwide", as_backend_objs=False
            ).sel(techs="test_supply_elec")
        )

    @pytest.mark.parametrize("bound", (("equals", "max")))
    def test_techs_flow_capacity_systemwide_no_constraint(self, simple_supply, bound):
        assert "flow_capacity_systemwide" not in simple_supply.backend.constraints

        # setting the constraint to infinity leads to no constraint being built
        m = build_model(
            {f"techs.test_supply_elec.constraints.flow_cap_{bound}_systemwide": np.inf},
            "simple_supply,two_hours,investment_costs",
        )
        m.build()
        assert "flow_capacity_systemwide" not in m.backend.constraints


@pytest.mark.skip(reason="to be reimplemented by comparison to LP files")
class TestDispatchConstraints:
    # dispatch.py
    def test_loc_tech_carriers_flow_out_max_constraint(self, simple_supply):
        """
        i for i in sets.loc_tech_carriers_prod
        if i not in sets.loc_tech_carriers_conversion_plus
        and i.rsplit('::', 1)[0] not in sets.loc_techs_milp
        """
        assert "flow_out_max" in simple_supply.backend.constraints

    def test_loc_tech_carriers_flow_out_max_milp_constraint(self, supply_milp):
        assert "flow_out_max" not in supply_milp.backend.constraints

    def test_loc_tech_carriers_flow_out_min_constraint(self, simple_supply):
        """
        i for i in sets.loc_tech_carriers_prod
        if i not in sets.loc_tech_carriers_conversion_plus
        and constraint_exists(model_run, i, 'constraints.flow_out_min_relative')
        and i.rsplit('::', 1)[0] not in sets.loc_techs_milp
        """
        assert "flow_out_min" not in simple_supply.backend.constraints

        m = build_model(
            {"techs.test_supply_elec.constraints.flow_out_min_relative": 0.1},
            "simple_supply,two_hours,investment_costs",
        )
        m.build()
        assert "flow_out_min" in m.backend.constraints

    def test_loc_tech_carriers_flow_out_min_milp_constraint(self, supply_milp):
        assert "flow_out_min" not in supply_milp.backend.constraints

        m = build_model(
            {"techs.test_supply_elec.constraints.flow_out_min_relative": 0.1},
            "supply_milp,two_hours,investment_costs",
        )
        m.build()
        assert "flow_out_min" not in m.backend.constraints

    def test_loc_tech_carriers_flow_in_max_constraint(self, simple_supply):
        """
        i for i in sets.loc_tech_carriers_con
        if i.rsplit('::', 1)[0] in sets.loc_techs_demand +
            sets.loc_techs_storage + sets.loc_techs_transmission
        and i.rsplit('::', 1)[0] not in sets.loc_techs_milp
        """
        assert "flow_in_max" in simple_supply.backend.constraints

    def test_loc_tech_carriers_flow_in_max_milp_constraint(self, supply_milp):
        assert "flow_in_max" in supply_milp.backend.constraints

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
            {"techs.test_supply_elec.constraints.flow_ramping": 0.1},
            "simple_supply,two_hours,investment_costs",
        )
        m.build()
        assert "ramping_up" in m.backend.constraints
        assert "ramping_down" in m.backend.constraints

        m = build_model(
            {"techs.test_conversion.constraints.flow_ramping": 0.1},
            "simple_conversion,two_hours,investment_costs",
        )
        m.build()
        assert "ramping_up" in m.backend.constraints
        assert "ramping_down" in m.backend.constraints


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

    def test_loc_tech_carriers_flow_out_max_milp_constraint(
        self, simple_supply, supply_milp, supply_purchase, conversion_plus_milp
    ):
        """
        i for i in sets.loc_tech_carriers_prod
        if i not in sets.loc_tech_carriers_conversion_plus
        and i.rsplit('::', 1)[0] in sets.loc_techs_milp
        """
        assert "flow_out_max_milp" not in simple_supply.backend.constraints
        assert "flow_out_max_milp" in supply_milp.backend.constraints

        assert "flow_out_max_milp" not in supply_purchase.backend.constraints
        assert "flow_out_max_milp" not in conversion_plus_milp.backend.constraints

    def test_loc_techs_flow_out_max_conversion_plus_milp_constraint(
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
            "flow_out_max_conversion_plus_milp" not in simple_supply.backend.constraints
        )
        assert (
            "flow_out_max_conversion_plus_milp" not in supply_milp.backend.constraints
        )
        assert (
            "flow_out_max_conversion_plus_milp"
            not in supply_purchase.backend.constraints
        )
        assert (
            "flow_out_max_conversion_plus_milp"
            in conversion_plus_milp.backend.constraints
        )
        assert (
            "flow_out_max_conversion_plus_milp"
            not in conversion_plus_purchase.backend.constraints
        )

    def test_loc_tech_carriers_flow_out_min_milp_constraint(self):
        """
        i for i in sets.loc_tech_carriers_prod
        if i not in sets.loc_tech_carriers_conversion_plus
        and constraint_exists(model_run, i.rsplit('::', 1)[0], 'constraints.flow_out_min_relative')
        and i.rsplit('::', 1)[0] in sets.loc_techs_milp
        """
        m = build_model(
            {"techs.test_supply_elec.constraints.flow_out_min_relative": 0.1},
            "simple_supply,two_hours,investment_costs",
        )
        m.build()
        assert "flow_out_min_milp" not in m.backend.constraints

        m = build_model(
            {"techs.test_supply_elec.constraints.flow_out_min_relative": 0.1},
            "supply_milp,two_hours,investment_costs",
        )
        m.build()
        assert "flow_out_min_milp" in m.backend.constraints

        m = build_model(
            {"techs.test_supply_elec.constraints.flow_out_min_relative": 0.1},
            "supply_purchase,two_hours,investment_costs",
        )
        m.build()
        assert "flow_out_min_milp" not in m.backend.constraints

        m = build_model(
            {"techs.test_supply_elec.constraints.flow_out_min_relative": 0.1},
            "conversion_plus_milp,two_hours,investment_costs",
        )
        m.build()
        assert "flow_out_min_milp" not in m.backend.constraints

        m = build_model(
            {"techs.test_conversion_plus.constraints.flow_out_min_relative": 0.1},
            "conversion_plus_milp,two_hours,investment_costs",
        )
        m.build()
        assert "flow_out_min_milp" not in m.backend.constraints

    def test_loc_techs_flow_out_min_conversion_plus_milp_constraint(self):
        """
        i for i in sets.loc_techs_conversion_plus
        if constraint_exists(model_run, i, 'constraints.flow_out_min_relative')
        and i in sets.loc_techs_milp
        """
        m = build_model(
            {"techs.test_supply_elec.constraints.flow_out_min_relative": 0.1},
            "simple_supply,two_hours,investment_costs",
        )
        m.build()
        assert "flow_out_min_conversion_plus_milp" not in m.backend.constraints

        m = build_model(
            {"techs.test_supply_elec.constraints.flow_out_min_relative": 0.1},
            "supply_milp,two_hours,investment_costs",
        )
        m.build()
        assert "flow_out_min_conversion_plus_milp" not in m.backend.constraints

        m = build_model(
            {"techs.test_supply_elec.constraints.flow_out_min_relative": 0.1},
            "conversion_plus_milp,two_hours,investment_costs",
        )
        m.build()
        assert "flow_out_min_conversion_plus_milp" not in m.backend.constraints

        m = build_model(
            {"techs.test_conversion_plus.constraints.flow_out_min_relative": 0.1},
            "conversion_plus_milp,two_hours,investment_costs",
        )
        m.build()
        assert "flow_out_min_conversion_plus_milp" in m.backend.constraints

        m = build_model(
            {"techs.test_conversion_plus.constraints.flow_out_min_relative": 0.1},
            "conversion_plus_purchase,two_hours,investment_costs",
        )
        m.build()
        assert "flow_out_min_conversion_plus_milp" not in m.backend.constraints

    def test_loc_tech_carriers_flow_in_max_milp_constraint(
        self, simple_supply, supply_milp, storage_milp, conversion_plus_milp
    ):
        """
        i for i in sets.loc_tech_carriers_con
        if i.rsplit('::', 1)[0] in sets.loc_techs_demand +
            sets.loc_techs_storage + sets.loc_techs_transmission
        and i.rsplit('::', 1)[0] in sets.loc_techs_milp
        """
        assert "flow_in_max_milp" not in simple_supply.backend.constraints
        assert "flow_in_max_milp" not in supply_milp.backend.constraints
        assert "flow_in_max_milp" in storage_milp.backend.constraints
        assert "flow_in_max_milp" not in conversion_plus_milp.backend.constraints

    def test_loc_techs_flow_capacity_units_milp_constraint(
        self, simple_supply, supply_milp, storage_milp, conversion_plus_milp
    ):
        """
        i for i in sets.loc_techs_milp
        if constraint_exists(model_run, i, 'constraints.flow_cap_per_unit')
        is not None
        """
        assert "flow_capacity_units_milp" not in simple_supply.backend.constraints
        assert "flow_capacity_units_milp" in supply_milp.backend.constraints
        assert "flow_capacity_units_milp" in storage_milp.backend.constraints
        assert "flow_capacity_units_milp" in conversion_plus_milp.backend.constraints

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

    @pytest.mark.xfail(reason="flow_cap_equals is no more")
    def test_loc_techs_flow_capacity_max_purchase_milp_constraint(
        self, simple_supply, supply_milp, supply_purchase
    ):
        """
        i for i in sets.loc_techs_purchase
        if (constraint_exists(model_run, i, 'constraints.flow_cap_equals') is not None
            or constraint_exists(model_run, i, 'constraints.flow_cap_max') is not None)
        """
        assert (
            "flow_capacity_max_purchase_milp" not in simple_supply.backend.constraints
        )
        assert "flow_capacity_max_purchase_milp" not in supply_milp.backend.constraints
        assert "flow_capacity_max_purchase_milp" in supply_purchase.backend.constraints

        m = build_model(
            {
                "techs.test_supply_elec.constraints": {
                    "flow_cap_max": None,
                    "flow_cap_equals": 15,
                }
            },
            "supply_purchase,two_hours,investment_costs",
        )
        m.build()
        assert "flow_capacity_max_purchase_milp" in m.backend.constraints

    def test_loc_techs_flow_capacity_min_purchase_milp_constraint(
        self, simple_supply, supply_milp, supply_purchase
    ):
        """
        i for i in sets.loc_techs_purchase
        if (not constraint_exists(model_run, i, 'constraints.flow_cap_equals')
            and constraint_exists(model_run, i, 'constraints.flow_cap_min'))
        """
        assert (
            "flow_capacity_min_purchase_milp" not in simple_supply.backend.constraints
        )
        assert "flow_capacity_min_purchase_milp" not in supply_milp.backend.constraints
        assert (
            "flow_capacity_min_purchase_milp" not in supply_purchase.backend.constraints
        )

        m = build_model(
            {
                "techs.test_supply_elec.constraints": {
                    "flow_cap_max": None,
                    "flow_cap_equals": 15,
                }
            },
            "supply_purchase,two_hours,investment_costs",
        )
        m.build()
        assert "flow_capacity_min_purchase_milp" not in m.backend.constraints

        m = build_model(
            {"techs.test_supply_elec.constraints.flow_cap_min": 10},
            "supply_purchase,two_hours,investment_costs",
        )
        m.build()
        assert "flow_capacity_min_purchase_milp" in m.backend.constraints

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
                or constraint_exists(model_run, i, 'constraints.flow_cap_min')))
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
                "flow_cap_per_unit": 5,
            },
        }
        m = build_model(override_max, "conversion_plus_milp,two_hours,investment_costs")
        m.build()
        assert "unit_capacity_max_systemwide_milp" in m.backend.constraints
        assert (
            m.backend.get_constraint(
                "unit_capacity_max_systemwide_milp", as_backend_objs=False
            )
            .sel(techs="test_conversion_plus")
            .ub.item()
            == 2
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
            "techs.test_supply_elec.constraints.units_max_systemwide": 1,
            "nodes.b.techs.test_supply_elec.costs.monetary.purchase": 1,
        }
        m = build_model(
            override_no_transmission,
            "supply_milp,two_hours,investment_costs",
            model_file="model_minimal.yaml",
        )
        m.build()
        assert "unit_capacity_max_systemwide_milp" in m.backend.constraints

    @pytest.mark.parametrize("tech", (("test_storage"), ("test_transmission_elec")))
    def test_asynchronous_flow_constraint(self, tech):
        """
        Binary switch for flow in/out can be activated using the option
        'asynchronous_flow'
        """
        m = build_model(
            {f"techs.{tech}.constraints.force_async_flow": True},
            "simple_storage,investment_costs",
        )
        m.build()
        assert "async_flow_switch" in m.backend.variables
        assert "async_flow_in_milp" in m.backend.constraints
        assert "async_flow_out_milp" in m.backend.constraints


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
            "config.init.time_subset": ["2005-01-01", "2005-01-04"],
            "config.init.time_cluster": "cluster_days.csv",
            "config.init.custom_math": ["storage_inter_cluster"]
            if storage_inter_cluster
            else [],
            "config.build.cyclic_storage": cyclic,
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
            override_dict={"config.solve": {"solver": "gurobi", "solver_io": "python"}},
        )
        model.build()
        return model

    def test_no_duplicate_log_message(self, caplog, gurobi_model):
        caplog.set_level(logging.DEBUG)
        gurobi_model.solve()
        all_log_messages = [r.msg for r in caplog.records]
        assert sum([i.find("Gurobi Optimizer") > -1 for i in all_log_messages]) == 1


class TestModelDataChecks:
    def test_source_equals_cannot_be_inf(self):
        override = {"techs.test_supply_elec.source_equals": np.inf}
        m = build_model(override_dict=override, scenario="simple_supply,one_day")

        with pytest.raises(exceptions.ModelError) as excinfo:
            m.build()
        assert check_error_or_warning(excinfo, "Cannot include infinite values")

    def test_storage_initial_fractional_value(self):
        """
        Check that the storage_initial value is a fraction
        """
        m = build_model(
            {"techs.test_storage.storage_initial": 5},
            "simple_storage,two_hours,investment_costs",
        )

        with pytest.raises(exceptions.ModelError) as error:
            m.build()
        assert check_error_or_warning(error, "values larger than 1 are not allowed")


class TestNewBackend:
    LOGGER = logging.getLogger("calliope.backend.backend_model")

    @pytest.fixture(scope="class")
    def simple_supply_longnames(self):
        m = build_model({}, "simple_supply,two_hours,investment_costs")
        m.build()
        m.backend.verbose_strings()
        return m

    @pytest.fixture
    def temp_path(self, tmpdir_factory):
        return tmpdir_factory.mktemp("custom_math")

    def test_new_build_has_backend(self, simple_supply):
        assert hasattr(simple_supply, "backend")

    def test_new_build_optimal(self, simple_supply):
        assert hasattr(simple_supply, "results")
        assert simple_supply._model_data.attrs["termination_condition"] == "optimal"

    @pytest.mark.parametrize("mode", ["operate", "spores"])
    def test_add_run_mode_custom_math(self, caplog, mode):
        caplog.set_level(logging.DEBUG)
        mode_custom_math = AttrDict.from_yaml(
            importlib.resources.files("calliope") / "math" / f"{mode}.yaml"
        )
        m = build_model({}, "simple_supply,two_hours,investment_costs")

        base_math = deepcopy(m.math)
        base_math.union(mode_custom_math, allow_override=True)

        backend = PyomoBackendModel(m.inputs, mode=mode)
        backend._add_run_mode_custom_math()

        assert f"Updating math formulation with {mode} mode custom math." in caplog.text

        assert m.math != base_math
        assert backend.inputs.attrs["math"].as_dict() == base_math.as_dict()

    def test_add_run_mode_custom_math_before_build(self, caplog, temp_path):
        """A user can override the run mode custom math by including it directly in the custom math string"""
        caplog.set_level(logging.DEBUG)
        custom_math = AttrDict({"variables": {"flow_cap": {"active": True}}})
        file_path = temp_path.join("custom-math.yaml")
        custom_math.to_yaml(file_path)

        m = build_model(
            {"config.init.custom_math": ["operate", str(file_path)]},
            "simple_supply,two_hours,investment_costs",
        )
        m.build(mode="operate")

        # We set operate mode explicitly in our custom math so it won't be added again
        assert (
            "Updating math formulation with operate mode custom math."
            not in caplog.text
        )

        # operate mode set it to false, then our custom math set it back to active
        assert m.math.variables.flow_cap.active
        # operate mode set it to false and our custom math did not override that
        assert not m.math.variables.storage_cap.active

    def test_run_mode_mismatch(self):
        m = build_model(
            {"config.init.custom_math": ["operate"]},
            "simple_supply,two_hours,investment_costs",
        )
        backend = PyomoBackendModel(m.inputs)
        with pytest.warns(exceptions.ModelWarning) as excinfo:
            backend._add_run_mode_custom_math()

        assert check_error_or_warning(
            excinfo, "Running in plan mode, but run mode(s) {'operate'}"
        )

    @pytest.mark.parametrize(
        "component_type", ["variable", "global_expression", "parameter", "constraint"]
    )
    def test_new_build_get_missing_component(self, simple_supply, component_type):
        with pytest.raises(KeyError):
            getattr(simple_supply.backend, f"get_{component_type}")("foo")

    def test_new_build_get_variable(self, simple_supply):
        var = simple_supply.backend.get_variable("flow_cap")
        assert (
            var.to_series().dropna().apply(lambda x: isinstance(x, pmo.variable)).all()
        )
        assert var.attrs == {
            "obj_type": "variables",
            "references": {
                "flow_in_max",
                "flow_out_max",
                "cost_investment",
                "cost_investment_flow_cap",
                "symmetric_transmission",
            },
            "description": "A technology's flow capacity, also known as its nominal or nameplate capacity.",
            "unit": "power",
            "coords_in_name": False,
        }

    def test_new_build_get_variable_as_vals(self, simple_supply):
        var = simple_supply.backend.get_variable("flow_cap", as_backend_objs=False)
        assert (
            not var.to_series()
            .dropna()
            .apply(lambda x: isinstance(x, pmo.variable))
            .any()
        )

    def test_new_build_get_parameter(self, simple_supply):
        param = simple_supply.backend.get_parameter("flow_in_eff")
        assert isinstance(param.item(), pmo.parameter)
        assert param.attrs == {
            "obj_type": "parameters",
            "is_result": 0,
            "original_dtype": np.dtype("float64"),
            "references": {"flow_in_inc_eff"},
            "coords_in_name": False,
        }

    def test_new_build_get_parameter_as_vals(self, simple_supply):
        param = simple_supply.backend.get_parameter(
            "flow_in_eff", as_backend_objs=False
        )
        assert param.dtype == np.dtype("float64")

    def test_new_build_get_global_expression(self, simple_supply):
        expr = simple_supply.backend.get_global_expression("cost_investment")
        assert (
            expr.to_series()
            .dropna()
            .apply(lambda x: isinstance(x, pmo.expression))
            .all()
        )
        assert expr.attrs == {
            "obj_type": "global_expressions",
            "references": {"cost"},
            "description": "The installation costs of a technology, including annualised investment costs and annual maintenance costs.",
            "unit": "cost",
            "coords_in_name": False,
        }

    def test_new_build_get_global_expression_as_str(self, simple_supply):
        expr = simple_supply.backend.get_global_expression(
            "cost", as_backend_objs=False
        )
        assert expr.to_series().dropna().apply(lambda x: isinstance(x, str)).all()

    def test_new_build_get_global_expression_as_vals(self, simple_supply):
        expr = simple_supply.backend.get_global_expression(
            "cost", as_backend_objs=False, eval_body=True
        )
        assert (
            expr.to_series().dropna().apply(lambda x: isinstance(x, (float, int))).all()
        )

    def test_new_build_get_constraint(self, simple_supply):
        constr = simple_supply.backend.get_constraint("system_balance")
        assert (
            constr.to_series()
            .dropna()
            .apply(lambda x: isinstance(x, pmo.constraint))
            .all()
        )
        assert constr.attrs == {
            "obj_type": "constraints",
            "references": set(),
            "description": "Set the global carrier balance of the optimisation problem by fixing the total production of a given carrier to equal the total consumption of that carrier at every node in every timestep.",
            "coords_in_name": False,
        }

    def test_new_build_get_constraint_as_str(self, simple_supply):
        constr = simple_supply.backend.get_constraint(
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

    def test_new_build_get_constraint_as_vals(self, simple_supply):
        constr = simple_supply.backend.get_constraint(
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
    def test_new_build_get_constraint_bounds(self, simple_supply, bound):
        constr = simple_supply.backend.get_constraint(
            "system_balance", as_backend_objs=False
        )
        assert (constr[bound].to_series().dropna() == 0).all()

    def test_solve_before_build(self):
        m = build_model({}, "simple_supply,two_hours,investment_costs")
        with pytest.raises(exceptions.ModelError) as excinfo:
            m.solve()
        assert check_error_or_warning(excinfo, "You must build the optimisation")

    def test_solve_after_solve(self, simple_supply):
        with pytest.raises(exceptions.ModelError) as excinfo:
            simple_supply.solve()
        assert check_error_or_warning(excinfo, "This model object already has results.")

    def test_solve_operate_not_allowed(self, simple_supply):
        simple_supply.backend.inputs.attrs["config"]["build"]["mode"] = "operate"
        simple_supply._model_data.attrs["allow_operate_mode"] = False

        try:
            with pytest.raises(exceptions.ModelError) as excinfo:
                simple_supply.solve(force=True)
            assert check_error_or_warning(excinfo, "Unable to run this model in op")
        except AssertionError as e:
            simple_supply.backend.inputs.attrs["config"]["build"]["mode"] = "plan"
            simple_supply._model_data.attrs["allow_operate_mode"] = True
            raise e
        else:
            simple_supply.backend.inputs.attrs["config"]["build"]["mode"] = "plan"
            simple_supply._model_data.attrs["allow_operate_mode"] = True

    def test_solve_warmstart_not_possible(self, simple_supply):
        with pytest.warns(exceptions.ModelWarning) as excinfo:
            simple_supply.solve(force=True, warmstart=True)
        assert check_error_or_warning(excinfo, "cbc, does not support warmstart")

    def test_solve_non_optimal(self, simple_supply):
        simple_supply.backend.update_parameter(
            "sink_equals",
            simple_supply.inputs.sink_equals.where(
                simple_supply.inputs.techs == "test_demand_elec"
            )
            * 100,
        )
        with pytest.warns(exceptions.BackendWarning) as excinfo:
            simple_supply.solve(force=True)

        assert check_error_or_warning(excinfo, "Model solution was non-optimal")
        assert simple_supply._model_data.attrs["termination_condition"] == "infeasible"
        assert not simple_supply.results.data_vars

    def test_raise_error_on_preexistence_same_type(self, simple_supply):
        with pytest.raises(exceptions.BackendError) as excinfo:
            simple_supply.backend.add_parameter("flow_out_eff", xr.DataArray(1))

        assert check_error_or_warning(
            excinfo,
            "Trying to add already existing `flow_out_eff` to backend model parameters.",
        )

    def test_raise_error_on_preexistence_diff_type(self, simple_supply):
        with pytest.raises(exceptions.BackendError) as excinfo:
            simple_supply.backend.add_parameter("flow_out", xr.DataArray(1))

        assert check_error_or_warning(
            excinfo,
            "Trying to add already existing *variable* `flow_out` as a backend model *parameter*.",
        )

    def test_raise_error_on_constraint_with_nan(self, simple_supply):
        """
        A very simple constraint: For each tech, let the annual and regional sum of `flow_out` be larger than 100.
        However, not every tech has the variable `flow_out`.
        How to solve it? Let the constraint be active only where flow_out exists by setting 'where' accordingly.
        """
        # add constraint without nan
        constraint_dict = {
            "foreach": ["techs", "carriers"],
            "equations": [
                {"expression": "sum(flow_out, over=[nodes, timesteps]) >= 100"}
            ],
            "where": "carrier_out",  # <- no error is raised because of this
        }
        constraint_name = "constraint-without-nan"

        simple_supply.backend.add_constraint(constraint_name, constraint_dict)

        assert (
            simple_supply.backend.get_constraint(constraint_name).name
            == constraint_name
        )

        # add constraint with nan
        constraint_dict = {
            "foreach": ["techs", "carriers"],
            "equations": [
                {"expression": "sum(flow_out, over=[nodes, timesteps]) >= 100"}
            ],
            # "where": "carrier_out",  # <- no error would be raised with this uncommented
        }
        constraint_name = "constraint-with-nan"

        with pytest.raises(exceptions.BackendError) as error:
            simple_supply.backend.add_constraint(constraint_name, constraint_dict)

        assert check_error_or_warning(
            error,
            f"constraints:{constraint_name}:0 | Missing a linear expression for some coordinates selected by 'where'. Adapting 'where' might help.",
        )

    def test_raise_error_on_expression_with_nan(self, simple_supply):
        """
        A very simple expression: The annual and regional sum of `flow_out` for each tech.
        However, not every tech has the variable `flow_out`.
        How to solve it? Let the constraint be active only where flow_out exists by setting 'where' accordingly.
        """
        # add expression without nan
        expression_dict = {
            "foreach": ["techs", "carriers"],
            "equations": [{"expression": "sum(flow_out, over=[nodes, timesteps])"}],
            "where": "carrier_out",  # <- no error is raised because of this
        }
        expression_name = "expression-without-nan"

        # add expression with nan
        simple_supply.backend.add_global_expression(expression_name, expression_dict)

        assert (
            simple_supply.backend.get_global_expression(expression_name).name
            == expression_name
        )

        expression_dict = {
            "foreach": ["techs", "carriers"],
            "equations": [{"expression": "sum(flow_out, over=[nodes, timesteps])"}],
            # "where": "carrier_out",  # <- no error would be raised with this uncommented
        }
        expression_name = "expression-with-nan"

        with pytest.raises(exceptions.BackendError) as error:
            simple_supply.backend.add_global_expression(
                expression_name, expression_dict
            )

        assert check_error_or_warning(
            error,
            f"global_expressions:{expression_name}:0 | Missing a linear expression for some coordinates selected by 'where'. Adapting 'where' might help.",
        )

    def test_raise_error_on_excess_dimensions(self, simple_supply):
        """
        A very simple constraint: For each tech, let the `flow_cap` be larger than 100.
        However, we forgot to include `nodes` in `foreach`.
        With `nodes` included, this constraint should build.
        """
        # add constraint without excess dimensions
        constraint_dict = {
            # as 'nodes' is listed here, the constraint will have no excess dimensions
            "foreach": ["techs", "nodes", "carriers"],
            "equations": [{"expression": "flow_cap >= 100"}],
        }
        constraint_name = "constraint-without-excess-dimensions"

        simple_supply.backend.add_constraint(constraint_name, constraint_dict)

        assert (
            simple_supply.backend.get_constraint(constraint_name).name
            == constraint_name
        )

        # add constraint with excess dimensions
        constraint_dict = {
            # as 'nodes' is not listed here, the constraint will have excess dimensions
            "foreach": ["techs", "carriers"],
            "equations": [{"expression": "flow_cap >= 100"}],
        }
        constraint_name = "constraint-with-excess-dimensions"

        with pytest.raises(exceptions.BackendError) as error:
            simple_supply.backend.add_constraint(constraint_name, constraint_dict)

        assert check_error_or_warning(
            error,
            f"constraints:{constraint_name}:0 | The linear expression array is indexed over dimensions not present in `foreach`: {{'nodes'}}",
        )

    @pytest.mark.parametrize(
        "component", ["parameters", "variables", "global_expressions", "constraints"]
    )
    def test_create_and_delete_pyomo_list(self, simple_supply, component):
        backend_instance = simple_supply.backend._instance
        simple_supply.backend._create_obj_list("foo", component)
        assert "foo" in getattr(backend_instance, component).keys()

        simple_supply.backend.delete_component("foo", component)
        assert "foo" not in getattr(backend_instance, component).keys()
        assert "foo" not in getattr(simple_supply.backend, component).keys()

    @pytest.mark.parametrize(
        "component", ["parameters", "variables", "global_expressions", "constraints"]
    )
    def test_delete_inexistent_pyomo_list(self, simple_supply, component):
        backend_instance = simple_supply.backend._instance
        assert "bar" not in getattr(backend_instance, component).keys()
        simple_supply.backend.delete_component("bar", component)
        assert "bar" not in getattr(backend_instance, component).keys()

    @pytest.mark.parametrize(
        ["component", "eq"],
        [("global_expressions", "flow_cap + 1"), ("constraints", "flow_cap >= 1")],
    )
    def test_add_allnull_expr_or_constr(self, simple_supply, component, eq):
        adder = getattr(simple_supply.backend, "add_" + component.removesuffix("s"))
        constr_dict = {
            "foreach": ["nodes", "techs"],
            "where": "True",
            "equations": [{"expression": eq, "where": "False"}],
        }
        adder("foo", constr_dict)

        assert "foo" not in getattr(simple_supply.backend._instance, component).keys()
        assert "foo" not in simple_supply.backend._dataset.data_vars.keys()

    def test_add_allnull_param_no_shape(self, simple_supply):
        simple_supply.backend.add_parameter("foo", xr.DataArray(np.nan))

        assert "foo" not in simple_supply.backend._instance.parameters.keys()
        # We keep it in the dataset since it might be fillna'd by another param later.
        assert "foo" in simple_supply.backend._dataset.data_vars.keys()
        del simple_supply.backend._dataset["foo"]

    def test_add_allnull_param_with_shape(self, simple_supply):
        nan_array = simple_supply._model_data.flow_cap_max.where(lambda x: x < 0)
        simple_supply.backend.add_parameter("foo", nan_array)

        assert "foo" not in simple_supply.backend._instance.parameters.keys()
        # We keep it in the dataset since it might be fillna'd by another param later.
        assert "foo" in simple_supply.backend._dataset.data_vars.keys()
        del simple_supply.backend._dataset["foo"]

    def test_add_allnull_var(self, simple_supply):
        simple_supply.backend.add_variable(
            "foo", {"foreach": ["nodes"], "where": "False"}
        )
        assert "foo" not in simple_supply.backend._instance.variables.keys()
        assert "foo" not in simple_supply.backend._dataset.data_vars.keys()

    def test_add_allnull_obj(self, simple_supply):
        eq = {"expression": "bigM", "where": "False"}
        simple_supply.backend.add_objective(
            "foo", {"equations": [eq, eq], "sense": "minimise"}
        )
        assert len(simple_supply.backend._instance.objectives) == 1
        assert "foo" not in simple_supply.backend._dataset.data_vars.keys()

    def test_add_two_same_obj(self, simple_supply):
        eq = {"expression": "bigM", "where": "True"}
        with pytest.raises(exceptions.BackendError) as excinfo:
            simple_supply.backend.add_objective(
                "foo", {"equations": [eq, eq], "sense": "minimise"}
            )
        assert check_error_or_warning(
            excinfo,
            "objectives:foo:1 | trying to set two equations for the same component.",
        )

    def test_add_valid_obj(self, simple_supply):
        eq = {"expression": "bigM", "where": "True"}
        simple_supply.backend.add_objective(
            "foo", {"equations": [eq], "sense": "minimise"}
        )
        assert "foo" in simple_supply.backend.objectives
        assert not simple_supply.backend.objectives.foo.item().active

    def test_object_string_representation(self, simple_supply):
        assert (
            simple_supply.backend.variables.flow_out.sel(
                nodes="a",
                techs="test_supply_elec",
                carriers="electricity",
                timesteps="2005-01-01 00:00",
            )
            .item()
            .name
            == "variables[flow_out][4]"
        )
        assert not simple_supply.backend.variables.flow_out.coords_in_name

    @pytest.mark.parametrize(
        ["objname", "dims", "objtype"],
        [
            (
                "flow_out",
                {
                    "nodes": "a",
                    "techs": "test_supply_elec",
                    "carriers": "electricity",
                    "timesteps": "2005-01-01 00:00",
                },
                "variables",
            ),
            ("flow_out_eff", {"techs": "test_supply_elec"}, "parameters"),
            (
                "system_balance",
                {
                    "nodes": "a",
                    "carriers": "electricity",
                    "timesteps": "2005-01-01 00:00",
                },
                "constraints",
            ),
        ],
    )
    def test_verbose_strings(self, simple_supply_longnames, objname, dims, objtype):
        obj = simple_supply_longnames.backend._dataset[objname]
        assert (
            obj.sel(dims).item().name
            == f"{objtype}[{objname}][{', '.join(dims[i] for i in obj.dims)}]"
        )
        assert obj.attrs["coords_in_name"]

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
        assert (
            obj.sel(dims).body.item()
            == f"(parameters[flow_in_eff]*variables[flow_in][{', '.join(dims[i] for i in obj.dims)}])"
        )
        assert obj.coords_in_name

    def test_verbose_strings_expression(self, simple_supply_longnames):
        dims = {"nodes": "a", "techs": "test_supply_elec", "costs": "monetary"}

        obj = simple_supply_longnames.backend.get_global_expression(
            "cost_investment", as_backend_objs=False
        )

        assert (
            "variables[flow_cap][a, test_supply_elec, electricity]"
            in obj.sel(dims).item()
        )
        assert "parameters[cost_interest_rate]" in obj.sel(dims).item()

        assert not obj.coords_in_name

    def test_verbose_strings_no_len(self, simple_supply_longnames):
        obj = simple_supply_longnames.backend.parameters.bigM

        assert obj.item().name == "parameters[bigM]"
        assert obj.coords_in_name

    def test_update_parameter(self, simple_supply):
        updated_param = simple_supply.inputs.flow_out_eff * 1000
        simple_supply.backend.update_parameter("flow_out_eff", updated_param)

        expected = simple_supply.backend.get_parameter(
            "flow_out_eff", as_backend_objs=False
        )
        assert expected.where(updated_param.notnull()).equals(updated_param)

    def test_update_parameter_one_val(self, caplog, simple_supply):
        updated_param = 1000
        new_dims = {"techs"}
        caplog.set_level(logging.DEBUG)

        simple_supply.backend.update_parameter("flow_out_eff", updated_param)

        assert (
            f"New values will be broadcast along the {new_dims} dimension(s)"
            in caplog.text
        )
        expected = simple_supply.backend.get_parameter(
            "flow_out_eff", as_backend_objs=False
        )
        assert (expected == updated_param).all()

    def test_update_parameter_replace_defaults(self, simple_supply):
        updated_param = simple_supply.inputs.flow_out_eff.fillna(0.1)

        simple_supply.backend.update_parameter("flow_out_eff", updated_param)

        expected = simple_supply.backend.get_parameter(
            "flow_out_eff", as_backend_objs=False
        )
        assert expected.equals(updated_param)

    def test_update_parameter_add_dim(self, caplog, simple_supply):
        """
        flow_out_eff doesn't have the time dimension in the simple model, we add it here.
        """
        updated_param = simple_supply.inputs.flow_out_eff.where(
            simple_supply.inputs.timesteps.notnull()
        )
        refs_to_update = [  # should be sorted alphabetically
            "balance_supply_no_storage",
            "balance_transmission",
            "flow_out_inc_eff",
        ]
        caplog.set_level(logging.DEBUG)

        simple_supply.backend.update_parameter("flow_out_eff", updated_param)

        assert (
            "Defining values for a previously fully/partially undefined parameter. "
            f"The optimisation problem components {refs_to_update} will be re-built."
            in caplog.text
        )

        expected = simple_supply.backend.get_parameter(
            "flow_out_eff", as_backend_objs=False
        )
        assert "timesteps" in expected.dims

    def test_update_parameter_replace_undefined(self, caplog, simple_supply):
        """source_eff isn't defined in the inputs, so is a dimensionless value in the pyomo object, assigned its default value."""
        updated_param = simple_supply.inputs.flow_out_eff

        refs_to_update = ["balance_supply_no_storage"]
        caplog.set_level(logging.DEBUG)

        simple_supply.backend.update_parameter("source_eff", updated_param)

        assert (
            "Defining values for a previously fully/partially undefined parameter. "
            f"The optimisation problem components {refs_to_update} will be re-built."
            in caplog.text
        )

        expected = simple_supply.backend.get_parameter(
            "source_eff", as_backend_objs=False
        )
        default_val = simple_supply._model_data.attrs["defaults"]["source_eff"]
        assert expected.equals(updated_param.fillna(default_val))

    def test_update_parameter_no_refs_to_update(self, simple_supply):
        """flow_cap_per_storage_cap_max isn't defined in the inputs, so is a dimensionless value in the pyomo object, assigned its default value.

        Updating it doesn't change the model in any way, because none of the existing constraints/expressions depend on it.
        Therefore, no warning is raised.
        """
        updated_param = 1

        simple_supply.backend.update_parameter(
            "flow_cap_per_storage_cap_max", updated_param
        )

        expected = simple_supply.backend.get_parameter(
            "flow_cap_per_storage_cap_max", as_backend_objs=False
        )
        assert expected == 1

    @pytest.mark.parametrize("bound", ["min", "max"])
    def test_update_variable_single_bound_single_val(self, simple_supply, bound):
        translator = {"min": "lb", "max": "ub"}

        simple_supply.backend.update_variable_bounds("flow_out", **{bound: 1})

        bound_vals = simple_supply.backend.get_variable_bounds("flow_out")[
            translator[bound]
        ]

        assert (bound_vals == 1).where(bound_vals.notnull()).all()

    def test_update_variable_bounds_single_val(self, simple_supply):
        simple_supply.backend.update_variable_bounds("flow_out", min=2, max=2)
        bound_vals = simple_supply.backend.get_variable_bounds("flow_out")
        assert (bound_vals == 2).where(bound_vals.notnull()).all().all()

    def test_update_variable_single_bound_multi_val(self, caplog, simple_supply):
        caplog.set_level(logging.INFO)
        bound_array = simple_supply.inputs.sink_equals.sel(techs="test_demand_elec")
        simple_supply.backend.update_variable_bounds("flow_in", min=bound_array)
        bound_vals = simple_supply.backend.get_variable_bounds("flow_in").lb
        assert "New `min` bounds will be broadcast" in caplog.text
        assert bound_vals.equals(
            bound_array.where(bound_vals.notnull()).transpose(*bound_vals.dims)
        )

    def test_update_variable_error_update_parameter_instead(self, simple_supply):
        with pytest.raises(exceptions.BackendError) as excinfo:
            simple_supply.backend.update_variable_bounds("flow_cap", min=1)
        assert check_error_or_warning(
            excinfo,
            "Cannot update variable bounds that have been set by parameters."
            " Use `update_parameter('flow_cap_min')` to update the min bound of flow_cap.",
        )

    @staticmethod
    def _is_fixed(val):
        if pd.notnull(val):
            return val.fixed
        else:
            return np.nan

    def test_fix_variable(self, simple_supply):
        simple_supply.backend.fix_variable("flow_cap")
        fixed = simple_supply.backend._apply_func(
            self._is_fixed, simple_supply.backend.variables.flow_cap
        )
        simple_supply.backend.unfix_variable("flow_cap")  # reset
        assert fixed.where(fixed.notnull()).all()

    def test_fix_variable_where(self, simple_supply):
        where = (
            simple_supply.inputs.flow_cap_max.notnull()
            & simple_supply.backend.variables.flow_cap.notnull()
        )
        simple_supply.backend.fix_variable("flow_cap", where=where)
        fixed = simple_supply.backend._apply_func(
            self._is_fixed, simple_supply.backend.variables.flow_cap
        )
        simple_supply.backend.unfix_variable("flow_cap")  # reset
        assert not fixed.sel(techs="test_demand_elec", carriers="electricity").any()
        assert fixed.where(where, other=True).all()

    def test_fix_variable_before_solve(self, simple_supply_longnames):
        with pytest.raises(exceptions.BackendError) as excinfo:
            simple_supply_longnames.backend.fix_variable("flow_cap")

        assert check_error_or_warning(
            excinfo,
            "Cannot fix variable values without already having solved the model successfully.",
        )

    def test_unfix_variable(self, simple_supply):
        simple_supply.backend.fix_variable("flow_cap")
        simple_supply.backend.unfix_variable("flow_cap")
        fixed = simple_supply.backend._apply_func(
            self._is_fixed, simple_supply.backend.variables.flow_cap
        )
        assert not fixed.where(fixed.notnull()).all()

    def test_unfix_variable_where(self, simple_supply):
        where = (
            simple_supply.inputs.flow_cap_max.notnull()
            & simple_supply.backend.variables.flow_cap.notnull()
        )
        simple_supply.backend.fix_variable("flow_cap")
        simple_supply.backend.unfix_variable("flow_cap", where=where)
        fixed = simple_supply.backend._apply_func(
            self._is_fixed, simple_supply.backend.variables.flow_cap
        )
        simple_supply.backend.unfix_variable("flow_cap")  # reset
        assert fixed.sel(techs="test_demand_elec").all()
        assert not fixed.where(where).all()
