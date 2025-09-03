import logging
from itertools import product

import numpy as np
import pyomo.core as po
import pyomo.kernel as pmo
import pytest  # noqa: F401
import xarray as xr
from pyomo.core.kernel.piecewise_library.transforms import piecewise_sos2

import calliope
import calliope.backend
import calliope.exceptions as exceptions

from .common.util import build_test_model as build_model
from .common.util import check_error_or_warning, check_variable_exists


@pytest.mark.xfail(
    reason="Need to reintroduce these checks in config/model_data_checks.yaml where it's reasonable to do so"
)
class TestChecks:
    @pytest.mark.parametrize("on", [True, False])
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
        assert m.attrs["config"].build.cyclic_storage is False

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
        assert "timesteps" in m.inputs[param].dims

        with pytest.warns(exceptions.ModelWarning):
            m.build()  # will fail to complete run if there's a problem

    @pytest.mark.parametrize("constr", ["max", "equals"])
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

    @pytest.mark.parametrize("constr", ["max", "equals"])
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
        ("source_unit", "constr"),
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

    def _model_operate_source_unit_without_area(source_unit: str):
        return build_model(
            {
                "techs.test_supply_elec": {
                    "constraints": {
                        "source_use_max": "file=supply_plus_resource.csv:1",
                        "flow_cap_max": 15,
                    },
                    "switches": {"force_source": True, "source_unit": source_unit},
                }
            },
            "simple_supply_and_supply_plus,operate,investment_costs",
        )

    def test_operate_source_unit_without_area_use_absolute(self):
        m = self._model_operate_source_unit_without_area("absolute")

        with pytest.warns(exceptions.ModelWarning) as warning:
            m.build()

        _warnings = [
            "Flow capacity constraint removed from 0::test_supply_elec as force_source is applied and source is not linked to flow (source_unit = `absolute`)"
        ]
        not_warnings = [
            "Area use constraint removed from 0::test_supply_elec as force_source is applied and source is not linked to flow (source_unit = `absolute`)",
            "Flow capacity constraint removed from 0::test_demand_elec as force_source is applied and source is not linked to flow (source_unit = `absolute`)",
            "Flow capacity constraint removed from 1::test_demand_elec as force_source is applied and source is not linked to flow (source_unit = `absolute`)",
        ]

        assert check_error_or_warning(warning, _warnings)
        assert not check_error_or_warning(warning, not_warnings)

    def test_operate_source_unit_without_area_use_per_cap(self):
        m = self._model_operate_source_unit_without_area("per_cap")

        with pytest.warns(exceptions.ModelWarning) as warning:
            m.build()

        _warnings = []
        not_warnings = [
            "Area use constraint removed from 0::test_supply_elec as force_source is applied and source is linked to flow using `per_cap`",
            "Flow capacity constraint removed from 0::test_supply_elec as force_source is applied and source is not linked to flow (source_unit = `absolute`)",
            "Flow capacity constraint removed from 0::test_demand_elec as force_source is applied and source is not linked to flow (source_unit = `absolute`)",
            "Flow capacity constraint removed from 1::test_demand_elec as force_source is applied and source is not linked to flow (source_unit = `absolute`)",
        ]

        assert check_error_or_warning(warning, _warnings)
        assert not check_error_or_warning(warning, not_warnings)

    def test_operate_source_unit_without_area_use_per_area(self):
        m = self._model_operate_source_unit_without_area("per_area")

        with pytest.raises(exceptions.ModelError) as error:
            with pytest.warns(exceptions.ModelWarning) as warning:
                m.build()

        _warnings = [
            "Flow capacity constraint removed from 0::test_supply_elec as force_source is applied and source is linked to flow using `per_area`"
        ]
        not_warnings = [
            "Area use constraint removed from 0::test_supply_elec as force_source is applied and source is linked to flow using `per_cap`",
            "Flow capacity constraint removed from 0::test_demand_elec as force_source is applied and source is not linked to flow (source_unit = `absolute`)",
            "Flow capacity constraint removed from 1::test_demand_elec as force_source is applied and source is not linked to flow (source_unit = `absolute`)",
        ]
        check_error_or_warning(
            error,
            "Operate mode: User must define a finite area_use "
            "(via area_use_equals or area_use_max) for 0::test_supply_elec",
        )
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

    @pytest.mark.parametrize("on", [True, False])
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
            assert np.isinf(m.results.source_cap.loc["a", "test_supply_plus"].item())
        elif on is True:
            assert not check_error_or_warning(
                warning, "Source capacity constraint defined and set to infinity"
            )
            assert m.results.source_cap.loc["a", "test_supply_plus"].item() == 1e6

    @pytest.mark.parametrize("on", [True, False])
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
            assert m.results.storage_initial.loc["a", "test_supply_plus"].item() == 0
        elif on is True:
            assert not check_error_or_warning(
                warning, "Initial stored carrier not defined"
            )
            assert m.results.storage_initial.loc["a", "test_supply_plus"].item() == 0.5


@pytest.mark.skip(reason="to be reimplemented by comparison to LP files")
class TestBalanceConstraints:
    def test_loc_carriers_system_balance_constraint(self, simple_supply):
        """sets.loc_carriers"""
        assert "system_balance" in simple_supply.backend.constraints

    def test_loc_techs_balance_supply_constraint(self):
        """sets.loc_techs_finite_resource_supply,"""
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
        """sets.loc_techs_finite_resource_demand,"""
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
        """sets.loc_techs_finite_resource_supply_plus,"""
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
        """sets.loc_techs_transmission,"""
        assert "balance_transmission" in simple_supply.backend.constraints

    def test_loc_techs_balance_supply_plus_constraint(
        self, simple_supply_and_supply_plus
    ):
        """sets.loc_techs_supply_plus,"""
        assert (
            "balance_supply_plus_with_storage"
            in simple_supply_and_supply_plus.backend.constraints
        )
        assert (
            "balance_supply_plus_no_storage"
            not in simple_supply_and_supply_plus.backend.constraints
        )

    def test_loc_techs_balance_storage_constraint(self, simple_storage):
        """sets.loc_techs_storage,"""
        assert "balance_storage" in simple_storage.backend.constraints
        assert "set_storage_initial" not in simple_storage.backend.constraints

    def test_loc_techs_balance_storage_discharge_depth_constraint(self):
        """sets.loc_techs_storage,"""
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
            m3.inputs.storage_initial.to_series().dropna()
            > m3.inputs.storage_discharge_depth.to_series().dropna()
        ).all()

    def test_storage_initial_constraint(self, simple_storage):
        """sets.loc_techs_store,"""
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
        """I for i in sets.carriers if i in model_run.model.get_key('reserve_margin', {}).keys()"""
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
        """sets.loc_techs_cost,"""
        assert "cost" in simple_supply.backend.expressions

    def test_loc_techs_cost_investment_constraint(self, simple_conversion):
        """sets.loc_techs_investment_cost,"""
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
        """I for i in sets.loc_techs_om_cost if i not in sets.loc_techs_conversion_plus + sets.loc_techs_conversion"""
        assert "cost_operation_variable" not in simple_conversion.backend.expressions

    @pytest.mark.parametrize(
        ("tech", "scenario", "cost"),
        [
            ("test_supply_elec", "simple_supply", "flow_out"),
            ("test_supply_elec", "simple_supply", "flow_in"),
            ("test_supply_plus", "simple_supply_and_supply_plus", "flow_in"),
            ("test_demand_elec", "simple_supply", "flow_in"),
            ("test_transmission_elec", "simple_supply", "flow_out"),
            ("test_conversion", "simple_conversion", "flow_in"),
            ("test_conversion_plus", "simple_conversion_plus", "flow_out"),
        ],
    )
    def test_loc_techs_cost_var_constraint(self, tech, scenario, cost):
        """I for i in sets.loc_techs_om_cost if i not in sets.loc_techs_conversion_plus + sets.loc_techs_conversion"""
        m = build_model(
            {f"techs.{tech}.costs.monetary.{cost}": 1}, f"{scenario},two_hours"
        )
        m.build()
        assert "cost_operation_variable" in m.backend.expressions

    def test_one_way_om_cost(self):
        """With one_way transmission, it should still be possible to set an flow_out cost."""
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
            m.backend.get_expression("cost_operation_variable", as_backend_objs=False),
            "flow_out",
            idx,
        )

        idx["nodes"] = "a"
        idx["techs"] = "test_transmission_elec:b"
        assert not check_variable_exists(
            m.backend.get_expression("cost_operation_variable", as_backend_objs=False),
            "flow_out",
            idx,
        )


@pytest.mark.skip(reason="to be reimplemented by comparison to LP files")
class TestExportConstraints:
    # export.py
    def test_loc_carriers_system_balance_no_export(self, simple_supply):
        """I for i in sets.loc_carriers if sets.loc_techs_export
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
        """sets.loc_tech_carriers_export,"""
        assert "export_balance" in supply_export.backend.constraints

    def test_loc_techs_update_costs_var_constraint(self, supply_export):
        """I for i in sets.loc_techs_om_cost if i in sets.loc_techs_export"""
        assert "cost_operation_variable" in supply_export.backend.expressions

        m = build_model(
            {"techs.test_supply_elec.costs.monetary.flow_out": 0.1},
            "supply_export,two_hours,investment_costs",
        )
        m.build()
        assert "cost_operation_variable" in m.backend.expressions

        assert check_variable_exists(
            m.backend.get_expression("cost_operation_variable", as_backend_objs=False),
            "flow_export",
        )

    def test_loc_tech_carriers_export_max_constraint(self):
        """I for i in sets.loc_tech_carriers_export
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
        """I for i in sets.loc_techs_store if i not in sets.loc_techs_milp"""
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
                    "purchased_units_max": 1,
                    "flow_cap_per_unit": 20,
                    "storage_cap_per_unit": 20,
                }
            },
            "simple_storage,two_hours,investment_costs",
        )
        m.build()
        assert "storage_capacity" not in m.backend.constraints

    @pytest.mark.parametrize(
        ("scenario", "tech", "override"),
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
        """I for i in sets.loc_techs_store if constraint_exists(model_run, i, 'constraints.flow_cap_per_storage_cap_max')"""
        m = build_model(
            {f"techs.{tech}.constraints.flow_cap_per_storage_cap_{override}": 0.5},
            f"{scenario},two_hours,investment_costs",
        )
        m.build()
        assert hasattr(
            m._backend_model,
            f"flow_capacity_per_storage_capacity_{override}_constraint",
        )

    @pytest.mark.filterwarnings("ignore:(?s).*Integer:calliope.exceptions.ModelWarning")
    @pytest.mark.parametrize("override", (["max", "min"]))
    def test_loc_techs_flow_capacity_milp_storage_constraint(self, override):
        """I for i in sets.loc_techs_store if constraint_exists(model_run, i, 'constraints.flow_cap_per_storage_cap_max')"""
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
        """I for i in sets.loc_techs_store if constraint_exists(model_run, i, 'constraints.flow_cap_per_storage_cap_max')"""
        with caplog.at_level(logging.INFO):
            m = build_model(model_file="flow_cap_per_storage_cap.yaml")

        m.build()
        assert not any(
            [
                hasattr(m._backend_model, f"flow_capacity_storage_{i}_constraint")
                for i in ["max", "min"]
            ]
        )

    @pytest.mark.parametrize("override", ([None, "max", "min"]))
    def test_loc_techs_resource_capacity_constraint(self, override):
        """I for i in sets.loc_techs_finite_resource_supply_plus
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
                {f"techs.test_supply_plus.constraints.resource_cap_{override}": 10},
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
        """I for i in sets.loc_techs_finite_resource_supply_plus
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
        """I for i in sets.loc_techs_area if i in sets.loc_techs_supply_plus"""
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
        """I for i in sets.loc_techs_area if i in sets.loc_techs_supply_plus
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
        """I for i in sets.locs
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
        """I for i in sets.loc_techs
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
        m = build_model(override, "simple_supply,two_hours,investment_costs")
        with pytest.raises(exceptions.ModelError) as error:
            m.build()

        assert check_error_or_warning(
            error,
            "Cannot use inf for flow_cap_equals for node, tech `('a', 'test_supply_elec')`",
        )

    @pytest.mark.parametrize("bound", ("max"))
    def test_techs_flow_capacity_systemwide_constraint(self, bound):
        """I for i in sets.techs
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

    @pytest.mark.parametrize("bound", (["equals", "max"]))
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
        """I for i in sets.loc_tech_carriers_prod
        if i not in sets.loc_tech_carriers_conversion_plus
        and i.rsplit('::', 1)[0] not in sets.loc_techs_milp
        """
        assert "flow_out_max" in simple_supply.backend.constraints

    def test_loc_tech_carriers_flow_out_max_milp_constraint(self, supply_milp):
        assert "flow_out_max" not in supply_milp.backend.constraints

    def test_loc_tech_carriers_flow_out_min_constraint(self, simple_supply):
        """I for i in sets.loc_tech_carriers_prod
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
        """I for i in sets.loc_tech_carriers_con
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
        """sets.loc_techs_finite_resource_supply_plus,"""
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
        """sets.loc_techs_store"""
        assert "storage_max" not in simple_supply.backend.constraints
        assert "storage_max" in simple_supply_and_supply_plus.backend.constraints
        assert "storage_max" in simple_storage.backend.constraints

    def test_loc_tech_carriers_ramping_constraint(self, simple_supply):
        """I for i in sets.loc_tech_carriers_prod
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
        """sets.loc_techs_milp,"""
        assert "unit_commitment_milp" not in simple_supply.backend.constraints
        assert "unit_commitment_milp" in supply_milp.backend.constraints
        assert "unit_commitment_milp" not in supply_purchase.backend.constraints

    def test_loc_techs_unit_capacity_milp_constraint(
        self, simple_supply, supply_milp, supply_purchase
    ):
        """sets.loc_techs_milp,"""
        assert "units" not in simple_supply.backend.variables
        assert "units" in supply_milp.backend.variables
        assert "units" not in supply_purchase.backend.variables

    def test_loc_tech_carriers_flow_out_max_milp_constraint(
        self, simple_supply, supply_milp, supply_purchase, conversion_plus_milp
    ):
        """I for i in sets.loc_tech_carriers_prod
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
        """I for i in sets.loc_techs_conversion_plus
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
        """I for i in sets.loc_tech_carriers_prod
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
        """I for i in sets.loc_techs_conversion_plus
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
        """I for i in sets.loc_tech_carriers_con
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
        """I for i in sets.loc_techs_milp
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
        """I for i in sets.loc_techs_milp if i in sets.loc_techs_store"""
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
        """I for i in sets.loc_techs_purchase
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
        """I for i in sets.loc_techs_purchase
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
        """I for i in set(sets.loc_techs_purchase).intersection(sets.loc_techs_store)"""
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
        """I for i in set(sets.loc_techs_purchase).intersection(sets.loc_techs_store)
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
        [
            ("simple_supply", ("not", "not"), {}),
            ("supply_milp", ("not", "not"), {}),
            (
                "supply_milp",
                ("not", "is"),
                {"techs.test_supply_elec.costs.monetary.purchase": 1},
            ),
            ("supply_purchase", ("is", "not"), {}),
        ],
    )
    def test_loc_techs_update_costs_investment_units_milp_constraint(
        self, scenario, exists, override_dict
    ):
        """I for i in sets.loc_techs_milp
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
        """sets.techs if unit_cap_max_systemwide or unit_cap_equals_systemwide"""
        override_max = {
            "links.a,b.exists": True,
            "techs.test_conversion_plus.constraints.units_max_systemwide": 2,
            "nodes.b.techs.test_conversion_plus.constraints": {
                "purchased_units_max": 2,
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
        """sets.techs if unit_cap_max_systemwide or unit_cap_equals_systemwide"""
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

    @pytest.mark.parametrize("tech", [("test_storage"), ("test_transmission_elec")])
    def test_asynchronous_flow_constraint(self, tech):
        """Binary switch for flow in/out can be activated using the option
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
        """sets.loc_techs_conversion,"""
        assert "balance_conversion" not in simple_supply.backend.constraints
        assert "balance_conversion" in simple_conversion.backend.constraints
        assert "balance_conversion" not in simple_conversion_plus.backend.constraints


@pytest.mark.skip(reason="to be reimplemented by comparison to LP files")
class TestNetworkConstraints:
    # network.py
    def test_loc_techs_symmetric_transmission_constraint(
        self, simple_supply, simple_conversion_plus
    ):
        """sets.loc_techs_transmission,"""
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
        self, storage_inter_cluster=True, cyclic=False, storage_initial=False
    ):
        override = {
            "config.init.time_subset": ["2005-01-01", "2005-01-04"],
            "config.init.time_cluster": "cluster_days_param",
            "config.init.extra_math": (
                ["storage_inter_cluster"] if storage_inter_cluster else []
            ),
            "data_definitions.cyclic_storage": cyclic,
            "data_tables.cluster_days": {
                "data": "data_tables/cluster_days.csv",
                "rows": "datesteps",
                "add_dims": {"parameters": "cluster_days_param"},
            },
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
        override = {"techs.test_supply_elec.source_use_equals": np.inf}
        m = build_model(override_dict=override, scenario="simple_supply,one_day")

        with pytest.raises(exceptions.ModelError) as excinfo:
            m.build()
        assert check_error_or_warning(excinfo, "Cannot include infinite values")

    def test_storage_initial_fractional_value(self):
        """Check that the storage_initial value is a fraction"""
        m = build_model(
            {"techs.test_storage.storage_initial": 5},
            "simple_storage,two_hours,investment_costs",
        )

        with pytest.raises(exceptions.ModelError) as error:
            m.build()
        assert check_error_or_warning(
            error, "requiring values within the interval [0, 1]"
        )


class TestNewBackend:
    LOGGER = logging.getLogger("calliope.backend.backend_model")

    @pytest.fixture(scope="class")
    def simple_supply_updated_cost_flow_cap(
        self, simple_supply: calliope.Model, dummy_int: int
    ) -> calliope.Model:
        simple_supply.backend.verbose_strings()
        simple_supply.backend.update_input("cost_flow_cap", dummy_int)
        return simple_supply

    @pytest.fixture
    def temp_path(self, tmpdir_factory):
        return tmpdir_factory.mktemp("custom_math")

    def test_add_run_mode_custom_math_before_build(self, caplog):
        """Run mode math is applied before anything else."""
        caplog.set_level(logging.DEBUG)
        custom_math = {"constraints": {"force_zero_area_use": {"active": True}}}

        m = build_model(
            {},
            "simple_supply,two_hours,investment_costs",
            mode="operate",
            math_dict=custom_math,
        )
        m.build(operate={"window": "12h", "horizon": "12h"})

        # operate mode set it to false, then our math set it back to active
        assert m.math.build.constraints["force_zero_area_use"].active
        # operate mode set it to false and our math did not override that
        assert not m.math.build.variables["storage_cap"].active

    def test_new_build_get_variable(self, simple_supply):
        """Check a decision variable has the correct data type and has all expected attributes."""
        var = simple_supply.backend.get_variable("flow_cap")
        assert (
            var.to_series().dropna().apply(lambda x: isinstance(x, pmo.variable)).all()
        )

    def test_new_build_get_variable_as_vals(self, simple_supply):
        var = simple_supply.backend.get_variable("flow_cap", as_backend_objs=False)
        assert (
            not var.to_series()
            .dropna()
            .apply(lambda x: isinstance(x, pmo.variable))
            .any()
        )

    def test_new_build_get_component_exists(self, simple_supply):
        param = simple_supply.backend._get_component("flow_in_eff", "parameters")
        assert isinstance(param, xr.DataArray)

    def test_new_build_get_component_does_not_exist(self, simple_supply):
        with pytest.raises(KeyError) as excinfo:
            simple_supply.backend._get_component("does_not_exist", "parameters")
        assert check_error_or_warning(excinfo, "Unknown parameter: does_not_exist")

    def test_new_build_get_component_wrong_group(self, simple_supply):
        with pytest.raises(KeyError) as excinfo:
            simple_supply.backend._get_component("flow_in_eff", "piecewise_constraints")
        assert check_error_or_warning(
            excinfo, "Unknown piecewise constraint: flow_in_eff"
        )

    def test_new_build_get_parameter(self, simple_supply):
        """Check a parameter has the correct data type and has all expected attributes."""
        param = simple_supply.backend.get_parameter("cost_flow_cap")
        assert (
            param.to_series()
            .dropna()
            .apply(lambda x: isinstance(x, pmo.parameter))
            .all()
        )

    def test_new_build_get_global_expression(self, simple_supply):
        """Check a global expression has the correct data type and has all expected attributes."""
        expr = simple_supply.backend.get_global_expression("cost_investment")
        assert (
            expr.to_series()
            .dropna()
            .apply(lambda x: isinstance(x, po.expr.ExpressionBase))
            .all()
        )

    def test_new_build_get_constraint(self, simple_supply):
        constr = simple_supply.backend.get_constraint("system_balance")
        assert (
            constr.to_series()
            .dropna()
            .apply(lambda x: isinstance(x, pmo.constraint))
            .all()
        )

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
            .apply(lambda x: isinstance(x, float | int))
            .all()
        )

    @pytest.mark.parametrize("bound", ["lb", "ub"])
    def test_new_build_get_constraint_bounds(self, simple_supply, bound):
        constr = simple_supply.backend.get_constraint(
            "system_balance", as_backend_objs=False
        )
        assert (constr[bound].to_series().dropna() == 0).all()

    def test_add_allnull_var(self, simple_supply):
        simple_supply.backend.add_variable(
            "foo_var",
            {"foreach": ["nodes"], "where": "False", "bounds": {"min": 0, "max": 1}},
        )
        assert "foo_var" not in simple_supply.backend._instance.variables.keys()

    @pytest.mark.parametrize(
        ("component", "eq"),
        [("global_expressions", "flow_cap + 1"), ("constraints", "flow_cap >= 1")],
    )
    def test_add_allnull_expr_or_constr(self, simple_supply, component, eq):
        adder = getattr(simple_supply.backend, "add_" + component.removesuffix("s"))
        constr_dict = {
            "foreach": ["nodes", "techs"],
            "where": "True",
            "equations": [{"expression": eq, "where": "False"}],
        }
        name = f"foo_{component}"
        adder(name, constr_dict)

        assert name not in getattr(simple_supply.backend._instance, component).keys()

    def test_add_allnull_param_no_shape(self, simple_supply):
        simple_supply.backend.add_parameter(
            "foo_param_no_dims", xr.DataArray(np.nan), {}
        )

        assert (
            "foo_param_no_dims" not in simple_supply.backend._instance.parameters.keys()
        )

    def test_add_allnull_param_with_shape(self, simple_supply):
        nan_array = simple_supply.inputs.flow_cap_max.where(lambda x: x < 0)
        simple_supply.backend.add_parameter("foo_param_dims", nan_array, {})

        assert "foo_param_dims" not in simple_supply.backend._instance.parameters.keys()
        del simple_supply.backend._dataset["foo_param_dims"]

    def test_add_constraint_with_nan(self, simple_supply):
        """Expect an error if adding a constraint with a NaN in one of the expressions."""
        # add constraint with nan
        constraint_dict = {
            "foreach": ["techs", "carriers"],
            "equations": [
                {"expression": "sum(flow_out, over=[nodes, timesteps]) >= 100"}
            ],
            # "where": "carrier_out",  # <- no error would be raised with this uncommented
        }
        constraint_name = "constraint-with-nan"

        with pytest.raises(calliope.exceptions.BackendError) as error:
            simple_supply.backend.add_constraint(constraint_name, constraint_dict)

        assert check_error_or_warning(
            error,
            "(constraints, constraint-with-nan) | constraint array includes item(s) that resolves to a simple boolean. "
            "There must be a math component defined on at least one side of the equation: [('test_demand_elec', 'electricity')]",
        )

    def test_solve_warmstart_not_possible(self, simple_supply):
        with pytest.warns(exceptions.ModelWarning) as excinfo:
            simple_supply.solve(solver="cbc", force=True, warmstart=True)
        assert check_error_or_warning(excinfo, "cbc, does not support warmstart")

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

    def test_add_valid_obj(self, simple_supply):
        eq = {"expression": "bigM", "where": "True"}
        simple_supply.backend.add_objective(
            "foo", {"equations": [eq], "sense": "minimise"}
        )
        assert "foo" in simple_supply.backend.objectives
        assert not simple_supply.backend.objectives.foo.item().active

    def test_default_objective_set(self, simple_supply):
        assert simple_supply.backend.objectives.min_cost_optimisation.item().active
        assert simple_supply.backend.objective == "min_cost_optimisation"

    def test_new_objective_set(self, simple_supply_build_func):
        simple_supply_build_func.backend.add_objective(
            "foo", {"equations": [{"expression": "bigM"}], "sense": "minimise"}
        )
        simple_supply_build_func.backend.set_objective("foo")

        assert simple_supply_build_func.backend.objectives.foo.item().active
        assert not simple_supply_build_func.backend.objectives.min_cost_optimisation.item().active
        assert simple_supply_build_func.backend.objective == "foo"

    def test_new_objective_set_log(self, caplog, simple_supply_build_func):
        caplog.set_level(logging.INFO)
        simple_supply_build_func.backend.add_objective(
            "foo", {"equations": [{"expression": "bigM"}], "sense": "minimise"}
        )
        simple_supply_build_func.backend.set_objective("foo")
        assert ":foo | Objective activated." in caplog.text
        assert ":min_cost_optimisation | Objective deactivated." in caplog.text

    @staticmethod
    def _is_fixed(val):
        return val.fixed

    def test_fix_variable(self, simple_supply):
        simple_supply.backend.fix_variable("flow_cap")
        fixed = simple_supply.backend._apply_func(
            self._is_fixed,
            simple_supply.backend.variables.flow_cap.notnull(),
            1,
            simple_supply.backend.variables.flow_cap,
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
            self._is_fixed,
            simple_supply.backend.variables.flow_cap.notnull(),
            1,
            simple_supply.backend.variables.flow_cap,
        )
        simple_supply.backend.unfix_variable("flow_cap")  # reset
        assert not fixed.sel(techs="test_demand_elec", carriers="electricity").any()
        assert fixed.where(where, other=True).all()

    def test_unfix_variable(self, simple_supply):
        simple_supply.backend.fix_variable("flow_cap")
        simple_supply.backend.unfix_variable("flow_cap")
        fixed = simple_supply.backend._apply_func(
            self._is_fixed,
            simple_supply.backend.variables.flow_cap.notnull(),
            1,
            simple_supply.backend.variables.flow_cap,
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
            self._is_fixed,
            simple_supply.backend.variables.flow_cap.notnull(),
            1,
            simple_supply.backend.variables.flow_cap,
        )
        simple_supply.backend.unfix_variable("flow_cap")  # reset
        assert fixed.sel(techs="test_demand_elec", carriers="electricity").all()
        assert not fixed.where(where).all()

    def test_save_logs(self, simple_supply, tmp_path):
        dir = tmp_path / "logs"
        simple_supply.solve(force=True, save_logs=str(dir))

        assert dir.exists()
        assert any(file.suffixes == [".pyomo", ".lp"] for file in dir.glob("*"))

    @pytest.fixture
    def new_global_expr_math(self, dummy_int):
        def _new_global_expr_math(order):
            updated_math = {
                "parameters": {"cost_new": {"default": dummy_int}},
                "global_expressions": {
                    "new_expr": {
                        "foreach": ["nodes", "techs", "costs"],
                        "where": "cost_new",
                        "equations": [{"expression": "source_cap * cost_new"}],
                    },
                    # cost_investment_source_cap exists in the pre-defined math.
                    "cost_investment_source_cap": {
                        "where": "source_cap",
                        "equations": [
                            {
                                "expression": "default_if_empty(cost_source_cap, 0) * source_cap + default_if_empty(new_expr, 0)"
                            }
                        ],
                    },
                },
            }
            if order is not None:
                updated_math["global_expressions"]["new_expr"]["order"] = order
            new_cost = {"data": dummy_int, "index": "monetary", "dims": "costs"}
            m = build_model(
                {"techs.test_supply_elec.cost_new": new_cost},
                "simple_supply,two_hours,investment_costs",
                pre_validate_math_strings=False,
                math_dict=updated_math,
            )
            return m

        return _new_global_expr_math

    def test_add_reordered_global_expression(self, new_global_expr_math):
        """Adding a new global expression with an appropriately small order should be added before a pre-defined global expression."""

        m = new_global_expr_math(-1)
        m.build(backend="pyomo")
        m.backend.verbose_strings()
        expr_to_check = (
            m.backend.get_global_expression(
                "cost_investment_source_cap", as_backend_objs=False
            )
            .to_series()
            .dropna()
        )
        new_expr_present = expr_to_check.str.contains(
            "parameters[cost_new][test_supply_elec, monetary]", regex=False
        )
        assert new_expr_present.all()

    @pytest.mark.parametrize("order", [0, None, 100])
    def test_add_reordered_global_expression_fails(self, new_global_expr_math, order):
        """Adding a new global expression without reordering will cause an error to be raised when evaluating the other global expression in which it has been referenced."""

        m = new_global_expr_math(order)
        with pytest.raises(
            exceptions.BackendError,
            match="Trying to access a math component that is not yet defined: new_expr.",
        ):
            m.build(backend="pyomo")


class TestVerboseStrings:
    @pytest.fixture(scope="class")
    def simple_supply_longnames(self):
        m = build_model({}, "simple_supply,two_hours,investment_costs")
        m.build()
        m.backend.verbose_strings()
        assert m.backend._has_verbose_strings
        return m

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

    def test_new_build_get_constraint_as_vals_no_solve(self, simple_supply_longnames):
        constr = simple_supply_longnames.backend.get_constraint(
            "system_balance", as_backend_objs=False, eval_body=True
        )
        assert (
            constr["body"]
            .to_series()
            .dropna()
            .apply(lambda x: isinstance(x, str))
            .all()
        )

    @pytest.mark.parametrize(
        ("objname", "dims", "objtype"),
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
            "techs": "test_supply_elec",
            "carriers": "electricity",
            "timesteps": "2005-01-01 00:00",
        }

        obj = simple_supply_longnames.backend.get_constraint(
            "balance_supply_no_storage", as_backend_objs=False
        )

        assert (
            obj.sel(dims).body.item()
            == "1/parameters[flow_out_eff][test_supply_elec]*variables[flow_out][a, test_supply_elec, electricity, 2005-01-01 00:00] - variables[source_use][a, test_supply_elec, 2005-01-01 00:00]"
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
        assert (
            "parameters[cost_flow_cap][test_supply_elec, monetary]"
            in obj.sel(dims).item()
        )

        assert not obj.coords_in_name

    def test_verbose_strings_no_len(self, simple_supply_longnames):
        obj = simple_supply_longnames.backend.parameters.bigM

        assert obj.item().name == "parameters[bigM]"
        assert obj.coords_in_name


class TestPiecewiseConstraints:
    def gen_params(self, data, index=[0, 1, 2], dim="breakpoints"):
        return {
            "data_definitions": {
                "piecewise_x": {"data": data, "index": index, "dims": dim},
                "piecewise_y": {
                    "data": [0, 1, 5],
                    "index": [0, 1, 2],
                    "dims": "breakpoints",
                },
            }
        }

    @pytest.fixture(scope="class")
    def working_math(self):
        return {
            "foreach": ["nodes", "techs", "carriers"],
            "where": "[test_supply_elec] in techs AND piecewise_x AND piecewise_y",
            "x_values": "piecewise_x",
            "x_expression": "flow_cap",
            "y_values": "piecewise_y",
            "y_expression": "sum(flow_in, over=timesteps)",
            "description": "FOO",
        }

    @pytest.fixture(scope="class")
    def add_math(self):
        return {
            "parameters": {"piecewise_x": {}, "piecewise_y": {}},
            "dimensions": {
                "breakpoints": {"dtype": "integer", "iterator": "breakpoint"}
            },
        }

    @pytest.fixture(scope="class")
    def working_params(self):
        return self.gen_params([0, 5, 10])

    @pytest.fixture(scope="class")
    def length_mismatch_params(self):
        return self.gen_params([0, 10], [0, 1])

    @pytest.fixture(scope="class")
    def not_reaching_var_bound_with_breakpoint_params(self):
        return self.gen_params([0, 5, 8])

    @pytest.fixture(scope="class")
    def working_model(self, working_params, working_math, add_math):
        m = build_model(
            working_params,
            "simple_supply,two_hours,investment_costs",
            math_dict=add_math,
        )
        m.build()
        m.backend.add_piecewise_constraint("foo_piecewise", working_math)
        return m

    def test_piecewise_type(self, working_model):
        """All piecewise elements are the correct Pyomo type."""
        constr = working_model.backend.get_piecewise_constraint("foo_piecewise")
        assert (
            constr.to_series()
            .dropna()
            .apply(lambda x: isinstance(x, piecewise_sos2))
            .all()
        )

    def test_piecewise_verbose(self, working_model):
        """All piecewise elements have the full set of dimensions when verbose."""
        working_model.backend.verbose_strings()
        constr = working_model.backend.get_piecewise_constraint("foo_piecewise")
        dims = {"nodes": "a", "techs": "test_supply_elec", "carriers": "electricity"}
        constraint_item = constr.sel(dims).item()
        assert (
            str(constraint_item)
            == f"piecewise_constraints[foo_piecewise][{', '.join(dims[i] for i in constr.dims)}]"
        )

    def test_fails_on_length_mismatch(
        self, length_mismatch_params, working_math, add_math
    ):
        """Expected error when number of breakpoints on X and Y don't match."""
        m = build_model(
            length_mismatch_params,
            "simple_supply,two_hours,investment_costs",
            math_dict=add_math,
        )
        m.build()
        with pytest.raises(exceptions.BackendError) as excinfo:
            m.backend.add_piecewise_constraint("foo_piecewise_fails", working_math)
        assert check_error_or_warning(
            excinfo,
            "The number of breakpoints (2) differs from the number of function values (3)",
        )

    def test_fails_on_not_reaching_bounds(
        self, not_reaching_var_bound_with_breakpoint_params, working_math, add_math
    ):
        """Expected error when breakpoints exceed upper bound of the variable (pyomo-specific error)."""
        m = build_model(
            not_reaching_var_bound_with_breakpoint_params,
            "simple_supply,two_hours,investment_costs",
            math_dict=add_math,
        )
        m.build()
        with pytest.raises(exceptions.BackendError) as excinfo:
            m.backend.add_piecewise_constraint("foo_piecewise_fails", working_math)
        assert check_error_or_warning(
            excinfo,
            [
                "(piecewise_constraints, foo_piecewise_fails) | Errors in generating piecewise constraint: Piecewise function domain does not include the upper bound",
                "ub = 10.0 > 8.0.",
            ],
        )
        assert not check_error_or_warning(excinfo, "To avoid this error")


class TestShadowPrices:
    @pytest.fixture
    def simple_supply(self):
        m = build_model({}, "simple_supply,two_hours,investment_costs")
        m.build()
        return m

    @pytest.fixture
    def supply_milp(self):
        m = build_model({}, "supply_milp,two_hours,investment_costs")
        m.build()
        return m

    @pytest.fixture
    def simple_supply_with_yaml_shadow_prices(self):
        m = build_model({}, "simple_supply,two_hours,investment_costs,shadow_prices")
        m.build()
        return m

    @pytest.fixture
    def simple_supply_yaml(self):
        m = build_model({}, "simple_supply,two_hours,investment_costs,shadow_prices")
        m.build()
        return m

    @pytest.fixture
    def simple_supply_yaml_invalid(self):
        m = build_model(
            {},
            "simple_supply,two_hours,investment_costs,shadow_prices_invalid_constraint",
        )
        m.build()
        return m

    @pytest.fixture
    def supply_milp_yaml(self):
        m = build_model({}, "supply_milp,two_hours,investment_costs,shadow_prices")
        m.build()
        return m

    def test_default_to_deactivated(self, simple_supply):
        assert not simple_supply.backend.shadow_prices.is_active

    def test_available_constraints(self, simple_supply):
        assert set(simple_supply.backend.shadow_prices.available_constraints) == set(
            simple_supply.backend.constraints.data_vars
        )

    def test_activate_continuous_model(self, simple_supply):
        simple_supply.backend.shadow_prices.activate()
        assert simple_supply.backend.shadow_prices.is_active

    def test_activate_milp_model(self, supply_milp):
        with pytest.warns(exceptions.BackendWarning):
            supply_milp.backend.shadow_prices.activate()
        assert not supply_milp.backend.shadow_prices.is_active

    def test_deactivate(self, simple_supply):
        simple_supply.backend.shadow_prices.activate()
        simple_supply.backend.shadow_prices.deactivate()
        assert not simple_supply.backend.shadow_prices.is_active

    def test_get_shadow_price(self, simple_supply):
        simple_supply.backend.shadow_prices.activate()
        simple_supply.solve(solver="glpk")
        shadow_prices = simple_supply.backend.shadow_prices.get("system_balance")
        assert shadow_prices.notnull().all()

    def test_get_shadow_price_some_nan(self, simple_supply):
        simple_supply.backend.shadow_prices.activate()
        simple_supply.solve(solver="glpk")
        shadow_prices = simple_supply.backend.shadow_prices.get("balance_demand")
        assert shadow_prices.notnull().any()
        assert shadow_prices.isnull().any()

    def test_shadow_prices_deactivated_with_cbc(self, simple_supply):
        simple_supply.backend.shadow_prices.activate()
        with pytest.warns(exceptions.ModelWarning) as warning:
            simple_supply.solve(solver="cbc")

        assert check_error_or_warning(warning, "Switching off shadow price tracker")
        assert not simple_supply.backend.shadow_prices.is_active
        shadow_prices = simple_supply.backend.shadow_prices.get("system_balance")
        assert shadow_prices.isnull().all()

    def test_yaml_continuous_model_tracked(self, simple_supply_yaml):
        # before solve, there are no constraints to track
        assert not simple_supply_yaml.backend.shadow_prices.tracked

        simple_supply_yaml.solve(solver="glpk")

        assert simple_supply_yaml.backend.shadow_prices.tracked == {
            "system_balance",
            "balance_demand",
        }

    def test_yaml_continuous_model_result(self, simple_supply_yaml):
        m = simple_supply_yaml
        m.solve(solver="glpk")
        assert m.results["shadow_price_system_balance"].sum().item() == pytest.approx(
            0.0005030505
        )
        assert m.results["shadow_price_balance_demand"].sum().item() == pytest.approx(
            0.0005030505
        )

    def test_yaml_milp_model(self, supply_milp_yaml):
        assert not supply_milp_yaml.backend.shadow_prices.is_active

    def test_yaml_with_invalid_constraint(self, simple_supply_yaml_invalid):
        m = simple_supply_yaml_invalid
        with pytest.warns(exceptions.ModelWarning) as warning:
            m.solve()
        assert check_error_or_warning(
            warning, "Invalid constraints {'flow_cap_max_foobar'}"
        )
        # Since we listed only one (invalid) constraint, tracking should not be active
        assert not m.backend.shadow_prices.is_active
