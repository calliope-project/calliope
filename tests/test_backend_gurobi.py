import importlib
import logging

import calliope.exceptions as exceptions
import numpy as np
import pandas as pd
import pytest  # noqa: F401
import xarray as xr

from .common.util import build_test_model as build_model
from .common.util import check_error_or_warning

if importlib.util.find_spec("gurobipy") is not None:
    import gurobipy


class TestNewBackend:
    pytest.importorskip("gurobipy")

    LOGGER = logging.getLogger("calliope.backend.backend_model")

    @pytest.fixture(scope="class")
    def simple_supply_longnames(self):
        m = build_model({}, "simple_supply,two_hours,investment_costs")
        m.build(backend="gurobi")
        m.backend.verbose_strings()
        return m

    @pytest.fixture(scope="class")
    def simple_supply_gurobi(self):
        m = build_model({}, "simple_supply,two_hours,investment_costs")
        m.build(backend="gurobi")
        m.solve()
        return m

    @pytest.fixture
    def temp_path(self, tmpdir_factory):
        return tmpdir_factory.mktemp("custom_math")

    def test_new_build_has_backend(self, simple_supply_gurobi):
        assert hasattr(simple_supply_gurobi, "backend")

    def test_new_build_optimal(self, simple_supply_gurobi):
        assert hasattr(simple_supply_gurobi, "results")
        assert (
            simple_supply_gurobi._model_data.attrs["termination_condition"] == "optimal"
        )

    @pytest.mark.parametrize(
        "component_type", ["variable", "global_expression", "parameter", "constraint"]
    )
    def test_new_build_get_missing_component(
        self, simple_supply_gurobi, component_type
    ):
        with pytest.raises(KeyError):
            getattr(simple_supply_gurobi.backend, f"get_{component_type}")("foo")

    def test_new_build_get_variable(self, simple_supply_gurobi):
        var = simple_supply_gurobi.backend.get_variable("flow_cap")
        assert (
            var.to_series().dropna().apply(lambda x: isinstance(x, gurobipy.Var)).all()
        )

    def test_new_build_get_variable_as_vals(self, simple_supply_gurobi):
        var = simple_supply_gurobi.backend.get_variable(
            "flow_cap", as_backend_objs=False
        )
        assert (
            not var.to_series()
            .dropna()
            .apply(lambda x: isinstance(x, gurobipy.Var))
            .any()
        )

    def test_new_build_get_parameter(self, simple_supply_gurobi):
        param = simple_supply_gurobi.backend.get_parameter("flow_in_eff")
        assert isinstance(param.item(), float)

    def test_new_build_get_parameter_as_vals(self, simple_supply_gurobi):
        param = simple_supply_gurobi.backend.get_parameter(
            "flow_in_eff", as_backend_objs=False
        )
        assert param.dtype == np.dtype("float64")

    def test_new_build_get_global_expression(self, simple_supply_gurobi):
        expr = simple_supply_gurobi.backend.get_global_expression("cost_investment")
        assert (
            expr.to_series()
            .dropna()
            .apply(lambda x: isinstance(x, gurobipy.LinExpr))
            .all()
        )

    def test_new_build_get_global_expression_as_str(self, simple_supply_gurobi):
        expr = simple_supply_gurobi.backend.get_global_expression(
            "cost", as_backend_objs=False
        )
        assert expr.to_series().dropna().apply(lambda x: isinstance(x, str)).all()

    def test_new_build_get_global_expression_as_vals(self, simple_supply_gurobi):
        expr = simple_supply_gurobi.backend.get_global_expression(
            "cost", as_backend_objs=False, eval_body=True
        )
        assert (
            expr.to_series().dropna().apply(lambda x: isinstance(x, (float, int))).all()
        )

    def test_new_build_get_constraint(self, simple_supply_gurobi):
        constr = simple_supply_gurobi.backend.get_constraint("system_balance")
        assert (
            constr.to_series()
            .dropna()
            .apply(lambda x: isinstance(x, gurobipy.Constr))
            .all()
        )

    def test_new_build_get_constraint_as_str(self, simple_supply_gurobi):
        with pytest.raises(exceptions.BackendError) as excinfo:
            simple_supply_gurobi.backend.get_constraint(
                "system_balance", as_backend_objs=False
            )
        assert check_error_or_warning(
            excinfo, "Cannot return a Gurobi constraint in string format"
        )

    def test_new_build_get_constraint_as_vals(self, simple_supply_gurobi):
        with pytest.raises(exceptions.BackendError) as excinfo:
            simple_supply_gurobi.backend.get_constraint(
                "system_balance", as_backend_objs=False, eval_body=True
            )
        assert check_error_or_warning(
            excinfo, "Cannot return a Gurobi constraint in string format"
        )

    def test_solve_non_optimal(self, simple_supply_gurobi):
        simple_supply_gurobi.backend.update_parameter(
            "sink_use_equals",
            simple_supply_gurobi.inputs.sink_use_equals.where(
                simple_supply_gurobi.inputs.techs == "test_demand_elec"
            )
            * 100,
        )
        with pytest.warns(exceptions.BackendWarning) as excinfo:
            simple_supply_gurobi.solve(force=True)

        assert check_error_or_warning(excinfo, "Model solution was non-optimal")
        assert (
            simple_supply_gurobi._model_data.attrs["termination_condition"]
            == "infeasible"
        )
        assert not simple_supply_gurobi.results.data_vars

    def test_raise_error_on_preexistence_same_type(self, simple_supply_gurobi):
        with pytest.raises(exceptions.BackendError) as excinfo:
            simple_supply_gurobi.backend.add_parameter("flow_out_eff", xr.DataArray(1))

        assert check_error_or_warning(
            excinfo,
            "Trying to add already existing `flow_out_eff` to backend model parameters.",
        )

    def test_raise_error_on_preexistence_diff_type(self, simple_supply_gurobi):
        with pytest.raises(exceptions.BackendError) as excinfo:
            simple_supply_gurobi.backend.add_parameter("flow_out", xr.DataArray(1))

        assert check_error_or_warning(
            excinfo,
            "Trying to add already existing *variable* `flow_out` as a backend model *parameter*.",
        )

    def test_add_constraint(self, simple_supply_gurobi):
        # add constraint without nan
        constraint_dict = {
            "foreach": ["techs", "carriers"],
            "equations": [
                {"expression": "sum(flow_out, over=[nodes, timesteps]) >= 100"}
            ],
            "where": "carrier_out",  # <- no error is raised because of this
        }
        constraint_name = "constraint-without-nan"

        simple_supply_gurobi.backend.add_constraint(constraint_name, constraint_dict)

        assert (
            simple_supply_gurobi.backend.get_constraint(constraint_name).name
            == constraint_name
        )

    def test_add_global_expression(self, simple_supply_gurobi):
        # add expression without nan
        expression_dict = {
            "foreach": ["techs", "carriers"],
            "equations": [{"expression": "sum(flow_out, over=[nodes, timesteps])"}],
            "where": "carrier_out",  # <- no error is raised because of this
        }
        expression_name = "expression-without-nan"

        # add expression with nan
        simple_supply_gurobi.backend.add_global_expression(
            expression_name, expression_dict
        )

        assert (
            simple_supply_gurobi.backend.get_global_expression(expression_name).name
            == expression_name
        )

    @pytest.mark.parametrize(
        ["component", "eq"],
        [("global_expressions", "flow_cap + 1"), ("constraints", "flow_cap >= 1")],
    )
    def test_add_allnull_expr_or_constr(self, simple_supply_gurobi, component, eq):
        adder = getattr(
            simple_supply_gurobi.backend, "add_" + component.removesuffix("s")
        )
        constr_dict = {
            "foreach": ["nodes", "techs"],
            "where": "True",
            "equations": [{"expression": eq, "where": "False"}],
        }
        adder("foo", constr_dict)

        assert "foo" not in simple_supply_gurobi.backend._dataset.data_vars.keys()

    def test_add_allnull_param_no_shape(self, simple_supply_gurobi):
        simple_supply_gurobi.backend.add_parameter("foo", xr.DataArray(np.nan))

        assert simple_supply_gurobi.backend._dataset["foo"].isnull().all()
        del simple_supply_gurobi.backend._dataset["foo"]

    def test_add_allnull_param_with_shape(self, simple_supply_gurobi):
        nan_array = simple_supply_gurobi._model_data.flow_cap_max.where(lambda x: x < 0)
        simple_supply_gurobi.backend.add_parameter("foo", nan_array)

        assert simple_supply_gurobi.backend._dataset["foo"].isnull().all()
        del simple_supply_gurobi.backend._dataset["foo"]

    def test_add_allnull_var(self, simple_supply_gurobi):
        simple_supply_gurobi.backend.add_variable(
            "foo", {"foreach": ["nodes"], "where": "False"}
        )
        assert "foo" not in simple_supply_gurobi.backend._dataset.data_vars.keys()

    def test_add_allnull_obj(self, simple_supply_gurobi):
        eq = {"expression": "bigM", "where": "False"}
        simple_supply_gurobi.backend.add_objective(
            "foo", {"equations": [eq, eq], "sense": "minimise"}
        )
        assert "foo" not in simple_supply_gurobi.backend._dataset.data_vars.keys()

    def test_add_two_same_obj(self, simple_supply_gurobi):
        eq = {"expression": "bigM", "where": "True"}
        with pytest.raises(exceptions.BackendError) as excinfo:
            simple_supply_gurobi.backend.add_objective(
                "foo", {"equations": [eq, eq], "sense": "minimise"}
            )
        assert check_error_or_warning(
            excinfo,
            "objectives:foo:1 | trying to set two equations for the same component.",
        )

    def test_add_valid_obj(self, simple_supply_gurobi):
        eq = {"expression": "bigM", "where": "True"}
        simple_supply_gurobi.backend.add_objective(
            "foo", {"equations": [eq], "sense": "minimise"}
        )
        assert "foo" in simple_supply_gurobi.backend.objectives

    def test_object_string_representation(self, simple_supply_gurobi):
        assert (
            simple_supply_gurobi.backend.variables.flow_out.sel(
                nodes="a",
                techs="test_supply_elec",
                carriers="electricity",
                timesteps="2005-01-01 00:00",
            )
            .item()
            .VarName
        ).startswith("C")
        assert not simple_supply_gurobi.backend.variables.flow_out.coords_in_name

    @pytest.mark.parametrize(
        ["objname", "dims", "objtype", "namegetter"],
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
                "VarName",
            ),
            (
                "system_balance",
                {
                    "nodes": "a",
                    "carriers": "electricity",
                    "timesteps": "2005-01-01 00:00",
                },
                "constraints",
                "ConstrName",
            ),
        ],
    )
    def test_verbose_strings(
        self, simple_supply_longnames, objname, dims, objtype, namegetter
    ):
        obj = simple_supply_longnames.backend._dataset[objname]
        assert (
            getattr(obj.sel(dims).item(), namegetter)
            == f"{objname}[{','.join(dims[i] for i in obj.dims)}]"
        )
        assert obj.attrs["coords_in_name"]

    def test_verbose_strings_expression(self, simple_supply_longnames):
        dims = {"nodes": "a", "techs": "test_supply_elec", "costs": "monetary"}

        obj = simple_supply_longnames.backend.get_global_expression(
            "cost_investment", as_backend_objs=False
        )

        assert "flow_cap[a,test_supply_elec,electricity]" in obj.sel(dims).item()
        # parameters are not gurobi objects, so we don't get their names in our strings
        assert "parameters[cost_interest_rate]" not in obj.sel(dims).item()

        assert not obj.coords_in_name

    def test_update_parameter(self, caplog, simple_supply_gurobi):
        caplog.set_level(logging.DEBUG)

        updated_param = simple_supply_gurobi.inputs.flow_out_eff * 1000
        simple_supply_gurobi.backend.update_parameter("flow_out_eff", updated_param)

        refs_to_update = [  # should be sorted alphabetically
            "balance_supply_no_storage",
            "balance_transmission",
            "flow_out_inc_eff",
        ]

        assert (
            f"The optimisation problem components {refs_to_update} will be re-built."
            in caplog.text
        )
        expected = simple_supply_gurobi.backend.get_parameter("flow_out_eff")
        assert expected.where(updated_param.notnull()).equals(updated_param)

    def test_update_parameter_one_val(self, caplog, simple_supply_gurobi):
        updated_param = 1000
        new_dims = {"techs"}
        caplog.set_level(logging.DEBUG)

        simple_supply_gurobi.backend.update_parameter("flow_out_eff", updated_param)
        refs_to_update = [  # should be sorted alphabetically
            "balance_supply_no_storage",
            "balance_transmission",
            "flow_out_inc_eff",
        ]

        assert (
            f"The optimisation problem components {refs_to_update} will be re-built."
            in caplog.text
        )

        assert (
            f"New values will be broadcast along the {new_dims} dimension(s)"
            in caplog.text
        )
        expected = simple_supply_gurobi.backend.get_parameter("flow_out_eff")
        assert (expected == updated_param).all()

    def test_update_parameter_replace_defaults(self, simple_supply_gurobi):
        updated_param = simple_supply_gurobi.inputs.flow_out_eff.fillna(0.1)

        simple_supply_gurobi.backend.update_parameter("flow_out_eff", updated_param)

        expected = simple_supply_gurobi.backend.get_parameter("flow_out_eff")
        assert expected.equals(updated_param)

    def test_update_parameter_add_dim(self, caplog, simple_supply_gurobi):
        """
        flow_out_eff doesn't have the time dimension in the simple model, we add it here.
        """
        updated_param = simple_supply_gurobi.inputs.flow_out_eff.where(
            simple_supply_gurobi.inputs.timesteps.notnull()
        )
        refs_to_update = [  # should be sorted alphabetically
            "balance_supply_no_storage",
            "balance_transmission",
            "flow_out_inc_eff",
        ]
        caplog.set_level(logging.DEBUG)

        simple_supply_gurobi.backend.update_parameter("flow_out_eff", updated_param)

        assert (
            f"The optimisation problem components {refs_to_update} will be re-built."
            in caplog.text
        )

        expected = simple_supply_gurobi.backend.get_parameter("flow_out_eff")
        assert "timesteps" in expected.dims

    def test_update_parameter_replace_undefined(self, caplog, simple_supply_gurobi):
        """source_eff isn't defined in the inputs, so is a dimensionless value in the pyomo object, assigned its default value."""
        updated_param = simple_supply_gurobi.inputs.flow_out_eff

        refs_to_update = ["balance_supply_no_storage"]
        caplog.set_level(logging.DEBUG)

        simple_supply_gurobi.backend.update_parameter("source_eff", updated_param)

        assert (
            f"The optimisation problem components {refs_to_update} will be re-built."
            in caplog.text
        )

        expected = simple_supply_gurobi.backend.get_parameter(
            "source_eff", as_backend_objs=False
        )
        default_val = simple_supply_gurobi._model_data.attrs["defaults"]["source_eff"]
        assert expected.equals(updated_param.fillna(default_val))

    def test_update_parameter_no_refs_to_update(self, simple_supply_gurobi):
        """flow_cap_per_storage_cap_max isn't defined in the inputs, so is a dimensionless value in the pyomo object, assigned its default value.

        Updating it doesn't change the model in any way, because none of the existing constraints/expressions depend on it.
        Therefore, no warning is raised.
        """
        updated_param = 1

        simple_supply_gurobi.backend.update_parameter(
            "flow_cap_per_storage_cap_max", updated_param
        )

        expected = simple_supply_gurobi.backend.get_parameter(
            "flow_cap_per_storage_cap_max"
        )
        assert expected == 1

    @pytest.mark.parametrize("bound", ["min", "max"])
    def test_update_variable_single_bound_single_val(self, simple_supply_gurobi, bound):
        translator = {"min": "lb", "max": "ub"}

        simple_supply_gurobi.backend.update_variable_bounds("flow_out", **{bound: 1})

        bound_vals = simple_supply_gurobi.backend.get_variable_bounds("flow_out")[
            translator[bound]
        ]

        assert (bound_vals == 1).where(bound_vals.notnull()).all()

    def test_update_variable_bounds_single_val(self, simple_supply_gurobi):
        simple_supply_gurobi.backend.update_variable_bounds("flow_out", min=2, max=2)
        bound_vals = simple_supply_gurobi.backend.get_variable_bounds("flow_out")
        assert (bound_vals == 2).where(bound_vals.notnull()).all().all()

    def test_update_variable_single_bound_multi_val(self, caplog, simple_supply_gurobi):
        caplog.set_level(logging.INFO)
        bound_array = simple_supply_gurobi.inputs.sink_use_equals.sel(
            techs="test_demand_elec"
        )
        simple_supply_gurobi.backend.update_variable_bounds("flow_in", min=bound_array)
        bound_vals = simple_supply_gurobi.backend.get_variable_bounds("flow_in").lb
        assert "New `min` bounds will be broadcast" in caplog.text
        assert bound_vals.equals(
            bound_array.where(bound_vals.notnull()).transpose(*bound_vals.dims)
        )

    def test_update_variable_error_update_parameter_instead(self, simple_supply_gurobi):
        with pytest.raises(exceptions.BackendError) as excinfo:
            simple_supply_gurobi.backend.update_variable_bounds("flow_cap", min=1)
        assert check_error_or_warning(
            excinfo,
            "Cannot update variable bounds that have been set by parameters."
            " Use `update_parameter('flow_cap_min')` to update the min bound of flow_cap.",
        )

    @staticmethod
    def _is_fixed(val):
        if pd.notnull(val):
            return val.lb == val.ub
        else:
            return np.nan

    def test_fix_variable(self, simple_supply_gurobi):
        simple_supply_gurobi.build(backend="gurobi", force=True)
        simple_supply_gurobi.solve(force=True)
        simple_supply_gurobi.backend.fix_variable("flow_cap")
        fixed = simple_supply_gurobi.backend._apply_func(
            self._is_fixed, simple_supply_gurobi.backend.variables.flow_cap
        )
        assert fixed.where(fixed.notnull()).all()

        # reset
        simple_supply_gurobi.build(backend="gurobi", force=True)
        simple_supply_gurobi.solve(force=True)

    def test_fix_variable_where(self, simple_supply_gurobi):
        simple_supply_gurobi.build(backend="gurobi", force=True)
        simple_supply_gurobi.solve(force=True)

        where = (
            simple_supply_gurobi.inputs.flow_cap_max.notnull()
            & simple_supply_gurobi.backend.variables.flow_cap.notnull()
        )
        simple_supply_gurobi.backend.fix_variable("flow_cap", where=where)
        fixed = simple_supply_gurobi.backend._apply_func(
            self._is_fixed, simple_supply_gurobi.backend.variables.flow_cap
        )

        assert not fixed.sel(techs="test_demand_elec", carriers="electricity").any()
        assert fixed.where(where, other=True).all()
        # reset
        simple_supply_gurobi.build(backend="gurobi", force=True)
        simple_supply_gurobi.solve(force=True)

    def test_fix_variable_before_solve(self, simple_supply_longnames):
        with pytest.raises(exceptions.BackendError) as excinfo:
            simple_supply_longnames.backend.fix_variable("flow_cap")

        assert check_error_or_warning(
            excinfo,
            "Cannot fix variable values without already having solved the model successfully.",
        )

    def test_unfix_variable(self, simple_supply_gurobi):
        with pytest.raises(exceptions.BackendError) as excinfo:
            simple_supply_gurobi.backend.unfix_variable("flow_cap")

        assert check_error_or_warning(
            excinfo,
            "Cannot unfix a variable using the Gurobi backend; you will need to rebuild your backend or update variable bounds to match the original bounds.",
        )


class TestShadowPrices:
    @pytest.fixture(scope="function")
    def simple_supply(self):
        m = build_model({}, "simple_supply,two_hours,investment_costs")
        m.build(backend="gurobi")
        return m

    @pytest.fixture(scope="function")
    def supply_milp(self):
        m = build_model({}, "supply_milp,two_hours,investment_costs")
        m.build(backend="gurobi")
        return m

    def test_always_active_in_gurobi(self, simple_supply):
        assert simple_supply.backend.shadow_prices.is_active

    def test_activate(self, simple_supply):
        simple_supply.backend.shadow_prices.activate()
        assert simple_supply.backend.shadow_prices.is_active

    def test_deactivate_doesnt_do_anything(self, simple_supply):
        simple_supply.backend.shadow_prices.activate()
        simple_supply.backend.shadow_prices.deactivate()
        assert simple_supply.backend.shadow_prices.is_active

    def test_get_shadow_price(self, simple_supply):
        simple_supply.backend.shadow_prices.activate()
        simple_supply.solve()
        shadow_prices = simple_supply.backend.shadow_prices.get("system_balance")
        assert shadow_prices.notnull().all()

    def test_get_shadow_price_some_nan(self, simple_supply):
        simple_supply.backend.shadow_prices.activate()
        simple_supply.solve()
        shadow_prices = simple_supply.backend.shadow_prices.get("balance_demand")
        assert shadow_prices.notnull().any()
        assert shadow_prices.isnull().any()

    def test_get_shadow_price_empty_milp(self, supply_milp):
        supply_milp.backend.shadow_prices.activate()
        supply_milp.solve()
        shadow_prices = supply_milp.backend.shadow_prices.get("system_balance")
        assert shadow_prices.isnull().all()
