import logging

import calliope.exceptions as exceptions
import gurobipy
import numpy as np
import pytest  # noqa: F401
import xarray as xr

from .common.util import build_test_model as build_model
from .common.util import check_error_or_warning


class TestNewBackend:
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
        assert param.attrs == {
            "obj_type": "parameters",
            "is_result": 0,
            "original_dtype": np.dtype("float64"),
            "references": {"flow_in_inc_eff"},
            "coords_in_name": False,
        }

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
        assert expr.attrs == {
            "obj_type": "global_expressions",
            "references": {"cost"},
            "description": "The installation costs of a technology, including annualised investment costs and annual maintenance costs.",
            "unit": "cost",
            "coords_in_name": False,
        }

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
        assert constr.attrs == {
            "obj_type": "constraints",
            "references": set(),
            "description": "Set the global carrier balance of the optimisation problem by fixing the total production of a given carrier to equal the total consumption of that carrier at every node in every timestep.",
            "coords_in_name": False,
        }

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
