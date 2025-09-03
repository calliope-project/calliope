import logging

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import calliope
import calliope.backend
from calliope.util import DATETIME_DTYPE

from .common.util import build_test_model as build_model
from .common.util import check_error_or_warning


@pytest.fixture(scope="class", params=["pyomo", "gurobi"])
def backend(request) -> str:
    if request.param == "gurobi":
        pytest.importorskip("gurobipy")
    return request.param


@pytest.fixture(scope="class")
def built_model_cls_longnames(backend) -> calliope.Model:
    m = build_model({}, "simple_supply,two_hours,investment_costs")
    m.build(backend=backend)
    m.backend.verbose_strings()
    return m


@pytest.fixture
def built_model_func_longnames(backend) -> calliope.Model:
    m = build_model(
        {}, "simple_supply,two_hours,investment_costs", pre_validate_math_strings=False
    )
    m.build(backend=backend)
    m.backend.verbose_strings()
    return m


@pytest.fixture
def solved_model_func(backend) -> calliope.Model:
    m = build_model(
        {}, "simple_supply,two_hours,investment_costs", pre_validate_math_strings=False
    )
    m.build(backend=backend)
    m.solve()
    return m


@pytest.fixture(scope="class")
def infeasible_model_cls(backend) -> calliope.Model:
    """If we increase demand, the solved model becomes infeasible"""
    m = build_model({}, "simple_supply,two_hours,investment_costs")
    m.build(backend=backend)
    m.backend.update_input(
        "sink_use_equals",
        m.inputs.sink_use_equals.where(m.inputs.techs == "test_demand_elec") * 100,
    )
    with pytest.warns(
        calliope.exceptions.BackendWarning, match="Model solution was non-optimal"
    ):
        m.solve(force=True)
    return m


@pytest.fixture(scope="class")
def solved_model_cls(backend) -> calliope.Model:
    m = build_model({}, "simple_supply,two_hours,investment_costs")
    m.build(backend=backend)
    m.solve()
    return m


@pytest.fixture(scope="class")
def solved_model_cls_multi_type_global_expression(backend) -> calliope.Model:
    new_global_expression = {
        "global_expressions.multi_type": {
            "foreach": ["nodes", "techs", "carriers"],
            "equations": [
                {  # expression
                    "where": "[test_demand_elec] in techs and [a] in nodes",
                    "expression": "1 + flow_cap",
                },
                {  # decision variable
                    "where": "[test_supply_elec] in techs and [a] in nodes",
                    "expression": "flow_cap",
                },
                {  # simple numeric
                    "where": "[test_demand_elec] in techs and [b] in nodes",
                    "expression": "1",
                },
                {  # mutable parameter
                    "where": "[test_supply_elec] in techs and [b] in nodes",
                    "expression": "flow_cap_max",
                },
            ],
        }
    }
    m = build_model(
        {"techs.test_link_a_b_heat.active": False},
        "simple_supply,two_hours,investment_costs",
        math_dict=new_global_expression,
    )
    m.build(backend=backend)
    m.solve()
    return m


@pytest.fixture
def built_model_func_updated_cost_flow_cap(backend, dummy_int: int) -> calliope.Model:
    m = build_model(
        {}, "simple_supply,two_hours,investment_costs", pre_validate_math_strings=False
    )
    m.build(backend=backend)
    m.backend.verbose_strings()
    m.backend.update_input("cost_flow_cap", dummy_int)
    return m


@pytest.fixture(scope="class")
def solved_model_milp_cls(backend) -> calliope.Model:
    m = build_model({}, "supply_milp,two_hours,investment_costs")
    m.build(backend=backend)
    m.solve()
    return m


class TestBackend:
    def test_has_backend(self, solved_model_cls):
        """building a model produces a backend."""
        assert hasattr(solved_model_cls, "backend")

    def test_to_lp(self, solved_model_cls, tmp_path):
        """Any built backend should be able to store an LP representation of the math."""
        filepath = tmp_path / "out.lp"
        solved_model_cls.backend.to_lp(filepath)


class TestOptimality:
    def test_optimal(self, solved_model_cls):
        """Solved model is optimal."""
        assert hasattr(solved_model_cls, "results")
        assert solved_model_cls.runtime.termination_condition == "optimal"

    def test_solve_non_optimal(self, infeasible_model_cls):
        """Infeasible models have no results"""
        assert infeasible_model_cls.runtime.termination_condition == "infeasible"
        assert not infeasible_model_cls.results.data_vars


class TestGetters:
    @pytest.fixture(scope="class")
    def variable(self, solved_model_cls):
        return solved_model_cls.backend.get_variable("flow_cap")

    @pytest.fixture(scope="class")
    def parameter(self, solved_model_cls):
        return solved_model_cls.backend.get_parameter("flow_in_eff")

    @pytest.fixture(scope="class")
    def constraint(self, solved_model_cls):
        return solved_model_cls.backend.get_constraint("system_balance")

    @pytest.fixture(scope="class")
    def global_expression(self, solved_model_cls):
        return solved_model_cls.backend.get_global_expression("cost_investment")

    @pytest.fixture(scope="class")
    def lookup(self, solved_model_cls):
        return solved_model_cls.backend.get_lookup("cap_method")

    @pytest.mark.parametrize(
        "component_type", ["variable", "global_expression", "parameter", "constraint"]
    )
    def test_get_missing_component(self, solved_model_cls, component_type):
        """Try and fail to get a component that doesn't exist in the backend dataset."""
        with pytest.raises(KeyError):
            getattr(solved_model_cls.backend, f"get_{component_type}")("foo")

    def test_get_variable_attrs(self, variable):
        """Check a decision variable has all expected attributes."""
        expected_keys = {
            "obj_type",
            "references",
            "coords_in_name",
            "title",
            "description",
            "unit",
            "default",
            "domain",
        }
        assert not expected_keys.symmetric_difference(variable.attrs.keys())

    def test_get_variable_obj_type(self, variable):
        """Check a decision variable has the correct obj_type."""
        assert variable.attrs["obj_type"] == "variables"

    def test_get_variable_refs(self, variable, solved_model_cls):
        """Check a decision variable has all expected references to other math components."""
        assert variable.attrs["references"] == {
            "flow_in_max",
            "flow_out_max",
            "cost_investment_flow_cap",
            "symmetric_transmission",
        }

    def test_get_variable_default(self, variable):
        """Check a decision variable has expected default val."""
        assert variable.attrs["default"] == 0

    def test_get_variable_coords_in_name(self, variable):
        """Check a decision variable does not have verbose strings activated."""
        assert variable.attrs["coords_in_name"] is False

    def test_get_parameter(self, parameter):
        """Check a parameter has all expected attributes."""
        assert parameter.attrs == {
            "obj_type": "parameters",
            "references": {"flow_in_inc_eff"},
            "coords_in_name": False,
            "title": "Inflow efficiency",
            "description": (
                "Conversion efficiency from `source`/`flow_in` (tech dependent) into the technology. "
                "Set as value between 1 (no loss) and 0 (all lost)."
            ),
            "default": 1.0,
            "unit": "unitless.",
        }

    def test_get_parameter_as_vals(self, solved_model_cls):
        """flow_in_eff is all float values when resolving the backend parameter objects."""
        param = solved_model_cls.backend.get_parameter(
            "flow_in_eff", as_backend_objs=False
        )
        assert param.dtype == np.dtype("float64")

    def test_get_parameter_as_vals_integers(self, solved_model_func, dummy_int):
        """Timestep values should have a datetime dtype when resolving the backend parameter objects."""
        solved_model_func.backend.add_parameter(
            "important_number", xr.DataArray(dummy_int), {}
        )
        param = solved_model_func.backend.get_parameter(
            "important_number", as_backend_objs=False
        )
        assert param.equals(xr.DataArray(dummy_int))

    def test_get_global_expression_attrs(self, global_expression):
        """Check a global expression has all expected attributes."""
        expected_keys = {
            "obj_type",
            "references",
            "title",
            "description",
            "unit",
            "default",
            "coords_in_name",
        }
        assert not expected_keys.symmetric_difference(global_expression.attrs.keys())

    def test_get_global_expression_obj_type(self, global_expression):
        """Check a global expression has expected obj_type."""
        assert global_expression.attrs["obj_type"] == "global_expressions"

    def test_get_global_expression_refs(self, global_expression):
        """Check a global expression has all expected math component refs."""
        assert global_expression.attrs["references"] == {
            "cost_investment_annualised",
            "cost_operation_fixed",
        }

    def test_get_global_expression_default(self, global_expression):
        """Check a global expression has expected default."""
        assert global_expression.attrs["default"] == 0

    def test_get_global_expression_coords_in_name(self, global_expression):
        """Check a global expression does not have verbose strings activated."""
        assert global_expression.attrs["coords_in_name"] is False

    def test_get_global_expression_as_str(self, solved_model_cls):
        """Resolving backend global expressions produces strings."""
        expr = solved_model_cls.backend.get_global_expression(
            "cost", as_backend_objs=False
        )
        assert expr.to_series().dropna().apply(lambda x: isinstance(x, str)).all()

    def test_get_global_expression_as_vals(self, solved_model_cls):
        """Evaluating backend global expressions of solved models produces their values in the optimal solution."""
        expr = solved_model_cls.backend.get_global_expression(
            "cost", as_backend_objs=False, eval_body=True
        )
        assert (
            expr.to_series().dropna().apply(lambda x: isinstance(x, float | int)).all()
        )

    def test_get_global_expression_as_vals_multitype(
        self, solved_model_cls_multi_type_global_expression
    ):
        """Evaluating backend global expressions of solved models produces their values in the optimal solution, no matter the type of the object."""

        expr = (
            solved_model_cls_multi_type_global_expression.backend.get_global_expression(
                "multi_type", as_backend_objs=False, eval_body=True
            )
        ).dropna("techs", how="all")
        expected = (
            pd.Series(
                {
                    ("a", "test_demand_elec", "electricity"): 6.0,  # flow_cap + 1
                    ("a", "test_supply_elec", "electricity"): 5.0,  # flow_cap
                    ("b", "test_demand_elec", "electricity"): 1.0,  # numeric 1
                    ("b", "test_supply_elec", "electricity"): 10.0,  # flow_cap_max
                }
            )
            .rename_axis(index=["nodes", "techs", "carriers"])
            .to_xarray()
            .broadcast_like(expr)
        )
        assert expr.equals(expected)

    def test_get_global_expression_as_vals_no_solve(self, built_model_cls_longnames):
        """Evaluating backend global expressions of built but not solved models produces their string representation."""
        expr = built_model_cls_longnames.backend.get_global_expression(
            "cost", as_backend_objs=False, eval_body=True
        )
        assert (
            expr.where(expr != "nan")
            .to_series()
            .dropna()
            .apply(lambda x: isinstance(x, str))
            .all()
        )

    def test_get_global_objective_as_str(self, solved_model_cls):
        """Resolving backend objectives produces strings."""
        obj = solved_model_cls.backend.get_objective(
            "min_cost_optimisation", as_backend_objs=False
        )
        assert isinstance(obj.item(), str)

    def test_get_objective_as_val(self, solved_model_cls):
        """Evaluating backend objective of solved models produces the objective function value in the optimal solution."""
        obj = solved_model_cls.backend.get_objective(
            "min_cost_optimisation", as_backend_objs=False, eval_body=True
        )
        assert isinstance(obj.item(), float)

    def test_get_objective_as_vals_no_solve(self, built_model_cls_longnames):
        """Evaluating backend objective of built but not solved models produces their string representation."""
        obj = built_model_cls_longnames.backend.get_objective(
            "min_cost_optimisation", as_backend_objs=False, eval_body=True
        )
        assert isinstance(obj.item(), str)

    def test_timeseries_dtype(self, built_model_cls_longnames):
        """Getting verbose strings leads to the timeseries being stringified then converted back to datetime."""
        expr = built_model_cls_longnames.backend.get_global_expression(
            "flow_out_inc_eff", as_backend_objs=False, eval_body=True
        )
        assert (
            expr.where(expr != "nan").to_series().dropna().str.contains("2005-").all()
        )
        assert (
            built_model_cls_longnames.backend._dataset.timesteps.dtype.kind
            == DATETIME_DTYPE
        )
        assert (
            built_model_cls_longnames.backend.inputs.timesteps.dtype.kind
            == DATETIME_DTYPE
        )

    def test_get_constraint_attrs(self, constraint):
        """Check a constraint has all expected attributes."""
        expected_keys = {
            "obj_type",
            "references",
            "description",
            "coords_in_name",
            "title",
        }

        assert not expected_keys.symmetric_difference(constraint.attrs.keys())

    def test_get_constraint_obj_type(self, constraint):
        """Check a constraint has expected object type."""
        assert constraint.attrs["obj_type"] == "constraints"

    def test_get_constraint_refs(self, constraint):
        """Check a constraint has expected refs to other math components (zero for constraints)."""
        assert constraint.attrs["references"] == set()

    def test_get_constraint_coords_in_name(self, constraint):
        """Check a constraint does not have verbose strings activated."""
        assert constraint.attrs["coords_in_name"] is False

    def test_get_lookup_obj_type(self, lookup):
        """Check that a lookup has the right object type set."""
        assert lookup.attrs["obj_type"] == "lookups"

    def test_get_lookup_refs(self, lookup):
        """Check that a lookup has zero refs to other components."""
        assert not lookup.attrs["references"]

    def test_get_lookup_expected_keys(self, lookup):
        """Check that a lookup displays its schema attributes."""
        expected_keys = {
            "obj_type",
            "references",
            "description",
            "coords_in_name",
            "title",
            "default",
            "dtype",
        }
        assert not expected_keys.symmetric_difference(lookup.attrs.keys())


class TestAdders:
    def test_raise_error_on_preexistence_same_type(self, solved_model_func):
        """Cannot add a parameter if one with the same name already exists"""
        with pytest.raises(calliope.exceptions.BackendError) as excinfo:
            solved_model_func.backend.add_parameter("flow_out_eff", xr.DataArray(1), {})

        assert check_error_or_warning(
            excinfo,
            "Trying to add already existing `flow_out_eff` to backend model parameters.",
        )

    def test_raise_error_on_preexistence_diff_type(self, solved_model_func):
        """Cannot add a component if one with the same name already exists, even if it is a different component type."""
        with pytest.raises(calliope.exceptions.BackendError) as excinfo:
            solved_model_func.backend.add_parameter("flow_out", xr.DataArray(1), {})

        assert check_error_or_warning(
            excinfo,
            "Trying to add already existing *variable* `flow_out` as a backend model *parameter*.",
        )

    def test_add_constraint(self, solved_model_func):
        """A very simple constraint: For each tech, let the annual and regional sum of `flow_out` be larger than 100.

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
        constraint_name = "constraint_without_nan"

        solved_model_func.backend.add_constraint(constraint_name, constraint_dict)

        assert (
            solved_model_func.backend.get_constraint(constraint_name).name
            == constraint_name
        )

    def test_add_global_expression(self, solved_model_func):
        """A very simple expression: The annual and regional sum of `flow_out` for each tech.

        However, not every tech has the variable `flow_out`.
        How to solve it? Let the constraint be active only where flow_out exists by setting 'where' accordingly.
        """
        # add expression without nan
        expression_dict = {
            "foreach": ["techs", "carriers"],
            "equations": [{"expression": "sum(flow_out, over=[nodes, timesteps])"}],
            "where": "carrier_out",  # <- no error is raised because of this
        }
        expression_name = "expression_without_nan"

        # add expression with nan
        solved_model_func.backend.add_global_expression(
            expression_name, expression_dict
        )

        assert (
            solved_model_func.backend.get_global_expression(expression_name).name
            == expression_name
        )

    def test_raise_error_on_excess_constraint_dimensions(self, solved_model_func):
        """A very simple constraint: For each tech, let the `flow_cap` be larger than 100.

        However, we forgot to include `nodes` in `foreach`.
        With `nodes` included, this constraint should build.
        """
        # add constraint with excess dimensions
        constraint_dict = {
            # as 'nodes' is not listed here, the constraint will have excess dimensions
            "foreach": ["techs", "carriers"],
            "equations": [{"expression": "flow_cap >= 100"}],
        }
        constraint_name = "constraint_with_excess_dimensions"

        with pytest.raises(calliope.exceptions.BackendError) as error:
            solved_model_func.backend.add_constraint(constraint_name, constraint_dict)

        assert check_error_or_warning(
            error,
            f"(constraints:{constraint_name}:0, flow_cap >= 100) | The left-hand side of the equation is indexed over dimensions not present in `foreach`: {{'nodes'}}",
        )

    def test_raise_error_on_excess_expression_dimensions(self, solved_model_func):
        """A very simple expression: For each tech, add a fixed quantity to `flow_cap`.

        However, we forgot to include `nodes` in `foreach`.
        With `nodes` included, this expression would build.
        """
        # add global expression with excess dimensions
        expr_dict = {
            # as 'nodes' is not listed here, the constraint will have excess dimensions
            "foreach": ["techs", "carriers"],
            "equations": [{"expression": "flow_cap + 1"}],
        }
        expr_name = "expr_with_excess_dimensions"

        with pytest.raises(calliope.exceptions.BackendError) as error:
            solved_model_func.backend.add_global_expression(expr_name, expr_dict)

        assert check_error_or_warning(
            error,
            f"global_expressions:{expr_name}:0 | The linear expression array is indexed over dimensions not present in `foreach`: {{'nodes'}}",
        )

    def test_add_two_same_expr_nodim(self, solved_model_func):
        """Cannot set multiple equation expressions for a dimensionless global expression"""
        eq = {"expression": "bigM"}
        with pytest.raises(calliope.exceptions.BackendError) as excinfo:
            solved_model_func.backend.add_global_expression(
                "foo", {"equations": [eq, eq]}
            )
        assert check_error_or_warning(
            excinfo,
            "global_expressions:foo:1 | trying to set two equations for the same component.",
        )

    def test_add_two_same_expr_with_shape(self, solved_model_func):
        """Cannot set multiple equation expressions for a global expression with dimensions"""
        eq = {"expression": "flow_cap + 1"}
        with pytest.raises(calliope.exceptions.BackendError) as excinfo:
            solved_model_func.backend.add_global_expression(
                "foo",
                {"foreach": ["techs", "carriers", "nodes"], "equations": [eq, eq]},
            )
        assert check_error_or_warning(
            excinfo,
            "global_expressions:foo:1 | trying to set two equations for the same index",
        )

    def test_add_two_same_expr_with_shape_partial(self, solved_model_func):
        """Cannot set multiple equation expressions for any array item in a global expression array."""
        eq1 = {
            "expression": "flow_cap + 1",
            "where": "[test_supply_elec, test_demand_elec] in techs",
        }
        eq2 = {"expression": "flow_cap + 1", "where": "[test_supply_elec] in techs"}
        with pytest.raises(calliope.exceptions.BackendError) as excinfo:
            solved_model_func.backend.add_global_expression(
                "foo",
                {"foreach": ["techs", "carriers", "nodes"], "equations": [eq1, eq2]},
            )
        assert check_error_or_warning(
            excinfo,
            "global_expressions:foo:1 | trying to set two equations for the same index",
        )

    def test_add_allnull_expr(self, solved_model_func, dummy_int):
        """If `where` string resolves to False in all array elements, the component will be built with its default."""
        constr_dict = {
            "foreach": ["nodes", "techs"],
            "equations": [{"expression": "flow_cap + 1", "where": "False"}],
            "default": dummy_int,
        }
        solved_model_func.backend.add_global_expression("foo", constr_dict)

        assert solved_model_func.backend._dataset["foo"].equals(xr.DataArray(np.nan))

    def test_add_allnull_constr(self, solved_model_func):
        """If `where` string resolves to False in all array elements, the component won't be built."""
        constr_dict = {
            "foreach": ["nodes", "techs"],
            "equations": [{"expression": "flow_cap <= 1", "where": "False"}],
        }
        solved_model_func.backend.add_constraint("foo", constr_dict)

        assert solved_model_func.backend.constraints["foo"].equals(xr.DataArray(np.nan))

    def test_add_allnull_param_no_shape(self, solved_model_func):
        """If parameter is Null, the component will still be added to the backend dataset in case it is filled by another parameter later."""
        solved_model_func.backend.add_parameter("foo", xr.DataArray(np.nan), {})

        assert solved_model_func.backend.parameters.foo.equals(xr.DataArray(np.nan))

    def test_add_allnull_param_with_shape(self, solved_model_func):
        """If parameter is Null in all array elements, the component will still be added to the backend dataset in case it is filled by another parameter later."""
        nan_array = solved_model_func.inputs.flow_cap_max.where(lambda x: x < 0)
        solved_model_func.backend.add_parameter("foo", nan_array, {})

        # We keep it in the dataset since it might be fillna'd by another param later.
        assert solved_model_func.backend.parameters["foo"].equals(xr.DataArray(np.nan))

    def test_add_allnull_var(self, solved_model_func, dummy_int):
        """If `where` string resolves to False in all array elements, the component won't be built."""
        solved_model_func.backend.add_variable(
            "foo",
            {
                "foreach": ["nodes"],
                "where": "False",
                "bounds": {"min": 0, "max": 1},
                "default": dummy_int,
            },
        )
        assert solved_model_func.backend.variables["foo"].equals(xr.DataArray(np.nan))

    def test_add_allnull_obj(self, solved_model_func):
        """If `where` string resolves to False in all array elements, the component won't be built."""
        eq = {"expression": "bigM", "where": "False"}
        solved_model_func.backend.add_objective(
            "foo", {"equations": [eq, eq], "sense": "minimise"}
        )
        assert solved_model_func.backend.objectives["foo"].equals(xr.DataArray(np.nan))

    def test_add_two_same_obj(self, solved_model_func):
        """Cannot set multiple equation expressions for an objective"""
        eq = {"expression": "bigM", "where": "True"}
        with pytest.raises(calliope.exceptions.BackendError) as excinfo:
            solved_model_func.backend.add_objective(
                "foo", {"equations": [eq, eq], "sense": "minimise"}
            )
        assert check_error_or_warning(
            excinfo,
            "objectives:foo:1 | trying to set two equations for the same component.",
        )


class TestUpdateParameter:
    def test_update_input(self, solved_model_func):
        """Updating a parameter where no Null values need to be rebuilt."""
        updated_param = solved_model_func.inputs.flow_out_eff * 1000
        solved_model_func.backend.update_input("flow_out_eff", updated_param)

        expected = solved_model_func.backend.get_parameter(
            "flow_out_eff", as_backend_objs=False
        )
        assert expected.where(updated_param.notnull()).equals(updated_param)

    def test_update_input_one_val(self, caplog, solved_model_func, dummy_int: int):
        """Updating a parameter where a single value needs broadcasting to the shape of the parameter, leading to any parameter Null values being rebuilt."""
        updated_param = dummy_int
        new_dims = {"techs"}
        caplog.set_level(logging.DEBUG)

        solved_model_func.backend.update_input("flow_out_eff", updated_param)

        assert (
            f"New values will be broadcast along the {new_dims} dimension(s)"
            in caplog.text
        )
        expected = solved_model_func.backend.get_parameter(
            "flow_out_eff", as_backend_objs=False
        )
        assert (expected == dummy_int).all()

    def test_update_input_replace_defaults(self, solved_model_func):
        """Updating a parameter that only exists in the backend thanks to its existence in the model definition schema."""
        updated_param = solved_model_func.inputs.flow_out_eff.fillna(0.1)

        solved_model_func.backend.update_input("flow_out_eff", updated_param)

        expected = solved_model_func.backend.get_parameter(
            "flow_out_eff", as_backend_objs=False
        )
        assert expected.equals(updated_param)

    def test_update_input_add_dim(self, caplog, solved_model_func):
        """flow_out_eff doesn't have the time dimension in the simple model, we add it here."""
        updated_param = solved_model_func.inputs.flow_out_eff.where(
            solved_model_func.inputs.timesteps.notnull()
        )
        refs_to_update = [  # should be sorted alphabetically
            "balance_supply_no_storage",
            "balance_transmission",
            "flow_out_inc_eff",
        ]
        caplog.set_level(logging.DEBUG)

        solved_model_func.backend.update_input("flow_out_eff", updated_param)

        assert (
            f"The optimisation problem components {refs_to_update} will be re-built."
            in caplog.text
        )

        expected = solved_model_func.backend.get_parameter(
            "flow_out_eff", as_backend_objs=False
        )
        assert "timesteps" in expected.dims

    def test_update_input_replace_undefined(self, caplog, solved_model_func):
        """source_eff isn't defined in the inputs, so is a dimensionless value in the pyomo object, assigned its default value."""
        updated_param = solved_model_func.inputs.flow_out_eff

        refs_to_update = ["balance_supply_no_storage"]
        caplog.set_level(logging.DEBUG)

        solved_model_func.backend.update_input("source_eff", updated_param)

        assert (
            f"The optimisation problem components {refs_to_update} will be re-built."
            in caplog.text
        )

        expected = solved_model_func.backend.get_parameter(
            "source_eff", as_backend_objs=False
        )
        assert expected.equals(updated_param)

    @pytest.mark.parametrize("model_suffix", ["_longnames", "_updated_cost_flow_cap"])
    @pytest.mark.parametrize(
        ("expr", "kwargs"),
        [
            ("cost_investment_flow_cap", {"carriers": "electricity"}),
            ("cost_investment", {}),
            ("cost", {}),
        ],
    )
    @pytest.mark.usefixtures(
        "built_model_func_longnames", "built_model_func_updated_cost_flow_cap"
    )
    def test_update_input_expr_refs_rebuilt(
        self, request: pytest.FixtureRequest, model_suffix: str, expr: str, kwargs: dict
    ):
        """Check that parameter re-definition propagates across all cross-referenced global expressions."""
        model: calliope.Model = request.getfixturevalue(
            "built_model_func" + model_suffix
        )
        expression_string = (
            model.backend.get_global_expression(expr, as_backend_objs=False)
            .sel(techs="test_demand_elec", **kwargs)
            .astype(str)
        )
        if model_suffix.endswith("updated_cost_flow_cap"):
            assert expression_string.str.contains("test_demand_elec").all()
        else:
            assert not (expression_string.str.contains("test_demand_elec").any())

    @pytest.mark.usefixtures(
        "built_model_func_longnames", "built_model_func_updated_cost_flow_cap"
    )
    @pytest.mark.parametrize("model_suffix", ["_longnames", "_updated_cost_flow_cap"])
    def test_update_input_refs_in_obj_func(
        self, request: pytest.FixtureRequest, model_suffix: str
    ):
        """Check that parameter re-definition propagates from global expressions to objective function."""
        model: calliope.Model = request.getfixturevalue(
            "built_model_func" + model_suffix
        )
        # TODO: update once we have a `get_objective` method that is backend-agnostic
        if isinstance(model.backend, calliope.backend.GurobiBackendModel):
            objective_string = str(
                model.backend.objectives.min_cost_optimisation.item()
            )
        elif isinstance(model.backend, calliope.backend.PyomoBackendModel):
            objective_string = str(
                model.backend.objectives.min_cost_optimisation.item().expr
            )
        if model_suffix.endswith("updated_cost_flow_cap"):
            assert "test_demand_elec" in objective_string
        else:
            # This ensures that the `updated_cost_flow_cap` test passing isn't a false negative.
            # If this fails, it means that `updated_cost_flow_cap` might be passing for reasons unrelated to a successful rebuild.
            assert "test_demand_elec" not in objective_string

    def test_update_input_no_refs_to_update(self, solved_model_func):
        """flow_cap_per_storage_cap_max isn't defined in the inputs, so is a dimensionless value in the pyomo object, assigned its default value.

        Updating it doesn't change the model in any way, because none of the existing constraints/expressions depend on it.
        Therefore, no warning is raised.
        """
        updated_param = 1

        solved_model_func.backend.update_input(
            "flow_cap_per_storage_cap_max", updated_param
        )

        expected = solved_model_func.backend.get_parameter(
            "flow_cap_per_storage_cap_max", as_backend_objs=False
        )
        assert expected == 1


class TestUpdateVariable:
    @pytest.mark.parametrize("bound", ["min", "max"])
    def test_update_variable_single_bound_single_val(
        self, solved_model_func, bound, dummy_int
    ):
        """Updating a variable lower and upper bounds one at a time"""
        translator = {"min": "lb", "max": "ub"}

        solved_model_func.backend.update_variable_bounds(
            "flow_out", **{bound: dummy_int}
        )

        bound_vals = solved_model_func.backend.get_variable_bounds("flow_out")[
            translator[bound]
        ]

        assert (bound_vals == dummy_int).where(bound_vals.notnull()).all()

    def test_update_variable_bounds_single_val(self, solved_model_func, dummy_int):
        """Updating a variable lower and upper bounds simultaneously."""
        solved_model_func.backend.update_variable_bounds(
            "flow_out", min=dummy_int, max=dummy_int
        )
        bound_vals = solved_model_func.backend.get_variable_bounds("flow_out")
        assert (bound_vals == dummy_int).where(bound_vals.notnull()).all().all()

    def test_update_variable_single_bound_multi_val(self, caplog, solved_model_func):
        """Updating a bound using an array of values."""
        caplog.set_level(logging.INFO)
        bound_array = solved_model_func.inputs.sink_use_equals.sel(
            techs="test_demand_elec"
        )
        solved_model_func.backend.update_variable_bounds("flow_in", min=bound_array)
        bound_vals = solved_model_func.backend.get_variable_bounds("flow_in").lb
        assert "New `min` bounds will be broadcast" in caplog.text
        assert bound_vals.equals(
            bound_array.where(bound_vals.notnull()).transpose(*bound_vals.dims)
        )

    def test_update_variable_error_update_input_instead(self, solved_model_func):
        """Check that expected error is raised if trying to update a variable bound that was set by a parameter."""
        with pytest.raises(calliope.exceptions.BackendError) as excinfo:
            solved_model_func.backend.update_variable_bounds("flow_cap", max=1)
        assert check_error_or_warning(
            excinfo,
            "Cannot update variable bounds that have been set by parameters."
            " Use `update_input('flow_cap_max')` to update the max bound of flow_cap.",
        )

    def test_fix_variable_before_solve(self, built_model_cls_longnames):
        """Cannot fix a variable before solving the model."""
        with pytest.raises(calliope.exceptions.BackendError) as excinfo:
            built_model_cls_longnames.backend.fix_variable("flow_cap")

        assert check_error_or_warning(
            excinfo,
            "Cannot fix variable values without already having solved the model successfully.",
        )


class TestMILP:
    def test_has_integer_or_binary_variables_lp(self, solved_model_cls):
        """LP models have no integer or binary variables."""
        assert not solved_model_cls.backend.has_integer_or_binary_variables

    def test_has_integer_or_binary_variables_milp(self, solved_model_milp_cls):
        """MILP models have integer / binary variables."""
        assert solved_model_milp_cls.backend.has_integer_or_binary_variables


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
            "y_expression": "source_cap",
            "description": "FOO",
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
    def missing_breakpoint_dims(self):
        return self.gen_params([0, 5, 10], dim="foobar")

    @pytest.fixture(scope="class")
    def add_math(self):
        return {
            "parameters": {"piecewise_x": {}, "piecewise_y": {}},
            "dimensions": {
                "breakpoints": {"dtype": "integer", "iterator": "breakpoint"}
            },
        }

    @pytest.fixture(scope="class")
    def working_model(self, backend, working_params, working_math, add_math):
        m = build_model(
            working_params,
            "simple_supply,two_hours,investment_costs",
            math_dict=add_math,
        )
        m.build(backend=backend)
        m.backend.add_piecewise_constraint("foo", working_math)
        return m

    @pytest.fixture(scope="class")
    def piecewise_constraint(self, working_model):
        return working_model.backend.get_piecewise_constraint("foo")

    def test_piecewise_attrs(self, piecewise_constraint):
        """Check a piecewise constraint has all expected attributes."""
        expected_keys = set(
            ["obj_type", "references", "title", "description", "coords_in_name"]
        )
        assert not expected_keys.symmetric_difference(piecewise_constraint.attrs.keys())

    def test_piecewise_obj_type(self, piecewise_constraint):
        """Check a piecewise constraint has expected object type."""
        assert piecewise_constraint.attrs["obj_type"] == "piecewise_constraints"

    def test_piecewise_refs(self, piecewise_constraint):
        """Check a piecewise constraint has expected refs to other math components (zero for piecewise constraints)."""
        assert not piecewise_constraint.attrs["references"]

    def test_piecewise_obj_coords_in_name(self, piecewise_constraint):
        """Check a piecewise constraint does not have verbose strings activated."""
        assert piecewise_constraint.attrs["coords_in_name"] is False

    @pytest.mark.parametrize(
        "var", ["flow_cap", "source_cap", "piecewise_x", "piecewise_y"]
    )
    def test_piecewise_upstream_refs(self, working_model, var):
        """Expected tracking of piecewise constraint in component reference chains."""
        assert "foo" in working_model.backend._dataset[var].attrs["references"]

    def test_fails_on_breakpoints_in_foreach(self, working_model, working_math):
        """Expected error when defining `breakpoints` in foreach."""
        failing_math = {"foreach": ["nodes", "techs", "carriers", "breakpoints"]}
        with pytest.raises(calliope.exceptions.BackendError) as excinfo:
            working_model.backend.add_piecewise_constraint(
                "bar", {**working_math, **failing_math}
            )
        assert check_error_or_warning(
            excinfo,
            "(piecewise_constraints, bar) | `breakpoints` dimension should not be in `foreach`",
        )

    def test_fails_on_no_breakpoints_in_params(
        self, missing_breakpoint_dims, working_math, backend, add_math
    ):
        """Expected error when parameter defining breakpoints isn't indexed over `breakpoints`."""
        m = build_model(
            missing_breakpoint_dims,
            "simple_supply,two_hours,investment_costs",
            math_dict=add_math,
        )
        m.build(backend=backend)
        with pytest.raises(calliope.exceptions.BackendError) as excinfo:
            m.backend.add_piecewise_constraint("bar", working_math)
        assert check_error_or_warning(
            excinfo,
            "(piecewise_constraints, bar) | `x_values` must be indexed over the `breakpoints` dimension",
        )
