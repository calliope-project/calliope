import logging

import numpy as np
import pytest
import xarray as xr

import calliope.exceptions as exceptions

from ..common.util import build_test_model as build_model
from ..common.util import check_error_or_warning

highspy = pytest.importorskip("highspy")


class TestNewBackend:
    @pytest.fixture(scope="class")
    def simple_supply_longnames(self):
        m = build_model({}, "simple_supply,two_hours,investment_costs")
        m.build(backend="highs")
        m.backend.verbose_strings()
        return m

    @pytest.fixture(scope="class")
    def simple_supply_highs(self):
        m = build_model({}, "simple_supply,two_hours,investment_costs")
        m.build(backend="highs")
        m.solve()
        return m

    @pytest.fixture
    def simple_supply_highs_func(self):
        m = build_model({}, "simple_supply,two_hours,investment_costs")
        m.build(backend="highs", pre_validate_math_strings=False)
        m.solve()
        return m

    @pytest.fixture
    def simple_supply_highs_func_new_objective(self, simple_supply_highs_func):
        simple_supply_highs_func.backend.add_objective(
            "foo",
            {
                "equations": [{"expression": "bigM"}],
                "sense": "minimise",
                "active": True,
            },
        )
        simple_supply_highs_func.backend.set_objective("foo")
        simple_supply_highs_func.backend.verbose_strings()
        return simple_supply_highs_func

    def test_new_build_get_variable(self, simple_supply_highs):
        var = simple_supply_highs.backend.get_variable("flow_cap")
        assert (
            var.to_series()
            .dropna()
            .apply(lambda x: isinstance(x, highspy.highs.highs_var))
            .all()
        )

    def test_new_build_get_variable_as_vals(self, simple_supply_highs):
        var = simple_supply_highs.backend.get_variable(
            "flow_cap", as_backend_objs=False
        )
        assert (
            not var.to_series()
            .dropna()
            .apply(lambda x: isinstance(x, highspy.highs.highs_var))
            .any()
        )

    def test_new_build_get_parameter(self, simple_supply_highs):
        param = simple_supply_highs.backend.get_parameter("flow_in_eff")
        assert isinstance(param.item(), float)

    def test_add_parameter_updates_math(self, simple_supply_highs_func):
        """Parameters unknown to the math must be registered in the math on addition."""
        backend = simple_supply_highs_func.backend
        assert "foo" not in backend.math["parameters"].root
        backend.add_parameter("foo", xr.DataArray(1), {})
        assert "foo" in backend.math["parameters"].root

    def test_new_build_get_global_expression(self, simple_supply_highs):
        expr = simple_supply_highs.backend.get_global_expression("cost_investment")
        assert (
            expr.to_series()
            .dropna()
            .apply(lambda x: isinstance(x, highspy.highs.highs_linear_expression))
            .all()
        )

    def test_from_highs_expr_fails_on_str_type(self, simple_supply_highs):
        """HiGHS backend fails if it finds a string in an expression array."""
        with pytest.raises(
            TypeError, match="Cannot convert highs object of type <class 'str'>"
        ):
            simple_supply_highs.backend._from_highs_expr("foo", col_values=[])

    def test_new_build_get_constraint(self, simple_supply_highs):
        constr = simple_supply_highs.backend.get_constraint("system_balance")
        assert (
            constr.to_series()
            .dropna()
            .apply(lambda x: isinstance(x, highspy.highs.highs_cons))
            .all()
        )

    def test_new_build_get_constraint_as_str(self, simple_supply_longnames):
        constr = simple_supply_longnames.backend.get_constraint(
            "system_balance", as_backend_objs=False
        )
        constr_str = constr.to_series().dropna().iloc[0]
        assert "flow_out[" in constr_str
        assert "==" in constr_str or "<=" in constr_str or ">=" in constr_str

    @pytest.mark.parametrize(
        ("expr_func", "expected"),
        [
            pytest.param(lambda var: -2.5 * var, "- 2.5*{name}", id="negative-coeff"),
            pytest.param(
                lambda var: 2.5 * var + 4, "2.5*{name} + 4", id="positive-constant"
            ),
            pytest.param(
                lambda var: 2 * var - 4, "2*{name} - 4", id="negative-constant"
            ),
            pytest.param(
                lambda var: highspy.highs.highs_linear_expression(4.0),
                "4",
                id="constant-only",
            ),
            pytest.param(
                lambda var: highspy.highs.highs_linear_expression(), "0", id="empty"
            ),
        ],
    )
    def test_linexpr_to_str(self, simple_supply_longnames, expr_func, expected):
        """Coefficients, constants and empty expressions must all be rendered."""
        backend = simple_supply_longnames.backend
        var = backend.get_variable("flow_cap").to_series().dropna().iloc[0]
        expr_str = backend._linexpr_to_str(expr_func(var))
        assert expr_str == expected.format(name=var.name)

    def test_expr_to_str_plain_number(self, simple_supply_highs):
        """Plain numbers in expression arrays must be rendered via `str`."""
        assert simple_supply_highs.backend._expr_to_str(1.5) == "1.5"

    def test_cons_to_str_inequality_bounds(self, simple_supply_highs_func):
        """One-sided and ranged constraint bounds must all be rendered."""
        backend = simple_supply_highs_func.backend
        instance = backend._instance
        var = backend.get_variable("flow_cap").to_series().dropna().iloc[0]
        name = backend._col_name(var.index)
        upper = instance.addConstr(1.0 * var <= 5)
        lower = instance.addConstr(1.0 * var >= 1)
        ranged = instance.addConstr((1.0 * var) == [1, 5])
        assert backend._cons_to_str(upper) == f"{name} <= 5"
        assert backend._cons_to_str(lower) == f"{name} >= 1"
        assert backend._cons_to_str(ranged) == f"1 <= {name} <= 5"

    def test_add_constraint_tiny_coefficient_error(self, simple_supply_highs_func):
        """highspy's bare exception on rejecting a constraint must become a BackendError."""
        with pytest.raises(
            exceptions.BackendError, match="Failed to add constraint `foo`"
        ):
            simple_supply_highs_func.backend.add_constraint(
                "foo",
                {
                    "equations": [
                        {
                            "expression": "1e-30 * sum(flow_cap, over=[nodes, techs, carriers]) >= 0"
                        }
                    ]
                },
            )

    def test_new_build_get_constraint_as_vals(self, simple_supply_highs):
        """Constraint bodies cannot be evaluated by the HiGHS backend."""
        with pytest.raises(exceptions.BackendError) as excinfo:
            simple_supply_highs.backend.get_constraint(
                "system_balance", as_backend_objs=False, eval_body=True
            )
        assert check_error_or_warning(
            excinfo, "Cannot return the evaluated body of a HiGHS constraint"
        )

    def test_get_variable_as_vals_unsolved(self, simple_supply_longnames):
        """Variable values are not valid before a solve, so strings must be returned."""
        var = simple_supply_longnames.backend.get_variable(
            "flow_cap", as_backend_objs=False
        )
        assert var.to_series().dropna().apply(lambda x: isinstance(x, str)).all()

    def test_get_variable_as_vals_scalar(self, simple_supply_highs_func):
        """Dimensionless variables must resolve to a scalar value array after a solve."""
        backend = simple_supply_highs_func.backend
        backend.add_variable("scalar_var", {"bounds": {"min": 0, "max": 1}})
        simple_supply_highs_func.solve(force=True)
        var = backend.get_variable("scalar_var", as_backend_objs=False)
        assert var.shape == ()
        assert 0 <= var.item() <= 1

    def test_add_valid_obj(self, simple_supply_highs):
        eq = {"expression": "bigM", "where": "True"}
        simple_supply_highs.backend.add_objective(
            "foo", {"equations": [eq], "sense": "minimise", "active": True}
        )
        assert "foo" in simple_supply_highs.backend.objectives

    def test_default_objective_set(self, simple_supply_longnames):
        obj = simple_supply_longnames.backend.get_objective(
            "min_cost_optimisation", as_backend_objs=False
        ).item()
        assert "flow_cap" in obj
        assert simple_supply_longnames.backend.objective == "min_cost_optimisation"

    def test_new_objective_set(self, simple_supply_highs_func_new_objective):
        assert simple_supply_highs_func_new_objective.backend.objective == "foo"

    def test_new_objective_set_log(self, caplog, simple_supply_highs_func):
        caplog.set_level(logging.INFO)
        simple_supply_highs_func.backend.add_objective(
            "foo",
            {
                "equations": [{"expression": "bigM"}],
                "sense": "minimise",
                "active": True,
            },
        )
        simple_supply_highs_func.backend.set_objective("foo")
        assert ":foo | Objective activated." in caplog.text

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
            == f"{objname}[{'__'.join(dims[i].replace(' ', '_') for i in obj.dims)}]"
        )
        assert obj.attrs["coords_in_name"]

    def test_verbose_strings_expression(self, simple_supply_longnames):
        dims = {"nodes": "a", "techs": "test_supply_elec", "costs": "monetary"}

        obj = simple_supply_longnames.backend.get_global_expression(
            "cost_investment", as_backend_objs=False
        )

        assert "flow_cap[" in obj.sel(dims).item()
        # parameters are not highs objects, so we don't get their names in our strings
        assert "cost_flow_cap" not in obj.sel(dims).item()

        assert not obj.coords_in_name

    def test_verbose_strings_after_rebuild(self, simple_supply_highs_func, dummy_int):
        """Verbose names must be re-applied to components rebuilt by `update_input`."""
        backend = simple_supply_highs_func.backend
        backend.verbose_strings()
        backend.update_input("flow_out_eff", 0.5)

        rebuilt_constr = backend._dataset["balance_supply_no_storage"]
        assert rebuilt_constr.attrs["coords_in_name"]
        sample_cons = rebuilt_constr.to_series().dropna().iloc[0]
        assert sample_cons.name.startswith("balance_supply_no_storage[")

    def _bounds(self, backend, name):
        return backend._apply_func(
            backend._from_highs_variable_bounds,
            backend.variables[name].notnull(),
            2,
            backend.variables[name],
        )

    def test_get_variable_bounds(self, simple_supply_highs):
        """Bounds must be returned as numeric values, not null placeholders."""
        bounds = simple_supply_highs.backend.get_variable_bounds("flow_out")
        assert bounds.lb.notnull().any()
        assert (bounds.lb == 0).where(bounds.lb.notnull()).all()
        assert (bounds.ub == np.inf).where(bounds.ub.notnull()).all()

    @pytest.mark.parametrize(("bound", "other_bound"), [("min", "ub"), ("max", "lb")])
    def test_update_variable_single_bound(
        self, simple_supply_highs_func, dummy_int, bound, other_bound
    ):
        """Updating one bound must change that bound and leave the other unchanged."""
        backend = simple_supply_highs_func.backend
        before = backend.get_variable_bounds("flow_out")
        backend.update_variable_bounds("flow_out", **{bound: dummy_int})
        after = backend.get_variable_bounds("flow_out")

        translator = {"min": "lb", "max": "ub"}
        changed = after[translator[bound]]
        assert changed.notnull().any()
        assert (changed == dummy_int).where(changed.notnull()).all()
        assert after[other_bound].equals(before[other_bound])

    def test_update_variable_bounds_all_dims(
        self, caplog, simple_supply_highs_func, dummy_int
    ):
        """Bounds carrying all variable dims must be applied without broadcasting."""
        caplog.set_level(logging.INFO)
        backend = simple_supply_highs_func.backend
        var = backend.variables["flow_out"]
        new_min = xr.full_like(var, dummy_int, dtype=float).where(var.notnull())
        backend.update_variable_bounds("flow_out", min=new_min)
        assert "will be broadcast" not in caplog.text
        bounds = backend.get_variable_bounds("flow_out")
        assert (bounds.lb == dummy_int).where(bounds.lb.notnull()).all()

    def test_fix_variable(self, simple_supply_highs_func):
        backend = simple_supply_highs_func.backend
        backend.fix_variable("flow_cap")
        lb, ub = self._bounds(backend, "flow_cap")
        fixed = (lb == ub).where(lb.notnull())
        assert fixed.all()

    def test_fix_variable_where(self, simple_supply_highs_func):
        backend = simple_supply_highs_func.backend
        where = (
            simple_supply_highs_func.inputs.flow_cap_max.notnull()
            & backend.variables.flow_cap.notnull()
        )
        backend.fix_variable("flow_cap", where=where)
        lb, ub = self._bounds(backend, "flow_cap")
        fixed = lb == ub
        assert not fixed.sel(techs="test_demand_elec", carriers="electricity").any()
        assert fixed.where(where, other=True).all()

    @pytest.mark.filterwarnings(
        "ignore:(?s).*Model solution was non-optimal:calliope.exceptions.BackendWarning"
    )
    def test_fix_variable_before_optimal_solve(self, simple_supply_highs_func):
        simple_supply_highs_func.backend.update_input("flow_cap_max", xr.DataArray(0))
        simple_supply_highs_func.solve(force=True)
        assert simple_supply_highs_func.runtime.termination_condition != "optimal"
        with pytest.raises(exceptions.BackendError) as excinfo:
            simple_supply_highs_func.backend.fix_variable("flow_cap")

        assert check_error_or_warning(
            excinfo,
            "Cannot fix variable values without already having solved the model successfully.",
        )

    def test_unfix_variable(self, simple_supply_highs):
        with pytest.raises(exceptions.BackendError) as excinfo:
            simple_supply_highs.backend.unfix_variable("flow_cap")

        assert check_error_or_warning(
            excinfo,
            "Cannot unfix a variable using the HiGHS backend; you will need to rebuild your backend or update variable bounds to match the original bounds.",
        )

    def test_set_solver_option(self, simple_supply_highs):
        simple_supply_highs.solve(force=True, solver_options={"time_limit": 100})
        status, val = simple_supply_highs.backend._instance.getOptionValue("time_limit")
        assert status == highspy.HighsStatus.kOk
        assert val == 100

    def test_small_matrix_value_persists(self, simple_supply_highs):
        """`small_matrix_value` must survive the `resetOptions` call in `_solve`.

        Otherwise constraints with tiny (but valid) coefficients added during
        `update_input`-triggered rebuilds are rejected by highspy >= 1.15.
        """
        simple_supply_highs.solve(force=True)
        status, val = simple_supply_highs.backend._instance.getOptionValue(
            "small_matrix_value"
        )
        assert status == highspy.HighsStatus.kOk
        assert val == 1e-12

    def test_update_input_tiny_coefficient(self, simple_supply_highs_func, dummy_int):
        """Constraint rebuilds with coefficients below 1e-9 must not error.

        Regression test: `1 / dummy_int` is ~2.7e-10, below the default HiGHS
        `small_matrix_value` tolerance; highspy >= 1.15 raises on the resulting
        warning status unless the tolerance is lowered.
        """
        simple_supply_highs_func.backend.update_input("flow_out_eff", dummy_int)
        expected = simple_supply_highs_func.backend.get_parameter(
            "flow_out_eff", as_backend_objs=False
        )
        assert (expected == dummy_int).all()

    def test_set_warmstart_no_warning(self, simple_supply_highs, recwarn):
        """HiGHS hot-starts from the previous solve; warmstart must not warn."""
        simple_supply_highs.solve(force=True, warmstart=True)
        assert simple_supply_highs.runtime.termination_condition == "optimal"
        assert not any("warmstart" in str(warning.message) for warning in recwarn)

    def test_save_logs(self, simple_supply_highs, tmp_path):
        dir = tmp_path / "logs"
        dir.mkdir()
        expected = dir / "highs.log"
        simple_supply_highs.solve(force=True, save_logs=str(dir))
        assert expected.exists()

    def test_to_lp_wrong_file_extension(self, simple_supply_highs, tmp_path):
        filepath = tmp_path / "out.txt"
        with pytest.raises(ValueError, match="File extension must be `.lp`"):
            simple_supply_highs.backend.to_lp(filepath)

    def test_to_lp(self, simple_supply_longnames, tmp_path):
        filepath = tmp_path / "out.lp"
        simple_supply_longnames.backend.to_lp(filepath)
        # verbose names contain characters that are invalid in LP files, so HiGHS
        # falls back to generic column/row names; we only check overall structure.
        lp_string = filepath.read_text()
        assert "min" in lp_string
        assert "obj:" in lp_string

    def test_has_integer_or_binary_variables_lp(self, simple_supply_highs):
        assert not simple_supply_highs.backend.has_integer_or_binary_variables

    def test_has_integer_or_binary_variables_milp(self):
        m = build_model({}, "supply_milp,two_hours,investment_costs")
        m.build(backend="highs")
        assert m.backend.has_integer_or_binary_variables

    def test_add_piecewise_constraint_not_implemented(self):
        m = build_model(
            {
                "data_definitions": {
                    "piecewise_x": {
                        "data": [0, 5, 10],
                        "index": [0, 1, 2],
                        "dims": "breakpoints",
                    },
                    "piecewise_y": {
                        "data": [0, 1, 5],
                        "index": [0, 1, 2],
                        "dims": "breakpoints",
                    },
                }
            },
            "simple_supply,two_hours,investment_costs",
            math_dict={
                "parameters": {"piecewise_x": {}, "piecewise_y": {}},
                "dimensions": {
                    "breakpoints": {"dtype": "integer", "iterator": "breakpoint"}
                },
            },
        )
        m.build(backend="highs")
        with pytest.raises(
            NotImplementedError,
            match="Piecewise constraints are not yet implemented for the HiGHS backend.",
        ):
            m.backend.add_piecewise_constraint(
                "foo",
                {
                    "foreach": ["nodes", "techs", "carriers"],
                    "where": "[test_supply_elec] in techs AND piecewise_x AND piecewise_y",
                    "x_values": "piecewise_x",
                    "x_expression": "flow_cap",
                    "y_values": "piecewise_y",
                    "y_expression": "source_cap",
                    "description": "FOO",
                    "active": True,
                },
            )


class TestDeleteComponent:
    """`delete_component` must keep stored object indices in sync with the instance.

    HiGHS shifts internal row/column indices on deletion; `_shift_stale_indices`
    corrects the copies held by the highspy wrapper objects. These tests validate
    that bookkeeping by round-tripping names through the instance and by re-solving.
    """

    @pytest.fixture
    def verbose_model(self):
        m = build_model({}, "simple_supply,two_hours,investment_costs")
        m.build(backend="highs")
        m.backend.verbose_strings()
        return m

    def test_delete_constraint_shifts_row_indices(self, verbose_model):
        """Names are stored in the instance pre-deletion, so a live name lookup by the
        stored (shifted) index validates the index bookkeeping."""
        backend = verbose_model.backend
        backend.delete_component("balance_demand", "constraints")
        assert "balance_demand" not in backend.constraints

        cons = (
            backend._dataset["system_balance"]
            .sel(nodes="a", carriers="electricity", timesteps="2005-01-01 00:00")
            .item()
        )
        assert cons.name == "system_balance[a__electricity__2005-01-01_00:00]"

    def test_delete_variable_shifts_col_indices(self, verbose_model):
        backend = verbose_model.backend
        backend.delete_component("flow_in", "variables")
        assert "flow_in" not in backend.variables

        var = (
            backend._dataset["flow_out"]
            .sel(
                nodes="a",
                techs="test_supply_elec",
                carriers="electricity",
                timesteps="2005-01-01 00:00",
            )
            .item()
        )
        assert (
            var.name == "flow_out[a__test_supply_elec__electricity__2005-01-01_00:00]"
        )

    def test_delete_variable_shifts_expression_indices(self, verbose_model):
        backend = verbose_model.backend
        backend.delete_component("flow_out", "variables")

        expr_str = (
            backend.get_global_expression("cost_investment", as_backend_objs=False)
            .to_series()
            .dropna()
            .iloc[0]
        )
        assert "flow_cap[" in expr_str

    @pytest.mark.parametrize(
        ("key", "component_type"),
        [("not_in_dataset", "constraints"), ("flow_cap", "constraints")],
    )
    def test_delete_missing_or_mismatched_is_noop(
        self, verbose_model, key, component_type
    ):
        """Deleting an unknown key or one of a different component type must do nothing."""
        backend = verbose_model.backend
        n_rows = backend._instance.getNumRow()
        backend.delete_component(key, component_type)
        assert backend._instance.getNumRow() == n_rows
        assert "flow_cap" in backend.variables

    def test_shift_stale_indices_without_deletions_is_noop(self, verbose_model):
        """A direct call with no deleted indices must leave stored indices untouched."""
        backend = verbose_model.backend
        var = backend._dataset["flow_cap"].to_series().dropna().iloc[0]
        index_before = var.index
        backend._shift_stale_indices([], "variables")
        assert var.index == index_before

    def test_update_input_rebuild_and_resolve(self):
        """Sequential rebuilds must leave a model that solves to the same objective as a once-updated build."""
        m = build_model({}, "simple_supply,two_hours,investment_costs")
        m.build(backend="highs")
        m.backend.update_input("flow_out_eff", 0.5)
        m.backend.update_input("flow_out_eff", 0.9)
        m.solve()

        fresh = build_model({}, "simple_supply,two_hours,investment_costs")
        fresh.build(backend="highs")
        fresh.backend.update_input("flow_out_eff", 0.9)
        fresh.solve()

        assert m.runtime.termination_condition == "optimal"
        assert np.isclose(m.results.cost.sum().item(), fresh.results.cost.sum().item())


class TestShadowPrices:
    @pytest.fixture
    def simple_supply(self):
        m = build_model({}, "simple_supply,two_hours,investment_costs")
        m.build(backend="highs")
        return m

    @pytest.fixture
    def supply_milp(self):
        m = build_model({}, "supply_milp,two_hours,investment_costs")
        m.build(backend="highs")
        return m

    def test_always_active_in_highs(self, simple_supply):
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
        """MILP solutions have no valid duals, so all shadow prices must be null."""
        supply_milp.backend.shadow_prices.activate()
        supply_milp.solve()
        shadow_prices = supply_milp.backend.shadow_prices.get("system_balance")
        assert shadow_prices.isnull().all()

    def test_get_shadow_price_missing_duals_interface(self, simple_supply, monkeypatch):
        """Shadow prices must fall back to null if highspy cannot provide duals."""
        simple_supply.solve()

        def _raise_attribute_error(val):
            raise AttributeError("no duals available")

        monkeypatch.setattr(
            simple_supply.backend._instance, "constrDuals", _raise_attribute_error
        )
        shadow_prices = simple_supply.backend.shadow_prices.get("system_balance")
        assert shadow_prices.isnull().all()

    def test_get_shadow_price_unsolved(self, simple_supply):
        """Shadow prices requested before a solve must be null, not garbage."""
        shadow_prices = simple_supply.backend.shadow_prices.get("system_balance")
        assert shadow_prices.isnull().all()

    def test_available_constraints(self, simple_supply):
        assert (
            "system_balance"
            in simple_supply.backend.shadow_prices.available_constraints
        )


class TestHighsImportHandling:
    """Test handling of highspy import failures."""

    def test_highs_backend_requires_highspy(self, monkeypatch):
        """Test that building with highs backend fails gracefully without highspy."""
        import sys

        # Mock highspy as unavailable
        monkeypatch.setitem(sys.modules, "highspy", None)

        m = build_model({}, "simple_supply,two_hours")
        with pytest.raises(ImportError, match="Install the `highspy` package"):
            m.build(backend="highs")
