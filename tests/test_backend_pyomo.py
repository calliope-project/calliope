import logging

import numpy as np
import pyomo.core as po
import pyomo.kernel as pmo
import pytest  # noqa: F401
import xarray as xr
from pyomo.core.kernel.piecewise_library.transforms import piecewise_sos2

import calliope
import calliope.exceptions as exceptions

from .common.util import build_test_model as build_model
from .common.util import check_error_or_warning


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
        with caplog.at_level(logging.DEBUG):
            gurobi_model.solve(solver_io="python")
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
                        "default": 0,
                    },
                    # cost_investment_source_cap exists in the pre-defined math.
                    "cost_investment_source_cap": {
                        "where": "source_cap",
                        "equations": [
                            {"expression": "cost_source_cap * source_cap + new_expr"}
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
            == "variables[flow_out][8]"
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
