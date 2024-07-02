import importlib

import calliope.exceptions as exceptions
import pytest  # noqa: F401
import xarray as xr

from .common.util import build_test_model as build_model
from .common.util import check_error_or_warning

if importlib.util.find_spec("gurobipy") is not None:
    import gurobipy


class TestNewBackend:
    pytest.importorskip("gurobipy")

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

    def test_new_build_get_global_expression(self, simple_supply_gurobi):
        expr = simple_supply_gurobi.backend.get_global_expression("cost_investment")
        assert (
            expr.to_series()
            .dropna()
            .apply(lambda x: isinstance(x, gurobipy.LinExpr))
            .all()
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
        ("objname", "dims", "objtype", "namegetter"),
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
            == f"{objname}[{', '.join(dims[i] for i in obj.dims)}]"
        )
        assert obj.attrs["coords_in_name"]

    def test_verbose_strings_expression(self, simple_supply_longnames):
        dims = {"nodes": "a", "techs": "test_supply_elec", "costs": "monetary"}

        obj = simple_supply_longnames.backend.get_global_expression(
            "cost_investment", as_backend_objs=False
        )

        assert "flow_cap[a, test_supply_elec, electricity]" in obj.sel(dims).item()
        # parameters are not gurobi objects, so we don't get their names in our strings
        assert "parameters[cost_interest_rate]" not in obj.sel(dims).item()

        assert not obj.coords_in_name

    @staticmethod
    def _is_fixed(val):
        return val.lb == val.ub

    def test_fix_variable(self, simple_supply_gurobi):
        simple_supply_gurobi.build(backend="gurobi", force=True)
        simple_supply_gurobi.solve(force=True)
        simple_supply_gurobi.backend.fix_variable("flow_cap")
        fixed = simple_supply_gurobi.backend._apply_func(
            self._is_fixed,
            simple_supply_gurobi.backend.variables.flow_cap.notnull(),
            1,
            simple_supply_gurobi.backend.variables.flow_cap,
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
            self._is_fixed,
            simple_supply_gurobi.backend.variables.flow_cap.notnull(),
            1,
            simple_supply_gurobi.backend.variables.flow_cap,
        )
        assert not fixed.sel(techs="test_demand_elec", carriers="electricity").any()
        assert fixed.where(where, other=True).all()
        # reset
        simple_supply_gurobi.build(backend="gurobi", force=True)
        simple_supply_gurobi.solve(force=True)

    def test_fix_variable_before_optimal_solve(self, simple_supply_gurobi):
        simple_supply_gurobi.backend.update_parameter("flow_cap_max", xr.DataArray(0))
        simple_supply_gurobi.solve(force=True)
        assert simple_supply_gurobi.results.termination_condition != "optimal"
        with pytest.raises(exceptions.BackendError) as excinfo:
            simple_supply_gurobi.backend.fix_variable("flow_cap")

        assert check_error_or_warning(
            excinfo,
            "Cannot fix variable values without already having solved the model successfully.",
        )

        # reset
        simple_supply_gurobi.backend.update_parameter(
            "flow_cap_max", simple_supply_gurobi.inputs.flow_cap_max
        )
        simple_supply_gurobi.solve(force=True)

    def test_unfix_variable(self, simple_supply_gurobi):
        with pytest.raises(exceptions.BackendError) as excinfo:
            simple_supply_gurobi.backend.unfix_variable("flow_cap")

        assert check_error_or_warning(
            excinfo,
            "Cannot unfix a variable using the Gurobi backend; you will need to rebuild your backend or update variable bounds to match the original bounds.",
        )

    def test_set_solver_option(self, simple_supply_gurobi):
        simple_supply_gurobi.solve(force=True, solver_options={"Threads": 1})
        assert simple_supply_gurobi.backend._instance.Params.Threads == 1
        assert (
            "Threads" in simple_supply_gurobi.backend._instance.Params._getChangeList()
        )

    def test_set_warmstart(self, simple_supply_gurobi):
        simple_supply_gurobi.solve(force=True, warmstart=True)
        assert simple_supply_gurobi.backend._instance.Params.LPWarmStart == 1

        # warmstart = 1 is the Gurobi default, so no change if warmstart=True
        assert (
            "LPWarmStart"
            not in simple_supply_gurobi.backend._instance.Params._getChangeList()
        )

    def test_unset_warmstart(self, simple_supply_gurobi):
        simple_supply_gurobi.solve(force=True, warmstart=False)
        assert simple_supply_gurobi.backend._instance.Params.LPWarmStart == 0
        assert (
            "LPWarmStart"
            in simple_supply_gurobi.backend._instance.Params._getChangeList()
        )

    def test_save_logs(self, simple_supply_gurobi, tmp_path):
        dir = tmp_path / "logs"
        dir.mkdir()
        expected = dir / "gurobi.log"
        simple_supply_gurobi.solve(force=True, save_logs=str(dir))

        assert (
            simple_supply_gurobi.backend._instance.Params.LogFile == expected.as_posix()
        )
        assert (
            "LogFile" in simple_supply_gurobi.backend._instance.Params._getChangeList()
        )
        assert expected.exists()

    def test_to_lp_wrong_file_extension(self, simple_supply_gurobi, tmp_path):
        filepath = tmp_path / "out.txt"
        with pytest.raises(ValueError, match="File extension must be `.lp`"):
            simple_supply_gurobi.backend.to_lp(filepath)


class TestShadowPrices:
    @pytest.fixture()
    def simple_supply(self):
        m = build_model({}, "simple_supply,two_hours,investment_costs")
        m.build(backend="gurobi")
        return m

    @pytest.fixture()
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
