import pytest  # noqa: F401

from .common.util import build_test_model as build_model
from .common.util import check_variable_exists


@pytest.mark.skip(reason="to be reimplemented by comparison to LP files")
class TestBuildConversionPlusConstraints:
    # conversion_plus.py
    def test_no_balance_conversion_plus_primary_constraint(self, simple_supply):
        """
        sets.loc_techs_conversion_plus,
        """
        assert (
            "balance_conversion_plus_primary" not in simple_supply.backend.constraints
        )

    @pytest.mark.parametrize(
        ("override", "flow"),
        ((None, None), (["electricity", "heat"], "out"), (["coal", "gas"], "in")),
    )
    def test_balance_conversion_plus_primary_constraint(self, override, flow):
        if override is not None:
            override_dict = {
                "techs.test_conversion_plus.essentials": {
                    f"carrier_{flow}": override,
                    f"primary_carrier_{flow}": "gas" if flow == "in" else "electricity",
                }
            }
        else:
            override_dict = {}
        m = build_model(
            override_dict, "simple_conversion_plus,two_hours,investment_costs"
        )
        m.build()
        assert "balance_conversion_plus_primary" in m.backend.constraints

    def test_loc_techs_flow_out_max_conversion_plus_constraint(
        self, simple_conversion_plus
    ):
        """
        i for i in sets.loc_techs_conversion_plus
        if i not in sets.loc_techs_milp
        """
        assert (
            "flow_out_max_conversion_plus" in simple_conversion_plus.backend.constraints
        )

    @pytest.mark.filterwarnings("ignore:(?s).*Integer:calliope.exceptions.ModelWarning")
    def test_loc_techs_flow_out_max_conversion_plus_milp_constraint(
        self, conversion_plus_milp
    ):
        assert (
            "flow_out_max_conversion_plus"
            not in conversion_plus_milp.backend.constraints
        )

    def test_loc_techs_flow_out_min_conversion_plus_constraint(
        self, simple_conversion_plus
    ):
        """
        i for i in sets.loc_techs_conversion_plus
        if constraint_exists(model_run, i, 'constraints.flow_out_min_relative')
        and i not in sets.loc_techs_milp
        """
        assert (
            "flow_out_min_conversion_plus"
            not in simple_conversion_plus.backend.constraints
        )

        m = build_model(
            {"techs.test_conversion_plus.constraints.flow_out_min_relative": 0.1},
            "simple_conversion_plus,two_hours,investment_costs",
        )
        m.build()
        assert "flow_out_min_conversion_plus" in m.backend.constraints

    @pytest.mark.filterwarnings("ignore:(?s).*Integer:calliope.exceptions.ModelWarning")
    def test_loc_techs_flow_out_min_conversion_plus_milp_constraint(
        self, conversion_plus_milp
    ):
        assert (
            "flow_out_min_conversion_plus"
            not in conversion_plus_milp.backend.constraints
        )

        m = build_model(
            {"techs.test_conversion_plus.constraints.flow_out_min_relative": 0.1},
            "conversion_plus_milp,two_hours,investment_costs",
        )
        m.build()
        assert "flow_out_min_conversion_plus" not in m.backend.constraints

    @pytest.mark.parametrize("flow", ("out", "in"))
    def test_loc_techs_cost_var_conversion_plus_constraint(self, flow):
        """
        sets.loc_techs_om_cost_conversion_plus,
        """

        # flow_out creates constraint and populates it with flow_out driven cost
        m = build_model(
            {f"techs.test_conversion_plus.costs.monetary.om_{flow}": 0.1},
            "simple_conversion_plus,two_hours,investment_costs",
        )
        m.build()
        assert m.backend.expressions.cost_var.notnull().any()
        assert check_variable_exists(
            m.backend.get_expression("cost_var", as_backend_objs=False),
            f"carrier_{flow}",
        )
        assert not check_variable_exists(
            m.backend.get_expression("cost_var", as_backend_objs=False),
            "flow_outin".replace(flow, ""),
        )
        assert (
            m.backend.expressions.cost_var.sel(techs="test_conversion_plus")
            .notnull()
            .all()
        )

    @pytest.mark.filterwarnings(
        "ignore:(?s).*`test_conversion_plus` gives a carrier ratio for `heat`:calliope.exceptions.ModelWarning"
    )
    def test_no_balance_conversion_plus_non_primary_constraint(self):
        """
        sets.loc_techs_in_2,
        """
        m = build_model(
            {"techs.test_conversion_plus.essentials.carrier_out_2": None},
            "simple_conversion_plus,two_hours,investment_costs",
        )
        m.build()
        assert "balance_conversion_plus_non_primary" not in m.backend.constraints

    @pytest.mark.filterwarnings(
        "ignore:(?s).*`test_conversion_plus` gives a carrier ratio for `heat`:calliope.exceptions.ModelWarning"
    )
    @pytest.mark.parametrize("tier", ("in_2", "in_3", "out_2", "out_3"))
    @pytest.mark.parametrize("carriers", ("coal", ["coal", "heat"]))
    def test_loc_techs_balance_conversion_plus_non_primary_constraint(
        self, tier, carriers
    ):
        """
        sets.loc_techs_in_2,
        """
        direction = tier.split("_")[0]
        if direction == "in":
            primary_carrier = "gas"
        if direction == "out":
            primary_carrier = "electricity"
        m = build_model(
            {
                "techs.test_conversion_plus.essentials": {
                    f"carrier_{tier}": carriers,
                    f"primary_carrier_{direction}": primary_carrier,
                }
            },
            "simple_conversion_plus,two_hours,investment_costs",
        )
        m.build()
        assert "balance_conversion_plus_non_primary" in m.backend.constraints

    def test_loc_tech_carrier_tiers_conversion_plus_zero_ratio_constraint(
        self, simple_conversion_plus
    ):
        """ """
        assert (
            "conversion_plus_flow_to_zero"
            not in simple_conversion_plus.backend.constraints
        )

        m = build_model(
            {
                "techs.test_conversion_plus.constraints.carrier_ratios.carrier_out_2.heat": "file=carrier_ratio.csv"
            },
            "simple_conversion_plus,two_hours,investment_costs",
        )
        m.build()
        assert "conversion_plus_flow_to_zero" not in m.backend.constraints

        m = build_model(
            {
                "techs.test_conversion_plus.constraints.carrier_ratios.carrier_out_2.heat": "file=carrier_ratio.csv"
            },
            "simple_conversion_plus,one_day,investment_costs",
        )
        m.build()
        assert "conversion_plus_flow_to_zero" in m.backend.constraints
        assert check_variable_exists(
            m.backend.get_constraint(
                "conversion_plus_flow_to_zero", as_backend_objs=False
            ),
            "flow_out",
        )

        m = build_model(
            {
                "techs.test_conversion_plus": {
                    "essentials.carrier_in": ["gas", "coal"],
                    "essentials.primary_carrier_in": "gas",
                    "constraints.carrier_ratios": {
                        "carrier_out_2.heat": "file=carrier_ratio.csv",
                        "carrier_in.coal": "file=carrier_ratio.csv",
                    },
                }
            },
            "simple_conversion_plus,one_day,investment_costs",
        )
        m.build()
        assert "conversion_plus_flow_to_zero" in m.backend.constraints
        assert check_variable_exists(
            m.backend.get_constraint(
                "conversion_plus_flow_to_zero", as_backend_objs=False
            ),
            "flow_out",
        )
        assert check_variable_exists(
            m.backend.get_constraint(
                "conversion_plus_flow_to_zero", as_backend_objs=False
            ),
            "flow_in",
        )


@pytest.mark.skip(reason="to be reimplemented by comparison to LP files")
class TestConversionPlusConstraintResults:
    def test_carrier_ratio(self):
        m = build_model(
            {"techs.test_conversion.costs.monetary.flow_out": 3},
            "conversion_and_conversion_plus,one_day,investment_costs",
        )
        m.build()
        m.solve()
        flow_out_conversion_plus = (
            m.results.flow_out.loc[{"techs": "test_conversion_plus"}]
            .sum("nodes")
            .to_pandas()
        )
        where_all_zero = (flow_out_conversion_plus.loc["heat"] == 0) & (
            flow_out_conversion_plus.loc["electricity"] == 0
        )
        assert all(
            (
                flow_out_conversion_plus.loc["heat"].where(~where_all_zero)
                / flow_out_conversion_plus.loc["electricity"].where(~where_all_zero)
            ).dropna()
            == 0.8
        )

    def test_carrier_ratio_from_file(self):
        m = build_model(
            {
                "techs.test_conversion_plus": {
                    "constraints.carrier_ratios": {
                        "carrier_out_2.heat": "file=carrier_ratio.csv",
                        "carrier_in.coal": "file=carrier_ratio.csv",
                    },
                    "essentials": {
                        "carrier_in": ["gas", "coal"],
                        "primary_carrier_in": "gas",
                    },
                    "costs.monetary": {"flow_in": 2, "flow_out": -1},
                }
            },
            "conversion_and_conversion_plus,one_day,investment_costs",
        )
        m.build()
        m.solve()
        flow_out_conversion_plus = (
            m.results.flow_out.loc[{"techs": "test_conversion_plus"}]
            .sum("nodes")
            .to_pandas()
        )
        flow_in_conversion_plus = -1 * (
            m.results.flow_in.loc[{"techs": "test_conversion_plus"}]
            .sum("nodes")
            .to_pandas()
        )
        where_all_zero = (flow_out_conversion_plus.loc["heat"] == 0) & (
            flow_out_conversion_plus.loc["electricity"] == 0
        )
        prod_ratios = (
            flow_out_conversion_plus.loc["heat"].where(~where_all_zero)
            / flow_out_conversion_plus.loc["electricity"].where(~where_all_zero)
        ).dropna()

        con_ratios = (
            flow_out_conversion_plus.loc["electricity"].where(~where_all_zero)
            / flow_in_conversion_plus.loc["coal"].where(~where_all_zero)
        ).dropna()

        assert (
            m._model_run.timeseries_data["carrier_ratio.csv"]
            .loc[:, "a"]
            .reindex(prod_ratios.index)
            == prod_ratios.astype(float).round(1)
        ).all()
        assert (
            m._model_run.timeseries_data["carrier_ratio.csv"]
            .loc[:, "a"]
            .reindex(con_ratios.index)
            == con_ratios.astype(float).round(1)
        ).all()
