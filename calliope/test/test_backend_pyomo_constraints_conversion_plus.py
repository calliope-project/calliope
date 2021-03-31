import pytest  # noqa: F401

from calliope.test.common.util import build_test_model as build_model
from calliope.test.common.util import check_variable_exists


class TestBuildConversionPlusConstraints:
    # conversion_plus.py
    def test_no_balance_conversion_plus_primary_constraint(self):
        """
        sets.loc_techs_conversion_plus,
        """
        m = build_model({}, "simple_supply,two_hours,investment_costs")
        m.run(build_only=True)
        assert not hasattr(
            m._backend_model, "balance_conversion_plus_primary_constraint"
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
        m.run(build_only=True)
        assert hasattr(m._backend_model, "balance_conversion_plus_primary_constraint")

    def test_loc_techs_carrier_production_max_conversion_plus_constraint(self):
        """
        i for i in sets.loc_techs_conversion_plus
        if i not in sets.loc_techs_milp
        """

        m = build_model({}, "simple_conversion_plus,two_hours,investment_costs")
        m.run(build_only=True)
        assert hasattr(
            m._backend_model, "carrier_production_max_conversion_plus_constraint"
        )

    @pytest.mark.filterwarnings("ignore:(?s).*Integer:calliope.exceptions.ModelWarning")
    def test_loc_techs_carrier_production_max_conversion_plus_milp_constraint(self):
        m = build_model({}, "conversion_plus_milp,two_hours,investment_costs")
        m.run(build_only=True)
        assert not hasattr(
            m._backend_model, "carrier_production_max_conversion_plus_constraint"
        )

    def test_loc_techs_carrier_production_min_conversion_plus_constraint(self):
        """
        i for i in sets.loc_techs_conversion_plus
        if constraint_exists(model_run, i, 'constraints.energy_cap_min_use')
        and i not in sets.loc_techs_milp
        """

        m = build_model({}, "simple_conversion_plus,two_hours,investment_costs")
        m.run(build_only=True)
        assert not hasattr(
            m._backend_model, "carrier_production_min_conversion_plus_constraint"
        )

        m = build_model(
            {"techs.test_conversion_plus.constraints.energy_cap_min_use": 0.1},
            "simple_conversion_plus,two_hours,investment_costs",
        )
        m.run(build_only=True)
        assert hasattr(
            m._backend_model, "carrier_production_min_conversion_plus_constraint"
        )

    @pytest.mark.filterwarnings("ignore:(?s).*Integer:calliope.exceptions.ModelWarning")
    def test_loc_techs_carrier_production_min_conversion_plus_milp_constraint(self):
        m = build_model({}, "conversion_plus_milp,two_hours,investment_costs")
        m.run(build_only=True)
        assert not hasattr(
            m._backend_model, "carrier_production_min_conversion_plus_constraint"
        )

        m = build_model(
            {"techs.test_conversion_plus.constraints.energy_cap_min_use": 0.1},
            "conversion_plus_milp,two_hours,investment_costs",
        )
        m.run(build_only=True)
        assert not hasattr(
            m._backend_model, "carrier_production_min_conversion_plus_constraint"
        )

    @pytest.mark.parametrize("flow", ("prod", "con"))
    def test_loc_techs_cost_var_conversion_plus_constraint(self, flow):
        """
        sets.loc_techs_om_cost_conversion_plus,
        """

        # om_prod creates constraint and populates it with carrier_prod driven cost
        m = build_model(
            {f"techs.test_conversion_plus.costs.monetary.om_{flow}": 0.1},
            "simple_conversion_plus,two_hours,investment_costs",
        )
        m.run(build_only=True)
        assert hasattr(m._backend_model, "cost_var")
        assert check_variable_exists(m._backend_model, "cost_var", f"carrier_{flow}")
        assert not check_variable_exists(
            m._backend_model, "cost_var", "carrier_prodcon".replace(flow, "")
        )
        assert all(
            "test_conversion_plus" in i for i in m._backend_model.cost_var._index
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
        m.run(build_only=True)
        assert not hasattr(
            m._backend_model, "balance_conversion_plus_non_primary_constraint"
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
        m.run(build_only=True)
        assert hasattr(
            m._backend_model, "balance_conversion_plus_non_primary_constraint"
        )

    def test_loc_tech_carrier_tiers_conversion_plus_zero_ratio_constraint(self):
        """"""

        m = build_model({}, "simple_conversion_plus,one_day,investment_costs")
        m.run(build_only=True)
        assert not hasattr(
            m._backend_model, "conversion_plus_prod_con_to_zero_constraint"
        )

        m = build_model(
            {
                "techs.test_conversion_plus.constraints.carrier_ratios.carrier_out_2.heat": "file=carrier_ratio.csv"
            },
            "simple_conversion_plus,two_hours,investment_costs",
        )
        m.run(build_only=True)
        assert not hasattr(
            m._backend_model, "conversion_plus_prod_con_to_zero_constraint"
        )

        m = build_model(
            {
                "techs.test_conversion_plus.constraints.carrier_ratios.carrier_out_2.heat": "file=carrier_ratio.csv"
            },
            "simple_conversion_plus,one_day,investment_costs",
        )
        m.run(build_only=True)
        assert hasattr(m._backend_model, "conversion_plus_prod_con_to_zero_constraint")
        assert check_variable_exists(
            m._backend_model,
            "conversion_plus_prod_con_to_zero_constraint",
            "carrier_prod",
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
        m.run(build_only=True)
        assert hasattr(m._backend_model, "conversion_plus_prod_con_to_zero_constraint")
        assert check_variable_exists(
            m._backend_model,
            "conversion_plus_prod_con_to_zero_constraint",
            "carrier_prod",
        )
        assert check_variable_exists(
            m._backend_model,
            "conversion_plus_prod_con_to_zero_constraint",
            "carrier_con",
        )


class TestConversionPlusConstraintResults:
    def test_carrier_ratio_from_file(self):
        m = build_model(
            {"techs.test_conversion.costs.monetary.om_prod": 3},
            "conversion_and_conversion_plus,one_day,investment_costs",
        )
        m.run()
        carrier_prod_conversion_plus = (
            m.results.carrier_prod.loc[{"techs": "test_conversion_plus"}]
            .sum("nodes")
            .to_pandas()
        )
        assert all(
            (
                carrier_prod_conversion_plus.loc["heat"]
                / carrier_prod_conversion_plus.loc["electricity"]
            ).dropna()
            == 0.8
        )

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
                    "costs.monetary": {"om_con": 3, "om_prod": 3},
                }
            },
            "conversion_and_conversion_plus,one_day,investment_costs",
        )
        m.run()
        carrier_prod_conversion_plus = (
            m.results.carrier_prod.loc[{"techs": "test_conversion_plus"}]
            .sum("nodes")
            .to_pandas()
        )
        carrier_con_conversion_plus = (
            m.results.carrier_con.loc[{"techs": "test_conversion_plus"}]
            .sum("nodes")
            .to_pandas()
        )

        prod_ratios = (
            carrier_prod_conversion_plus.loc["heat"]
            / carrier_prod_conversion_plus.loc["electricity"]
        ).dropna()
        con_ratios = (
            carrier_con_conversion_plus.loc["coal"]
            / carrier_prod_conversion_plus.sum(axis=0)
        ).dropna()

        assert (
            m._model_run.timeseries_data["carrier_ratio.csv"]
            .loc[:, "a"]
            .reindex(prod_ratios.index)
            == prod_ratios.round(1)
        ).all()
        assert (
            m._model_run.timeseries_data["carrier_ratio.csv"]
            .loc[:, "a"]
            .reindex(con_ratios.index)
            == con_ratios.round(1)
        ).all()
