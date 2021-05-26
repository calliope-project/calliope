import pytest  # pylint: disable=unused-import

from calliope.test.common.util import build_test_model as build_model
from calliope.test.common.util import check_variable_exists


class TestBuildConversionPlusConstraints:
    # conversion_plus.py
    def test_loc_techs_balance_conversion_plus_primary_constraint(self):
        """
        sets.loc_techs_conversion_plus,
        """
        m = build_model({}, "simple_supply,two_hours,investment_costs")
        m.run(build_only=True)
        assert not hasattr(
            m._backend_model, "balance_conversion_plus_primary_constraint"
        )

        m = build_model({}, "simple_conversion_plus,two_hours,investment_costs")
        m.run(build_only=True)
        assert hasattr(m._backend_model, "balance_conversion_plus_primary_constraint")

        m = build_model(
            {
                "techs.test_conversion_plus.essentials.carrier_out": [
                    "electricity",
                    "heat",
                ]
            },
            "simple_conversion_plus,two_hours,investment_costs",
        )
        m.run(build_only=True)
        assert hasattr(m._backend_model, "balance_conversion_plus_primary_constraint")

        m = build_model(
            {
                "techs.test_conversion_plus.essentials": {
                    "carrier_in": ["coal", "gas"],
                    "primary_carrier_in": "gas",
                }
            },
            "simple_conversion_plus,two_hours,investment_costs",
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

    def test_loc_techs_cost_var_conversion_plus_constraint(self):
        """
        sets.loc_techs_om_cost_conversion_plus,
        """
        # no conversion_plus = no constraint
        m = build_model(
            {"techs.test_supply_elec.costs.monetary.om_prod": 0.1},
            "simple_supply,two_hours,investment_costs",
        )
        m.run(build_only=True)
        assert not hasattr(m._backend_model, "cost_var_conversion_plus_constraint")

        # no conversion_plus = no constraint
        m = build_model(
            {"techs.test_conversion.costs.monetary.om_prod": 0.1},
            "simple_conversion,two_hours,investment_costs",
        )
        m.run(build_only=True)
        assert not hasattr(m._backend_model, "cost_var_conversion_plus_constraint")

        # no variable costs for conversion_plus = no constraint
        m = build_model({}, "simple_conversion_plus,two_hours,investment_costs")
        m.run(build_only=True)
        assert not hasattr(m._backend_model, "cost_var_conversion_plus_constraint")

        # om_prod creates constraint and populates it with carrier_prod driven cost
        m = build_model(
            {"techs.test_conversion_plus.costs.monetary.om_prod": 0.1},
            "simple_conversion_plus,two_hours,investment_costs",
        )
        m.run(build_only=True)
        assert hasattr(m._backend_model, "cost_var_conversion_plus_constraint")
        assert check_variable_exists(
            m._backend_model, "cost_var_conversion_plus_constraint", "carrier_prod"
        )
        assert not check_variable_exists(
            m._backend_model, "cost_var_conversion_plus_constraint", "carrier_con"
        )

        # om_con creates constraint and populates it with carrier_con driven cost
        m = build_model(
            {"techs.test_conversion_plus.costs.monetary.om_con": 0.1},
            "simple_conversion_plus,two_hours,investment_costs",
        )
        m.run(build_only=True)
        assert hasattr(m._backend_model, "cost_var_conversion_plus_constraint")
        assert check_variable_exists(
            m._backend_model, "cost_var_conversion_plus_constraint", "carrier_con"
        )
        assert not check_variable_exists(
            m._backend_model, "cost_var_conversion_plus_constraint", "carrier_prod"
        )

    def test_loc_techs_balance_conversion_plus_in_2_constraint(self):
        """
        sets.loc_techs_in_2,
        """

        m = build_model({}, "simple_conversion_plus,two_hours,investment_costs")
        m.run(build_only=True)
        assert not hasattr(m._backend_model, "balance_conversion_plus_in_2_constraint")

        m = build_model(
            {
                "techs.test_conversion_plus.essentials": {
                    "carrier_in_2": "coal",
                    "primary_carrier_in": "gas",
                }
            },
            "simple_conversion_plus,two_hours,investment_costs",
        )
        m.run(build_only=True)
        assert hasattr(m._backend_model, "balance_conversion_plus_in_2_constraint")

        m = build_model(
            {
                "techs.test_conversion_plus.essentials": {
                    "carrier_in_2": ["coal", "heat"],
                    "primary_carrier_in": "gas",
                }
            },
            "simple_conversion_plus,two_hours,investment_costs",
        )
        m.run(build_only=True)
        assert hasattr(m._backend_model, "balance_conversion_plus_in_2_constraint")

    def test_loc_techs_balance_conversion_plus_in_3_constraint(self):
        """
        sets.loc_techs_in_3,
        """

        m = build_model({}, "simple_conversion_plus,two_hours,investment_costs")
        m.run(build_only=True)
        assert not hasattr(m._backend_model, "balance_conversion_plus_in_3_constraint")

        m = build_model(
            {
                "techs.test_conversion_plus.essentials": {
                    "carrier_in_3": "coal",
                    "primary_carrier_in": "gas",
                }
            },
            "simple_conversion_plus,two_hours,investment_costs",
        )
        m.run(build_only=True)
        assert hasattr(m._backend_model, "balance_conversion_plus_in_3_constraint")

        m = build_model(
            {
                "techs.test_conversion_plus.essentials": {
                    "carrier_in_3": ["coal", "heat"],
                    "primary_carrier_in": "gas",
                }
            },
            "simple_conversion_plus,two_hours,investment_costs",
        )
        m.run(build_only=True)
        assert hasattr(m._backend_model, "balance_conversion_plus_in_3_constraint")

    def test_loc_techs_balance_conversion_plus_out_2_constraint(self):
        """
        sets.loc_techs_out_2,
        """

        m = build_model(
            {"techs.test_conversion_plus.essentials.carrier_out_2": ["coal", "heat"]},
            "simple_conversion_plus,two_hours,investment_costs",
        )
        m.run(build_only=True)
        assert hasattr(m._backend_model, "balance_conversion_plus_out_2_constraint")

    def test_loc_techs_balance_conversion_plus_out_3_constraint(self):
        """
        sets.loc_techs_out_3,
        """

        m = build_model({}, "simple_conversion_plus,two_hours,investment_costs")
        m.run(build_only=True)
        assert not hasattr(m._backend_model, "balance_conversion_plus_out_3_constraint")

        m = build_model(
            {"techs.test_conversion_plus.essentials.carrier_out_3": "coal"},
            "simple_conversion_plus,two_hours,investment_costs",
        )
        m.run(build_only=True)
        assert hasattr(m._backend_model, "balance_conversion_plus_out_3_constraint")

        m = build_model(
            {"techs.test_conversion_plus.essentials.carrier_out_3": ["coal", "heat"]},
            "simple_conversion_plus,two_hours,investment_costs",
        )
        m.run(build_only=True)
        assert hasattr(m._backend_model, "balance_conversion_plus_out_3_constraint")

    def test_loc_tech_carrier_tiers_conversion_plus_zero_ratio_constraint(self):
        """ """

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
        assert hasattr(
            m._backend_model,
            "loc_tech_carrier_tiers_conversion_plus_zero_ratio_constraint",
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
        assert hasattr(
            m._backend_model,
            "loc_tech_carrier_tiers_conversion_plus_zero_ratio_constraint",
        )


class TestConversionPlusConstraintResults:
    def test_carrier_ratio_from_file(self):
        m = build_model(
            {"techs.test_conversion.costs.monetary.om_prod": 3},
            "conversion_and_conversion_plus,one_day,investment_costs",
        )
        m.run()
        carrier_prod_conversion_plus = (
            (
                m.get_formatted_array("carrier_prod").loc[
                    {"techs": "test_conversion_plus"}
                ]
            )
            .sum("locs")
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
            (
                m.get_formatted_array("carrier_prod").loc[
                    {"techs": "test_conversion_plus"}
                ]
            )
            .sum("locs")
            .to_pandas()
        )
        carrier_con_conversion_plus = (
            (
                m.get_formatted_array("carrier_con").loc[
                    {"techs": "test_conversion_plus"}
                ]
            )
            .sum("locs")
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
            .loc[:, "0"]
            .reindex(prod_ratios.index)
            == prod_ratios.round(1)
        ).all()
        assert (
            m._model_run.timeseries_data["carrier_ratio.csv"]
            .loc[:, "0"]
            .reindex(con_ratios.index)
            == con_ratios.round(1)
        ).all()
