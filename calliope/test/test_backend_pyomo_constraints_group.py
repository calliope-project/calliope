import os

import pytest
from pytest import approx
import pandas as pd


import calliope
from calliope.core.attrdict import AttrDict
from calliope.test.common.util import build_test_model as build_model
from calliope.test.common.util import check_error_or_warning


def get_supply_conversion_techs(model):
    available_techs = [
        "elec_to_heat_cool_unlinked",
        "cheap_elec_supply",
        "elec_to_heat_cool_linked",
        "elec_to_heat",
        "expensive_elec_supply",
        "normal_elec_supply",
        "cheap_heat_supply",
        "cheap_cool_supply",
        "normal_heat_supply",
        "expensive_heat_supply",
        "normal_cool_supply",
        "expensive_cool_supply",
        "elec_supply_plus",
    ]
    return [i for i in available_techs if i in model._model_data.techs.values]


@pytest.fixture(scope="module")
def model_file():
    return os.path.join("model_config_group", "base_model.yaml")


# Group constraints, i.e. those that can be defined on a system/subsystem scale
@pytest.mark.filterwarnings(
    "ignore:(?s).*All technologies were requested:calliope.exceptions.ModelWarning"
)
class TestBuildGroupConstraints:

    metadata_yaml = os.path.join(
        os.path.dirname(__file__),
        "common",
        "test_model",
        "model_config_group",
        "scenarios_metadata.yaml",
    )

    scenario_metadata = AttrDict.from_yaml(metadata_yaml)
    test_vars = [
        (k, v["constraint"], v["group_name"], v["loc_techs"])
        for k, v in scenario_metadata.items()
    ]

    @pytest.mark.parametrize("scenario, constraints, group_names, loc_techs", test_vars)
    def test_build_group_constraint(
        self, scenario, constraints, group_names, loc_techs
    ):

        scenario_model = build_model(
            model_file=os.path.join("model_config_group", "base_model.yaml"),
            scenario=scenario,
        )

        assert all(
            "group_names_" + i in scenario_model._model_data.dims for i in constraints
        )
        assert all(
            group_names[i]
            in scenario_model._model_data["group_names_" + constraints[i]].values
            for i in range(len(constraints))
        )
        assert all(
            "group_" + i in scenario_model._model_data.data_vars.keys()
            for i in constraints
        )
        assert all(
            len(
                set(loc_techs[i])
                - set(
                    scenario_model._model_data.get(
                        "group_constraint_loc_techs_" + group_names[i],
                        scenario_model._model_data.get(
                            "group_constraint_loc_tech_carriers_" + group_names[i],
                            set(),
                        ),
                    ).values
                )
            )
            == 0
            for i in range(len(group_names))
        )

        scenario_model.run(build_only=True)
        assert all(
            hasattr(scenario_model._backend_model, "group_" + i + "_constraint")
            for i in constraints
        )


@pytest.mark.filterwarnings(
    "ignore:(?s).*Not all requested techs:calliope.exceptions.ModelWarning"
)
class TestGroupConstraints:
    def test_no_group_constraint(self):
        model = build_model(model_file="group_constraint_general.yaml")
        model.run()
        expensive_generation = (
            model.get_formatted_array("carrier_prod")
            .loc[{"techs": "expensive_supply"}]
            .sum()
            .item()
        )
        assert expensive_generation == 0

    def test_switched_off_group_constraint(self):
        model = build_model(
            model_file="group_constraint_general.yaml",
            scenario="switching_off_group_constraint",
        )
        model.run()
        expensive_generation = (
            model.get_formatted_array("carrier_prod")
            .loc[{"techs": "expensive_supply"}]
            .sum()
            .item()
        )
        assert expensive_generation == 0

    def test_group_constraint_with_several_constraints_infeasible(self):

        model = build_model(
            model_file="group_constraint_general.yaml",
            scenario="several_infeasible_group_constraints",
        )
        model.run()
        assert model._model_data.termination_condition != "optimal"

    def test_group_constraint_with_several_constraints_feasible(self):

        model = build_model(
            model_file="group_constraint_general.yaml",
            scenario="several_feasible_group_constraints",
        )
        model.run()
        assert model._model_data.energy_cap.loc["0::expensive_supply"].item() <= 6
        assert model._model_data.energy_cap.loc["0::expensive_supply"].item() >= 5

    @pytest.mark.parametrize(
        ("scenario", "message"),
        [
            (
                "several_w_carrier_group_constraints",
                "Can only handle one constraint in a group constraint if one of them is carrier-based",
            ),
            (
                "several_carriers_group_constraints",
                "Can only handle one carrier per group constraint that is carrier-based",
            ),
        ],
    )
    def test_group_constraint_with_several_carriers(self, scenario, message):

        with pytest.raises(calliope.exceptions.ModelError) as excinfo:
            build_model(model_file="group_constraint_general.yaml", scenario=scenario)

        assert check_error_or_warning(excinfo, message)


@pytest.mark.filterwarnings(
    "ignore:(?s).*Not all requested techs:calliope.exceptions.ModelWarning"
)
class TestDemandShareGroupConstraints:
    def test_no_demand_share_constraint(self):
        model = build_model(model_file="group_constraint_demand_share.yaml")
        model.run()
        expensive_generation = (
            model.get_formatted_array("carrier_prod")
            .to_dataframe()
            .reset_index()
            .groupby("techs")
            .carrier_prod.sum()
            .loc["expensive_elec_supply"]
        )
        assert expensive_generation == 0

    def test_systemwide_demand_share_max_constraint(self):
        model = build_model(
            model_file="group_constraint_demand_share.yaml",
            scenario="demand_share_max_systemwide",
        )
        model.run()
        cheap_generation = (
            model.get_formatted_array("carrier_prod")
            .loc[{"techs": "cheap_elec_supply", "carriers": "electricity"}]
            .sum()
        )
        demand = -1 * model.get_formatted_array("carrier_con").sum()
        # assert share in each timestep is 0.6
        assert (cheap_generation / demand).round(5) <= 0.3

    def test_systemwide_demand_share_min_constraint(self):
        model = build_model(
            model_file="group_constraint_demand_share.yaml",
            scenario="demand_share_min_systemwide",
        )
        model.run()
        expensive_generation = (
            model.get_formatted_array("carrier_prod")
            .loc[{"techs": "expensive_elec_supply", "carriers": "electricity"}]
            .sum()
        )
        demand = -1 * model.get_formatted_array("carrier_con").sum()
        # assert share in each timestep is 0.6
        assert (expensive_generation / demand).round(5) >= 0.6

    def test_systemwide_demand_share_equals_constraint(self):
        model = build_model(
            model_file="group_constraint_demand_share.yaml",
            scenario="demand_share_equals_systemwide",
        )
        model.run()
        expensive_generation = (
            model.get_formatted_array("carrier_prod")
            .loc[{"techs": "expensive_elec_supply", "carriers": "electricity"}]
            .sum()
        )
        demand = -1 * model.get_formatted_array("carrier_con").sum()
        # assert share in each timestep is 0.6
        assert (expensive_generation / demand).round(5) == 0.6

    def test_location_specific_demand_share_max_constraint(self):
        model = build_model(
            model_file="group_constraint_demand_share.yaml",
            scenario="demand_share_max_location_0",
        )
        model.run()
        generation = (
            model.get_formatted_array("carrier_prod")
            .sum(dim="timesteps")
            .loc[{"carriers": "electricity"}]
        )
        demand0 = (
            -model.get_formatted_array("carrier_con").loc[{"locs": "0"}].sum().item()
        )
        cheap_generation0 = generation.loc[
            {"locs": "0", "techs": "cheap_elec_supply"}
        ].item()
        expensive_generation1 = generation.loc[
            {"locs": "1", "techs": "expensive_elec_supply"}
        ].item()
        assert round(cheap_generation0 / demand0, 5) <= 0.3
        assert expensive_generation1 == 0

    def test_location_specific_demand_share_max_constraint_two_techs(self):
        model = build_model(
            model_file="group_constraint_demand_share.yaml",
            scenario="demand_share_max_location_0_two_techs",
        )
        model.run()
        generation = (
            model.get_formatted_array("carrier_prod")
            .sum(dim="timesteps")
            .loc[{"carriers": "electricity"}]
        )
        demand0 = (
            -model.get_formatted_array("carrier_con").loc[{"locs": "0"}].sum().item()
        )
        generation0 = (
            generation.loc[
                {"locs": "0", "techs": ["cheap_elec_supply", "normal_elec_supply"]}
            ]
            .sum("techs")
            .item()
        )
        assert round(generation0 / demand0, 5) <= 0.4

    def test_location_specific_demand_share_min_constraint(self):
        model = build_model(
            model_file="group_constraint_demand_share.yaml",
            scenario="demand_share_min_location_0",
        )
        model.run()
        generation = (
            model.get_formatted_array("carrier_prod")
            .sum(dim="timesteps")
            .loc[{"carriers": "electricity"}]
        )
        demand0 = (
            -model.get_formatted_array("carrier_con").loc[{"locs": "0"}].sum().item()
        )
        expensive_generation0 = generation.loc[
            {"locs": "0", "techs": "expensive_elec_supply"}
        ].item()
        expensive_generation1 = generation.loc[
            {"locs": "1", "techs": "expensive_elec_supply"}
        ].item()
        assert round(expensive_generation0 / demand0, 5) >= 0.6
        assert expensive_generation1 == 0

    def test_multiple_group_constraints(self):
        model = build_model(
            model_file="group_constraint_demand_share.yaml",
            scenario="multiple_constraints",
        )
        model.run()
        generation = model.get_formatted_array("carrier_prod").sum(
            dim=("timesteps", "locs", "carriers")
        )
        demand = -model.get_formatted_array("carrier_con").sum().item()
        cheap_generation = generation.loc[{"techs": "cheap_elec_supply"}].item()
        expensive_generation = generation.loc[{"techs": "expensive_elec_supply"}].item()

        assert round(expensive_generation / demand, 5) >= 0.6
        assert round(cheap_generation / demand, 5) <= 0.3

    def test_multiple_group_carriers(self):
        model = build_model(
            model_file="group_constraint_demand_share.yaml",
            scenario="multiple_carriers_max",
        )
        model.run()
        generation = model.get_formatted_array("carrier_prod").sum(
            dim=("timesteps", "locs")
        )
        demand = -model.get_formatted_array("carrier_con").sum(
            dim=("timesteps", "locs")
        )
        cheap_generation_elec = generation.loc[
            {"techs": "cheap_elec_supply", "carriers": "electricity"}
        ].item()
        demand_elec = demand.loc[
            {"techs": "electricity_demand", "carriers": "electricity"}
        ].item()
        cheap_generation_heat = generation.loc[
            {"techs": "cheap_heat_supply", "carriers": "heat"}
        ].item()
        demand_heat = demand.loc[{"techs": "heat_demand", "carriers": "heat"}].item()

        assert round(cheap_generation_elec / demand_elec, 5) <= 0.3
        assert round(cheap_generation_heat / demand_heat, 5) <= 0.5

    def test_multiple_group_carriers_constraints(self):
        model = build_model(
            model_file="group_constraint_demand_share.yaml",
            scenario="multiple_constraints_carriers",
        )
        model.run()
        generation = model.get_formatted_array("carrier_prod").sum(
            dim=("timesteps", "locs")
        )
        demand = -model.get_formatted_array("carrier_con").sum(
            dim=("timesteps", "locs")
        )
        cheap_generation_elec = generation.loc[
            {"techs": "cheap_elec_supply", "carriers": "electricity"}
        ].item()
        expensive_generation_elec = generation.loc[
            {"techs": "expensive_elec_supply", "carriers": "electricity"}
        ].item()
        demand_elec = demand.loc[
            {"techs": "electricity_demand", "carriers": "electricity"}
        ].item()
        cheap_generation_heat = generation.loc[
            {"techs": "cheap_heat_supply", "carriers": "heat"}
        ].item()
        expensive_generation_heat = generation.loc[
            {"techs": "expensive_heat_supply", "carriers": "heat"}
        ].item()
        demand_heat = demand.loc[{"techs": "heat_demand", "carriers": "heat"}].item()

        assert round(cheap_generation_elec / demand_elec, 5) <= 0.3
        assert round(expensive_generation_elec / demand_elec, 5) >= 0.6
        assert round(cheap_generation_heat / demand_heat, 5) <= 0.5
        assert round(expensive_generation_heat / demand_heat, 5) >= 0.4

    def test_different_locations_per_group_constraint(self):
        model = build_model(
            model_file="group_constraint_demand_share.yaml",
            scenario="different_locations_per_group",
        )
        model.run()
        generation = model.get_formatted_array("carrier_prod").sum(
            dim=("timesteps", "carriers")
        )
        demand = -model.get_formatted_array("carrier_con").sum(dim=("timesteps"))
        cheap_generation_0 = generation.loc[
            {"techs": "cheap_elec_supply", "locs": "0"}
        ].item()
        expensive_generation_0 = generation.loc[
            {"techs": "expensive_elec_supply", "locs": "0"}
        ].item()
        cheap_generation_1 = generation.loc[
            {"techs": "cheap_elec_supply", "locs": "1"}
        ].item()
        expensive_generation_1 = generation.loc[
            {"techs": "expensive_elec_supply", "locs": "1"}
        ].item()
        demand_elec_0 = demand.loc[
            {"techs": "electricity_demand", "carriers": "electricity", "locs": "0"}
        ].item()
        demand_elec_1 = demand.loc[
            {"techs": "electricity_demand", "carriers": "electricity", "locs": "1"}
        ].item()

        assert round(expensive_generation_0 / demand_elec_0, 5) >= 0.6
        assert expensive_generation_1 / demand_elec_1 == 0
        assert (
            round(
                (cheap_generation_0 + cheap_generation_1)
                / (demand_elec_0 + demand_elec_1),
                5,
            )
            <= 0.3
        )

    def test_transmission_not_included_in_demand(self):
        model = build_model(
            model_file="group_constraint_demand_share.yaml",
            scenario="transmission_not_included_in_demand",
        )
        model.run()
        assert model.results.termination_condition == "optimal"
        generation = model.get_formatted_array("carrier_prod").sum(
            dim=("timesteps", "carriers")
        )
        demand = -model.get_formatted_array("carrier_con").sum(
            dim=("timesteps", "carriers")
        )

        assert (
            generation.sel(
                locs="1",
                techs=[
                    "normal_elec_supply",
                    "cheap_elec_supply",
                    "expensive_elec_supply",
                ],
            )
            .sum(dim="techs")
            .item()
        ) == approx(0)

        cheap_elec_supply_0 = generation.sel(locs="0", techs="cheap_elec_supply").item()
        demand_0 = demand.sel(locs="0", techs="electricity_demand").item()

        assert round(cheap_elec_supply_0 / demand_0, 5) <= 0.4

    def test_demand_share_per_timestep_max(self):
        model = build_model(
            model_file="group_constraint_demand_share.yaml",
            scenario="demand_share_per_timestep_max",
        )
        model.run()
        cheap_generation = (
            model.get_formatted_array("carrier_prod")
            .loc[{"techs": "cheap_elec_supply", "carriers": "electricity"}]
            .sum("locs")
        )
        demand = -1 * model.get_formatted_array("carrier_con").sum("locs")
        # assert share in each timestep is 0.6
        assert ((cheap_generation / demand).round(5) <= 0.3).all()

    def test_demand_share_per_timestep_min(self):
        model = build_model(
            model_file="group_constraint_demand_share.yaml",
            scenario="demand_share_per_timestep_min",
        )
        model.run()
        expensive_generation = (
            model.get_formatted_array("carrier_prod")
            .loc[{"techs": "expensive_elec_supply", "carriers": "electricity"}]
            .sum("locs")
        )
        demand = -1 * model.get_formatted_array("carrier_con").sum("locs")
        # assert share in each timestep is 0.6
        assert ((expensive_generation / demand).round(5) >= 0.6).all()

    def test_demand_share_per_timestep_equals(self):
        model = build_model(
            model_file="group_constraint_demand_share.yaml",
            scenario="demand_share_per_timestep_equals",
        )
        model.run()
        expensive_generation = (
            model.get_formatted_array("carrier_prod")
            .loc[{"techs": "expensive_elec_supply", "carriers": "electricity"}]
            .sum("locs")
        )
        demand = -1 * model.get_formatted_array("carrier_con").sum("locs")
        # assert share in each timestep is 0.6
        assert ((expensive_generation / demand).round(5) == 0.6).all()


@pytest.mark.filterwarnings(
    "ignore:(?s).*Not all requested techs:calliope.exceptions.ModelWarning"
)
class TestDemandShareDecisionGroupConstraints:
    @staticmethod
    def get_shares(model, sumlocs=True, carrier="electricity", con_contains=""):
        demand = -1 * model.get_formatted_array("carrier_con").loc[
            {"carriers": carrier}
        ].where(lambda x: x.techs.str.contains(con_contains)).sum("techs", min_count=1)
        supply = model.get_formatted_array("carrier_prod").loc[{"carriers": carrier}]
        if sumlocs:
            return supply.sum("locs", min_count=1) / demand.sum("locs", min_count=1)
        else:
            return supply / demand

    def test_demand_share_per_timestep_decision_inf(self):
        model = build_model(
            model_file="group_constraint_demand_share_decision.yaml",
            scenario="demand_share_per_timestep_decision_inf",
        )
        model.run()
        shares = self.get_shares(model)

        assert all(shares.loc[{"techs": "cheap_elec_supply"}] == 0.1875)
        assert all(shares.loc[{"techs": "normal_elec_supply"}] == 0.8125)

    def test_demand_share_per_timestep_decision_simple(self):
        model = build_model(
            model_file="group_constraint_demand_share_decision.yaml",
            scenario="demand_share_per_timestep_decision_simple",
        )
        model.run()
        shares = self.get_shares(model)

        assert all(shares.sel(techs="cheap_elec_supply") == 0.1875)
        assert all(shares.sel(techs="normal_elec_supply") == 0.8125)
        assert model._model_data.demand_share_per_timestep_decision.sum().item() == 1

    def test_demand_share_per_timestep_decision_not_one(self):
        model = build_model(
            model_file="group_constraint_demand_share_decision.yaml",
            scenario="demand_share_per_timestep_decision_not_one",
        )
        model.run()
        shares = self.get_shares(model)

        assert all(shares.sel(techs="cheap_elec_supply") == 0.1)
        assert all(shares.sel(techs="normal_elec_supply") == 0.9)
        assert all(shares.sel(techs="expensive_elec_supply") == 0)
        assert model._model_data.demand_share_per_timestep_decision.sum().item() == 0.9

    def test_demand_share_per_timestep_decision_per_location(self):
        model = build_model(
            model_file="group_constraint_demand_share_decision.yaml",
            scenario="demand_share_per_timestep_decision_per_location",
        )
        model.run()
        shares = self.get_shares(model, sumlocs=False)

        assert all(shares.sel(techs="cheap_elec_supply", locs="0") == 0.25)
        assert all(shares.sel(techs="normal_elec_supply", locs="0") == 0.75)

        assert all(shares.sel(techs="cheap_elec_supply", locs="1") == 0.125)
        assert all(shares.sel(techs="normal_elec_supply", locs="1") == 0.875)

    def test_demand_share_per_timestep_decision_inf_with_transmission(self):
        model = build_model(
            model_file="group_constraint_demand_share_decision.yaml",
            scenario="demand_share_per_timestep_decision_inf,with_electricity_transmission",
        )
        model.run()
        shares = self.get_shares(model, sumlocs=False)

        assert all(shares.sel(techs="cheap_elec_supply", locs="0") == 0.125)
        assert all(shares.sel(techs="normal_elec_supply", locs="0") == 0.875)
        assert all(shares.sel(techs="electricity_transmission:0", locs="1") == 1.0)

    def test_demand_share_per_timestep_decision_inf_with_heat_constrain_electricity(
        self,
    ):
        model = build_model(
            model_file="group_constraint_demand_share_decision.yaml",
            scenario="demand_share_per_timestep_decision_inf,with_electricity_conversion_tech",
        )
        model.run()
        shares = self.get_shares(model)

        assert all(shares.sel(techs="cheap_elec_supply") == approx(0.1759259))
        assert all(shares.sel(techs="normal_elec_supply") == approx(0.82407407))

    def test_demand_share_per_timestep_decision_inf_with_heat_constrain_heat(self):
        model = build_model(
            model_file="group_constraint_demand_share_decision.yaml",
            scenario="demand_share_per_timestep_decision_inf_with_heat,with_electricity_conversion_tech",
        )
        model.run()
        shares = self.get_shares(model, carrier="heat")

        assert all(shares.sel(techs="elec_to_heat") == approx(0.5))
        assert all(shares.sel(techs="heating") == approx(0.5))

    def test_demand_share_per_timestep_decision_inf_with_heat_constrain_heat_and_electricity(
        self,
    ):
        model = build_model(
            model_file="group_constraint_demand_share_decision.yaml",
            scenario="demand_share_per_timestep_decision_inf_with_heat,demand_share_per_timestep_decision_not_one,with_electricity_conversion_tech",
        )
        model.run()
        shares_heat = self.get_shares(model, carrier="heat", con_contains="demand")
        shares_elec = self.get_shares(
            model, carrier="electricity", con_contains="demand"
        )

        assert all(shares_heat.sel(techs="elec_to_heat") == 0.5)
        assert all(shares_heat.sel(techs="heating") == 0.5)
        assert all(shares_elec.sel(techs="normal_elec_supply") == approx(0.9))
        assert all(shares_elec.sel(techs="cheap_elec_supply").round(5) >= 0.1)

    @pytest.mark.parametrize("relax", [0, 0.01, 0.05, 0.1])
    def test_demand_share_per_timestep_decision_relax(self, relax):
        model = build_model(
            model_file="group_constraint_demand_share_decision.yaml",
            scenario="demand_share_per_timestep_decision_not_one",
            override_dict={
                "run.relax_constraint.demand_share_per_timestep_decision_main_constraint": relax
            },
        )
        model.run()
        shares = self.get_shares(model)

        assert all(shares.sel(techs="expensive_elec_supply") == 0)
        assert model._model_data.demand_share_per_timestep_decision.sum().item() == 0.9

        if relax == 0:
            assert all(shares.sel(techs="cheap_elec_supply") == 0.1)
            assert all(shares.sel(techs="normal_elec_supply") == 0.9)
        else:
            assert not all(shares.sel(techs="cheap_elec_supply") == 0.1)
            assert not all(shares.sel(techs="normal_elec_supply") == 0.9)
            assert all(
                shares.sel(techs="normal_elec_supply").round(5) >= 0.9 * (1 - relax)
            )
            assert all(
                shares.sel(techs="normal_elec_supply").round(5) <= 0.9 * (1 + relax)
            )


@pytest.mark.filterwarnings(
    "ignore:(?s).*Not all requested techs:calliope.exceptions.ModelWarning"
)
class TestResourceAreaGroupConstraints:
    def test_no_resource_area_constraint(self, model_file):
        model = build_model(model_file=model_file)
        model.run()
        capacity = model.get_formatted_array("resource_area")
        expensive_capacity = (
            capacity.loc[{"techs": "expensive_elec_supply"}].sum().item()
        )
        assert expensive_capacity == 0

    def test_resource_area_max_supply_constraint(self, model_file):
        model = build_model(model_file=model_file, scenario="resource_area_max_supply")
        model.run()
        capacity = model.get_formatted_array("resource_area")
        cheap_capacity = capacity.loc[{"techs": "expensive_elec_supply"}].sum().item()
        assert round(cheap_capacity, 5) <= 28

    def test_resource_area_min_supply_constraint(self, model_file):
        model = build_model(model_file=model_file, scenario="resource_area_min_supply")
        model.run()
        capacity = model.get_formatted_array("resource_area")
        expensive_capacity = (
            capacity.loc[{"techs": "expensive_elec_supply"}].sum().item()
        )
        assert round(expensive_capacity, 5) >= 12

    def test_resource_area_equals_supply_constraint(self, model_file):
        model = build_model(
            model_file=model_file, scenario="resource_area_equals_supply"
        )
        model.run()
        capacity = model.get_formatted_array("resource_area")
        expensive_capacity = (
            capacity.loc[{"techs": "expensive_elec_supply"}].sum().item()
        )
        assert expensive_capacity == approx(20)

    def test_resource_area_min_max_supply_constraint(self, model_file):
        model = build_model(
            model_file=model_file, scenario="resource_area_min_max_supply"
        )
        model.run()
        capacity = model.get_formatted_array("resource_area")
        cheap_capacity = capacity.loc[{"techs": "cheap_elec_supply"}].sum().item()
        expensive_capacity = (
            capacity.loc[{"techs": "expensive_elec_supply"}].sum().item()
        )
        assert round(cheap_capacity, 5) <= 8
        assert round(expensive_capacity, 5) >= 12

    def test_resource_area_max_supply_loc_1_constraint(self, model_file):
        model = build_model(
            model_file=model_file, scenario="resource_area_max_supply_loc_1"
        )
        model.run()
        capacity = model.get_formatted_array("resource_area")
        cheap_capacity1 = capacity.loc[
            {"locs": "1", "techs": "cheap_elec_supply"}
        ].item()
        expensive_capacity0 = capacity.loc[
            {"locs": "0", "techs": "expensive_elec_supply"}
        ].item()
        assert round(cheap_capacity1, 5) <= 8
        assert expensive_capacity0 == 0

    def test_resource_area_min_supply_loc_0_constraint(self, model_file):
        model = build_model(
            model_file=model_file, scenario="resource_area_min_supply_loc_0"
        )
        model.run()
        capacity = model.get_formatted_array("resource_area")
        expensive_capacity0 = capacity.loc[
            {"locs": "0", "techs": "expensive_elec_supply"}
        ].item()
        expensive_capacity1 = capacity.loc[
            {"locs": "1", "techs": "expensive_elec_supply"}
        ].item()
        assert round(expensive_capacity0, 5) >= 12
        assert expensive_capacity1 == 0

    def test_resource_area_min_max_supply_loc0_1_constraint(self, model_file):
        model = build_model(
            model_file=model_file, scenario="resource_area_min_max_supply_loc_0_1"
        )
        model.run()
        capacity = model.get_formatted_array("resource_area")
        cheap_capacity1 = capacity.loc[
            {"locs": "1", "techs": "cheap_elec_supply"}
        ].item()
        expensive_capacity0 = capacity.loc[
            {"locs": "0", "techs": "expensive_elec_supply"}
        ].item()
        expensive_capacity1 = capacity.loc[
            {"locs": "1", "techs": "expensive_elec_supply"}
        ].item()
        assert round(cheap_capacity1, 5) <= 8
        assert round(expensive_capacity0, 5) >= 12
        assert expensive_capacity1 == 0

    # All technologies, but insufficient resource_area_max for enough installed capacity to meet demand
    @pytest.mark.filterwarnings(
        "ignore:(?s).*['conversion', 'conversion_plus', 'demand', 'storage', "
        "'transmission'].*:calliope.exceptions.ModelWarning"
    )
    def test_resource_area_max_all_techs_infeasible_constraint(self, model_file):
        model = build_model(
            model_file=model_file, scenario="resource_area_max_all_techs_infeasible"
        )

        model.run()

        assert model._model_data.attrs["termination_condition"] != "optimal"


@pytest.mark.filterwarnings(
    "ignore:(?s).*Not all requested techs:calliope.exceptions.ModelWarning"
)
class TestCostCapGroupConstraint:
    def test_systemwide_cost_max_constraint(self):
        model = build_model(
            model_file="group_constraint_cost_cap.yaml",
            scenario="cheap_cost_max_systemwide",
        )
        model.run()
        cheap_cost = (
            (
                model.get_formatted_array("cost").loc[
                    {"costs": "monetary", "techs": "cheap_polluting_supply"}
                ]
            )
            .sum()
            .item()
        )
        assert round(cheap_cost, 5) <= 30

    def test_systemwide_cost_investment_max_constraint(self):
        model = build_model(
            model_file="group_constraint_cost_cap.yaml",
            scenario="cheap_cost_investment_max_systemwide",
        )
        model.run()
        cheap_cost = (
            (
                model.get_formatted_array("cost").loc[
                    {"costs": "monetary", "techs": "cheap_polluting_supply"}
                ]
            )
            .sum()
            .item()
        )
        cheap_cost_investment = (
            (
                model.get_formatted_array("cost_investment").loc[
                    {"costs": "monetary", "techs": "cheap_polluting_supply"}
                ]
            )
            .sum()
            .item()
        )
        assert cheap_cost > cheap_cost_investment
        assert round(cheap_cost_investment, 5) <= 4

    def test_systemwide_cost_var_max_constraint(self):
        model = build_model(
            model_file="group_constraint_cost_cap.yaml",
            scenario="cheap_cost_var_max_systemwide",
        )
        model.run()
        cheap_cost = (
            (
                model.get_formatted_array("cost").loc[
                    {"costs": "monetary", "techs": "cheap_polluting_supply"}
                ]
            )
            .sum()
            .item()
        )
        cheap_cost_var = (
            (
                model.get_formatted_array("cost_var").loc[
                    {"costs": "monetary", "techs": "cheap_polluting_supply"}
                ]
            )
            .sum()
            .item()
        )
        assert cheap_cost > cheap_cost_var
        assert round(cheap_cost_var, 5) <= 200

    def test_systemwide_cost_min_constraint(self):
        model = build_model(
            model_file="group_constraint_cost_cap.yaml",
            scenario="expensive_cost_min_systemwide",
        )
        model.run()
        expensive_cost = (
            (
                model.get_formatted_array("cost").loc[
                    {"costs": "monetary", "techs": "expensive_clean_supply"}
                ]
            )
            .sum()
            .item()
        )
        assert round(expensive_cost, 5) >= 600

    def test_systemwide_cost_equals_constraint(self):
        model = build_model(
            model_file="group_constraint_cost_cap.yaml",
            scenario="cheap_cost_equals_systemwide",
        )
        model.run()
        cheap_cost = (
            (
                model.get_formatted_array("cost").loc[
                    {"costs": "monetary", "techs": "cheap_polluting_supply"}
                ]
            )
            .sum()
            .item()
        )
        assert cheap_cost == approx(210)

    def test_location_specific_cost_max_constraint(self):
        model = build_model(
            model_file="group_constraint_cost_cap.yaml",
            scenario="cheap_cost_max_location_0",
        )
        model.run()
        cheap_cost0 = (
            (
                model.get_formatted_array("cost").loc[
                    {
                        "costs": "monetary",
                        "techs": "cheap_polluting_supply",
                        "locs": "0",
                    }
                ]
            )
            .sum()
            .item()
        )
        assert round(cheap_cost0, 5) <= 10

    def test_systemwide_emissions_max_constraint(self):
        model = build_model(
            model_file="group_constraint_cost_cap.yaml",
            scenario="emissions_max_systemwide",
        )
        model.run()
        emissions = (
            (model.get_formatted_array("cost").loc[{"costs": "emissions"}]).sum().item()
        )
        assert round(emissions, 5) <= 400

    def test_location_specific_emissions_max_constraint(self):
        model = build_model(
            model_file="group_constraint_cost_cap.yaml",
            scenario="emissions_max_location_0",
        )
        model.run()
        emissions0 = (
            (model.get_formatted_array("cost").loc[{"costs": "emissions", "locs": "0"}])
            .sum()
            .item()
        )
        assert round(emissions0, 5) <= 200

    def test_systemwide_clean_emissions_max_constraint(self):
        model = build_model(
            model_file="group_constraint_cost_cap.yaml",
            scenario="clean_emissions_max_systemwide",
        )
        model.run()
        clean_emissions = (
            (
                model.get_formatted_array("cost").loc[
                    {"costs": "emissions", "techs": "expensive_clean_supply"}
                ]
            )
            .sum()
            .item()
        )
        assert round(clean_emissions, 5) <= 300

    def test_multiple_costs_constraint(self):
        model = build_model(
            model_file="group_constraint_cost_cap.yaml",
            scenario="multiple_costs_constraint",
        )
        model.run()
        emissions = (
            (model.get_formatted_array("cost").loc[{"costs": "emissions"}]).sum().item()
        )
        expensive_cost = (
            (
                model.get_formatted_array("cost").loc[
                    {"costs": "monetary", "techs": "expensive_clean_supply"}
                ]
            )
            .sum()
            .item()
        )
        assert round(emissions, 5) <= 400
        assert round(expensive_cost, 5) <= 600

    def test_different_locations_per_cost_group_constraint(self):
        model = build_model(
            model_file="group_constraint_cost_cap.yaml",
            scenario="different_locations_per_group",
        )
        model.run()
        cheap_cost = (
            (
                model.get_formatted_array("cost").loc[
                    {"costs": "monetary", "techs": "cheap_polluting_supply"}
                ]
            )
            .sum()
            .item()
        )
        cheap_cost0 = (
            (
                model.get_formatted_array("cost").loc[
                    {
                        "costs": "monetary",
                        "techs": "cheap_polluting_supply",
                        "locs": "0",
                    }
                ]
            )
            .sum()
            .item()
        )
        assert round(cheap_cost, 5) <= 30
        assert round(cheap_cost0, 5) <= 10

    def test_different_techs_per_cost_group_constraint(self):
        model = build_model(
            model_file="group_constraint_cost_cap.yaml",
            scenario="different_techs_per_group",
        )
        model.run()
        emissions = (
            (model.get_formatted_array("cost").loc[{"costs": "emissions"}]).sum().item()
        )
        clean_emissions = (
            (
                model.get_formatted_array("cost").loc[
                    {"costs": "emissions", "techs": "expensive_clean_supply"}
                ]
            )
            .sum()
            .item()
        )
        assert round(emissions, 5) <= 400
        assert round(clean_emissions, 5) <= 300


@pytest.mark.filterwarnings(
    "ignore:(?s).*Not all requested techs:calliope.exceptions.ModelWarning"
)
class TestSupplyShareGroupConstraints:
    def test_no_carrier_prod_share_constraint(self):
        model = build_model(model_file="group_constraint_carrier_prod_share.yaml")
        model.run()
        expensive_generation = (
            model.get_formatted_array("carrier_prod")
            .loc[{"techs": "expensive_supply"}]
            .sum()
        ).item()
        assert expensive_generation == 0

    def test_systemwide_carrier_prod_share_max_constraint(self):
        model = build_model(
            model_file="group_constraint_carrier_prod_share.yaml",
            scenario="carrier_prod_share_max_systemwide",
        )
        model.run()
        cheap_generation = (
            model.get_formatted_array("carrier_prod")
            .to_dataframe()
            .reset_index()
            .groupby("techs")
            .carrier_prod.sum()
            .transform(lambda x: x / x.sum())
            .loc["cheap_supply"]
        )
        assert round(cheap_generation, 5) <= 0.4

    def test_systemwide_carrier_prod_share_min_constraint(self):
        model = build_model(
            model_file="group_constraint_carrier_prod_share.yaml",
            scenario="carrier_prod_share_min_systemwide",
        )
        model.run()
        expensive_generation = (
            model.get_formatted_array("carrier_prod")
            .to_dataframe()
            .reset_index()
            .groupby("techs")
            .carrier_prod.sum()
            .transform(lambda x: x / x.sum())
            .loc["expensive_supply"]
        )
        assert round(expensive_generation, 5) >= 0.6

    def test_location_specific_carrier_prod_share_max_constraint(self):
        model = build_model(
            model_file="group_constraint_carrier_prod_share.yaml",
            scenario="carrier_prod_share_max_location_0",
        )
        model.run()
        generation = (
            model.get_formatted_array("carrier_prod")
            .sum(dim="timesteps")
            .loc[{"carriers": "electricity"}]
        )
        cheap_generation0 = generation.loc[
            {"locs": "0", "techs": "cheap_supply"}
        ].item()
        expensive_generation0 = generation.loc[
            {"locs": "0", "techs": "expensive_supply"}
        ].item()
        expensive_generation1 = generation.loc[
            {"locs": "1", "techs": "expensive_supply"}
        ].item()
        assert (
            round(cheap_generation0 / (cheap_generation0 + expensive_generation0), 5)
            <= 0.4
        )
        assert expensive_generation1 == 0

    def test_location_specific_carrier_prod_share_min_constraint(self):
        model = build_model(
            model_file="group_constraint_carrier_prod_share.yaml",
            scenario="carrier_prod_share_min_location_0",
        )
        model.run()
        generation = (
            model.get_formatted_array("carrier_prod")
            .sum(dim="timesteps")
            .loc[{"carriers": "electricity"}]
        )
        cheap_generation0 = generation.loc[
            {"locs": "0", "techs": "cheap_supply"}
        ].item()
        expensive_generation0 = generation.loc[
            {"locs": "0", "techs": "expensive_supply"}
        ].item()
        expensive_generation1 = generation.loc[
            {"locs": "1", "techs": "expensive_supply"}
        ].item()
        assert (
            round(
                expensive_generation0 / (cheap_generation0 + expensive_generation0), 5
            )
            >= 0.6
        )
        assert expensive_generation1 == 0

    def test_carrier_prod_share_with_transmission(self):
        model = build_model(
            model_file="group_constraint_carrier_prod_share.yaml",
            scenario="carrier_prod_share_min_systemwide,transmission_link",
        )
        model.run()
        expensive_generation = (
            model.get_formatted_array("carrier_prod")
            .loc[{"techs": "expensive_supply"}]
            .sum()
            .item()
        )
        supply = (
            model._model_data.carrier_prod.loc[
                {
                    "loc_tech_carriers_prod": model._model_data.loc_tech_carriers_supply_conversion_all.values
                }
            ]
            .sum()
            .item()
        )

        assert round(expensive_generation / supply, 5) >= 0.6

    def test_carrier_prod_share_with_storage(self):
        model = build_model(
            model_file="group_constraint_carrier_prod_share.yaml",
            scenario="carrier_prod_share_min_systemwide,storage_tech",
        )
        model.run()
        expensive_generation = (
            model.get_formatted_array("carrier_prod")
            .loc[{"techs": "expensive_supply"}]
            .sum()
            .item()
        )
        supply = (
            model._model_data.carrier_prod.loc[
                {
                    "loc_tech_carriers_prod": model._model_data.loc_tech_carriers_supply_conversion_all.values
                }
            ]
            .sum()
            .item()
        )

        assert round(expensive_generation / supply, 5) >= 0.6

    def test_carrier_prod_share_per_timestep_max(self):
        model = build_model(
            model_file="group_constraint_carrier_prod_share.yaml",
            scenario="carrier_prod_share_per_timestep_max",
        )
        model.run()
        cheap_elec_supply = (
            model.get_formatted_array("carrier_prod")
            .loc[{"techs": "cheap_supply", "carriers": "electricity"}]
            .sum("locs")
        )
        supply = (
            model.get_formatted_array("carrier_prod")
            .loc[{"carriers": "electricity"}]
            .sum("locs")
            .sum("techs")
        )
        # assert share in each timestep is 0.4
        assert ((cheap_elec_supply / supply).round(5) <= 0.4).all()

    def test_carrier_prod_share_per_timestep_min(self):
        model = build_model(
            model_file="group_constraint_carrier_prod_share.yaml",
            scenario="carrier_prod_share_per_timestep_min",
        )
        model.run()
        expensive_elec_supply = (
            model.get_formatted_array("carrier_prod")
            .loc[{"techs": "expensive_supply", "carriers": "electricity"}]
            .sum("locs")
        )
        supply = (
            model.get_formatted_array("carrier_prod")
            .loc[{"carriers": "electricity"}]
            .sum("locs")
            .sum("techs")
        )
        # assert share in each timestep is 0.6
        assert ((expensive_elec_supply / supply).round(5) >= 0.6).all()

    def test_carrier_prod_share_per_timestep_equals(self):
        model = build_model(
            model_file="group_constraint_carrier_prod_share.yaml",
            scenario="carrier_prod_share_per_timestep_equals",
        )
        model.run()
        expensive_elec_supply = (
            model.get_formatted_array("carrier_prod")
            .loc[{"techs": "expensive_supply", "carriers": "electricity"}]
            .sum("locs")
        )
        supply = (
            model.get_formatted_array("carrier_prod")
            .loc[{"carriers": "electricity"}]
            .sum("locs")
            .sum("techs")
        )
        # assert share in each timestep is 0.6
        assert ((expensive_elec_supply / supply).round(5) == 0.6).all()


class TestEnergyCapShareGroupConstraints:
    @pytest.fixture(scope="module")
    def model_file(self):
        return os.path.join("model_config_group", "base_model.yaml")

    def test_no_energy_cap_share_constraint(self, model_file):
        model = build_model(model_file=model_file)
        model.run()
        capacity = model.get_formatted_array("energy_cap")
        expensive_capacity = (
            capacity.loc[{"techs": "expensive_elec_supply"}].sum().item()
        )
        assert expensive_capacity == 0

    def test_energy_cap_share_max_supply_constraint(self, model_file):
        model = build_model(
            model_file=model_file, scenario="energy_cap_share_max_supply"
        )
        model.run()
        capacity = model.get_formatted_array("energy_cap")
        capacity_supply_conversion_all = capacity.loc[
            {"techs": get_supply_conversion_techs(model)}
        ]
        cheap_capacity = capacity.loc[{"techs": "expensive_elec_supply"}].sum().item()
        assert (
            round(cheap_capacity / capacity_supply_conversion_all.sum().item(), 5)
            <= 0.5
        )

    def test_energy_cap_share_min_supply_constraint(self, model_file):
        model = build_model(
            model_file=model_file, scenario="energy_cap_share_min_supply"
        )
        model.run()
        capacity = model.get_formatted_array("energy_cap")
        capacity_supply_conversion_all = capacity.loc[
            {"techs": get_supply_conversion_techs(model)}
        ]
        expensive_capacity = (
            capacity.loc[{"techs": "expensive_elec_supply"}].sum().item()
        )
        assert (
            round(expensive_capacity / capacity_supply_conversion_all.sum().item(), 5)
            >= 0.4
        )

    def test_energy_cap_share_equals_supply_constraint(self, model_file):
        model = build_model(
            model_file=model_file, scenario="energy_cap_share_equals_supply"
        )
        model.run()
        capacity = model.get_formatted_array("energy_cap")
        capacity_supply_conversion_all = capacity.loc[
            {"techs": get_supply_conversion_techs(model)}
        ]
        expensive_capacity = (
            capacity.loc[{"techs": "expensive_elec_supply"}].sum().item()
        )
        assert (
            expensive_capacity / capacity_supply_conversion_all.sum().item()
            == approx(0.3)
        )

    def test_energy_cap_share_min_max_supply_constraint(self, model_file):
        model = build_model(
            model_file=model_file, scenario="energy_cap_share_min_max_supply"
        )
        model.run()
        capacity = model.get_formatted_array("energy_cap")
        capacity_supply_conversion_all = capacity.loc[
            {"techs": get_supply_conversion_techs(model)}
        ]
        cheap_capacity = capacity.loc[{"techs": "cheap_elec_supply"}].sum().item()
        expensive_capacity = (
            capacity.loc[{"techs": "expensive_elec_supply"}].sum().item()
        )
        assert (
            round(cheap_capacity / capacity_supply_conversion_all.sum().item(), 5)
            <= 0.2
        )
        assert (
            round(expensive_capacity / capacity_supply_conversion_all.sum().item(), 5)
            >= 0.4
        )

    def test_energy_cap_share_max_supply_loc_1_constraint(self, model_file):
        model = build_model(
            model_file=model_file, scenario="energy_cap_share_max_supply_loc_1"
        )
        model.run()
        capacity = model.get_formatted_array("energy_cap")
        capacity_supply_conversion_all_loc1 = capacity.loc[
            {"locs": "1", "techs": get_supply_conversion_techs(model)}
        ]
        cheap_capacity1 = capacity.loc[
            {"locs": "1", "techs": "cheap_elec_supply"}
        ].item()
        expensive_capacity0 = capacity.loc[
            {"locs": "0", "techs": "expensive_elec_supply"}
        ].item()
        assert (
            round(cheap_capacity1 / capacity_supply_conversion_all_loc1.sum().item(), 5)
            <= 0.2
        )
        assert expensive_capacity0 == 0

    def test_energy_cap_share_min_supply_loc_0_constraint(self, model_file):
        model = build_model(
            model_file=model_file, scenario="energy_cap_share_min_supply_loc_0"
        )
        model.run()
        capacity = model.get_formatted_array("energy_cap")
        capacity_supply_conversion_all_loc0 = capacity.loc[
            {"locs": "0", "techs": get_supply_conversion_techs(model)}
        ]
        expensive_capacity0 = capacity.loc[
            {"locs": "0", "techs": "expensive_elec_supply"}
        ].item()
        expensive_capacity1 = capacity.loc[
            {"locs": "1", "techs": "expensive_elec_supply"}
        ].item()
        assert (
            round(
                expensive_capacity0 / capacity_supply_conversion_all_loc0.sum().item(),
                5,
            )
            >= 0.4
        )
        assert expensive_capacity1 == 0

    def test_energy_cap_share_min_max_supply_loc0_1_constraint(self, model_file):
        model = build_model(
            model_file=model_file, scenario="energy_cap_share_min_max_supply_loc_0_1"
        )
        model.run()
        capacity = model.get_formatted_array("energy_cap")
        capacity_supply_conversion_all_loc0 = capacity.loc[
            {"locs": "0", "techs": get_supply_conversion_techs(model)}
        ]
        capacity_supply_conversion_all_loc1 = capacity.loc[
            {"locs": "1", "techs": get_supply_conversion_techs(model)}
        ]
        cheap_capacity1 = capacity.loc[
            {"locs": "1", "techs": "cheap_elec_supply"}
        ].item()
        expensive_capacity0 = capacity.loc[
            {"locs": "0", "techs": "expensive_elec_supply"}
        ].item()
        expensive_capacity1 = capacity.loc[
            {"locs": "1", "techs": "expensive_elec_supply"}
        ].item()
        assert (
            round(cheap_capacity1 / capacity_supply_conversion_all_loc1.sum().item(), 5)
            <= 0.2
        )
        assert (
            round(
                expensive_capacity0 / capacity_supply_conversion_all_loc0.sum().item(),
                5,
            )
            >= 0.4
        )
        assert expensive_capacity1 == 0

    # All conversion technologies with insufficient energy_cap_max to use the
    # cheap direct heat/cooling supply techs
    def test_energy_cap_share_max_non_conversion_all_constraint(self, model_file):
        model = build_model(
            model_file=model_file, scenario="energy_cap_share_max_non_conversion_all"
        )
        model.run()
        capacity = model.get_formatted_array("energy_cap")
        capacity_supply_conversion_all_loc1 = capacity.loc[
            {"locs": "1", "techs": get_supply_conversion_techs(model)}
        ]
        cheap_heat = capacity.loc[{"locs": "1", "techs": "cheap_heat_supply"}].item()
        cheap_cool = capacity.loc[{"locs": "1", "techs": "cheap_cool_supply"}].item()
        assert (
            cheap_heat + cheap_cool
        ) / capacity_supply_conversion_all_loc1.sum().item() <= 0.1

    # All technologies, but insufficient energy_cap_max for enough installed capacity to meet demand
    @pytest.mark.filterwarnings(
        "ignore:(?s).*`['demand']` have been ignored*:calliope.exceptions.ModelWarning"
    )
    def test_energy_cap_share_max_all_techs_infeasible_constraint(self, model_file):
        model = build_model(
            model_file=model_file, scenario="energy_cap_share_max_all_techs_infeasible"
        )

        model.run()

        assert model._model_data.attrs["termination_condition"] != "optimal"


class TestEnergyCapGroupConstraints:
    def test_no_energy_cap_constraint(self, model_file):
        model = build_model(model_file=model_file)
        model.run()
        capacity = model.get_formatted_array("energy_cap")
        expensive_capacity = (
            capacity.loc[{"techs": "expensive_elec_supply"}].sum().item()
        )
        assert expensive_capacity == 0

    def test_energy_cap_max_supply_constraint(self, model_file):
        model = build_model(model_file=model_file, scenario="energy_cap_max_supply")
        model.run()
        capacity = model.get_formatted_array("energy_cap")
        cheap_capacity = capacity.loc[{"techs": "cheap_elec_supply"}].sum().item()
        assert round(cheap_capacity, 5) <= 14

    def test_energy_cap_min_supply_constraint(self, model_file):
        model = build_model(model_file=model_file, scenario="energy_cap_min_supply")
        model.run()
        capacity = model.get_formatted_array("energy_cap")
        expensive_capacity = (
            capacity.loc[{"techs": "expensive_elec_supply"}].sum().item()
        )
        assert round(expensive_capacity, 5) >= 6

    def test_energy_cap_equals_supply_constraint(self, model_file):
        model = build_model(model_file=model_file, scenario="energy_cap_equals_supply")
        model.run()
        capacity = model.get_formatted_array("energy_cap")
        expensive_capacity = (
            capacity.loc[{"techs": "expensive_elec_supply"}].sum().item()
        )
        assert expensive_capacity == approx(10)

    def test_energy_cap_min_max_supply_constraint(self, model_file):
        model = build_model(model_file=model_file, scenario="energy_cap_min_max_supply")
        model.run()
        capacity = model.get_formatted_array("energy_cap")
        cheap_capacity = capacity.loc[{"techs": "cheap_elec_supply"}].sum().item()
        expensive_capacity = (
            capacity.loc[{"techs": "expensive_elec_supply"}].sum().item()
        )
        assert round(cheap_capacity, 5) <= 4
        assert round(expensive_capacity, 5) >= 6

    def test_energy_cap_max_supply_loc_1_constraint(self, model_file):
        model = build_model(
            model_file=model_file, scenario="energy_cap_max_supply_loc_1"
        )
        model.run()
        capacity = model.get_formatted_array("energy_cap")
        cheap_capacity1 = capacity.loc[
            {"locs": "1", "techs": "cheap_elec_supply"}
        ].item()
        expensive_capacity0 = capacity.loc[
            {"locs": "0", "techs": "expensive_elec_supply"}
        ].item()
        assert round(cheap_capacity1, 5) <= 4
        assert expensive_capacity0 == 0

    def test_energy_cap_min_supply_loc_0_constraint(self, model_file):
        model = build_model(
            model_file=model_file, scenario="energy_cap_min_supply_loc_0"
        )
        model.run()
        capacity = model.get_formatted_array("energy_cap")
        expensive_capacity0 = capacity.loc[
            {"locs": "0", "techs": "expensive_elec_supply"}
        ].item()
        expensive_capacity1 = capacity.loc[
            {"locs": "1", "techs": "expensive_elec_supply"}
        ].item()
        assert round(expensive_capacity0, 5) >= 6
        assert expensive_capacity1 == 0

    def test_energy_cap_min_max_supply_loc0_1_constraint(self, model_file):
        model = build_model(
            model_file=model_file, scenario="energy_cap_min_max_supply_loc_0_1"
        )
        model.run()
        capacity = model.get_formatted_array("energy_cap")
        cheap_capacity1 = capacity.loc[
            {"locs": "1", "techs": "cheap_elec_supply"}
        ].item()
        expensive_capacity0 = capacity.loc[
            {"locs": "0", "techs": "expensive_elec_supply"}
        ].item()
        expensive_capacity1 = capacity.loc[
            {"locs": "1", "techs": "expensive_elec_supply"}
        ].item()
        assert round(cheap_capacity1, 5) <= 4
        assert round(expensive_capacity0, 5) >= 6
        assert expensive_capacity1 == 0

    # All conversion technologies with insufficient energy_cap_max to use the
    # cheap direct heat/cooling supply techs
    def test_energy_cap_max_non_conversion_all_constraint(self, model_file):
        model = build_model(
            model_file=model_file, scenario="energy_cap_max_non_conversion_all"
        )
        model.run()
        capacity = model.get_formatted_array("energy_cap")
        elec_to_heat = capacity.loc[{"locs": "1", "techs": "elec_to_heat"}].item()
        elec_to_heat_cool_linked = capacity.loc[
            {"locs": "1", "techs": "elec_to_heat_cool_linked"}
        ].item()
        elec_to_heat_cool_unlinked = capacity.loc[
            {"locs": "1", "techs": "elec_to_heat_cool_unlinked"}
        ].item()
        cheap_heat = capacity.loc[{"locs": "1", "techs": "cheap_heat_supply"}].item()
        cheap_cool = capacity.loc[{"locs": "1", "techs": "cheap_cool_supply"}].item()
        assert cheap_heat == approx(0.8333, rel=0.001)
        assert cheap_cool == approx(0.1667, rel=0.001)
        assert elec_to_heat_cool_linked == approx(1.6667, rel=0.001)
        assert elec_to_heat_cool_unlinked == approx(0)
        assert elec_to_heat == approx(0)

    # All technologies, but insufficient energy_cap_max for enough installed capacity to meet demand
    @pytest.mark.filterwarnings(
        "ignore:(?s).*`['demand']` have been ignored*:calliope.exceptions.ModelWarning"
    )
    def test_energy_cap_max_all_techs_infeasible_constraint(self, model_file):
        model = build_model(
            model_file=model_file, scenario="energy_cap_max_all_techs_infeasible"
        )

        model.run()

        assert model._model_data.attrs["termination_condition"] != "optimal"


class TestStorageCapGroupConstraints:
    def test_no_storage_cap_constraint(self):
        model = build_model(model_file="group_constraint_storage_cap.yaml")
        model.run()
        capacity = model.get_formatted_array("storage_cap")
        assert capacity.loc[{"techs": "expensive_elec_storage"}].sum().item() == 0

    def test_storage_cap_max_constraint(self):
        model = build_model(
            model_file="group_constraint_storage_cap.yaml", scenario="storage_cap_max"
        )
        model.run()
        capacity = model.get_formatted_array("storage_cap")
        cheap_capacity = capacity.loc[{"techs": "cheap_elec_storage"}].sum().item()
        assert round(cheap_capacity, 5) <= 5

    def test_storage_cap_min_constraint(self):
        model = build_model(
            model_file="group_constraint_storage_cap.yaml", scenario="storage_cap_min"
        )
        model.run()
        capacity = model.get_formatted_array("storage_cap")
        expensive_capacity = (
            capacity.loc[{"techs": "expensive_elec_storage"}].sum().item()
        )
        assert round(expensive_capacity, 5) >= 4

    def test_storage_cap_equals_constraint(self):
        model = build_model(
            model_file="group_constraint_storage_cap.yaml",
            scenario="storage_cap_equals",
        )
        model.run()
        capacity = model.get_formatted_array("storage_cap")
        expensive_capacity = (
            capacity.loc[{"techs": "expensive_elec_storage"}].sum().item()
        )
        assert expensive_capacity == approx(6)

    def test_storage_cap_min_max_constraint(self):
        model = build_model(
            model_file="group_constraint_storage_cap.yaml",
            scenario="storage_cap_min_max",
        )
        model.run()
        capacity = model.get_formatted_array("storage_cap")
        cheap_capacity = capacity.loc[{"techs": "cheap_elec_storage"}].sum().item()
        expensive_capacity = (
            capacity.loc[{"techs": "expensive_elec_storage"}].sum().item()
        )
        assert round(cheap_capacity, 5) <= 2
        assert round(expensive_capacity, 5) >= 2


class TestNetImportShareGroupConstraints:
    @pytest.fixture
    def results_for_scenario(self, model_file):
        def __run(scenario=None):
            model = build_model(model_file=model_file, scenario=scenario)
            model.run()
            return model

        return __run

    @staticmethod
    def retrieve_imports(results):
        carrier_prod = results.get_formatted_array("carrier_prod").sel(
            carriers="electricity"
        )
        return carrier_prod.sel(
            techs=["transmission" in str(tech) for tech in carrier_prod.techs]
        ).sum(["techs", "timesteps"])

    @staticmethod
    def retrieve_exports(results):
        carrier_prod = results.get_formatted_array("carrier_con").sel(
            carriers="electricity"
        )
        return (
            carrier_prod.sel(
                techs=["transmission" in str(tech) for tech in carrier_prod.techs]
            ).sum(["techs", "timesteps"])
        ) * (-1)

    @staticmethod
    def retrieve_net_imports(results):
        imports = TestNetImportShareGroupConstraints.retrieve_imports(results)
        exports = TestNetImportShareGroupConstraints.retrieve_exports(results)
        return imports - exports

    @staticmethod
    def retrieve_demand(results):
        return (
            results.get_formatted_array("carrier_con")
            .sel(carriers="electricity", techs="electricity_demand")
            .sum(["timesteps"])
        )

    def test_only_imports_without_constraint(self, results_for_scenario):
        results = results_for_scenario("expensive-1")
        demand = self.retrieve_demand(results).sel(locs="1").item()
        net_imports = self.retrieve_net_imports(results).sel(locs="1").item()
        assert net_imports == pytest.approx(-demand)

    def test_only_imports_with_wrong_carrier(self, results_for_scenario):
        with pytest.warns(calliope.exceptions.ModelWarning) as warn:
            results = results_for_scenario("expensive-1,other-carrier")
        assert check_error_or_warning(
            warn,
            "Constraint group `example_net_import_share_constraint` will be completely ignored",
        )
        demand = self.retrieve_demand(results).sel(locs="1").item()
        net_imports = self.retrieve_net_imports(results).sel(locs="1").item()
        assert net_imports == pytest.approx(-demand)

    def test_no_net_imports(self, results_for_scenario):
        results = results_for_scenario("expensive-1,no-net-imports")
        net_imports = self.retrieve_net_imports(results).sel(locs="1").item()
        assert net_imports == pytest.approx(0)

    def test_no_imports_explicit_tech(self, results_for_scenario):
        results = results_for_scenario("expensive-1,no-net-imports-explicit-tech")
        net_imports = self.retrieve_net_imports(results).sel(locs="1").item()
        assert net_imports == pytest.approx(0)

    def test_some_imports_allowed(self, results_for_scenario):
        results = results_for_scenario("expensive-1,some-imports-allowed")
        demand = self.retrieve_demand(results).sel(locs="1").item()
        net_imports = self.retrieve_net_imports(results).sel(locs="1").item()
        assert net_imports <= -0.2 * demand

    def test_some_imports_enforced(self, results_for_scenario):
        results = results_for_scenario("expensive-1,some-imports-enforced")
        demand = self.retrieve_demand(results).sel(locs="1").item()
        net_imports = self.retrieve_net_imports(results).sel(locs="1").item()
        assert net_imports == pytest.approx(-0.2 * demand)

    def test_some_expensive_imports_enforced(self, results_for_scenario):
        results = results_for_scenario("expensive-1,some-expensive-imports-enforced")
        demand = self.retrieve_demand(results).sel(locs="0").item()
        net_imports = self.retrieve_net_imports(results).sel(locs="0").item()
        assert net_imports >= -0.2 * demand

    def test_ignores_imports_within_group(self, results_for_scenario):
        with pytest.warns(calliope.exceptions.ModelWarning) as warn:
            results = results_for_scenario("expensive-1,ignores-imports-within-group")
        assert check_error_or_warning(
            warn,
            "Constraint group `example_net_import_share_constraint` will be completely ignored",
        )

        demand = self.retrieve_demand(results).sel(locs="1").item()
        net_imports = self.retrieve_imports(results).sel(locs="1").item()
        assert net_imports == pytest.approx(-demand)

    def test_allows_gross_imports(self, results_for_scenario):
        results = results_for_scenario("no-net-imports,alternating-costs")
        gross_imports = self.retrieve_imports(results).sel(locs="1").item()
        assert gross_imports > 0

    def test_no_net_imports_despite_gross_imports(self, results_for_scenario):
        results = results_for_scenario("no-net-imports,alternating-costs")
        net_imports = self.retrieve_net_imports(results).sel(locs="1")
        assert net_imports <= 0


class TestCarrierProdGroupConstraints:
    def test_no_carrier_prod_constraint(self, model_file):
        model = build_model(model_file=model_file)
        model.run()
        prod = model.get_formatted_array("carrier_prod")
        expensive_prod = prod.loc[{"techs": "expensive_elec_supply"}].sum().item()
        assert expensive_prod == 0

    def test_carrier_prod_max_supply_constraint(self, model_file):
        model = build_model(model_file=model_file, scenario="carrier_prod_max_supply")
        model.run()
        prod = model.get_formatted_array("carrier_prod")
        cheap_prod = prod.loc[{"techs": "cheap_elec_supply"}].sum().item()
        assert round(cheap_prod, 5) <= 40

    def test_carrier_prod_min_supply_constraint(self, model_file):
        model = build_model(model_file=model_file, scenario="carrier_prod_min_supply")
        model.run()
        prod = model.get_formatted_array("carrier_prod")
        expensive_prod = prod.loc[{"techs": "expensive_elec_supply"}].sum().item()
        assert round(expensive_prod, 5) >= 10

    def test_carrier_prod_equals_supply_constraint(self, model_file):
        model = build_model(
            model_file=model_file, scenario="carrier_prod_equals_supply"
        )
        model.run()
        prod = model.get_formatted_array("carrier_prod")
        expensive_prod = prod.loc[{"techs": "expensive_elec_supply"}].sum().item()
        assert expensive_prod == approx(30)

    def test_carrier_prod_min_max_supply_constraint(self, model_file):
        model = build_model(
            model_file=model_file, scenario="carrier_prod_min_max_supply"
        )
        model.run()
        prod = model.get_formatted_array("carrier_prod")
        cheap_prod = prod.loc[{"techs": "cheap_elec_supply"}].sum().item()
        expensive_prod = prod.loc[{"techs": "expensive_elec_supply"}].sum().item()
        assert round(cheap_prod, 5) <= 40
        assert round(expensive_prod, 5) >= 10

    def test_carrier_prod_max_supply_loc_1_constraint(self, model_file):
        model = build_model(
            model_file=model_file, scenario="carrier_prod_max_supply_loc_1"
        )
        model.run()
        prod = model.get_formatted_array("carrier_prod")
        cheap_prod1 = prod.loc[{"locs": "1", "techs": "cheap_elec_supply"}].sum().item()
        expensive_prod0 = (
            prod.loc[{"locs": "0", "techs": "expensive_elec_supply"}].sum().item()
        )
        assert round(cheap_prod1, 5) <= 40
        assert expensive_prod0 == 0

    def test_carrier_prod_min_supply_loc_0_constraint(self, model_file):
        model = build_model(
            model_file=model_file, scenario="carrier_prod_min_supply_loc_0"
        )
        model.run()
        prod = model.get_formatted_array("carrier_prod")
        expensive_prod0 = (
            prod.loc[{"locs": "0", "techs": "expensive_elec_supply"}].sum().item()
        )
        expensive_prod1 = (
            prod.loc[{"locs": "1", "techs": "expensive_elec_supply"}].sum().item()
        )
        assert round(expensive_prod0, 5) >= 10
        assert expensive_prod1 == 0

    def test_carrier_prod_min_max_supply_loc0_1_constraint(self, model_file):
        model = build_model(
            model_file=model_file, scenario="carrier_prod_min_max_supply_loc_0_1"
        )
        model.run()
        prod = model.get_formatted_array("carrier_prod")
        cheap_prod1 = prod.loc[{"locs": "1", "techs": "cheap_elec_supply"}].sum().item()
        expensive_prod0 = (
            prod.loc[{"locs": "0", "techs": "expensive_elec_supply"}].sum().item()
        )
        expensive_prod1 = (
            prod.loc[{"locs": "1", "techs": "expensive_elec_supply"}].sum().item()
        )
        assert round(cheap_prod1, 5) <= 40
        assert round(expensive_prod0, 5) >= 10
        assert expensive_prod1 == 0

    # All conversion technologies with insufficient supply_max to use the
    # cheap direct heat/cooling supply techs
    def test_carrier_prod_max_non_conversion_all_constraint(self, model_file):
        with pytest.warns(calliope.exceptions.ModelWarning) as warn:
            model = build_model(
                model_file=model_file, scenario="carrier_prod_max_non_conversion_all"
            )
        assert check_error_or_warning(
            warn,
            "Constraint group `example_carrier_prod_constraint` will be completely ignored ",
        )
        model.run()
        prod = model.get_formatted_array("carrier_prod")
        elec_to_heat = prod.loc[{"locs": "1", "techs": "elec_to_heat"}].sum().item()
        elec_to_heat_cool_linked = prod.loc[
            {"locs": "1", "techs": "elec_to_heat_cool_linked"}
        ].sum("timesteps")
        elec_to_heat_cool_unlinked = (
            prod.loc[{"locs": "1", "techs": "elec_to_heat_cool_unlinked"}].sum().item()
        )
        cheap_heat = prod.loc[{"locs": "1", "techs": "cheap_heat_supply"}].sum().item()
        cheap_cool = prod.loc[{"locs": "1", "techs": "cheap_cool_supply"}].sum().item()
        assert round(cheap_heat, 5) <= 10
        assert round(cheap_cool, 5) <= 5
        assert round(elec_to_heat_cool_linked.loc[{"carriers": "cool"}].item()) >= 1
        assert round(elec_to_heat_cool_linked.loc[{"carriers": "heat"}].item()) >= 5
        assert elec_to_heat_cool_unlinked == 0
        assert elec_to_heat == 0

    # All technologies, but insufficient supply_max for enough installed capacity to meet demand
    @pytest.mark.filterwarnings(
        "ignore:(?s).*`['demand', 'storage', 'transmission']` have been ignored*:calliope.exceptions.ModelWarning"
    )
    def test_carrier_prod_max_all_techs_infeasible_constraint(self, model_file):
        model = build_model(
            model_file=model_file, scenario="carrier_prod_max_all_techs_infeasible"
        )

        model.run()

        assert model._model_data.attrs["termination_condition"] != "optimal"


@pytest.mark.filterwarnings(
    "ignore:(?s).*defines force_resource but not a finite resourc.*:calliope.exceptions.ModelWarning"
)
class TestCarrierConGroupConstraints:
    def test_no_carrier_con_constraint(self, model_file):
        model = build_model(model_file=model_file)
        model.run()
        con = model.get_formatted_array("carrier_con")
        dem = con.loc[{"techs": "electricity_demand"}].sum().item()
        assert dem == -90

    def test_carrier_con_max_demand_constraint(self, model_file):
        model = build_model(model_file=model_file, scenario="carrier_con_max_demand")
        model.run()
        con = model.get_formatted_array("carrier_con")
        dem = con.loc[{"techs": "electricity_demand"}].sum().item()
        assert round(dem, 5) >= -80

    def test_carrier_con_min_demand_constraint(self, model_file):
        model = build_model(model_file=model_file, scenario="carrier_con_min_demand")
        model.run()
        con = model.get_formatted_array("carrier_con")
        dem = con.loc[{"techs": "electricity_demand"}].sum().item()
        assert round(dem, 5) <= -40

    def test_carrier_con_equals_demand_constraint(self, model_file):
        model = build_model(model_file=model_file, scenario="carrier_con_equals_demand")
        model.run()
        con = model.get_formatted_array("carrier_con")
        dem = con.loc[{"techs": "electricity_demand"}].sum().item()
        assert dem == approx(-60)

    def test_carrier_con_max_demand_loc_1_constraint(self, model_file):
        model = build_model(
            model_file=model_file, scenario="carrier_con_max_demand_loc_1"
        )
        model.run()
        con = model.get_formatted_array("carrier_con")
        dem1 = con.loc[{"techs": "electricity_demand", "locs": "1"}].sum().item()
        dem0 = con.loc[{"techs": "electricity_demand", "locs": "0"}].sum().item()
        assert round(dem1, 5) >= -80
        assert dem0 == 0

    def test_carrier_con_max_multi_demand_constraint(self, model_file):
        model = build_model(
            model_file=model_file, scenario="carrier_con_max_multi_demand"
        )
        model.run()
        con = model.get_formatted_array("carrier_con")
        dem_heat = con.loc[{"techs": "heat_demand"}].sum().item()
        dem_cool = con.loc[{"techs": "cool_demand"}].sum().item()
        assert round(dem_heat, 5) >= -10
        assert round(dem_cool, 5) >= -5

    def test_carrier_con_min_conversion_constraint(self, model_file):
        model = build_model(
            model_file=model_file, scenario="carrier_con_min_conversion"
        )
        model.run()
        con = model.get_formatted_array("carrier_con")
        conversion_con = con.loc[{"techs": "elec_to_heat"}].sum().item()
        assert round(conversion_con, 5) <= -1

    def test_carrier_con_min_conversion_plus_constraint(self, model_file):
        model = build_model(
            model_file=model_file, scenario="carrier_con_min_conversion_plus"
        )
        model.run()
        con = model.get_formatted_array("carrier_con")
        conversion_con = con.loc[{"techs": "elec_to_heat_cool_unlinked"}].sum().item()
        assert round(conversion_con, 5) <= 1
