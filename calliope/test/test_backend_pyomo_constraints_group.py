import pytest
from pytest import approx
import pandas as pd

import calliope
from calliope.test.common.util import build_test_model as build_model


# Group constraints, i.e. those that can be defined on a system/subsystem scale
@pytest.mark.filterwarnings("ignore:(?s).*Not all requested techs:calliope.exceptions.ModelWarning")
class TestBuildGroupConstraints:

    def _build_group_model(self, scenario, model_file):
        model = build_model(
            model_file=model_file + '.yaml',
            scenario=scenario
        )
        model.run(build_only=True)

        return model

    _test_vars = [
        ('demand_share_max_systemwide', ['example_demand_share_max_constraint'], ['demand_share_max'], 'demand_share'),
        ('demand_share_min_systemwide', ['example_demand_share_min_constraint'], ['demand_share_min'], 'demand_share'),
        ('demand_share_max_location_0', ['example_demand_share_max_constraint'], ['demand_share_max'], 'demand_share'),
        ('demand_share_min_location_0', ['example_demand_share_min_constraint'], ['demand_share_min'], 'demand_share'),
        ('multiple_constraints', ['example_demand_share_max_constraint', 'example_demand_share_min_constraint'], ['demand_share_max', 'demand_share_min'], 'demand_share'),
        ('cheap_cost_max_systemwide', ['example_cost_max_constraint'], ['cost_max'], 'model_cost_cap'),
        ('expensive_cost_min_systemwide', ['example_cost_min_constraint'], ['cost_min'], 'model_cost_cap'),
        ('cheap_cost_equals_systemwide', ['example_cost_equals_constraint'], ['cost_equals'], 'model_cost_cap'),
        ('cheap_cost_max_location_0', ['example_cheap_cost_max_location_0_constraint'], ['cost_max'], 'model_cost_cap'),
        ('multiple_costs_constraint', ['example_emissions_max_constraint', 'example_cost_min_constraint'], ['cost_max', 'cost_min'], 'model_cost_cap'),
        ('cheap_cost_var_max_systemwide', ['example_cost_var_max_constraint'], ['cost_var_max'], 'model_cost_cap'),
        ('cheap_cost_investment_max_systemwide', ['example_cost_investment_max_constraint'], ['cost_investment_max'], 'model_cost_cap')
    ]

    @pytest.mark.parametrize("scenario,group_name,constraints,model_file", _test_vars)
    def test_constraint_names(self, scenario, group_name, constraints, model_file):
        model = self._build_group_model(scenario, model_file)
        assert all('group_names_' + i in model._model_data.dims for i in constraints)

    @pytest.mark.parametrize("scenario,group_name,constraints,model_file", _test_vars)
    def test_group_names(self, scenario, group_name, constraints, model_file):
        model = self._build_group_model(scenario, model_file)
        assert all(
            group_name[i] in model._model_data['group_names_' + constraints[i]].values
            for i in range(len(constraints))
        )

    @pytest.mark.parametrize("scenario,group_name,constraints,model_file", _test_vars)
    def test_group_variables(self, scenario, group_name, constraints, model_file):
        model = self._build_group_model(scenario, model_file)
        assert all('group_' + i in model._model_data.data_vars.keys() for i in constraints)

    @pytest.mark.parametrize("scenario,group_name,constraints,model_file", _test_vars)
    def test_group_constraints(self, scenario, group_name, constraints, model_file):
        model = self._build_group_model(scenario, model_file)
        assert all(hasattr(model._backend_model, 'group_' + i + '_constraint')
                   for i in constraints)



@pytest.mark.filterwarnings("ignore:(?s).*Not all requested techs:calliope.exceptions.ModelWarning")
class TestGroupConstraints:

    def test_no_group_constraint(self):
        model = build_model(model_file="group_constraints.yaml")
        model.run()
        expensive_generation = (
            model.get_formatted_array("carrier_prod")
                 .loc[{'techs': 'expensive_supply'}].sum().item()
        )
        assert expensive_generation == 0

    def test_switched_off_group_constraint(self):
        model = build_model(
            model_file="group_constraints.yaml",
            scenario="switching_off_group_constraint"
        )
        model.run()
        expensive_generation = (
            model.get_formatted_array("carrier_prod")
                 .loc[{'techs': 'expensive_supply'}].sum().item()
        )
        assert expensive_generation == 0

    @pytest.mark.xfail(reason="Check not yet implemented.")
    def test_group_constraint_without_technology(self):
        model = build_model(
            model_file='group_constraints.yaml',
            scenario='group_constraint_without_tech'
        )
        with pytest.raises(calliope.exceptions.ModelError):
            model.run()

    def test_group_constraint_with_several_constraints(self):
        model = build_model(
            model_file='group_constraints.yaml',
            scenario='several_group_constraints'
        )
        model.run()
        expensive_generation = (model.get_formatted_array("carrier_prod")
                                     .to_dataframe()
                                     .reset_index()
                                     .groupby("techs")
                                     .carrier_prod
                                     .sum()
                                     .transform(lambda x: x / x.sum())
                                     .loc["expensive_supply"])
        assert round(expensive_generation, 5) >= 0.8


@pytest.mark.filterwarnings("ignore:(?s).*Not all requested techs:calliope.exceptions.ModelWarning")
class TestDemandShareGroupConstraints:
    def test_no_demand_share_constraint(self):
        model = build_model(model_file='demand_share.yaml')
        model.run()
        expensive_generation = (model.get_formatted_array("carrier_prod")
                                     .to_dataframe()
                                     .reset_index()
                                     .groupby("techs")
                                     .carrier_prod
                                     .sum()
                                     .loc["expensive_elec_supply"])
        assert expensive_generation == 0

    def test_systemwide_demand_share_max_constraint(self):
        model = build_model(
            model_file='demand_share.yaml',
            scenario='demand_share_max_systemwide'
        )
        model.run()
        cheap_generation = model.get_formatted_array("carrier_prod").loc[{'techs': "cheap_elec_supply", 'carriers': "electricity"}].sum()
        demand = -1 * model.get_formatted_array("carrier_con").sum()
        # assert share in each timestep is 0.6
        assert (cheap_generation / demand).round(5) <= 0.3

    def test_systemwide_demand_share_min_constraint(self):
        model = build_model(
            model_file='demand_share.yaml',
            scenario='demand_share_min_systemwide'
        )
        model.run()
        expensive_generation = model.get_formatted_array("carrier_prod").loc[{'techs': "expensive_elec_supply", 'carriers': "electricity"}].sum()
        demand = -1 * model.get_formatted_array("carrier_con").sum()
        # assert share in each timestep is 0.6
        assert (expensive_generation / demand).round(5) >= 0.6

    def test_systemwide_demand_share_equals_constraint(self):
        model = build_model(
            model_file='demand_share.yaml',
            scenario='demand_share_equals_systemwide'
        )
        model.run()
        expensive_generation = model.get_formatted_array("carrier_prod").loc[{'techs': "expensive_elec_supply", 'carriers': "electricity"}].sum()
        demand = -1 * model.get_formatted_array("carrier_con").sum()
        # assert share in each timestep is 0.6
        assert (expensive_generation / demand).round(5) == 0.6

    def test_location_specific_demand_share_max_constraint(self):
        model = build_model(
            model_file='demand_share.yaml',
            scenario='demand_share_max_location_0'
        )
        model.run()
        generation = model.get_formatted_array("carrier_prod").sum(dim='timesteps').loc[{'carriers': "electricity"}]
        demand0 = -model.get_formatted_array("carrier_con").loc[{'locs': '0'}].sum().item()
        cheap_generation0 = generation.loc[{'locs': "0", 'techs': "cheap_elec_supply"}].item()
        expensive_generation1 = generation.loc[{'locs': "1", 'techs': "expensive_elec_supply"}].item()
        assert round(cheap_generation0 / demand0, 5) <= 0.3
        assert expensive_generation1 == 0

    def test_location_specific_demand_share_max_constraint_two_techs(self):
        model = build_model(
            model_file='demand_share.yaml',
            scenario='demand_share_max_location_0_two_techs'
        )
        model.run()
        generation = model.get_formatted_array("carrier_prod").sum(dim='timesteps').loc[{'carriers': "electricity"}]
        demand0 = -model.get_formatted_array("carrier_con").loc[{'locs': '0'}].sum().item()
        generation0 = generation.loc[{'locs': "0", 'techs': ["cheap_elec_supply", "normal_elec_supply"]}].sum('techs').item()
        assert round(generation0 / demand0, 5) <= 0.4

    def test_location_specific_demand_share_min_constraint(self):
        model = build_model(
            model_file='demand_share.yaml',
            scenario='demand_share_min_location_0'
        )
        model.run()
        generation = model.get_formatted_array("carrier_prod").sum(dim='timesteps').loc[{'carriers': "electricity"}]
        demand0 = -model.get_formatted_array("carrier_con").loc[{'locs': '0'}].sum().item()
        expensive_generation0 = generation.loc[{'locs': "0", 'techs': "expensive_elec_supply"}].item()
        expensive_generation1 = generation.loc[{'locs': "1", 'techs': "expensive_elec_supply"}].item()
        assert round(expensive_generation0 / demand0, 5) >= 0.6
        assert expensive_generation1 == 0

    def test_multiple_group_constraints(self):
        model = build_model(
            model_file='demand_share.yaml',
            scenario='multiple_constraints'
        )
        model.run()
        generation = (model.get_formatted_array("carrier_prod")
                           .sum(dim=('timesteps', 'locs', 'carriers')))
        demand = -model.get_formatted_array("carrier_con").sum().item()
        cheap_generation = generation.loc[{'techs': 'cheap_elec_supply'}].item()
        expensive_generation = generation.loc[{'techs': 'expensive_elec_supply'}].item()

        assert round(expensive_generation / demand, 5) >= 0.6
        assert round(cheap_generation / demand, 5) <= 0.3

    def test_multiple_group_carriers(self):
        model = build_model(
            model_file='demand_share.yaml',
            scenario='multiple_carriers_max'
        )
        model.run()
        generation = (model.get_formatted_array("carrier_prod")
                           .sum(dim=('timesteps', 'locs')))
        demand = (-model.get_formatted_array("carrier_con")
                        .sum(dim=('timesteps', 'locs')))
        cheap_generation_elec = generation.loc[{'techs': 'cheap_elec_supply', 'carriers': 'electricity'}].item()
        demand_elec = demand.loc[{'techs': 'electricity_demand', 'carriers': 'electricity'}].item()
        cheap_generation_heat = generation.loc[{'techs': 'cheap_heat_supply', 'carriers': 'heat'}].item()
        demand_heat = demand.loc[{'techs': 'heat_demand', 'carriers': 'heat'}].item()

        assert round(cheap_generation_elec / demand_elec, 5) <= 0.3
        assert round(cheap_generation_heat / demand_heat, 5) <= 0.5

    def test_multiple_group_carriers_constraints(self):
        model = build_model(
            model_file='demand_share.yaml',
            scenario='multiple_constraints_carriers'
        )
        model.run()
        generation = (model.get_formatted_array("carrier_prod")
                           .sum(dim=('timesteps', 'locs')))
        demand = (-model.get_formatted_array("carrier_con")
                        .sum(dim=('timesteps', 'locs')))
        cheap_generation_elec = generation.loc[{'techs': 'cheap_elec_supply', 'carriers': 'electricity'}].item()
        expensive_generation_elec = generation.loc[{'techs': 'expensive_elec_supply', 'carriers': 'electricity'}].item()
        demand_elec = demand.loc[{'techs': 'electricity_demand', 'carriers': 'electricity'}].item()
        cheap_generation_heat = generation.loc[{'techs': 'cheap_heat_supply', 'carriers': 'heat'}].item()
        expensive_generation_heat = generation.loc[{'techs': 'expensive_heat_supply', 'carriers': 'heat'}].item()
        demand_heat = demand.loc[{'techs': 'heat_demand', 'carriers': 'heat'}].item()

        assert round(cheap_generation_elec / demand_elec, 5) <= 0.3
        assert round(expensive_generation_elec / demand_elec, 5) >= 0.6
        assert round(cheap_generation_heat / demand_heat, 5) <= 0.5
        assert round(expensive_generation_heat / demand_heat, 5) >= 0.4

    def test_different_locations_per_group_constraint(self):
        model = build_model(
            model_file='demand_share.yaml',
            scenario='different_locations_per_group'
        )
        model.run()
        generation = (model.get_formatted_array("carrier_prod")
                           .sum(dim=('timesteps', 'carriers')))
        demand = (-model.get_formatted_array("carrier_con")
                        .sum(dim=('timesteps')))
        cheap_generation_0 = generation.loc[{'techs': 'cheap_elec_supply', 'locs': '0'}].item()
        expensive_generation_0 = generation.loc[{'techs': 'expensive_elec_supply', 'locs': '0'}].item()
        cheap_generation_1 = generation.loc[{'techs': 'cheap_elec_supply', 'locs': '1'}].item()
        expensive_generation_1 = generation.loc[{'techs': 'expensive_elec_supply', 'locs': '1'}].item()
        demand_elec_0 = demand.loc[{'techs': 'electricity_demand', 'carriers': 'electricity', 'locs': '0'}].item()
        demand_elec_1 = demand.loc[{'techs': 'electricity_demand', 'carriers': 'electricity', 'locs': '1'}].item()

        assert round(expensive_generation_0 / demand_elec_0, 5) >= 0.6
        assert expensive_generation_1 / demand_elec_1 == 0
        assert round((cheap_generation_0 + cheap_generation_1) / (demand_elec_0 + demand_elec_1), 5) <= 0.3

    def test_transmission_not_included_in_demand(self):
        model = build_model(
            model_file='demand_share.yaml',
            scenario='transmission_not_included_in_demand'
        )
        model.run()
        assert model.results.termination_condition == "optimal"
        generation = (model.get_formatted_array("carrier_prod")
                           .sum(dim=('timesteps', 'carriers')))
        demand = (-model.get_formatted_array("carrier_con")
                        .sum(dim=('timesteps', 'carriers')))

        assert (generation.sel(locs="1", techs=["normal_elec_supply", "cheap_elec_supply", "expensive_elec_supply"])
                          .sum(dim="techs")
                          .item()) == pytest.approx(0)

        cheap_elec_supply_0 = generation.sel(locs="0", techs="cheap_elec_supply").item()
        demand_0 = demand.sel(locs="0", techs="electricity_demand").item()

        assert round(cheap_elec_supply_0 / demand_0, 5) <= 0.4

    def test_demand_share_per_timestep_max(self):
        model = build_model(
            model_file='demand_share.yaml',
            scenario='demand_share_per_timestep_max'
        )
        model.run()
        cheap_generation = model.get_formatted_array("carrier_prod").loc[{'techs': "cheap_elec_supply", 'carriers': "electricity"}].sum('locs')
        demand = -1 * model.get_formatted_array("carrier_con").sum('locs')
        # assert share in each timestep is 0.6
        assert ((cheap_generation / demand).round(5) <= 0.3).all()

    def test_demand_share_per_timestep_min(self):
        model = build_model(
            model_file='demand_share.yaml',
            scenario='demand_share_per_timestep_min'
        )
        model.run()
        expensive_generation = model.get_formatted_array("carrier_prod").loc[{'techs': "expensive_elec_supply", 'carriers': "electricity"}].sum('locs')
        demand = -1 * model.get_formatted_array("carrier_con").sum('locs')
        # assert share in each timestep is 0.6
        assert ((expensive_generation / demand).round(5) >= 0.6).all()

    def test_demand_share_per_timestep_equals(self):
        model = build_model(
            model_file='demand_share.yaml',
            scenario='demand_share_per_timestep_equals'
        )
        model.run()
        expensive_generation = model.get_formatted_array("carrier_prod").loc[{'techs': "expensive_elec_supply", 'carriers': "electricity"}].sum('locs')
        demand = -1 * model.get_formatted_array("carrier_con").sum('locs')
        # assert share in each timestep is 0.6
        assert ((expensive_generation / demand).round(5) == 0.6).all()

    def test_demand_share_per_timestep_decision_inf(self):
        model = build_model(
            model_file='demand_share_decision.yaml',
            scenario='demand_share_per_timestep_decision_inf'
        )
        model.run()
        demand = -1 * model.get_formatted_array("carrier_con").loc[{'carriers': "electricity"}].sum('locs').sum('techs').to_pandas()
        supply = model.get_formatted_array("carrier_prod").loc[{'carriers': "electricity"}].sum('locs').to_pandas().T
        shares = supply.div(demand, axis=0)

        assert (shares['cheap_elec_supply'] == 0.1875).all()
        assert (shares['normal_elec_supply'] == 0.8125).all()

    def test_demand_share_per_timestep_decision_simple(self):
        model = build_model(
            model_file='demand_share_decision.yaml',
            scenario='demand_share_per_timestep_decision_simple'
        )
        model.run()
        demand = -1 * model.get_formatted_array("carrier_con").loc[{'carriers': "electricity"}].sum('locs').sum('techs').to_pandas()
        supply = model.get_formatted_array("carrier_prod").loc[{'carriers': "electricity"}].sum('locs').to_pandas().T
        shares = supply.div(demand, axis=0)

        assert (shares['cheap_elec_supply'] == 0.1875).all()
        assert (shares['normal_elec_supply'] == 0.8125).all()

    def test_demand_share_per_timestep_decision_not_one(self):
        model = build_model(
            model_file='demand_share_decision.yaml',
            scenario='demand_share_per_timestep_decision_not_one'
        )
        model.run()
        demand = -1 * model.get_formatted_array("carrier_con").loc[{'carriers': "electricity"}].sum('locs').sum('techs').to_pandas()
        supply = model.get_formatted_array("carrier_prod").loc[{'carriers': "electricity"}].sum('locs').to_pandas().T
        shares = supply.div(demand, axis=0)

        assert (shares['cheap_elec_supply'] == 0.1).all()
        assert (shares['normal_elec_supply'] == 0.9).all()
        assert shares['expensive_elec_supply'].mean() + shares['normal_elec_supply'].mean() == 0.9

    def test_demand_share_per_timestep_decision_per_location(self):
        model = build_model(
            model_file='demand_share_decision.yaml',
            scenario='demand_share_per_timestep_decision_per_location'
        )
        model.run()
        demand = -1 * model.get_formatted_array("carrier_con").loc[{'carriers': "electricity"}].sum('techs').to_series()
        supply = model.get_formatted_array("carrier_prod").loc[{'carriers': "electricity"}].to_series()
        shares = supply.div(demand, axis=0)

        assert (shares.loc[pd.IndexSlice['0', :, :]].unstack()['cheap_elec_supply'] == 0.25).all()
        assert (shares.loc[pd.IndexSlice['0', :, :]].unstack()['normal_elec_supply'] == 0.75).all()

        assert (shares.loc[pd.IndexSlice['1', :, :]].unstack()['cheap_elec_supply'] == 0.125).all()
        assert (shares.loc[pd.IndexSlice['1', :, :]].unstack()['normal_elec_supply'] == 0.875).all()

    def test_demand_share_per_timestep_decision_inf_with_transmission(self):
        model = build_model(
            model_file='demand_share_decision.yaml',
            scenario='demand_share_per_timestep_decision_inf,with_electricity_transmission'
        )
        model.run()
        demand = -1 * model.get_formatted_array("carrier_con").loc[{'carriers': "electricity"}].sum('techs').to_series()
        supply = model.get_formatted_array("carrier_prod").loc[{'carriers': "electricity"}].to_series()
        shares = supply.div(demand, axis=0)

        assert all([i == pytest.approx(0.12373737) for i in shares.loc[pd.IndexSlice['0', :, :]].unstack()['cheap_elec_supply'].values])
        assert all([i == pytest.approx(0.87626263) for i in shares.loc[pd.IndexSlice['0', :, :]].unstack()['normal_elec_supply'].values])
        assert (shares.loc[pd.IndexSlice['1', :, :]].unstack()['electricity_transmission:0'] == 1.0).all()

    def test_demand_share_per_timestep_decision_inf_with_heat_cosntrain_electricity(self):
        model = build_model(
            model_file='demand_share_decision.yaml',
            scenario='demand_share_per_timestep_decision_inf,with_electricity_conversion_tech'
        )
        model.run()
        demand = -1 * model.get_formatted_array("carrier_con").loc[{'carriers': "electricity"}].sum(['locs', 'techs']).to_pandas()
        supply = model.get_formatted_array("carrier_prod").loc[{'carriers': "electricity"}].sum('locs').to_pandas().T
        shares = supply.div(demand, axis=0)

        assert all([i == pytest.approx(0.175926) for i in shares['cheap_elec_supply'].values])
        assert all([i == pytest.approx(0.824074) for i in shares['normal_elec_supply'].values])

    def test_demand_share_per_timestep_decision_inf_with_heat_constrain_heat(self):
        model = build_model(
            model_file='demand_share_decision.yaml',
            scenario='demand_share_per_timestep_decision_inf_with_heat,with_electricity_conversion_tech'
        )
        model.run()
        demand = -1 * model.get_formatted_array("carrier_con").loc[{'carriers': "heat"}].sum(['locs', 'techs']).to_pandas()
        supply = model.get_formatted_array("carrier_prod").loc[{'carriers': "heat"}].sum('locs').to_pandas().T
        shares = supply.div(demand, axis=0)

        assert all([i == pytest.approx(0.5) for i in shares['elec_to_heat'].values])
        assert all([i == pytest.approx(0.5) for i in shares['heating'].values])

    def test_demand_share_per_timestep_decision_inf_with_heat_constrain_heat_and_electricity(self):
        model = build_model(
            model_file='demand_share_decision.yaml',
            scenario='demand_share_per_timestep_decision_inf_with_heat,demand_share_per_timestep_decision_not_one,with_electricity_conversion_tech'
        )
        model.run()
        demand_heat = -1 * model.get_formatted_array("carrier_con").loc[{'carriers': "heat"}].sum(['locs', 'techs']).to_pandas()
        supply_heat = model.get_formatted_array("carrier_prod").loc[{'carriers': "heat"}].sum('locs').to_pandas().T
        shares_heat = supply_heat.div(demand_heat, axis=0)

        demand_elec = -1 * (
            model._model_data.carrier_con.loc[{
                'loc_tech_carriers_con': [
                    i for i in model._model_data.loc_tech_carriers_demand.values
                    if 'electricity' in i
                ]
            }].sum(['loc_tech_carriers_con']).to_pandas()
        )
        supply_elec = model.get_formatted_array("carrier_prod").loc[{'carriers': "electricity"}].sum('locs').to_pandas().T
        shares_elec = supply_elec.div(demand_elec, axis=0)

        assert all([i == pytest.approx(0.5) for i in shares_heat['elec_to_heat'].values])
        assert all([i == pytest.approx(0.5) for i in shares_heat['heating'].values])
        assert all([i == pytest.approx(0.9) for i in shares_elec['normal_elec_supply'].values])
        assert all([round(i, 5) >= 0.1 for i in shares_elec['cheap_elec_supply'].values])


@pytest.mark.filterwarnings("ignore:(?s).*Not all requested techs:calliope.exceptions.ModelWarning")
class TestResourceAreaGroupConstraints:

    def test_no_energy_cap_share_constraint(self):
        model = build_model(model_file='resource_area.yaml')
        model.run()
        cheap_resource_area = (model.get_formatted_array("resource_area")
                                    .loc[{'techs': "cheap_supply"}].sum()).item()
        assert cheap_resource_area == 40

    def test_systemwide_resource_area_max_constraint(self):
        model = build_model(
            model_file='resource_area.yaml',
            scenario='resource_area_max_systemwide'
        )
        model.run()
        cheap_resource_area = (model.get_formatted_array("resource_area")
                                    .loc[{'techs': "cheap_supply"}].sum()).item()
        assert cheap_resource_area == 20

    def test_systemwide_resource_area_min_constraint(self):
        model = build_model(
            model_file='resource_area.yaml',
            scenario='resource_area_min_systemwide'
        )
        model.run()
        resource_area = model.get_formatted_array("resource_area")
        assert resource_area.loc[{'techs': "cheap_supply"}].sum().item() == 0
        assert resource_area.loc[{'techs': "expensive_supply"}].sum().item() == 20

    def test_location_specific_resource_area_max_constraint(self):
        model = build_model(
            model_file='resource_area.yaml',
            scenario='resource_area_max_location_0'
        )
        model.run()
        resource_area = model.get_formatted_array("resource_area")
        cheap_resource_area0 = resource_area.loc[{'locs': "0", 'techs': 'cheap_supply'}].item()
        cheap_resource_area1 = resource_area.loc[{'locs': "1", 'techs': 'cheap_supply'}].item()
        assert cheap_resource_area0 == 10
        assert cheap_resource_area1 == 20

    def test_location_specific_resource_area_min_constraint(self):
        model = build_model(
            model_file='resource_area.yaml',
            scenario='resource_area_min_location_0'
        )
        model.run()
        resource_area = model.get_formatted_array("resource_area")
        expensive_resource_area0 = resource_area.loc[{'locs': "0", 'techs': "expensive_supply"}].item()
        expensive_resource_area1 = resource_area.loc[{'locs': "1", 'techs': "expensive_supply"}].item()
        assert expensive_resource_area0 == 10
        assert expensive_resource_area1 == 0


@pytest.mark.filterwarnings("ignore:(?s).*Not all requested techs:calliope.exceptions.ModelWarning")
class TestCostCapGroupConstraint:

    def test_systemwide_cost_max_constraint(self):
        model = build_model(
            model_file='model_cost_cap.yaml',
            scenario='cheap_cost_max_systemwide'
        )
        model.run()
        cheap_cost = (model.get_formatted_array('cost')
                           .loc[{'costs': 'monetary', 'techs': 'cheap_polluting_supply'}]).sum().item()
        assert round(cheap_cost, 5) <= 30

    def test_systemwide_cost_investment_max_constraint(self):
        model = build_model(
            model_file='model_cost_cap.yaml',
            scenario='cheap_cost_investment_max_systemwide'
        )
        model.run()
        cheap_cost = (model.get_formatted_array('cost')
                           .loc[{'costs': 'monetary', 'techs': 'cheap_polluting_supply'}]).sum().item()
        cheap_cost_investment = (model.get_formatted_array('cost_investment')
                                      .loc[{'costs': 'monetary', 'techs': 'cheap_polluting_supply'}]).sum().item()
        assert cheap_cost > cheap_cost_investment
        assert round(cheap_cost_investment, 5) <= 4

    def test_systemwide_cost_var_max_constraint(self):
        model = build_model(
            model_file='model_cost_cap.yaml',
            scenario='cheap_cost_var_max_systemwide'
        )
        model.run()
        cheap_cost = (model.get_formatted_array('cost')
                           .loc[{'costs': 'monetary', 'techs': 'cheap_polluting_supply'}]).sum().item()
        cheap_cost_var = (model.get_formatted_array('cost_var')
                               .loc[{'costs': 'monetary', 'techs': 'cheap_polluting_supply'}]).sum().item()
        assert cheap_cost > cheap_cost_var
        assert round(cheap_cost_var, 5) <= 200

    def test_systemwide_cost_min_constraint(self):
        model = build_model(
            model_file='model_cost_cap.yaml',
            scenario='expensive_cost_min_systemwide'
        )
        model.run()
        expensive_cost = (model.get_formatted_array('cost')
                               .loc[{'costs': 'monetary', 'techs': 'expensive_clean_supply'}]).sum().item()
        assert round(expensive_cost, 5) >= 600

    def test_systemwide_cost_equals_constraint(self):
        model = build_model(
            model_file='model_cost_cap.yaml',
            scenario='cheap_cost_equals_systemwide'
        )
        model.run()
        cheap_cost = (model.get_formatted_array('cost')
                           .loc[{'costs': 'monetary', 'techs': 'cheap_polluting_supply'}]).sum().item()
        assert cheap_cost == approx(210)

    def test_location_specific_cost_max_constraint(self):
        model = build_model(
            model_file='model_cost_cap.yaml',
            scenario='cheap_cost_max_location_0'
        )
        model.run()
        cheap_cost0 = (model.get_formatted_array('cost')
                            .loc[{'costs': 'monetary',
                                  'techs': 'cheap_polluting_supply',
                                  'locs': '0'}]).sum().item()
        assert round(cheap_cost0, 5) <= 10

    def test_systemwide_emissions_max_constraint(self):
        model = build_model(
            model_file='model_cost_cap.yaml',
            scenario='emissions_max_systemwide'
        )
        model.run()
        emissions = (model.get_formatted_array('cost')
                          .loc[{'costs': 'emissions'}]).sum().item()
        assert round(emissions, 5) <= 400

    def test_location_specific_emissions_max_constraint(self):
        model = build_model(
            model_file='model_cost_cap.yaml',
            scenario='emissions_max_location_0'
        )
        model.run()
        emissions0 = (model.get_formatted_array('cost')
                           .loc[{'costs': 'emissions',
                                 'locs': '0'}]).sum().item()
        assert round(emissions0, 5) <= 200

    def test_systemwide_clean_emissions_max_constraint(self):
        model = build_model(
            model_file='model_cost_cap.yaml',
            scenario='clean_emissions_max_systemwide'
        )
        model.run()
        clean_emissions = (model.get_formatted_array('cost')
                                .loc[{'costs': 'emissions',
                                      'techs': 'expensive_clean_supply'}]).sum().item()
        assert round(clean_emissions, 5) <= 300

    def test_multiple_costs_constraint(self):
        model = build_model(
            model_file='model_cost_cap.yaml',
            scenario='multiple_costs_constraint'
        )
        model.run()
        emissions = (model.get_formatted_array('cost')
                          .loc[{'costs': 'emissions'}]).sum().item()
        expensive_cost = (model.get_formatted_array('cost')
                               .loc[{'costs': 'monetary',
                                     'techs': 'expensive_clean_supply'}]).sum().item()
        assert round(emissions, 5) <= 400
        assert round(expensive_cost, 5) <= 600

    def test_different_locations_per_cost_group_constraint(self):
        model = build_model(
            model_file='model_cost_cap.yaml',
            scenario='different_locations_per_group'
        )
        model.run()
        cheap_cost = (model.get_formatted_array('cost')
                           .loc[{'costs': 'monetary', 'techs': 'cheap_polluting_supply'}]).sum().item()
        cheap_cost0 = (model.get_formatted_array('cost')
                            .loc[{'costs': 'monetary',
                                  'techs': 'cheap_polluting_supply',
                                  'locs': '0'}]).sum().item()
        assert round(cheap_cost, 5) <= 30
        assert round(cheap_cost0, 5) <= 10

    def test_different_techs_per_cost_group_constraint(self):
        model = build_model(
            model_file='model_cost_cap.yaml',
            scenario='different_techs_per_group'
        )
        model.run()
        emissions = (model.get_formatted_array('cost')
                          .loc[{'costs': 'emissions'}]).sum().item()
        clean_emissions = (model.get_formatted_array('cost')
                                .loc[{'costs': 'emissions',
                                      'techs': 'expensive_clean_supply'}]).sum().item()
        assert round(emissions, 5) <= 400
        assert round(clean_emissions, 5) <= 300


@pytest.mark.filterwarnings("ignore:(?s).*Not all requested techs:calliope.exceptions.ModelWarning")
class TestSupplyShareGroupConstraints:

    def test_no_supply_share_constraint(self):
        model = build_model(model_file='supply_share.yaml')
        model.run()
        expensive_generation = (model.get_formatted_array("carrier_prod")
                                     .loc[{'techs': "expensive_supply"}].sum()).item()
        assert expensive_generation == 0

    def test_systemwide_supply_share_max_constraint(self):
        model = build_model(
            model_file='supply_share.yaml',
            scenario='supply_share_max_systemwide'
        )
        model.run()
        cheap_generation = (model.get_formatted_array("carrier_prod")
                                 .to_dataframe()
                                 .reset_index()
                                 .groupby("techs")
                                 .carrier_prod
                                 .sum()
                                 .transform(lambda x: x / x.sum())
                                 .loc["cheap_supply"])
        assert round(cheap_generation, 5) <= 0.4

    def test_systemwide_supply_share_min_constraint(self):
        model = build_model(
            model_file='supply_share.yaml',
            scenario='supply_share_min_systemwide'
        )
        model.run()
        expensive_generation = (model.get_formatted_array("carrier_prod")
                                     .to_dataframe()
                                     .reset_index()
                                     .groupby("techs")
                                     .carrier_prod
                                     .sum()
                                     .transform(lambda x: x / x.sum())
                                     .loc["expensive_supply"])
        assert round(expensive_generation, 5) >= 0.6

    def test_location_specific_supply_share_max_constraint(self):
        model = build_model(
            model_file='supply_share.yaml',
            scenario='supply_share_max_location_0'
        )
        model.run()
        generation = (model.get_formatted_array("carrier_prod")
                           .sum(dim='timesteps').loc[{'carriers': 'electricity'}])
        cheap_generation0 = generation.loc[{'locs': "0", 'techs': "cheap_supply"}].item()
        expensive_generation0 = generation.loc[{'locs': "0", 'techs': "expensive_supply"}].item()
        expensive_generation1 = generation.loc[{'locs': "1", 'techs': "expensive_supply"}].item()
        assert round(cheap_generation0 / (cheap_generation0 + expensive_generation0), 5) <= 0.4
        assert expensive_generation1 == 0

    def test_location_specific_supply_share_min_constraint(self):
        model = build_model(
            model_file='supply_share.yaml',
            scenario='supply_share_min_location_0'
        )
        model.run()
        generation = (model.get_formatted_array("carrier_prod")
                           .sum(dim='timesteps').loc[{'carriers': 'electricity'}])
        cheap_generation0 = generation.loc[{'locs': "0", 'techs': "cheap_supply"}].item()
        expensive_generation0 = generation.loc[{'locs': "0", 'techs': "expensive_supply"}].item()
        expensive_generation1 = generation.loc[{'locs': "1", 'techs': "expensive_supply"}].item()
        assert round(expensive_generation0 / (cheap_generation0 + expensive_generation0), 5) >= 0.6
        assert expensive_generation1 == 0

    def test_supply_share_with_transmission(self):
        model = build_model(
            model_file='supply_share.yaml',
            scenario='supply_share_min_systemwide,transmission_link'
        )
        model.run()
        expensive_generation = (
            model.get_formatted_array("carrier_prod")
                 .loc[{"techs": "expensive_supply"}].sum().item()
        )
        supply = (
            model._model_data.carrier_prod
            .loc[{"loc_tech_carriers_prod":
                  model._model_data.loc_tech_carriers_supply_conversion_all.values}]
            .sum().item()
        )

        assert round(expensive_generation / supply, 5) >= 0.6

    def test_supply_share_with_storage(self):
        model = build_model(
            model_file='supply_share.yaml',
            scenario='supply_share_min_systemwide,storage_tech'
        )
        model.run()
        expensive_generation = (
            model.get_formatted_array("carrier_prod")
                 .loc[{"techs": "expensive_supply"}].sum().item()
        )
        supply = (
            model._model_data.carrier_prod
            .loc[{"loc_tech_carriers_prod":
                  model._model_data.loc_tech_carriers_supply_conversion_all.values}]
            .sum().item()
        )

        assert round(expensive_generation / supply, 5) >= 0.6

    def test_supply_share_per_timestep_max(self):
        model = build_model(
            model_file='supply_share.yaml',
            scenario='supply_share_per_timestep_max'
        )
        model.run()
        cheap_supply = model.get_formatted_array("carrier_prod").loc[{'techs': "cheap_supply", 'carriers': "electricity"}].sum('locs')
        supply = model.get_formatted_array("carrier_prod").loc[{'carriers': "electricity"}].sum('locs').sum('techs')
        # assert share in each timestep is 0.4
        assert ((cheap_supply / supply).round(5) <= 0.4).all()

    def test_supply_share_per_timestep_min(self):
        model = build_model(
            model_file='supply_share.yaml',
            scenario='supply_share_per_timestep_min'
        )
        model.run()
        expensive_supply = model.get_formatted_array("carrier_prod").loc[{'techs': "expensive_supply", 'carriers': "electricity"}].sum('locs')
        supply = model.get_formatted_array("carrier_prod").loc[{'carriers': "electricity"}].sum('locs').sum('techs')
        # assert share in each timestep is 0.6
        assert ((expensive_supply / supply).round(5) >= 0.6).all()

    def test_supply_share_per_timestep_equals(self):
        model = build_model(
            model_file='supply_share.yaml',
            scenario='supply_share_per_timestep_equals'
        )
        model.run()
        expensive_supply = model.get_formatted_array("carrier_prod").loc[{'techs': "expensive_supply", 'carriers': "electricity"}].sum('locs')
        supply = model.get_formatted_array("carrier_prod").loc[{'carriers': "electricity"}].sum('locs').sum('techs')
        # assert share in each timestep is 0.6
        assert ((expensive_supply / supply).round(5) == 0.6).all()


@pytest.mark.filterwarnings("ignore:(?s).*Not all requested techs:calliope.exceptions.ModelWarning")
class TestEnergyCapShareGroupConstraints:

    def test_no_energy_cap_share_constraint(self):
        model = build_model(model_file='energy_cap_share.yaml')
        model.run()
        expensive_capacity = (model.get_formatted_array("energy_cap")
                                   .loc[{'techs': "expensive_supply"}].sum())
        assert expensive_capacity == 0

    def test_systemwide_energy_cap_share_max_constraint(self):
        model = build_model(
            model_file='energy_cap_share.yaml',
            scenario='energy_cap_share_max_systemwide'
        )
        model.run()
        cheap_capacity = (model.get_formatted_array("energy_cap")
                               .to_dataframe()
                               .reset_index()
                               .groupby("techs")
                               .energy_cap
                               .sum()
                               .loc[["expensive_supply", "cheap_supply"]]  # remove demand
                               .transform(lambda x: x / x.sum())
                               .loc["cheap_supply"])
        assert cheap_capacity <= 0.4

    def test_systemwide_energy_cap_share_min_constraint(self):
        model = build_model(
            model_file='energy_cap_share.yaml',
            scenario='energy_cap_share_min_systemwide'
        )
        model.run()
        expensive_capacity = (model.get_formatted_array("energy_cap")
                                   .to_dataframe()
                                   .reset_index()
                                   .groupby("techs")
                                   .energy_cap
                                   .sum()
                                   .loc[["expensive_supply", "cheap_supply"]]  # remove demand
                                   .transform(lambda x: x / x.sum())
                                   .loc["expensive_supply"])
        assert expensive_capacity >= 0.6

    def test_location_specific_energy_cap_share_max_constraint(self):
        model = build_model(
            model_file='energy_cap_share.yaml',
            scenario='energy_cap_share_max_location_0'
        )
        model.run()
        capacity = model.get_formatted_array("energy_cap")
        cheap_capacity0 = capacity.loc[{'locs': "0", 'techs': "cheap_supply"}].item()
        expensive_capacity0 = capacity.loc[{'locs': "0", 'techs': "expensive_supply"}].item()
        expensive_capacity1 = capacity.loc[{'locs': "1", 'techs': "expensive_supply"}].item()
        assert cheap_capacity0 / (cheap_capacity0 + expensive_capacity0) <= 0.4
        assert expensive_capacity1 == 0

    def test_location_specific_energy_cap_share_min_constraint(self):
        model = build_model(
            model_file='energy_cap_share.yaml',
            scenario='energy_cap_share_min_location_0'
        )
        model.run()
        capacity = model.get_formatted_array("energy_cap")
        cheap_capacity0 = capacity.loc[{'locs': "0", 'techs': "cheap_supply"}].item()
        expensive_capacity0 = capacity.loc[{'locs': "0", 'techs': "expensive_supply"}].item()
        expensive_capacity1 = capacity.loc[{'locs': "1", 'techs': "expensive_supply"}].item()
        assert expensive_capacity0 / (cheap_capacity0 + expensive_capacity0) >= 0.6
        assert expensive_capacity1 == 0


@pytest.mark.filterwarnings("ignore:(?s).*Not all requested techs:calliope.exceptions.ModelWarning")
class TestEnergyCapGroupConstraints:

    def test_no_energy_cap_constraint(self):
        model = build_model(model_file='energy_cap.yaml')
        model.run()
        expensive_capacity = (model.get_formatted_array("energy_cap")
                                   .loc[{'techs': "expensive_supply"}].sum()).item()
        assert expensive_capacity == 0

    def test_systemwide_energy_cap_max_constraint(self):
        model = build_model(
            model_file='energy_cap.yaml',
            scenario='energy_cap_max_systemwide'
        )
        model.run()
        cheap_capacity = (model.get_formatted_array("energy_cap")
                               .loc[{'techs': "cheap_supply"}].sum()).item()
        assert round(cheap_capacity, 5) <= 14

    def test_systemwide_energy_cap_min_constraint(self):
        model = build_model(
            model_file='energy_cap.yaml',
            scenario='energy_cap_min_systemwide'
        )
        model.run()
        expensive_capacity = (model.get_formatted_array("energy_cap")
                                   .loc[{'techs': "expensive_supply"}].sum()).item()
        assert round(expensive_capacity, 5) >= 6

    def test_location_specific_energy_cap_max_constraint(self):
        model = build_model(
            model_file='energy_cap.yaml',
            scenario='energy_cap_max_location_0'
        )
        model.run()
        capacity = model.get_formatted_array("energy_cap")
        cheap_capacity0 = capacity.loc[{'locs': "0", 'techs': "cheap_supply"}].item()
        expensive_capacity1 = capacity.loc[{'locs': "1", 'techs': "expensive_supply"}].item()
        assert round(cheap_capacity0, 5) <= 4
        assert expensive_capacity1 == 0

    def test_location_specific_energy_cap_min_constraint(self):
        model = build_model(
            model_file='energy_cap.yaml',
            scenario='energy_cap_min_location_0'
        )
        model.run()
        capacity = model.get_formatted_array("energy_cap")
        expensive_capacity0 = capacity.loc[{'locs': "0", 'techs': "expensive_supply"}].item()
        expensive_capacity1 = capacity.loc[{'locs': "1", 'techs': "expensive_supply"}].item()
        assert round(expensive_capacity0, 5) >= 6
        assert expensive_capacity1 == 0