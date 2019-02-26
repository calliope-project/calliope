from pytest import approx

import calliope
from calliope.test.common.util import build_test_model as build_model


class TestNationalScaleExampleModelSenseChecks:
    def test_group_share_prod_min(self):
        model = calliope.examples.national_scale(
            scenario='cold_fusion_with_production_share'
        )
        model.run()

        df_carrier_prod = (
            model.get_formatted_array('carrier_prod')
                 .loc[dict(carriers='power')].sum('locs').sum('timesteps')
                 .to_pandas()
        )

        prod_share = (
            df_carrier_prod.loc[['cold_fusion', 'csp']].sum() /
            df_carrier_prod.loc[['ccgt', 'cold_fusion', 'csp']].sum()
        )

        assert prod_share == approx(0.85)

    def test_group_share_cap_max(self):
        model = calliope.examples.national_scale(
            scenario='cold_fusion_with_capacity_share'
        )
        model.run()

        cap_share = (
            model.get_formatted_array('energy_cap').to_pandas().loc[:, ['cold_fusion', 'csp']].sum().sum() /
            model.get_formatted_array('energy_cap').to_pandas().loc[:, ['ccgt', 'cold_fusion', 'csp']].sum().sum()
        )

        assert cap_share == approx(0.2)

    def test_systemwide_equals(self):
        model = calliope.examples.national_scale(
            override_dict={
                'techs.ccgt.constraints.energy_cap_max_systemwide': 10000,
                'techs.ac_transmission.constraints.energy_cap_equals_systemwide': 6000
            }
        )
        model.run()
        # Check that setting `_equals` to a finite value leads to forcing
        assert (
            model.get_formatted_array('energy_cap').loc[{'techs': 'ccgt'}].sum() == 10000
        )
        assert (
            model.get_formatted_array('energy_cap').loc[{'techs': 'ac_transmission:region1'}].sum() == 6000
        )

    def test_reserve_margin(self):
        model = calliope.examples.national_scale(
            scenario='reserve_margin'
        )

        model.run()

        # constraint_string = '-Inf : -1.1 * ( carrier_con[region1::demand_power::power,2005-01-05 16:00:00] + carrier_con[region2::demand_power::power,2005-01-05 16:00:00] ) / timestep_resolution[2005-01-05 16:00:00] - energy_cap[region1::ccgt] - energy_cap[region1-3::csp] - energy_cap[region1-2::csp] - energy_cap[region1-1::csp] :   0.0'

        # FIXME: capture Pyomo's print output...
        # assert constraint_string in model._backend_model.reserve_margin_constraint.pprint()

        assert float(model.results.cost.sum()) == approx(282487.35489)


class TestModelSettings:
    def test_feasibility(self):

        def override(feasibility, cap_val):
            override_dict = {
                'locations.0.techs': {'test_supply_elec': {}, 'test_demand_elec': {}},
                'links.0,1.exists': False,
                # pick a time subset where demand is uniformally -10 throughout
                'model.subset_time': ['2005-01-01 06:00:00', '2005-01-01 08:00:00'],
                'run.ensure_feasibility': feasibility, 'run.bigM': 1e3,
                # Allow setting resource and energy_cap_max/equals to force infeasibility
                'techs.test_supply_elec.constraints': {
                    'resource': cap_val, 'energy_eff': 1, 'energy_cap_equals': 15,
                    'force_resource': True
                }
            }

            return override_dict

        # Feasible case, unmet_demand/unused_supply is deleted
        model_10 = build_model(
            override_dict=override(True, 10),
            scenario='investment_costs'
        )
        model_10.run()
        for i in ['unmet_demand', 'unused_supply']:
            assert hasattr(model_10._backend_model, i)
            assert i not in model_10._model_data.data_vars.keys()

        # Infeasible case, unmet_demand is required
        model_5 = build_model(
            override_dict=override(True, 5),
            scenario='investment_costs'
        )
        model_5.run()
        assert hasattr(model_5._backend_model, 'unmet_demand')
        assert hasattr(model_5._backend_model, 'unused_supply')
        assert model_5._model_data['unmet_demand'].sum() == 15
        assert 'unused_supply' not in model_5._model_data.data_vars.keys()

        # Infeasible case, unused_supply is required
        model_15 = build_model(
            override_dict=override(True, 15),
            scenario='investment_costs'
        )
        model_15.run()
        assert hasattr(model_15._backend_model, 'unmet_demand')
        assert hasattr(model_15._backend_model, 'unused_supply')
        assert model_15._model_data['unmet_demand'].sum() == -15
        assert 'unused_supply' not in model_15._model_data.data_vars.keys()

        assert (
            model_15._backend_model.obj.expr() - model_10._backend_model.obj.expr() ==
            approx(1e3 * 15)
        )

        assert (
            model_5._backend_model.obj.expr() - model_10._backend_model.obj.expr() ==
            approx(1e3 * 15)
        )

        # Infeasible cases = non-optimal termination
        # too much supply
        model = build_model(
            override_dict=override(False, 15),
            scenario='investment_costs'
        )
        model.run()
        assert not hasattr(model._backend_model, 'unmet_demand')
        assert not hasattr(model._backend_model, 'unused_supply')
        assert not model._model_data.attrs['termination_condition'] == 'optimal'

        # too little supply
        model = build_model(
            override_dict=override(False, 5),
            scenario='investment_costs'
        )
        model.run()
        assert not model._model_data.attrs['termination_condition'] == 'optimal'


class TestGroupConstraints:
    def test_no_demand_share_constraint(self):
        model = build_model(model_file='model_demand_share.yaml')
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
            model_file='model_demand_share.yaml',
            scenario='demand_share_max_systemwide'
        )
        model.run()
        cheap_generation = (model.get_formatted_array("carrier_prod")
                                 .to_dataframe()
                                 .reset_index()
                                 .groupby("techs")
                                 .carrier_prod
                                 .sum()
                                 .transform(lambda x: x / x.sum())
                                 .loc["cheap_elec_supply"])
        assert cheap_generation <= 0.3

    def test_systemwide_demand_share_min_constraint(self):
        model = build_model(
            model_file='model_demand_share.yaml',
            scenario='demand_share_min_systemwide'
        )
        model.run()
        expensive_generation = (model.get_formatted_array("carrier_prod")
                                     .to_dataframe()
                                     .reset_index()
                                     .groupby("techs")
                                     .carrier_prod
                                     .sum()
                                     .transform(lambda x: x / x.sum())
                                     .loc["expensive_elec_supply"])
        assert expensive_generation >= 0.6

    def test_location_specific_demand_share_max_constraint(self):
        model = build_model(
            model_file='model_demand_share.yaml',
            scenario='demand_share_max_location_0'
        )
        model.run()
        generation = (model.get_formatted_array("carrier_prod")
                           .sum(dim='timesteps')
                           .to_dataframe()["carrier_prod"])
        demand0 = -model.get_formatted_array("carrier_con").loc[{'locs': '0'}].sum().item()
        cheap_generation0 = generation.loc[("0", "cheap_elec_supply", "electricity")]
        expensive_generation1 = generation.loc[("1", "expensive_elec_supply", "electricity")]
        assert cheap_generation0 / demand0 <= 0.3
        assert expensive_generation1 == 0

    def test_location_specific_demand_share_min_constraint(self):
        model = build_model(
            model_file='model_demand_share.yaml',
            scenario='demand_share_min_location_0'
        )
        model.run()
        generation = (model.get_formatted_array("carrier_prod")
                           .sum(dim='timesteps')
                           .to_dataframe()["carrier_prod"])
        demand0 = -model.get_formatted_array("carrier_con").loc[{'locs': '0'}].sum().item()
        expensive_generation0 = generation.loc[("0", "expensive_elec_supply", "electricity")]
        expensive_generation1 = generation.loc[("1", "expensive_elec_supply", "electricity")]
        assert expensive_generation0 / demand0 >= 0.6
        assert expensive_generation1 == 0

    def test_multiple_group_constraints(self):
        model = build_model(
            model_file='model_demand_share.yaml',
            scenario='multiple_constraints'
        )
        model.run()
        generation = (model.get_formatted_array("carrier_prod")
                           .sum(dim=('timesteps', 'locs', 'carriers')))
        demand = -model.get_formatted_array("carrier_con").sum().item()
        cheap_generation = generation.loc[{'techs': 'cheap_elec_supply'}].item()
        expensive_generation = generation.loc[{'techs': 'expensive_elec_supply'}].item()

        assert expensive_generation / demand >= 0.6
        assert cheap_generation / demand <= 0.3

    def test_multiple_group_carriers(self):
        model = build_model(
            model_file='model_demand_share.yaml',
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

        assert cheap_generation_elec / demand_elec <= 0.3
        assert cheap_generation_heat / demand_heat <= 0.5

    def test_multiple_group_carriers_constraints(self):
        model = build_model(
            model_file='model_demand_share.yaml',
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

        assert cheap_generation_elec / demand_elec <= 0.3
        assert expensive_generation_elec / demand_elec >= 0.6
        assert cheap_generation_heat / demand_heat <= 0.5
        assert expensive_generation_heat / demand_heat >= 0.4

    def test_different_locatinos_per_group_constraint(self):
        model = build_model(
            model_file='model_demand_share.yaml',
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

        assert expensive_generation_0 / demand_elec_0 >= 0.6
        assert expensive_generation_1 / demand_elec_1 == 0
        assert (cheap_generation_0 + cheap_generation_1) / (demand_elec_0 + demand_elec_1) <= 0.3


class TestResourceAreaGroupConstraints:

    def test_no_energy_cap_share_constraint(self):
        model = build_model(model_file='resource_area.yaml')
        model.run()
        cheap_resource_area = (model.get_formatted_array("resource_area")
                                    .to_dataframe()
                                    .reset_index()
                                    .groupby("techs")
                                    .resource_area
                                    .sum()
                                    .loc["cheap_supply"])
        assert cheap_resource_area == 40

    def test_systemwide_resource_area_max_constraint(self):
        model = build_model(
            model_file='resource_area.yaml',
            scenario='resource_area_max_systemwide'
        )
        model.run()
        cheap_resource_area = (model.get_formatted_array("resource_area")
                                    .to_dataframe()
                                    .reset_index()
                                    .groupby("techs")
                                    .resource_area
                                    .sum()
                                    .loc["cheap_supply"])
        assert cheap_resource_area == 20

    def test_systemwide_resource_area_min_constraint(self):
        model = build_model(
            model_file='resource_area.yaml',
            scenario='resource_area_min_systemwide'
        )
        model.run()
        resource_area = (model.get_formatted_array("resource_area")
                              .to_dataframe()
                              .reset_index()
                              .groupby("techs")
                              .resource_area
                              .sum())
        assert resource_area["cheap_supply"] == 0
        assert resource_area["expensive_supply"] == 20

    def test_location_specific_resource_area_max_constraint(self):
        model = build_model(
            model_file='resource_area.yaml',
            scenario='resource_area_max_location_0'
        )
        model.run()
        resource_area = (model.get_formatted_array("resource_area")
                              .to_dataframe()["resource_area"])
        cheap_resource_area0 = resource_area.loc[("0", "cheap_supply")]
        cheap_resource_area1 = resource_area.loc[("1", "cheap_supply")]
        assert cheap_resource_area0 == 10
        assert cheap_resource_area1 == 20

    def test_location_specific_resource_area_min_constraint(self):
        model = build_model(
            model_file='resource_area.yaml',
            scenario='resource_area_min_location_0'
        )
        model.run()
        resource_area = (model.get_formatted_array("resource_area")
                              .to_dataframe()["resource_area"])
        expensive_resource_area0 = resource_area.loc[("0", "expensive_supply")]
        expensive_resource_area1 = resource_area.loc[("1", "expensive_supply")]
        assert expensive_resource_area0 == 10
        assert expensive_resource_area1 == 0
