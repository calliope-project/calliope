import pytest
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
