"""
Copyright (C) 2013-2017 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

costs.py
~~~~~~~~

Cost constraints.

"""

import pyomo.core as po  # pylint: disable=import-error

from calliope.backend.pyomo.util import \
    param_getter, \
    get_timestep_weight


def load_cost_constraints(backend_model):
    model_data_dict = backend_model.__calliope_model_data__

    backend_model.cost_constraint = po.Constraint(
        backend_model.costs,
        backend_model.loc_techs_cost,
        rule=cost_constraint_rule
    )

    if 'loc_techs_investment_cost' in model_data_dict['sets']:
        backend_model.cost_investment_constraint = po.Constraint(
            backend_model.costs,
            backend_model.loc_techs_investment_cost,
            rule=cost_investment_constraint_rule
        )

    if 'loc_techs_om_cost' in model_data_dict['sets']:
        backend_model.cost_var_constraint = po.Constraint(
            backend_model.costs,
            backend_model.loc_techs_om_cost,
            backend_model.timesteps,
            rule=cost_var_constraint_rule
        )


def cost_constraint_rule(backend_model, cost, loc_tech):
    if loc_tech in backend_model.loc_techs_investment_cost:
        cost_investment = backend_model.cost_investment[cost, loc_tech]
    else:
        cost_investment = 0

    if loc_tech in backend_model.loc_techs_om_cost:
        cost_var = sum(backend_model.cost_var[cost, loc_tech, timestep] for timestep in backend_model.timesteps)
    else:
        cost_var = 0

    return (
        backend_model.cost[cost, loc_tech] == cost_investment + cost_var
    )

def cost_investment_constraint_rule(backend_model, cost, loc_tech):
    model_data_dict = backend_model.__calliope_model_data__

    def _get_investment_cost(capacity_decision_variable, calliope_set):
        """
        Conditionally add investment costs, if the relevant set of technologies
        exists. Both inputs are strings.
        """
        if hasattr(backend_model, calliope_set) and loc_tech in getattr(backend_model, calliope_set):
            _cost = (getattr(backend_model, capacity_decision_variable)[loc_tech] *
                param_getter(backend_model, 'cost_' + capacity_decision_variable, (cost, loc_tech)))
            return _cost
        else: return 0

    cost_energy_cap = (backend_model.energy_cap[loc_tech]
        * param_getter(backend_model, 'cost_energy_cap', (cost, loc_tech)))

    cost_storage_cap = _get_investment_cost('storage_cap', 'loc_techs_store')
    cost_resource_cap = _get_investment_cost('resource_cap', 'loc_techs_supply_plus')
    cost_resource_area = _get_investment_cost('resource_area', 'loc_techs_area')

    cost_om_annual_investment_fraction = param_getter(backend_model, 'cost_om_annual_investment_fraction', (cost, loc_tech))
    cost_om_annual = param_getter(backend_model, 'cost_om_annual', (cost, loc_tech))

    if hasattr(backend_model, 'loc_techs_purchase') and loc_tech in backend_model.loc_techs_purchase:
        cost_purchase = param_getter(backend_model, 'cost_purchase', (cost, loc_tech))
        cost_of_purchase = backend_model.purchased[loc_tech] * cost_purchase
    elif hasattr(backend_model, 'loc_techs_milp') and loc_tech in backend_model.loc_techs_milp:
        cost_purchase = param_getter(backend_model, 'cost_purchase', (cost, loc_tech))
        cost_of_purchase = backend_model.units[loc_tech] * cost_purchase
    else:
        cost_of_purchase = 0

    ts_weight = get_timestep_weight(backend_model)
    depreciation_rate = model_data_dict['data']['cost_depreciation_rate'][(cost, loc_tech)]

    cost_con = (
        depreciation_rate * ts_weight *
        (cost_energy_cap + cost_storage_cap + cost_resource_cap +
         cost_resource_area + cost_of_purchase)
    )

    # Transmission technologies exist at two locations, thus their cost is divided by 2
    if loc_tech in backend_model.loc_techs_transmission:
            cost_con = cost_con / 2

    cost_fractional_om = cost_om_annual_investment_fraction * cost_con
    cost_fixed_om = cost_om_annual * backend_model.energy_cap[loc_tech] * ts_weight

    return (
        backend_model.cost_investment[cost, loc_tech] ==
        cost_fractional_om + cost_fixed_om + cost_con
    )


def cost_var_constraint_rule(backend_model, cost, loc_tech, timestep):
    model_data_dict = backend_model.__calliope_model_data__

    cost_om_prod = param_getter(backend_model, 'cost_om_prod', (cost, loc_tech, timestep))
    cost_om_con = param_getter(backend_model, 'cost_om_con', (cost, loc_tech, timestep))
    weight = model_data_dict['data']['timestep_weights'][timestep]

    if hasattr(backend_model, 'loc_techs_export') and loc_tech in backend_model.loc_techs_export:
        export = backend_model.export[loc_tech, timestep]
        cost_export = param_getter(backend_model, 'cost_export', (cost, loc_tech, timestep)) * export
    else:
        export = 0
        cost_export = 0

    loc_tech_carrier = model_data_dict['data']['lookup_loc_techs'][loc_tech]

    if cost_om_prod:
        cost_prod = cost_om_prod * weight * backend_model.carrier_prod[loc_tech_carrier, timestep]
    else:
        cost_prod = 0

    if hasattr(backend_model, 'loc_techs_supply_plus') and loc_tech in backend_model.loc_techs_supply_plus and cost_om_con:
        resource_eff = param_getter(backend_model, 'resource_eff', (loc_tech, timestep))
        if resource_eff > 0:  # In case resource_eff is zero, to avoid an infinite value
            # Dividing by r_eff here so we get the actual r used, not the r
            # moved into storage...
            cost_con = cost_om_con * weight * (backend_model.resource[loc_tech, timestep] / resource_eff)
        else:
            cost_con = 0
    elif hasattr(backend_model, 'loc_techs_supply') and loc_tech in backend_model.loc_techs_supply and cost_om_con:
        energy_eff = param_getter(backend_model, 'energy_eff', (loc_tech, timestep))
        if energy_eff > 0:  # in case energy_eff is zero, to avoid an infinite value
            cost_con = cost_om_con * weight * (backend_model.carrier_prod[loc_tech_carrier, timestep] / energy_eff)
        else:
            cost_con = 0
    else:
        cost_con = 0

    return (backend_model.cost_var[cost, loc_tech, timestep] ==
            cost_prod + cost_con + cost_export)
