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

    cost_energy_cap = param_getter(backend_model, 'cost_energy_cap', (cost, loc_tech))
    cost_storage_cap = param_getter(backend_model, 'cost_storage_cap', (cost, loc_tech))
    cost_resource_cap = param_getter(backend_model, 'cost_resource_cap', (cost, loc_tech))
    cost_resource_area = param_getter(backend_model, 'cost_resource_area', (cost, loc_tech))
    cost_purchase = param_getter(backend_model, 'cost_purchase', (cost, loc_tech))
    cost_om_annual_investment_fraction = param_getter(backend_model, 'cost_om_annual_investment_fraction', (cost, loc_tech))
    cost_om_annual = param_getter(backend_model, 'cost_om_annual', (cost, loc_tech))

    if hasattr(backend_model, 'loc_techs_purchase') and loc_tech in backend_model.loc_techs_purchase:
        cost_of_purchase = backend_model.purchased[loc_tech] * cost_purchase
    elif hasattr(backend_model, 'loc_techs_milp') and loc_tech in backend_model.loc_techs_milp:
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

    # Tranmission technologies exist at two locations, thus their cost is divided by 2
    if loc_tech in backend_model.loc_techs_transmission:
            cost_con = cost_con / 2

    cost_fractional_om = cost_om_annual_investment_fraction * cost_con
    cost_fixed_om = cost_om_annual * backend_model.energy_cap[loc_tech] * ts_weight

    return (
        backend_model.cost_investment[cost, loc_tech] ==
        cost_fractional_om + cost_fixed_om + cost_con
    )


def cost_var_constraint_rule(backend_model, cost, loc_tech_om_cost, timestep):
    return po.Constraint.NoConstraint
    # backend_model.cost_var = po.Var(backend_model.loc_techs_om_cost, backend_model.costs, backend_model.timesteps, within=po.Reals)
