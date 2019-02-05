"""
Copyright (C) 2013-2019 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

piecewise.py
~~~~~~~~~

Piecewise linearised constraints.

"""

import pyomo.core as po  # pylint: disable=import-error

from calliope.backend.pyomo.util import \
    get_param, \
    get_timestep_weight, \
    loc_tech_is_in


def load_constraints(backend_model):
    sets = backend_model.__calliope_model_data__['sets']

    if 'loc_techs_piecewise_investment_cost' in sets:
        backend_model.piecewise_investment_cost_constraint = po.Constraint(
            backend_model.costs,
            backend_model.loc_techs_piecewise_investment_cost,
            backend_model.slope_intercepts,
            rule=piecewise_investment_cost_constraint_rule
        )
    if 'loc_techs_piecewise_om_cost' in sets:
        backend_model.piecewise_om_cost_constraint = po.Constraint(
            backend_model.costs,
            backend_model.loc_techs_piecewise_om_cost,
            backend_model.slope_intercepts,
            backend_model.timesteps,
            rule=piecewise_om_cost_constraint_rule
        )


def piecewise_investment_cost_constraint_rule(backend_model, cost, loc_tech, slope_intercept):
    """
    Calculate costs from capacity decision variables.

    .. container:: scrolling-wrapper

        .. math::=

    """
    model_data_dict = backend_model.__calliope_model_data__

    def _get_investment_cost(capacity_decision_variable, calliope_set):
        """
        Conditionally add investment costs, if the relevant set of technologies
        exists. Both inputs are strings.
        """
        if loc_tech_is_in(backend_model, loc_tech, calliope_set):
            _cost = get_param(backend_model, 'p_cost_' + capacity_decision_variable, (cost, loc_tech, slope_intercept))
            if _cost is not None:
                slope, intercept = (int(i) for i in po.value(_cost).split('::'))
                return (
                    slope * getattr(backend_model, capacity_decision_variable)[loc_tech]
                    + intercept * backend_model.purchased[loc_tech]
                )
        else:
            return None

    cost_energy_cap = _get_investment_cost('energy_cap', 'loc_techs')
    cost_storage_cap = _get_investment_cost('storage_cap', 'loc_techs_store')
    cost_resource_cap = _get_investment_cost('resource_cap', 'loc_techs_supply_plus')
    cost_resource_area = _get_investment_cost('resource_area', 'loc_techs_area')

    ts_weight = get_timestep_weight(backend_model)
    depreciation_rate = model_data_dict['data']['cost_depreciation_rate'].get((cost, loc_tech), 0)

    cost_con = None
    for i in [cost_energy_cap, cost_storage_cap, cost_resource_cap, cost_resource_area]:
        if i is not None:
            if cost_con is None:
                cost_con = i
            else:
                cost_con += i
    if cost_con is None:
        return po.NoConstraint
    else:
        cost_con *= depreciation_rate * ts_weight

    # Transmission technologies exist at two locations, thus their cost is divided by 2
    if loc_tech_is_in(backend_model, loc_tech, 'loc_techs_transmission'):
            cost_con = cost_con / 2

    backend_model.cost_investment_rhs[cost, loc_tech].expr = cost_con

    return (
        backend_model.cost_investment[cost, loc_tech] >=
        backend_model.cost_investment_rhs[cost, loc_tech]
    )


def piecewise_om_cost_constraint_rule(backend_model, cost, loc_tech, slope_intercept, timestep):
    """

    .. container:: scrolling-wrapper

        .. math::

    """
    model_data_dict = backend_model.__calliope_model_data__['data']

    cost_om_prod = get_param(backend_model, 'p_cost_om_prod', (cost, loc_tech, slope_intercept, timestep))
    cost_om_con = get_param(backend_model, 'p_cost_om_con', (cost, loc_tech, slope_intercept, timestep))

    weight = backend_model.timestep_weights[timestep]

    loc_tech_carrier = model_data_dict['lookup_loc_techs'][loc_tech]

    if cost_om_prod is not None:
        slope, intercept = (int(i) for i in po.value(cost_om_prod).split('::'))
        cost_prod = slope * weight * backend_model.carrier_prod[loc_tech_carrier, timestep] + intercept * backend_model.purchased[loc_tech]
    else:
        cost_prod = None

    if loc_tech_is_in(backend_model, loc_tech, 'loc_techs_supply_plus') and cost_om_con is not None:
        (int(i) for i in po.value(cost_om_con).split('::'))
        cost_con = slope * weight * backend_model.resource_con[loc_tech, timestep] + intercept * backend_model.purchased[loc_tech]

    elif loc_tech_is_in(backend_model, loc_tech, 'loc_techs_supply') and cost_om_con is not None:
        (int(i) for i in po.value(cost_om_con).split('::'))
        energy_eff = get_param(backend_model, 'energy_eff', (loc_tech, timestep))
        if po.value(energy_eff) > 0:  # in case energy_eff is zero, to avoid an infinite value
            cost_con = slope * weight * (backend_model.carrier_prod[loc_tech_carrier, timestep] / energy_eff) + intercept * backend_model.purchased[loc_tech]
        else:
            cost_con = None
    else:
        cost_con = None

    if cost_prod is not None and cost_con is not None:
        return backend_model.cost_var[cost, loc_tech, timestep] >= cost_prod + cost_con
    elif cost_prod is not None:
        return backend_model.cost_var[cost, loc_tech, timestep] >= cost_prod
    elif cost_con is not None:
        return backend_model.cost_var[cost, loc_tech, timestep] >= cost_con
    else:
        po.NoConstraint