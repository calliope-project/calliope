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
    sets = backend_model.__calliope_model_data['sets']

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
            backend_model.timesteps,  # hardcoded for kyoto project
            rule=piecewise_om_cost_constraint_rule
        )


def piecewise_investment_cost_constraint_rule(backend_model, cost, loc_tech, slope_intercept):
    """
    Calculate costs from capacity decision variables.

    .. container:: scrolling-wrapper

        .. math::=

    """
    model_data_dict = backend_model.__calliope_model_data

    def _get_investment_cost(capacity_decision_variable, calliope_set):
        """
        Conditionally add investment costs, if the relevant set of technologies
        exists. Both inputs are strings.
        """
        if loc_tech_is_in(backend_model, loc_tech, calliope_set):
            _cost = get_param(backend_model, 'p_cost_' + capacity_decision_variable, (cost, loc_tech, slope_intercept))
            if _cost is not None and po.value(_cost) != 'nan':
                slope, intercept = (float(i) for i in po.value(_cost).split('::'))
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
        return po.Constraint.NoConstraint
    else:
        cost_con *= depreciation_rate * ts_weight

    # Transmission technologies exist at two locations, thus their cost is divided by 2
    if loc_tech_is_in(backend_model, loc_tech, 'loc_techs_transmission'):
            cost_con = cost_con / 2

    return (
        backend_model.cost_investment[cost, loc_tech] >= cost_con
    )


def piecewise_om_cost_constraint_rule(backend_model, cost, loc_tech, slope_intercept, step):
    """

    .. container:: scrolling-wrapper

        .. math::

    """
    model_data_dict = backend_model.__calliope_model_data['data']

    cost_om_prod = get_param(backend_model, 'p_cost_om_prod', (cost, loc_tech, slope_intercept, step))
    cost_om_con = get_param(backend_model, 'p_cost_om_con', (cost, loc_tech, slope_intercept, step))

    loc_tech_carrier = model_data_dict['lookup_loc_techs'][loc_tech]

    # TODO: remove monthwise step hardcoding
    last_timestep = [timestep for timestep in backend_model.timesteps if timestep.month == step.month][-1]
    if step != last_timestep:
        return backend_model.cost_var[cost, loc_tech, step] == 0

    if (po.value(cost_om_prod) != 'nan' and cost_om_prod is not None):
        slope, intercept = (float(i) for i in po.value(cost_om_prod).split('::'))
        cost_prod = (
            slope * sum(
                backend_model.timestep_weights[timestep] * backend_model.carrier_prod[loc_tech_carrier, timestep]
                # TODO: remove monthwise step hardcoding
                for timestep in backend_model.timesteps if timestep.month == step.month
            ) + intercept * backend_model.purchased[loc_tech]
        )
    else:
        cost_prod = None

    if loc_tech_is_in(backend_model, loc_tech, 'loc_techs_supply_plus') and po.value(cost_om_con) != 'nan' and cost_om_con is not None:
        slope, intercept = (float(i) for i in po.value(cost_om_prod).split('::'))
        cost_con = (
            slope * sum(
                backend_model.timestep_weights[timestep] * backend_model.resource_con[loc_tech, timestep]
                # TODO: remove monthwise step hardcoding
                for timestep in backend_model.timesteps if timestep.month == step.month
            ) + intercept * backend_model.purchased[loc_tech]
        )

    elif loc_tech_is_in(backend_model, loc_tech, 'loc_techs_supply') and po.value(cost_om_con) != 'nan' and cost_om_con is not None:
        slope, intercept = (float(i) for i in po.value(cost_om_prod).split('::'))
        # TODO: add in a check for energy_eff == 0, as this will create an infinite value
        cost_con = (
            slope * sum(
                backend_model.timestep_weights[timestep] *
                (backend_model.carrier_prod[loc_tech_carrier, timestep] /
                    get_param(backend_model, 'energy_eff', (loc_tech, timestep)))
                # TODO: remove monthwise step hardcoding
                for timestep in backend_model.timesteps if timestep.month == step.month
            ) + intercept * backend_model.purchased[loc_tech]
        )
    else:
        cost_con = None

    if cost_prod is not None and cost_con is not None:
        return backend_model.cost_var[cost, loc_tech, last_timestep] >= cost_prod + cost_con
    elif cost_prod is not None:
        return backend_model.cost_var[cost, loc_tech, last_timestep] >= cost_prod
    elif cost_con is not None:
        return backend_model.cost_var[cost, loc_tech, last_timestep] >= cost_con
    else:
        return po.Constraint.NoConstraint
