"""
Copyright (C) 2013-2019 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

group.py
~~~~~~~~

Group constraints.

"""

import numpy as np
import pyomo.core as po  # pylint: disable=import-error

ORDER = 20  # order in which to invoke constraints relative to other constraint files


def load_constraints(backend_model):
    model_data_dict = backend_model.__calliope_model_data['data']

    if 'group_demand_share_min' in model_data_dict:
        backend_model.group_demand_share_min_constraint = po.Constraint(
            backend_model.group_names_demand_share_min,
            backend_model.carriers,
            ['min'], rule=demand_share_constraint_rule
        )
    if 'group_demand_share_max' in model_data_dict:
        backend_model.group_demand_share_max_constraint = po.Constraint(
            backend_model.group_names_demand_share_max,
            backend_model.carriers,
            ['max'], rule=demand_share_constraint_rule
        )
    if 'group_supply_share_min' in model_data_dict:
        backend_model.group_supply_share_min_constraint = po.Constraint(
            backend_model.group_names_supply_share_min,
            backend_model.carriers,
            ['min'], rule=supply_share_constraint_rule
        )
    if 'group_supply_share_max' in model_data_dict:
        backend_model.group_supply_share_max_constraint = po.Constraint(
            backend_model.group_names_supply_share_max,
            backend_model.carriers,
            ['max'], rule=supply_share_constraint_rule
        )
    if 'group_energy_cap_share_min' in model_data_dict:
        backend_model.group_energy_cap_share_min_constraint = po.Constraint(
            backend_model.group_names_energy_cap_share_min,
            ['min'], rule=energy_cap_share_constraint_rule
        )
    if 'group_energy_cap_share_max' in model_data_dict:
        backend_model.group_energy_cap_share_max_constraint = po.Constraint(
            backend_model.group_names_energy_cap_share_max,
            ['max'], rule=energy_cap_share_constraint_rule
        )
    if 'group_energy_cap_min' in model_data_dict:
        backend_model.group_energy_cap_min_constraint = po.Constraint(
            backend_model.group_names_energy_cap_min,
            ['min'], rule=energy_cap_constraint_rule
        )
    if 'group_energy_cap_max' in model_data_dict:
        backend_model.group_energy_cap_max_constraint = po.Constraint(
            backend_model.group_names_energy_cap_max,
            ['max'], rule=energy_cap_constraint_rule
        )
    if 'group_resource_area_min' in model_data_dict:
        backend_model.group_resource_area_min_constraint = po.Constraint(
            backend_model.group_names_resource_area_min,
            ['min'], rule=resource_area_constraint_rule
        )
    if 'group_resource_area_max' in model_data_dict:
        backend_model.group_resource_area_max_constraint = po.Constraint(
            backend_model.group_names_resource_area_max,
            ['max'], rule=resource_area_constraint_rule
        )

    for sense in ['min', 'max', 'equals']:
        if 'group_cost_{}'.format(sense) in model_data_dict:
            setattr(
                backend_model, 'group_cost_{}_constraint'.format(sense),
                po.Constraint(getattr(backend_model, 'group_names_cost_{}'.format(sense)),
                              backend_model.costs, [sense], rule=cost_cap_constraint_rule)
            )
        if 'group_cost_var_{}'.format(sense) in model_data_dict:
            setattr(
                backend_model, 'group_cost_var_{}_constraint'.format(sense),
                po.Constraint(getattr(backend_model, 'group_names_cost_var_{}'.format(sense)),
                              backend_model.costs, [sense], rule=cost_var_cap_constraint_rule)
            )
        if 'group_cost_investment_{}'.format(sense) in model_data_dict:
            setattr(
                backend_model, 'group_cost_investment_{}_constraint'.format(sense),
                po.Constraint(getattr(backend_model, 'group_names_cost_investment_{}'.format(sense)),
                              backend_model.costs, [sense], rule=cost_investment_cap_constraint_rule)
            )


def equalizer(lhs, rhs, sign):
    if sign == 'max':
        return lhs <= rhs
    elif sign == 'min':
        return lhs >= rhs
    elif sign == 'equals':
        return lhs == rhs
    else:
        raise ValueError('Invalid sign: {}'.format(sign))


def demand_share_constraint_rule(backend_model, group_name, carrier, what):
    """
    Enforces shares of demand of a carrier to be met by the given groups
    of technologies at the given locations. The share is relative
    to ``demand`` technologies only.

    .. container:: scrolling-wrapper

        .. math::

            \\sum_{loc::tech::carrier \\in given\\_group, timestep \\in timesteps} carrier_{prod}(loc::tech::carrier, timestep) \\leq
            share \\times \\sum_{loc::tech:carrier \\in loc\\_techs\\_demand \\in given\\_locations, timestep\\in timesteps}
            carrier_{con}(loc::tech::carrier, timestep)

    """
    model_data_dict = backend_model.__calliope_model_data['data']
    share = model_data_dict['group_demand_share_{}'.format(what)].get(
        (carrier, group_name), np.nan
    )
    # FIXME uncomment this once Bryn has merged his changes
    # and import again: from calliope.backend.pyomo.util import get_param
    # share = get_param(
    #     backend_model,
    #     'group_demand_share_{}'.format(what), (carrier, constraint_group)
    # )

    if np.isnan(share):
        return po.Constraint.NoConstraint
    else:
        lhs_loc_techs = getattr(
            backend_model,
            'group_constraint_loc_techs_{}'.format(group_name)
        )
        lhs_locs = set(loc_tech.split('::')[0] for loc_tech in lhs_loc_techs)
        lhs_loc_tech_carriers = [
            i for i in backend_model.loc_tech_carriers_prod
            if i.rsplit('::', 1)[0] in lhs_loc_techs and i.split('::')[-1] == carrier
        ]
        rhs_loc_tech_carriers = [
            i for i in backend_model.loc_tech_carriers_demand
            if i.split('::')[0] in lhs_locs and i.split('::')[-1] == carrier
        ]

        lhs = sum(
            backend_model.carrier_prod[loc_tech_carrier, timestep]
            for loc_tech_carrier in lhs_loc_tech_carriers
            for timestep in backend_model.timesteps
        )
        rhs = share * -1 * sum(
            backend_model.carrier_con[loc_tech_carrier, timestep]
            for loc_tech_carrier in rhs_loc_tech_carriers
            for timestep in backend_model.timesteps
        )

        return equalizer(lhs, rhs, what)


def supply_share_constraint_rule(backend_model, constraint_group, carrier, what):
    """
    Enforces shares of carrier_prod for groups of technologies and locations. The
    share is relative to ``supply`` and ``supply_plus`` technologies only.

    .. container:: scrolling-wrapper

        .. math::

            \\sum_{loc::tech::carrier \\in given\\_group, timestep \\in timesteps} carrier_{prod}(loc::tech::carrier, timestep) \\leq
            share \\times \\sum_{loc::tech:carrier \\in loc\\_tech\\_carriers\\_supply\\_all \\in given\\_locations, timestep\\in timesteps}
            carrier_{prod}(loc::tech::carrier, timestep)

    """
    model_data_dict = backend_model.__calliope_model_data['data']
    share = model_data_dict['group_supply_share_{}'.format(what)][(carrier, constraint_group)]

    if np.isnan(share):
        return po.Constraint.NoConstraint
    else:
        lhs_loc_techs = getattr(
            backend_model,
            'group_constraint_loc_techs_{}'.format(constraint_group)
        )
        lhs_locs = [loc_tech.split('::')[0] for loc_tech in lhs_loc_techs]
        rhs_loc_techs = [
            i for i in backend_model.loc_techs_supply_all
            if i.split('::')[0] in lhs_locs
        ]

        lhs = sum(
            backend_model.carrier_prod[loc_tech + '::' + carrier, timestep]
            for loc_tech in lhs_loc_techs
            for timestep in backend_model.timesteps
        )
        rhs = share * sum(
            backend_model.carrier_prod[loc_tech + '::' + carrier, timestep]
            for loc_tech in rhs_loc_techs
            for timestep in backend_model.timesteps
        )

        return equalizer(lhs, rhs, what)


def energy_cap_share_constraint_rule(backend_model, constraint_group, what):
    """
    Enforces shares of energy_cap for groups of technologies and locations. The
    share is relative to ``supply`` and ``supply_plus`` technologies only.

    .. container:: scrolling-wrapper

        .. math::

            \\sum_{loc::tech \\in given\\_group} energy_{cap}(loc::tech) \\leq
            share \\times \\sum_{loc::tech \\in loc\\_tech\\_supply\\_all \\in given\\_locations} energy_{cap}(loc::tech)
    """
    model_data_dict = backend_model.__calliope_model_data['data']
    share = model_data_dict['group_energy_cap_share_{}'.format(what)][(constraint_group)]

    if np.isnan(share):
        return po.Constraint.NoConstraint
    else:
        lhs_loc_techs = getattr(
            backend_model,
            'group_constraint_loc_techs_{}'.format(constraint_group)
        )
        lhs_locs = [loc_tech.split('::')[0] for loc_tech in lhs_loc_techs]
        rhs_loc_techs = [
            i for i in backend_model.loc_techs_supply_all
            if i.split('::')[0] in lhs_locs
        ]

        lhs = sum(
            backend_model.energy_cap[loc_tech]
            for loc_tech in lhs_loc_techs
        )
        rhs = share * sum(
            backend_model.energy_cap[loc_tech]
            for loc_tech in rhs_loc_techs
        )

        return equalizer(lhs, rhs, what)


def energy_cap_constraint_rule(backend_model, constraint_group, what):
    """
    Enforce upper and lower bounds for energy_cap of energy_cap
    for groups of technologies and locations.

    .. container:: scrolling-wrapper

        .. math::

            \\sum_{loc::tech \\in given\\_group} energy_{cap}(loc::tech) \\leq energy\\_cap\\_max\\\\

            \\sum_{loc::tech \\in given\\_group} energy_{cap}(loc::tech) \\geq energy\\_cap\\_min

    """
    model_data_dict = backend_model.__calliope_model_data['data']
    threshold = model_data_dict['group_energy_cap_{}'.format(what)][(constraint_group)]

    if np.isnan(threshold):
        return po.Constraint.NoConstraint
    else:
        lhs_loc_techs = getattr(
            backend_model,
            'group_constraint_loc_techs_{}'.format(constraint_group)
        )
        lhs = sum(
            backend_model.energy_cap[loc_tech]
            for loc_tech in lhs_loc_techs
        )
        rhs = threshold

        return equalizer(lhs, rhs, what)


def cost_cap_constraint_rule(backend_model, group_name, cost, what):
    """
    Limit cost for a specific cost class to a certain value,
    i.e. Ɛ-constrained costs,
    for groups of technologies and locations.

    .. container:: scrolling-wrapper

        .. math::

            \\sum{loc::tech \\in loc\\_techs_{group\\_name}, timestep \\in timesteps}
            \\boldsymbol{cost}(cost, loc::tech, timestep)
            \\begin{cases}
                \\leq cost\\_max(cost)
                \\geq cost\\_min(cost)
                = cost\\_equals(cost)
            \\end{cases}

    """

    loc_techs = [i for i in getattr(
        backend_model,
        'group_constraint_loc_techs_{}'.format(group_name)
    ) if i in backend_model.loc_techs_cost]

    model_data_dict = backend_model.__calliope_model_data['data']
    cost_cap = model_data_dict['group_cost_{}'.format(what)].get(
        (cost, group_name), np.nan
    )

    if np.isnan(cost_cap):
        return po.Constraint.NoConstraint

    sum_cost = sum(backend_model.cost[cost, loc_tech] for loc_tech in loc_techs)

    return equalizer(sum_cost, cost_cap, what)


def cost_investment_cap_constraint_rule(backend_model, group_name, cost, what):
    """
    Limit investment costs specific to a cost class to a
    certain value, i.e. Ɛ-constrained costs,
    for groups of technologies and locations.

    .. container:: scrolling-wrapper

        .. math::

            \\sum{loc::tech \\in loc\\_techs_{group\\_name}, timestep \\in timesteps}
            \\boldsymbol{cost\\_{investment}}(cost, loc::tech, timestep)
            \\begin{cases}
                \\leq cost\\_investment\\_max(cost)
                \\geq cost\\_investment\\_min(cost)
                = cost\\_investment\\_equals(cost)
            \\end{cases}

    """

    loc_techs = [i for i in getattr(
        backend_model,
        'group_constraint_loc_techs_{}'.format(group_name)
    ) if i in backend_model.loc_techs_investment_cost]

    model_data_dict = backend_model.__calliope_model_data['data']
    cost_cap = model_data_dict['group_cost_investment_{}'.format(what)].get(
        (cost, group_name), np.nan
    )

    if np.isnan(cost_cap):
        return po.Constraint.NoConstraint

    sum_cost = sum(backend_model.cost_investment[cost, loc_tech] for loc_tech in loc_techs)

    return equalizer(sum_cost, cost_cap, what)


def cost_var_cap_constraint_rule(backend_model, group_name, cost, what):
    """
    Limit variable costs specific to a cost class
    to a certain value, i.e. Ɛ-constrained costs,
    for groups of technologies and locations.

    .. container:: scrolling-wrapper

        .. math::

            \\sum{loc::tech \\in loc\\_techs_{group\\_name}, timestep \\in timesteps}
            \\boldsymbol{cost\\_{var}}(cost, loc::tech, timestep)
            \\begin{cases}
                \\leq cost\\_var\\_max(cost)
                \\geq cost\\_var\\_min(cost)
                = cost\\_var\\_equals(cost)
            \\end{cases}

    """

    loc_techs = [i for i in getattr(
        backend_model,
        'group_constraint_loc_techs_{}'.format(group_name)
    ) if i in backend_model.loc_techs_om_cost]

    model_data_dict = backend_model.__calliope_model_data['data']
    cost_cap = model_data_dict['group_cost_var_{}'.format(what)].get(
        (cost, group_name), np.nan
    )

    if np.isnan(cost_cap):
        return po.Constraint.NoConstraint

    sum_cost = sum(
        backend_model.cost_var[cost, loc_tech, timestep]
        for loc_tech in loc_techs for timestep in backend_model.timesteps
    )

    return equalizer(sum_cost, cost_cap, what)


def resource_area_constraint_rule(backend_model, constraint_group, what):
    """
    Enforce upper and lower bounds of resource_area for groups of
    technologies and locations.

    .. container:: scrolling-wrapper

        .. math::

            \\boldsymbol{resource_{area}}(loc::tech) \\leq group\\_resource\\_area\\_max\\\\

            \\boldsymbol{resource_{area}}(loc::tech) \\geq group\\_resource\\_area\\_min

    """
    model_data_dict = backend_model.__calliope_model_data['data']
    threshold = model_data_dict['group_resource_area_{}'.format(what)][(constraint_group)]

    if np.isnan(threshold):
        return po.Constraint.NoConstraint
    else:
        lhs_loc_techs = getattr(
            backend_model,
            'group_constraint_loc_techs_{}'.format(constraint_group)
        )

        lhs = sum(
            backend_model.resource_area[loc_tech]
            for loc_tech in lhs_loc_techs
        )
        rhs = threshold

        return equalizer(lhs, rhs, what)
