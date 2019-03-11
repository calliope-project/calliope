"""
Copyright (C) 2013-2019 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

group.py
~~~~~~~~

Group constraints.

"""

import numpy as np
import pyomo.core as po  # pylint: disable=import-error


def load_constraints(backend_model):
    model_data_dict = backend_model.__calliope_model_data__['data']

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
    TODO write docstring
    """
    model_data_dict = backend_model.__calliope_model_data__['data']
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
            i for i in backend_model.loc_tech_carriers_con
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


def cost_cap_constraint_rule(backend_model, group_name, cost, what):
    """
    Limit system-wide cost for a specific cost class to a certain value, i.e. Ɛ-constrained costs

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

    model_data_dict = backend_model.__calliope_model_data__['data']
    cost_cap = model_data_dict['group_cost_{}'.format(what)].get(
        (cost, group_name), np.nan
    )

    if np.isnan(cost_cap):
        return po.Constraint.NoConstraint

    sum_cost = sum(backend_model.cost[cost, loc_tech] for loc_tech in loc_techs)

    return equalizer(sum_cost, cost_cap, what)


def cost_investment_cap_constraint_rule(backend_model, group_name, cost, what):
    """
    Limit system-wide investment costs specific to a cost class to a
    certain value, i.e. Ɛ-constrained costs

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

    model_data_dict = backend_model.__calliope_model_data__['data']
    cost_cap = model_data_dict['group_cost_investment_{}'.format(what)].get(
        (cost, group_name), np.nan
    )

    if np.isnan(cost_cap):
        return po.Constraint.NoConstraint

    sum_cost = sum(backend_model.cost_investment[cost, loc_tech] for loc_tech in loc_techs)

    return equalizer(sum_cost, cost_cap, what)


def cost_var_cap_constraint_rule(backend_model, group_name, cost, what):
    """
    Limit system-wide variable costs specific to a cost class
    to a certain value, i.e. Ɛ-constrained costs

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

    model_data_dict = backend_model.__calliope_model_data__['data']
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
