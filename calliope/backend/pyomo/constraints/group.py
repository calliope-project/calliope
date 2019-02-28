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
    if 'group_energy_cap_share_min' in model_data_dict:
        backend_model.group_energy_cap_share_min_constraint = po.Constraint(
            backend_model.constraint_groups,
            ['min'], rule=energy_cap_share_constraint_rule
        )
    if 'group_energy_cap_share_max' in model_data_dict:
        backend_model.group_energy_cap_share_max_constraint = po.Constraint(
            backend_model.constraint_groups,
            ['max'], rule=energy_cap_share_constraint_rule
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


def energy_cap_share_constraint_rule(backend_model, constraint_group, what):
    """
    Enforces shares of energy_cap for groups of technologies and locations. The
    share is relative to ``supply`` and ``supply_plus`` technologies only.

    .. container:: scrolling-wrapper

        .. math::

            \\sum_{loc::tech \\in given\\_group} energy_{cap}(loc::tech) \\leq
            share \\times \\sum_{loc::tech \\in loc\\_tech\\_supply\\_all \\in given\\_locations} energy_{cap}(loc::tech)
    """
    model_data_dict = backend_model.__calliope_model_data__['data']
    share = model_data_dict['group_energy_cap_share_{}'.format(what)][(constraint_group)]
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
            'group_constraint_loc_techs_{}'.format(constraint_group)
        )
        lhs_locs = [loc_tech.split('::')[0] for loc_tech in lhs_loc_techs]
        rhs_loc_techs = [
            i for i in backend_model.loc_techs_supply
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
