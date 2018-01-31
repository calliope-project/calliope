"""
Copyright (C) 2013-2017 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

policy.py
~~~~~~~~~

Policy constraints.

"""

import pyomo.core as po  # pylint: disable=import-error


def load_constraints(backend_model):
    sets = backend_model.__calliope_model_data__['sets']

    if 'techlists_group_share_energy_cap_min_constraint' in sets:
        for techlist in backend_model.techlists_group_share_energy_cap_min_constraint:
            group_share_energy_cap_constraint_rule(backend_model, techlist, what='min')

    if 'techlists_group_share_energy_cap_max_constraint' in sets:
        for techlist in backend_model.techlists_group_share_energy_cap_max_constraint:
            group_share_energy_cap_constraint_rule(backend_model, techlist, what='max')

    if 'techlists_group_share_energy_cap_equals_constraint' in sets:
        for techlist in backend_model.techlists_group_share_energy_cap_equals_constraint:
            group_share_energy_cap_constraint_rule(backend_model, techlist, what='equals')

    if 'techlists_carrier_group_share_carrier_prod_min_constraint' in sets:
        for techlist_carrier in backend_model.techlists_carrier_group_share_carrier_prod_min_constraint:
            group_share_carrier_prod_constraint_rule(backend_model, techlist_carrier, what='min')

    if 'techlists_carrier_group_share_carrier_prod_max_constraint' in sets:
        for techlist_carrier in backend_model.techlists_carrier_group_share_carrier_prod_max_constraint:
            group_share_carrier_prod_constraint_rule(backend_model, techlist_carrier, what='max')

    if 'techlists_carrier_group_share_carrier_prod_equals_constraint' in sets:
        for techlist_carrier in backend_model.techlists_carrier_group_share_carrier_prod_equals_constraint:
            group_share_carrier_prod_constraint_rule(backend_model, techlist_carrier, what='equals')


def equalizer(lhs, rhs, sign):
    if sign == 'max':
        return lhs <= rhs
    elif sign == 'min':
        return lhs >= rhs
    elif sign == 'equals':
        return lhs == rhs
    else:
        raise ValueError('Invalid sign: {}'.format(sign))


def group_share_energy_cap_constraint_rule(backend_model, techlist, what):
    model_data_dict = backend_model.__calliope_model_data__['data']
    fraction = model_data_dict['group_share_energy_cap_{}'.format(what)][techlist]

    rhs_loc_techs = backend_model.loc_techs_supply
    lhs_loc_techs = [
        i for i in backend_model.loc_techs_supply
        if i.split('::')[0] in techlist.split(',')
    ]

    rhs = (fraction * sum(backend_model.energy_cap[loc_tech] for loc_tech in rhs_loc_techs))
    lhs = sum(backend_model.energy_cap[loc_tech] for loc_tech in lhs_loc_techs)

    return equalizer(lhs, rhs, what)


def group_share_carrier_prod_constraint_rule(backend_model, techlist_carrier, what):
    model_data_dict = backend_model.__calliope_model_data__['data']
    techlist, carrier = techlist_carrier.split('::')
    fraction = model_data_dict['group_share_carrier_prod_{}'.format(what)][(carrier, techlist)]

    rhs_loc_tech_carriers = [
        i for i in backend_model.loc_tech_carriers_supply_all
        if i.split('::')[-1] == carrier
    ]
    lhs_loc_tech_carriers = [
        i for i in backend_model.loc_tech_carriers_supply_all
        if i.split('::')[0] in techlist.split(',')
        and i.split('::')[-1] == carrier
    ]
    rhs = (fraction * sum(
        backend_model.carrier_prod[loc_tech_carrier, timestep]
        for loc_tech_carrier in rhs_loc_tech_carriers for timestep in backend_model.timesteps
    ))
    lhs = sum(
        backend_model.carrier_prod[loc_tech_carrier, timestep]
        for loc_tech_carrier in lhs_loc_tech_carriers for timestep in backend_model.timesteps
    )

    return equalizer(lhs, rhs, what)
