"""
Copyright (C) 2013-2017 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

energy_balance.py
~~~~~~~~~~~~~~~~~

Energy balance constraints.

"""

import pyomo.core as po  # pylint: disable=import-error

from calliope.backend.pyomo.util import \
    get_param, \
    get_previous_timestep, \
    get_loc_tech_carriers


def load_constraints(backend_model):
    sets = backend_model.__calliope_model_data__['sets']


    if 'loc_carriers_system_balance_constraint' in sets:
        backend_model.system_balance = po.Expression(
            backend_model.loc_carriers_system_balance_constraint,
            backend_model.timesteps,
            initialize=0.0
        )

        backend_model.system_balance_constraint = po.Constraint(
            backend_model.loc_carriers_system_balance_constraint,
            backend_model.timesteps,
            rule=system_balance_constraint_rule
        )

    if 'loc_techs_balance_supply_constraint' in sets:
        backend_model.balance_supply_constraint = po.Constraint(
            backend_model.loc_techs_balance_supply_constraint,
            backend_model.timesteps,
            rule=balance_supply_constraint_rule
        )

    if 'loc_techs_balance_demand_constraint' in sets:
        backend_model.balance_demand_constraint = po.Constraint(
            backend_model.loc_techs_balance_demand_constraint,
            backend_model.timesteps,
            rule=balance_demand_constraint_rule
        )

    if 'loc_techs_balance_transmission_constraint' in sets:
        backend_model.balance_transmission_constraint = po.Constraint(
            backend_model.loc_techs_balance_transmission_constraint,
            backend_model.timesteps,
            rule=balance_transmission_constraint_rule
        )

    if 'loc_techs_resource_availability_supply_plus_constraint' in sets:
        backend_model.balance_supply_plus_constraint = po.Constraint(
            backend_model.loc_techs_resource_availability_supply_plus_constraint,
            backend_model.timesteps,
            rule=balance_supply_plus_constraint_rule
        )

    if 'loc_techs_balance_supply_plus_constraint' in sets:
        backend_model.resource_availability_supply_plus_constraint = po.Constraint(
            backend_model.loc_techs_balance_supply_plus_constraint,
            backend_model.timesteps,
            rule=resource_availability_supply_plus_constraint_rule
        )

    if 'loc_techs_balance_storage_constraint' in sets:
        backend_model.balance_storage_constraint = po.Constraint(
            backend_model.loc_techs_balance_storage_constraint,
            backend_model.timesteps,
            rule=balance_storage_constraint_rule
        )

    if 'carriers_reserve_margin_constraint' in sets:
        backend_model.reserve_margin_constraint = po.Constraint(
            backend_model.carriers_reserve_margin_constraint,
            rule=reserve_margin_constraint_rule
        )


def system_balance_constraint_rule(backend_model, loc_carrier, timestep):
    """
    System balance ensures that, within each location, the production and
    consumption of each carrier is balanced.
    """
    prod, con, export = get_loc_tech_carriers(backend_model, loc_carrier)

    backend_model.system_balance[loc_carrier, timestep].expr = (
        sum(backend_model.carrier_prod[loc_tech_carrier, timestep] for loc_tech_carrier in prod) +
        sum(backend_model.carrier_con[loc_tech_carrier, timestep] for loc_tech_carrier in con)
    )

    return backend_model.system_balance[loc_carrier, timestep] == 0


def balance_supply_constraint_rule(backend_model, loc_tech, timestep):
    """
    Limit production from supply techs to their available resource
    """
    model_data_dict = backend_model.__calliope_model_data__['data']

    resource = get_param(backend_model, 'resource', (loc_tech, timestep))
    energy_eff = get_param(backend_model, 'energy_eff', (loc_tech, timestep))
    resource_scale = get_param(backend_model, 'resource_scale', loc_tech)
    force_resource = get_param(backend_model, 'force_resource', loc_tech)
    loc_tech_carrier = model_data_dict['lookup_loc_techs'][loc_tech]
    min_use = get_param(backend_model, 'resource_min_use', (loc_tech, timestep))

    if energy_eff == 0:
        return backend_model.carrier_prod[loc_tech_carrier, timestep] == 0
    else:
        carrier_prod = backend_model.carrier_prod[loc_tech_carrier, timestep] / energy_eff

    if loc_tech in backend_model.loc_techs_area:
        available_resource = resource * resource_scale * backend_model.resource_area[loc_tech]
    else:
        available_resource = resource * resource_scale

    if force_resource:
        return carrier_prod == available_resource
    elif min_use:
        return min_use * available_resource <= carrier_prod <= available_resource
    else:
        return carrier_prod <= available_resource


def balance_demand_constraint_rule(backend_model, loc_tech, timestep):
    """
    Limit consumption from demand techs to their required resource
    """
    model_data_dict = backend_model.__calliope_model_data__['data']

    resource = get_param(backend_model, 'resource', (loc_tech, timestep))
    energy_eff = get_param(backend_model, 'energy_eff', (loc_tech, timestep))
    resource_scale = get_param(backend_model, 'resource_scale', loc_tech)
    force_resource = get_param(backend_model, 'force_resource', loc_tech)

    loc_tech_carrier = model_data_dict['lookup_loc_techs'][loc_tech]
    carrier_con = backend_model.carrier_con[loc_tech_carrier, timestep] * energy_eff

    if loc_tech in backend_model.loc_techs_area:
        required_resource = resource * resource_scale * backend_model.resource_area[loc_tech]
    else:
        required_resource = resource * resource_scale

    if force_resource:
        return carrier_con == required_resource
    else:
        return carrier_con >= required_resource


def resource_availability_supply_plus_constraint_rule(backend_model, loc_tech, timestep):
    """
    Limit production from supply_plus techs to their available resource
    """
    resource = get_param(backend_model, 'resource', (loc_tech, timestep))
    resource_eff = get_param(backend_model, 'resource_eff', (loc_tech, timestep))
    resource_scale = get_param(backend_model, 'resource_scale', loc_tech)
    force_resource = get_param(backend_model, 'force_resource', loc_tech)

    if loc_tech in backend_model.loc_techs_area:
        available_resource = resource * resource_scale * backend_model.resource_area[loc_tech] * resource_eff
    else:
        available_resource = resource * resource_scale * resource_eff

    if force_resource:
        return backend_model.resource_con[loc_tech, timestep] == available_resource
    else:
        return backend_model.resource_con[loc_tech, timestep] <= available_resource


def balance_transmission_constraint_rule(backend_model, loc_tech, timestep):
    """
    Balance carrier production and consumption of transmission technologies
    """
    model_data_dict = backend_model.__calliope_model_data__['data']

    energy_eff = get_param(backend_model, 'energy_eff', (loc_tech, timestep))
    loc_tech_carrier = model_data_dict['lookup_loc_techs'][loc_tech]
    remote_loc_tech = model_data_dict['lookup_remotes'][loc_tech]
    remote_loc_tech_carrier = model_data_dict['lookup_loc_techs'][remote_loc_tech]

    if remote_loc_tech in backend_model.loc_techs_transmission:
        return (
            backend_model.carrier_prod[loc_tech_carrier, timestep] ==
            -1 * backend_model.carrier_con[remote_loc_tech_carrier, timestep] *
            energy_eff
        )
    else:
        return po.Constraint.NoConstraint


def balance_supply_plus_constraint_rule(backend_model, loc_tech, timestep):
    """
    Balance carrier production and resource consumption of supply_plus technologies
    alongside any use of resource storage
    """
    model_data_dict = backend_model.__calliope_model_data__['data']
    sets = backend_model.__calliope_model_data__['sets']

    energy_eff = get_param(backend_model, 'energy_eff', (loc_tech, timestep))
    parasitic_eff = get_param(backend_model, 'parasitic_eff', (loc_tech, timestep))
    total_eff = energy_eff * parasitic_eff

    if total_eff == 0:
        carrier_prod = 0
    else:
        loc_tech_carrier = model_data_dict['lookup_loc_techs'][loc_tech]
        carrier_prod = backend_model.carrier_prod[loc_tech_carrier, timestep] / total_eff

    # A) Case where no storage allowed
    if 'loc_techs_store' not in sets or loc_tech not in backend_model.loc_techs_store:
        return backend_model.resource_con[loc_tech, timestep] == carrier_prod

    # B) Case where storage is allowed
    else:
        resource = backend_model.resource_con[loc_tech, timestep]
        if backend_model.timesteps.order_dict[timestep] == 0:
            storage_previous_step = get_param(backend_model, 'storage_initial', loc_tech)
        else:
            storage_loss = get_param(backend_model, 'storage_loss', loc_tech)
            previous_step = get_previous_timestep(backend_model, timestep)
            time_resolution = model_data_dict['timestep_resolution'][previous_step]
            storage_previous_step = (
                ((1 - storage_loss) ** time_resolution) *
                backend_model.storage[loc_tech, previous_step]
            )

        return (
            backend_model.storage[loc_tech, timestep] ==
            storage_previous_step + resource - carrier_prod
        )


def balance_storage_constraint_rule(backend_model, loc_tech, timestep):
    """
    Balance carrier production and consumption of storage technologies,
    alongside any use of the stored volume
    """
    model_data_dict = backend_model.__calliope_model_data__['data']

    energy_eff = get_param(backend_model, 'energy_eff', (loc_tech, timestep))

    if energy_eff == 0:
        carrier_prod = 0
    else:
        loc_tech_carrier = model_data_dict['lookup_loc_techs'][loc_tech]
        carrier_prod = backend_model.carrier_prod[loc_tech_carrier, timestep] / energy_eff

    carrier_con = backend_model.carrier_con[loc_tech_carrier, timestep] * energy_eff

    if backend_model.timesteps.order_dict[timestep] == 0:
        storage_previous_step = get_param(backend_model, 'storage_initial', loc_tech)
    else:
        storage_loss = get_param(backend_model, 'storage_loss', loc_tech)
        previous_step = get_previous_timestep(backend_model, timestep)
        time_resolution = model_data_dict['timestep_resolution'][previous_step]
        storage_previous_step = (
            ((1 - storage_loss) ** time_resolution) *
            backend_model.storage[loc_tech, previous_step]
        )

    return (
        backend_model.storage[loc_tech, timestep] ==
        storage_previous_step - carrier_prod - carrier_con
    )


def reserve_margin_constraint_rule(backend_model, carrier):
    model_data_dict = backend_model.__calliope_model_data__['data']

    reserve_margin = model_data_dict['reserve_margin'][carrier]
    max_demand_timestep = model_data_dict['max_demand_timesteps'][carrier]
    max_demand_time_res = model_data_dict['timestep_resolution'][max_demand_timestep]

    return (
        sum(  # Sum all demand for this carrier and timestep
            backend_model.carrier_con[loc_tech_carrier, max_demand_timestep]
            for loc_tech_carrier in backend_model.loc_tech_carriers_demand
        ) * -1 * (1 / max_demand_time_res)
        >=
        sum(  # Sum all supply capacity for this carrier
            backend_model.energy_cap[loc_tech_carrier.rsplit('::', 1)[0]]
            for loc_tech_carrier in backend_model.loc_tech_carriers_supply_all
            if loc_tech_carrier.rsplit('::', 1)[1] == carrier
        ) * (1 + reserve_margin)
    )
