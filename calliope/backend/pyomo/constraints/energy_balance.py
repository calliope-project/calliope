"""
Copyright (C) 2013-2017 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

energy_balance.py
~~~~~~~~~~~~~~~~~

Energy balance constraints.

"""

import pyomo.core as po  # pylint: disable=import-error


def load_energy_balance_constraints(backend_model):
    model_data_dict = backend_model.__calliope_model_data__

    backend_model.balance_supply_constraint = po.Constraint(
        backend_model.loc_techs_finite_resource_supply, backend_model.timesteps,
        rule=balance_supply_constraint_rule
    )

    backend_model.balance_demand_constraint = po.Constraint(
        backend_model.loc_techs_finite_resource_demand, backend_model.timesteps,
        rule=balance_demand_constraint_rule
    )

    if 'loc_techs_transmission' in model_data_dict['sets']:
        backend_model.balance_transmission_constraint = po.Constraint(
            backend_model.loc_techs_transmission, backend_model.timesteps,
            rule=balance_transmission_constraint_rule
        )

    if 'loc_techs_supply_plus' in model_data_dict['sets']:
        backend_model.balance_supply_plus_constraint = po.Constraint(
            backend_model.loc_techs_supply_plus, backend_model.timesteps,
            rule=balance_supply_plus_constraint_rule
        )

        backend_model.resource_availability_supply_plus_constraint = po.Constraint(
            backend_model.loc_techs_finite_resource_supply_plus, backend_model.timesteps,
            rule=resource_availability_supply_plus_constraint_rule
        )

    if 'loc_techs_store' in model_data_dict['sets']:
        backend_model.balance_storage_constraint = po.Constraint(
            backend_model.loc_techs_store, backend_model.timesteps,
            rule=balance_storage_constraint_rule
        )


def balance_supply_constraint_rule(backend_model, loc_tech, timestep):
    model_data_dict = backend_model.__calliope_model_data__

    resource = model_data_dict['resource'][(loc_tech, timestep)]
    resource_scale = model_data_dict['resource_scale'][(loc_tech, timestep)]
    force_resource = model_data_dict['force_resource'][(loc_tech, timestep)]
    energy_eff = model_data_dict['energy_eff'][(loc_tech, timestep)]

    if energy_eff == 0:
        carrier_prod = 0
    else:
        loc_tech_carrier = model_data_dict['lookup_loc_tech_carriers'][loc_tech]
        carrier_prod = backend_model.carrier_prod[loc_tech_carrier, timestep] / energy_eff

    if loc_tech in backend_model.loc_techs_area:
        resource_avail = resource * resource_scale * backend_model.resource_area[loc_tech]
    else:
        resource_avail = resource * resource_scale

    if force_resource:
        return carrier_prod == resource_avail
    else:
        return carrier_prod <= resource_avail


def balance_demand_constraint_rule(backend_model, loc_tech, timestep):
    model_data_dict = backend_model.__calliope_model_data__

    resource = model_data_dict['resource'][(loc_tech, timestep)]
    resource_scale = model_data_dict['resource_scale'][(loc_tech, timestep)]
    force_resource = model_data_dict['force_resource'][(loc_tech, timestep)]
    energy_eff = model_data_dict['energy_eff'][(loc_tech, timestep)]

    loc_tech_carrier = model_data_dict['lookup_loc_tech_carriers'][loc_tech]
    carrier_con = backend_model.carrier_con[loc_tech_carrier, timestep] * energy_eff

    if loc_tech in backend_model.loc_tech_area:
        r_avail = resource * resource_scale * backend_model.r_area[loc_tech]
    else:
        r_avail = resource * resource_scale

    if force_resource:
        return carrier_con == r_avail
    else:
        return carrier_con >= r_avail


def resource_availability_supply_plus_constraint_rule(backend_model, loc_tech, timestep):
    model_data_dict = backend_model.__calliope_model_data__

    resource = model_data_dict['resource'][(loc_tech, timestep)]
    resource_scale = model_data_dict['resource_scale'][(loc_tech, timestep)]
    force_resource = model_data_dict['force_resource'][(loc_tech, timestep)]
    resource_eff = model_data_dict['resource_eff'][(loc_tech, timestep)]

    if loc_tech in backend_model.loc_tech_area:
        resource_avail = resource * resource_scale * backend_model.resource_area[loc_tech] * resource_eff
    else:
        resource_avail = resource * resource_scale * resource_eff

    if force_resource:
        return backend_model.resource[loc_tech, timestep] == resource_avail
    else:
        return backend_model.resource[loc_tech, timestep] <= resource_avail


def balance_transmission_constraint_rule(backend_model, loc_tech, timestep):
    model_data_dict = backend_model.__calliope_model_data__

    energy_eff = model_data_dict['energy_eff'][(loc_tech, timestep)]
    loc_tech_carrier = model_data_dict['lookup_loc_tech_carriers'][loc_tech]
    remote_loc_tech = model_data_dict['lookup_remotes'][loc_tech]
    remote_loc_tech_carrier = model_data_dict['lookup_loc_tech_carriers'][remote_loc_tech]

    if remote_loc_tech in backend_model.loc_tech_transmission:
        return (
            backend_model.carrier_prod[loc_tech_carrier, timestep] ==
            -1 * backend_model.carrier_con[remote_loc_tech_carrier, timestep] *
            energy_eff
        )
    else:
        return po.Constraint.NoConstraint


def balance_supply_plus_constraint_rule(backend_model, loc_tech, timestep):
    model_data_dict = backend_model.__calliope_model_data__

    energy_eff = model_data_dict['energy_eff'][(loc_tech, timestep)]
    parasitic_eff = model_data_dict['parasitic_eff'][(loc_tech, timestep)]
    total_eff = energy_eff * parasitic_eff

    if total_eff == 0:
        carrier_prod = 0
    else:
        loc_tech_carrier = model_data_dict['lookup_loc_tech_carriers'][loc_tech]
        carrier_prod = backend_model.carrier_prod[loc_tech_carrier, timestep] / total_eff

    # A) Case where no storage allowed
    if loc_tech not in backend_model.loc_techs_store:
        return backend_model.resource[loc_tech, timestep] == carrier_prod

    # B) Case where storage is allowed
    else:
        resource = backend_model.resource[loc_tech, timestep]
        if backend_model.timesteps.order_dict[timestep] == 0:  # FIXME?
            storage_previous_step = model_data_dict['storage_initial'][loc_tech]
        else:
            storage_loss = model_data_dict['storage_loss'][loc_tech]
            time_resolution = model_data_dict['time_resolution'][timestep]
            previous_step = backend_model.timesteps[backend_model.timesteps.order_dict[timestep] - 1]
            storage_previous_step = (
                ((1 - storage_loss) ** time_resolution[previous_step]) *
                backend_model.storage[loc_tech, previous_step]
            )

        return (
            backend_model.storage[loc_tech, timestep] ==
            storage_previous_step + resource - carrier_prod
        )


def balance_storage_constraint_rule(backend_model, loc_tech, timestep):
    model_data_dict = backend_model.__calliope_model_data__

    energy_eff = model_data_dict['energy_eff'][(loc_tech, timestep)]
    parasitic_eff = model_data_dict['parasitic_eff'][(loc_tech, timestep)]
    total_eff = energy_eff * parasitic_eff

    if total_eff == 0:
        carrier_prod = 0
    else:
        loc_tech_carrier = model_data_dict['lookup_loc_tech_carriers'][loc_tech]
        carrier_prod = backend_model.carrier_prod[loc_tech_carrier, timestep] / total_eff

    carrier_con = backend_model.carrier_con[loc_tech_carrier, timestep] * total_eff

    if backend_model.timesteps.order_dict[timestep] == 0:
        storage_previous_step = model_data_dict['storage_initial'][loc_tech]
    else:
        storage_loss = model_data_dict['storage_loss'][loc_tech]
        time_resolution = model_data_dict['time_resolution'][timestep]
        previous_step = backend_model.timesteps[backend_model.timesteps.order_dict[timestep] - 1]
        storage_previous_step = (
            ((1 - storage_loss) ** time_resolution[previous_step]) *
            backend_model.storage[loc_tech, previous_step]
        )

    return (
        backend_model.storage[loc_tech, timestep] ==
        storage_previous_step - carrier_prod - carrier_con
    )
