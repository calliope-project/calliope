"""
Copyright (C) 2013-2017 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

dispatch.py
~~~~~~~~~~~~~~~~~

Energy dispatch constraints, limiting production/consumption to the capacities
of the technologies

"""

import pyomo.core as po  # pylint: disable=import-error

from calliope.backend.pyomo.util import get_param, get_loc_tech


def load_dispatch_constraints(backend_model):
    sets = backend_model.__calliope_model_data__['sets']

    backend_model.carrier_production_max_constraint = po.Constraint(
        backend_model.loc_tech_carriers_prod, backend_model.timesteps,
        rule=carrier_production_max_constraint_rule
    )

    backend_model.carrier_production_min_constraint = po.Constraint(
        backend_model.loc_tech_carriers_prod, backend_model.timesteps,
        rule=carrier_production_min_constraint_rule
    )

    backend_model.carrier_consumption_max_constraint = po.Constraint(
        backend_model.loc_tech_carriers_con, backend_model.timesteps,
        rule=carrier_consumption_max_constraint_rule
    )

    if 'loc_techs_supply_plus' in sets:
        backend_model.resource_max_constraint = po.Constraint(
            backend_model.loc_techs_supply_plus, backend_model.timesteps,
            rule=resource_max_constraint_rule
        )

    if 'loc_techs_store' in sets:
        backend_model.storage_max_constraint = po.Constraint(
            backend_model.loc_techs_store, backend_model.timesteps,
            rule=storage_max_constraint_rule
        )


def carrier_production_max_constraint_rule(backend_model, loc_tech_carrier, timestep):
    """
    Set maximum carrier production. All technologies.
    """
    loc_tech = get_loc_tech(loc_tech_carrier)
    sets = backend_model.__calliope_model_data__['sets']
    carrier_prod = backend_model.carrier_prod[loc_tech_carrier, timestep]
    timestep_resolution = get_param(backend_model, 'timestep_resolution', timestep)
    parasitic_eff = get_param(backend_model, 'parasitic_eff', (loc_tech, timestep))

    if 'loc_tech_milp' in sets and loc_tech in backend_model.loc_tech_milp:
        energy_cap = get_param(backend_model, 'energy_cap_per_unit', loc_tech)
        return carrier_prod <= (
            backend_model.operating_units[loc_tech, timestep] *
            timestep_resolution * energy_cap * parasitic_eff
        )
    else:
        return carrier_prod <= (
            backend_model.energy_cap[loc_tech] * timestep_resolution * parasitic_eff
        )


def carrier_production_min_constraint_rule(backend_model, loc_tech_carrier, timestep):
    """
    Set minimum carrier production. All technologies except conversion_plus
    """
    loc_tech = get_loc_tech(loc_tech_carrier)
    sets = backend_model.__calliope_model_data__['sets']
    carrier_prod = backend_model.carrier_prod[loc_tech_carrier, timestep]
    timestep_resolution = get_param(backend_model, 'timestep_resolution', timestep)
    min_use = get_param(backend_model, 'energy_cap_min_use', (loc_tech, timestep))

    if not min_use:
        return po.Constraint.NoConstraint

    if 'loc_tech_milp' in sets and loc_tech in backend_model.loc_tech_milp:
        energy_cap = get_param(backend_model, 'energy_cap_per_unit', loc_tech)
        return carrier_prod >= (
            backend_model.operating_units[loc_tech, timestep] *
            timestep_resolution * energy_cap * min_use
        )
    else:
        return carrier_prod >= (
            backend_model.energy_cap[loc_tech] * timestep_resolution * min_use
        )


# FIXME: should this be only built over demand? All other technologies have an
# energy balance constraint forcing carrier_con to be a function of carrier_prod
# (so are limited by the carrier_production_max_constraint_rule).
def carrier_consumption_max_constraint_rule(backend_model, loc_tech_carrier, timestep):
    """
    Set maximum carrier consumption. All technologies.
    """
    loc_tech = get_loc_tech(loc_tech_carrier)
    sets = backend_model.__calliope_model_data__['sets']
    carrier_con = backend_model.carrier_con[loc_tech_carrier, timestep]
    timestep_resolution = get_param(backend_model, 'timestep_resolution', timestep)

    if 'loc_tech_milp' in sets and loc_tech in backend_model.loc_tech_milp:
        energy_cap = get_param(backend_model, 'energy_cap_per_unit', loc_tech)
        return carrier_con >= (-1 *
            backend_model.operating_units[loc_tech, timestep] *
            timestep_resolution * energy_cap
        )
    else:
        return carrier_con >= (-1 *
            backend_model.energy_cap[loc_tech] * timestep_resolution
        )


def resource_max_constraint_rule(backend_model, loc_tech, timestep):
    """
    Set maximum resource supply. Supply_plus techs only.
    """
    timestep_resolution = get_param(backend_model, 'timestep_resolution', timestep)

    return backend_model.resource_con[loc_tech, timestep] <= (
        timestep_resolution * backend_model.resource_cap[loc_tech])


def storage_max_constraint_rule(backend_model, loc_tech, timestep):
    """
    Set maximum stored energy. Supply_plus & storage techs only.
    """
    return (backend_model.storage[loc_tech, timestep]
        <= backend_model.storage_cap[loc_tech])
