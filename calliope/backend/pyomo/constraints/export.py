"""
Copyright (C) 2013-2017 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

Export.py
~~~~~~~~~~~~~~~~~

Energy export constraints.

"""

import pyomo.core as po  # pylint: disable=import-error

from calliope.backend.pyomo.util import \
    get_param, \
    get_loc_tech_carriers, \
    get_loc_tech, \
    loc_tech_is_in


def load_export_constraints(backend_model):
    model_data_dict = backend_model.__calliope_model_data__

    for loc_carrier, timestep in backend_model.loc_carriers * backend_model.timesteps:
        update_system_balance_constraint(backend_model, loc_carrier, timestep)

    backend_model.export_balance_constraint = po.Constraint(
        backend_model.loc_tech_carriers_export, backend_model.timesteps,
        rule=export_balance_constraint_rule
    )

    for cost, loc_tech, timestep in (backend_model.costs *
        backend_model.loc_techs_costs_export * backend_model.timesteps):
        update_costs_var_constraint(backend_model, cost, loc_tech, timestep)

    backend_model.export_max_constraint = po.Constraint(
        backend_model.loc_tech_carriers_export, backend_model.timesteps,
        rule=export_max_constraint_rule
    )


def update_system_balance_constraint(backend_model, loc_carrier, timestep):
    prod, con, export = get_loc_tech_carriers(backend_model, loc_carrier)

    backend_model.system_balance[loc_carrier, timestep].expr += -1 * (
        sum(backend_model.carrier_export[loc_tech_carrier, timestep]
            for loc_tech_carrier in export)
    )

    return None


def export_balance_constraint_rule(backend_model, loc_tech_carrier, timestep):
    # Ensuring no technology can 'pass' its export capability to another
    # technology with the same carrier_out,
    # by limiting its export to the capacity of its production

    return (backend_model.carrier_prod[loc_tech_carrier, timestep] >=
            backend_model.carrier_export[loc_tech_carrier, timestep])


def update_costs_var_constraint(backend_model, cost, loc_tech, timestep):
    model_data_dict = backend_model.__calliope_model_data__

    loc_tech_carrier = model_data_dict['data']['lookup_loc_techs_export'][(loc_tech)]
    weight = model_data_dict['data']['timestep_weights'][timestep]

    cost_export = (
        get_param(backend_model, 'cost_export', (cost, loc_tech, timestep))
        * backend_model.carrier_prod[loc_tech_carrier, timestep]
        * weight
    )

    backend_model.cost_var_rhs[cost, loc_tech, timestep].expr += cost_export


def export_max_constraint_rule(backend_model, loc_tech_carrier, timestep):
    """
    Set maximum export. All exporting technologies.
    """

    loc_tech = get_loc_tech(loc_tech_carrier)
    sets = backend_model.__calliope_model_data__['sets']

    if loc_tech_is_in(backend_model, loc_tech, 'loc_tech_milp'):
        operating_units = backend_model.operating_units[loc_tech, timestep]
    else:
        operating_units = 1

    export_cap = get_param(backend_model, 'export_cap', loc_tech)
    if export_cap:
        return (backend_model.carrier_export[loc_tech_carrier, timestep] <=
                export_cap * operating_units)
    else:
        return po.Constraint.Skip
