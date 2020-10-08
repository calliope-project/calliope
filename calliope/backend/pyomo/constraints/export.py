"""
Copyright (C) since 2013 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

Export.py
~~~~~~~~~~~~~~~~~

Energy export constraints.

"""

import pyomo.core as po  # pylint: disable=import-error

from calliope.backend.pyomo.util import (
    get_param,
    get_loc_tech_carriers,
    get_loc_tech,
    loc_tech_is_in,
)

ORDER = 30  # order in which to invoke constraints relative to other constraint files


def load_constraints(backend_model):
    sets = backend_model.__calliope_model_data["sets"]

    if "loc_carriers_update_system_balance_constraint" in sets:
        for loc_carrier, timestep in (
            backend_model.loc_carriers_update_system_balance_constraint
            * backend_model.timesteps
        ):
            update_system_balance_constraint(backend_model, loc_carrier, timestep)

    if "loc_tech_carriers_export_balance_constraint" in sets:
        backend_model.export_balance_constraint = po.Constraint(
            backend_model.loc_tech_carriers_export_balance_constraint,
            backend_model.timesteps,
            rule=export_balance_constraint_rule,
        )

    if "loc_techs_update_costs_var_constraint" in sets:
        for cost, loc_tech, timestep in (
            backend_model.costs
            * backend_model.loc_techs_update_costs_var_constraint
            * backend_model.timesteps
        ):
            update_costs_var_constraint(backend_model, cost, loc_tech, timestep)

    if "loc_tech_carriers_export_max_constraint" in sets:
        backend_model.export_max_constraint = po.Constraint(
            backend_model.loc_tech_carriers_export_max_constraint,
            backend_model.timesteps,
            rule=export_max_constraint_rule,
        )


def update_system_balance_constraint(backend_model, loc_carrier, timestep):
    """
    Update system balance constraint (from energy_balance.py) to include export

    Math given in :func:`~calliope.backend.pyomo.constraints.energy_balance.system_balance_constraint_rule`
    """
    prod, con, export = get_loc_tech_carriers(backend_model, loc_carrier)

    backend_model.system_balance[loc_carrier, timestep].expr += -1 * (
        sum(
            backend_model.carrier_export[loc_tech_carrier, timestep]
            for loc_tech_carrier in export
        )
    )

    return None


def export_balance_constraint_rule(backend_model, loc_tech_carrier, timestep):
    """
    Ensure no technology can 'pass' its export capability to another technology
    with the same carrier_out, by limiting its export to the capacity of its production


    .. container:: scrolling-wrapper

        .. math::

            \\boldsymbol{carrier_{prod}}(loc::tech::carrier, timestep)
            \\geq \\boldsymbol{carrier_{export}}(loc::tech::carrier, timestep)
            \\quad \\forall loc::tech::carrier \\in locs::tech::carriers_{export},
            \\forall timestep \\in timesteps
    """

    return (
        backend_model.carrier_prod[loc_tech_carrier, timestep]
        >= backend_model.carrier_export[loc_tech_carrier, timestep]
    )


def update_costs_var_constraint(backend_model, cost, loc_tech, timestep):
    """
    Update time varying cost constraint (from costs.py) to include export

    .. container:: scrolling-wrapper

        .. math::

            \\boldsymbol{cost_{var}}(cost, loc::tech, timestep) +=
            cost_{export}(cost, loc::tech, timestep) \\times
            \\boldsymbol{carrier_{export}}(loc::tech::carrier, timestep)
            * timestep_{weight} \\quad \\forall cost \\in costs,
            \\forall loc::tech \\in loc::techs_{cost_{var}, export},
            \\forall timestep \\in timesteps

    """
    model_data_dict = backend_model.__calliope_model_data["data"]

    loc_tech_carrier = model_data_dict["lookup_loc_techs_export"][(loc_tech)]
    weight = backend_model.timestep_weights[timestep]

    cost_export = (
        get_param(backend_model, "cost_export", (cost, loc_tech, timestep))
        * backend_model.carrier_export[loc_tech_carrier, timestep]
        * weight
    )

    backend_model.cost_var_rhs[cost, loc_tech, timestep].expr += cost_export


def export_max_constraint_rule(backend_model, loc_tech_carrier, timestep):
    """
    Set maximum export. All exporting technologies.

    .. container:: scrolling-wrapper

        .. math::

            \\boldsymbol{carrier_{export}}(loc::tech::carrier, timestep)
            \\leq export_{cap}(loc::tech)
            \\quad \\forall loc::tech::carrier \\in locs::tech::carriers_{export},
            \\forall timestep \\in timesteps

    If the technology is defined by integer units, not a continuous capacity,
    this constraint becomes:

    .. container:: scrolling-wrapper

        .. math::

            \\boldsymbol{carrier_{export}}(loc::tech::carrier, timestep)
            \\leq export_{cap}(loc::tech) \\times
            \\boldsymbol{operating_{units}}(loc::tech, timestep)

    """

    loc_tech = get_loc_tech(loc_tech_carrier)

    if loc_tech_is_in(backend_model, loc_tech, "loc_techs_milp"):
        operating_units = backend_model.operating_units[loc_tech, timestep]
    else:
        operating_units = 1

    export_cap = get_param(backend_model, "export_cap", loc_tech)
    return (
        backend_model.carrier_export[loc_tech_carrier, timestep]
        <= export_cap * operating_units
    )
