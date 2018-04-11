"""
Copyright (C) 2013-2018 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

energy_balance.py
~~~~~~~~~~~~~~~~~

Energy balance constraints.

"""

import pyomo.core as po  # pylint: disable=import-error

from calliope.backend.pyomo.util import \
    get_param, \
    get_previous_timestep, \
    get_loc_tech_carriers, \
    loc_tech_is_in


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


def system_balance_constraint_rule(backend_model, loc_carrier, timestep):
    """
    System balance ensures that, within each location, the production and
    consumption of each carrier is balanced.

    .. container:: scrolling-wrapper

        .. math::

            \\sum_{loc::tech::carrier_{prod} \\in loc::carrier} \\boldsymbol{carrier_{prod}}(loc::tech::carrier, timestep)
            + \\sum_{loc::tech::carrier_{con} \\in loc::carrier} \\boldsymbol{carrier_{con}}(loc::tech::carrier, timestep)
            + \\sum_{loc::tech::carrier_{export} \\in loc::carrier} \\boldsymbol{carrier_{export}}(loc::tech::carrier, timestep)
            \\quad \\forall loc::carrier \\in loc::carriers, \\forall timestep \\in timesteps

    """
    prod, con, export = get_loc_tech_carriers(backend_model, loc_carrier)
    if hasattr(backend_model, 'unmet_demand'):
        unmet_demand = backend_model.unmet_demand[loc_carrier, timestep]
    else:
        unmet_demand = 0

    backend_model.system_balance[loc_carrier, timestep].expr = (
        sum(backend_model.carrier_prod[loc_tech_carrier, timestep] for loc_tech_carrier in prod) +
        sum(backend_model.carrier_con[loc_tech_carrier, timestep] for loc_tech_carrier in con) +
        unmet_demand
    )

    return backend_model.system_balance[loc_carrier, timestep] == 0


def balance_supply_constraint_rule(backend_model, loc_tech, timestep):
    """
    Limit production from supply techs to their available resource

    .. container:: scrolling-wrapper

        .. math::

            min\_use(loc::tech) \\times available\_resource(loc::tech, timestep) \\leq
            \\frac{\\boldsymbol{carrier_{prod}}(loc::tech::carrier, timestep)}{\\eta_{energy}(loc::tech, timestep)}
            \\geq available\_resource(loc::tech, timestep)
            \\quad \\forall loc::tech \\in loc::techs_{supply}, \\forall timestep \\in timesteps

    If :math:`force\_resource(loc::tech)` is set:

    .. container:: scrolling-wrapper

        .. math::

            \\frac{\\boldsymbol{carrier_{prod}}(loc::tech::carrier, timestep)}{\\eta_{energy}(loc::tech, timestep)}
            = available\_resource(loc::tech, timestep)
            \\quad \\forall loc::tech \\in loc::techs_{supply}, \\forall timestep \\in timesteps

    Where:

    .. container:: scrolling-wrapper

        .. math::

            available\_resource(loc::tech, timestep) = resource(loc::tech, timestep)
            \\times resource\_scale(loc::tech)

    if :math:`loc::tech` is in :math:`loc::techs_{area}`:

    .. container:: scrolling-wrapper

        .. math::

            available\_resource(loc::tech, timestep) = resource(loc::tech, timestep)
            \\times resource\_scale(loc::tech) \\times \\boldsymbol{resource_{area}}(loc::tech)

    """
    model_data_dict = backend_model.__calliope_model_data__['data']

    resource = get_param(backend_model, 'resource', (loc_tech, timestep))
    energy_eff = get_param(backend_model, 'energy_eff', (loc_tech, timestep))
    resource_scale = get_param(backend_model, 'resource_scale', loc_tech)
    force_resource = get_param(backend_model, 'force_resource', loc_tech)
    loc_tech_carrier = model_data_dict['lookup_loc_techs'][loc_tech]
    min_use = get_param(backend_model, 'resource_min_use', (loc_tech, timestep))

    if po.value(energy_eff) == 0:
        return backend_model.carrier_prod[loc_tech_carrier, timestep] == 0
    else:
        carrier_prod = backend_model.carrier_prod[loc_tech_carrier, timestep] / energy_eff

    if loc_tech_is_in(backend_model, loc_tech, 'loc_techs_area'):
        available_resource = resource * resource_scale * backend_model.resource_area[loc_tech]
    else:
        available_resource = resource * resource_scale

    if po.value(force_resource):
        return carrier_prod == available_resource
    elif min_use:
        return min_use * available_resource <= carrier_prod <= available_resource
    else:
        return carrier_prod <= available_resource


def balance_demand_constraint_rule(backend_model, loc_tech, timestep):
    """
    Limit consumption from demand techs to their required resource.

    .. container:: scrolling-wrapper

        .. math::

            \\boldsymbol{carrier_{con}}(loc::tech::carrier, timestep) \\times \\eta_{energy}(loc::tech, timestep) \\geq
            required\_resource(loc::tech, timestep) \\quad \\forall loc::tech \\in loc::techs_{demand},
            \\forall timestep \\in timesteps

    If :math:`force\_resource(loc::tech)` is set:

    .. container:: scrolling-wrapper

        .. math::

            \\boldsymbol{carrier_{con}}(loc::tech::carrier, timestep) \\times \\eta_{energy}(loc::tech, timestep) =
            required\_resource(loc::tech, timestep)
            \\quad \\forall loc::tech \\in loc::techs_{demand}, \\forall timestep \\in timesteps

    Where:

    .. container:: scrolling-wrapper

        .. math::

            required\_resource(loc::tech, timestep) = resource(loc::tech, timestep)
            \\times resource\_scale(loc::tech)

    if :math:`loc::tech` is in :math:`loc::techs_{area}`:

    .. container:: scrolling-wrapper

        .. math::

            required\_resource(loc::tech, timestep) = resource(loc::tech, timestep)
            \\times resource\_scale(loc::tech) \\times \\boldsymbol{resource_{area}}(loc::tech)

    """
    model_data_dict = backend_model.__calliope_model_data__['data']

    resource = get_param(backend_model, 'resource', (loc_tech, timestep))
    energy_eff = get_param(backend_model, 'energy_eff', (loc_tech, timestep))
    resource_scale = get_param(backend_model, 'resource_scale', loc_tech)
    force_resource = get_param(backend_model, 'force_resource', loc_tech)

    loc_tech_carrier = model_data_dict['lookup_loc_techs'][loc_tech]
    carrier_con = backend_model.carrier_con[loc_tech_carrier, timestep] * energy_eff

    if loc_tech_is_in(backend_model, loc_tech, 'loc_techs_area'):
        required_resource = resource * resource_scale * backend_model.resource_area[loc_tech]
    else:
        required_resource = resource * resource_scale

    if po.value(force_resource):
        return carrier_con == required_resource
    else:
        return carrier_con >= required_resource


def resource_availability_supply_plus_constraint_rule(backend_model, loc_tech, timestep):
    """
    Limit production from supply_plus techs to their available resource.

    .. container:: scrolling-wrapper

        .. math::

            \\boldsymbol{resource_{con}}(loc::tech, timestep)
            \\leq available\_resource(loc::tech, timestep)
            \\quad \\forall loc::tech \\in loc::techs_{supply^{+}}, \\forall timestep \\in timesteps

    If :math:`force\_resource(loc::tech)` is set:

    .. container:: scrolling-wrapper

        .. math::

            \\boldsymbol{resource_{con}}(loc::tech, timestep)
            = available\_resource(loc::tech, timestep)
            \\quad \\forall loc::tech \\in loc::techs_{supply^{+}}, \\forall timestep \\in timesteps

    Where:

    .. container:: scrolling-wrapper

        .. math::

            available\_resource(loc::tech, timestep) = resource(loc::tech, timestep)
            \\times resource_{scale}(loc::tech)

    if :math:`loc::tech` is in :math:`loc::techs_{area}`:

    .. container:: scrolling-wrapper

        .. math::

            available\_resource(loc::tech, timestep) = resource(loc::tech, timestep)
            \\times resource_{scale}(loc::tech)
            \\times resource_{area}(loc::tech)

    """
    resource = get_param(backend_model, 'resource', (loc_tech, timestep))
    resource_scale = get_param(backend_model, 'resource_scale', loc_tech)
    force_resource = get_param(backend_model, 'force_resource', loc_tech)

    if loc_tech_is_in(backend_model, loc_tech, 'loc_techs_area'):
        available_resource = resource * resource_scale * backend_model.resource_area[loc_tech]
    else:
        available_resource = resource * resource_scale

    if po.value(force_resource):
        return backend_model.resource_con[loc_tech, timestep] == available_resource
    else:
        return backend_model.resource_con[loc_tech, timestep] <= available_resource


def balance_transmission_constraint_rule(backend_model, loc_tech, timestep):
    """
    Balance carrier production and consumption of transmission technologies

    .. container:: scrolling-wrapper

        .. math::

            -1 * \\boldsymbol{carrier_{con}}(loc_{from}::tech:loc_{to}::carrier, timestep)
            \\times \\eta_{energy}(loc::tech, timestep)
            = \\boldsymbol{carrier_{prod}}(loc_{to}::tech:loc_{from}::carrier, timestep)
            \\quad \\forall loc::tech:loc \in locs::techs:locs_{transmission},
            \\forall timestep \in timesteps

    Where a link is the connection between :math:`loc_{from}::tech:loc_{to}`
    and :math:`loc_{to}::tech:loc_{from}` for locations `to` and `from`.

    """
    model_data_dict = backend_model.__calliope_model_data__['data']

    energy_eff = get_param(backend_model, 'energy_eff', (loc_tech, timestep))
    loc_tech_carrier = model_data_dict['lookup_loc_techs'][loc_tech]
    remote_loc_tech = model_data_dict['lookup_remotes'][loc_tech]
    remote_loc_tech_carrier = model_data_dict['lookup_loc_techs'][remote_loc_tech]

    if (loc_tech_is_in(backend_model, remote_loc_tech, 'loc_techs_transmission')
        and get_param(backend_model, 'energy_prod', (loc_tech)) == 1):
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
    alongside any use of resource storage.

    .. _system_balance_constraint:

    .. container:: scrolling-wrapper

        .. math::

            \\boldsymbol{storage}(loc::tech, timestep) =
            \\boldsymbol{storage}(loc::tech, timestep_{previous})
            \\times (1 - storage\_loss(loc::tech, timestep))^{timestep\_resolution(timestep)} +
            \\boldsymbol{resource_{con}}(loc::tech, timestep) \\times \\eta_{resource}(loc::tech, timestep) -
            \\frac{\\boldsymbol{carrier_{prod}}(loc::tech::carrier, timestep)}{\\eta_{energy}(loc::tech, timestep) \\times \\eta_{parasitic}(loc::tech, timestep)}
            \\quad \\forall loc::tech \\in loc::techs_{supply^{+}}, \\forall timestep \\in timesteps

    If *no* storage is defined for the technology, this reduces to:

    .. container:: scrolling-wrapper

        .. math::

            \\boldsymbol{resource_{con}}(loc::tech, timestep) \\times \\eta_{resource}(loc::tech, timestep) =
            \\frac{\\boldsymbol{carrier_{prod}}(loc::tech::carrier, timestep)}{\\eta_{energy}(loc::tech, timestep) \\times \\eta_{parasitic}(loc::tech, timestep)}
            \\quad \\forall loc::tech \\in loc::techs_{supply^{+}}, \\forall timestep \\in timesteps

    """

    model_data_dict = backend_model.__calliope_model_data__['data']

    resource_eff = get_param(backend_model, 'resource_eff', (loc_tech, timestep))
    energy_eff = get_param(backend_model, 'energy_eff', (loc_tech, timestep))
    parasitic_eff = get_param(backend_model, 'parasitic_eff', (loc_tech, timestep))
    total_eff = energy_eff * parasitic_eff

    if po.value(total_eff) == 0:
        carrier_prod = 0
    else:
        loc_tech_carrier = model_data_dict['lookup_loc_techs'][loc_tech]
        carrier_prod = backend_model.carrier_prod[loc_tech_carrier, timestep] / total_eff

    # A) Case where no storage allowed
    if not loc_tech_is_in(backend_model, loc_tech, 'loc_techs_store'):
        return backend_model.resource_con[loc_tech, timestep] * resource_eff == carrier_prod

    # B) Case where storage is allowed
    else:
        resource = backend_model.resource_con[loc_tech, timestep] * resource_eff
        if backend_model.timesteps.order_dict[timestep] == 0:
            storage_previous_step = get_param(backend_model, 'storage_initial', loc_tech)
        else:
            storage_loss = get_param(backend_model, 'storage_loss', loc_tech)
            previous_step = get_previous_timestep(backend_model, timestep)
            time_resolution = backend_model.timestep_resolution[previous_step]
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
    alongside any use of the stored volume.

    .. container:: scrolling-wrapper

        .. math::

            \\boldsymbol{storage}(loc::tech, timestep) =
            \\boldsymbol{storage}(loc::tech, timestep_{previous})
            \\times (1 - storage\_loss(loc::tech, timestep))^{resolution(timestep)}
            - \\boldsymbol{carrier_{con}}(loc::tech::carrier, timestep)
            \\times \\eta_{energy}(loc::tech, timestep)
            - \\frac{\\boldsymbol{carrier_{prod}}(loc::tech::carrier, timestep)}{\\eta_{energy}(loc::tech, timestep)}
            \\quad \\forall loc::tech \\in loc::techs_{storage}, \\forall timestep \\in timesteps
    """
    model_data_dict = backend_model.__calliope_model_data__['data']

    energy_eff = get_param(backend_model, 'energy_eff', (loc_tech, timestep))

    if po.value(energy_eff) == 0:
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
        time_resolution = backend_model.timestep_resolution[previous_step]
        storage_previous_step = (
            ((1 - storage_loss) ** time_resolution) *
            backend_model.storage[loc_tech, previous_step]
        )

    return (
        backend_model.storage[loc_tech, timestep] ==
        storage_previous_step - carrier_prod - carrier_con
    )
