"""
Copyright (C) since 2013 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

energy_balance.py
~~~~~~~~~~~~~~~~~

Energy balance constraints.

"""

import pyomo.core as po

from calliope.backend.pyomo.util import (
    get_param,
    get_previous_timestep,
)


def system_balance_constraint_rule(backend_model, carrier, node, timestep):
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

    def _sum(var_name):
        return po.quicksum(
            getattr(backend_model, var_name)[carrier, node, tech, timestep]
            for tech in backend_model.techs
            if [carrier, node, tech, timestep]
            in getattr(backend_model, var_name)._index
        )

    carrier_prod = _sum("carrier_prod")
    carrier_con = _sum("carrier_con")

    if hasattr(backend_model, "carrier_export"):
        carrier_export = _sum("carrier_export")
    else:
        carrier_export = 0

    if hasattr(backend_model, "unmet_demand"):
        unmet_demand = backend_model.unmet_demand[carrier, node, timestep]
    else:
        unmet_demand = 0
    if hasattr(backend_model, "unused_supply"):
        unused_supply = backend_model.unused_supply[carrier, node, timestep]
    else:
        unused_supply = 0

    return (
        carrier_prod + carrier_con - carrier_export + unmet_demand + unused_supply == 0
    )


def balance_supply_constraint_rule(backend_model, carrier, node, tech, timestep):
    """
    Limit production from supply techs to their available resource

    .. container:: scrolling-wrapper

        .. math::

            min\\_use(loc::tech) \\times available\\_resource(loc::tech, timestep) \\leq
            \\frac{\\boldsymbol{carrier_{prod}}(loc::tech::carrier, timestep)}{\\eta_{energy}(loc::tech, timestep)}
            \\geq available\\_resource(loc::tech, timestep)
            \\quad \\forall loc::tech \\in loc::techs_{supply}, \\forall timestep \\in timesteps

    If :math:`force\\_resource(loc::tech)` is set:

    .. container:: scrolling-wrapper

        .. math::

            \\frac{\\boldsymbol{carrier_{prod}}(loc::tech::carrier, timestep)}{\\eta_{energy}(loc::tech, timestep)}
            = available\\_resource(loc::tech, timestep)
            \\quad \\forall loc::tech \\in loc::techs_{supply}, \\forall timestep \\in timesteps

    Where:

    .. container:: scrolling-wrapper

        .. math::

            available\\_resource(loc::tech, timestep) = resource(loc::tech, timestep)
            \\times resource\\_scale(loc::tech)

    if :math:`loc::tech` is in :math:`loc::techs_{area}`:

    .. container:: scrolling-wrapper

        .. math::

            available\\_resource(loc::tech, timestep) = resource(loc::tech, timestep)
            \\times resource\\_scale(loc::tech) \\times \\boldsymbol{resource_{area}}(loc::tech)

    """

    resource = get_param(backend_model, "resource", (node, tech, timestep))
    energy_eff = get_param(backend_model, "energy_eff", (node, tech, timestep))
    resource_scale = get_param(backend_model, "resource_scale", (node, tech))
    min_use = get_param(backend_model, "resource_min_use", (node, tech, timestep))

    if po.value(energy_eff) == 0:
        return backend_model.carrier_prod[carrier, node, tech, timestep] == 0
    else:
        carrier_prod = (
            backend_model.carrier_prod[carrier, node, tech, timestep] / energy_eff
        )

    if backend_model.resource_unit[node, tech].value == "energy_per_area":
        available_resource = (
            resource * resource_scale * backend_model.resource_area[node, tech]
        )
    elif backend_model.resource_unit[node, tech].value == "energy_per_cap":
        available_resource = (
            resource * resource_scale * backend_model.energy_cap[node, tech]
        )
    else:
        available_resource = resource * resource_scale

    # 1 represents boolean True here
    if backend_model.force_resource[node, tech].value == 1:
        return carrier_prod == available_resource
    elif min_use:
        return min_use * available_resource <= carrier_prod <= available_resource
    else:
        return carrier_prod <= available_resource


def balance_demand_constraint_rule(backend_model, carrier, node, tech, timestep):
    """
    Limit consumption from demand techs to their required resource.

    .. container:: scrolling-wrapper

        .. math::

            \\boldsymbol{carrier_{con}}(loc::tech::carrier, timestep) \\times \\eta_{energy}(loc::tech, timestep) \\geq
            required\\_resource(loc::tech, timestep) \\quad \\forall loc::tech \\in loc::techs_{demand},
            \\forall timestep \\in timesteps

    If :math:`force\\_resource(loc::tech)` is set:

    .. container:: scrolling-wrapper

        .. math::

            \\boldsymbol{carrier_{con}}(loc::tech::carrier, timestep) \\times \\eta_{energy}(loc::tech, timestep) =
            required\\_resource(loc::tech, timestep)
            \\quad \\forall loc::tech \\in loc::techs_{demand}, \\forall timestep \\in timesteps

    Where:

    .. container:: scrolling-wrapper

        .. math::

            required\\_resource(loc::tech, timestep) = resource(loc::tech, timestep)
            \\times resource\\_scale(loc::tech)

    if :math:`loc::tech` is in :math:`loc::techs_{area}`:

    .. container:: scrolling-wrapper

        .. math::

            required\\_resource(loc::tech, timestep) = resource(loc::tech, timestep)
            \\times resource\\_scale(loc::tech) \\times \\boldsymbol{resource_{area}}(loc::tech)

    """

    resource = get_param(backend_model, "resource", (node, tech, timestep))
    energy_eff = get_param(backend_model, "energy_eff", (node, tech, timestep))
    resource_scale = get_param(backend_model, "resource_scale", (node, tech))

    carrier_con = backend_model.carrier_con[carrier, node, tech, timestep] * energy_eff

    if backend_model.resource_unit[node, tech].value == "energy_per_area":
        required_resource = (
            resource * resource_scale * backend_model.resource_area[node, tech]
        )
    elif backend_model.resource_unit[node, tech].value == "energy_per_cap":
        required_resource = (
            resource * resource_scale * backend_model.energy_cap[node, tech]
        )
    else:
        required_resource = resource * resource_scale

    # We save the expression to the backend_model so it can be used elsewhere
    backend_model.required_resource[node, tech, timestep].expr = required_resource

    # 1 represents boolean True here
    if backend_model.force_resource[node, tech].value == 1:
        return carrier_con == backend_model.required_resource[node, tech, timestep]
    else:
        return carrier_con >= backend_model.required_resource[node, tech, timestep]


def resource_availability_supply_plus_constraint_rule(
    backend_model, node, tech, timestep
):
    """
    Limit production from supply_plus techs to their available resource.

    .. container:: scrolling-wrapper

        .. math::

            \\boldsymbol{resource_{con}}(loc::tech, timestep)
            \\leq available\\_resource(loc::tech, timestep)
            \\quad \\forall loc::tech \\in loc::techs_{supply^{+}}, \\forall timestep \\in timesteps

    If :math:`force\\_resource(loc::tech)` is set:

    .. container:: scrolling-wrapper

        .. math::

            \\boldsymbol{resource_{con}}(loc::tech, timestep)
            = available\\_resource(loc::tech, timestep)
            \\quad \\forall loc::tech \\in loc::techs_{supply^{+}}, \\forall timestep \\in timesteps

    Where:

    .. container:: scrolling-wrapper

        .. math::

            available\\_resource(loc::tech, timestep) = resource(loc::tech, timestep)
            \\times resource_{scale}(loc::tech)

    if :math:`loc::tech` is in :math:`loc::techs_{area}`:

    .. container:: scrolling-wrapper

        .. math::

            available\\_resource(loc::tech, timestep) = resource(loc::tech, timestep)
            \\times resource_{scale}(loc::tech)
            \\times resource_{area}(loc::tech)

    """
    resource = get_param(backend_model, "resource", (node, tech, timestep))
    resource_scale = get_param(backend_model, "resource_scale", (node, tech))

    if backend_model.resource_unit[node, tech].value == "energy_per_area":
        available_resource = (
            resource * resource_scale * backend_model.resource_area[node, tech]
        )
    elif backend_model.resource_unit[node, tech].value == "energy_per_cap":
        available_resource = (
            resource * resource_scale * backend_model.energy_cap[node, tech]
        )
    else:
        available_resource = resource * resource_scale

    # 1 represents boolean True here
    if backend_model.force_resource[node, tech].value == 1:
        return backend_model.resource_con[node, tech, timestep] == available_resource
    else:
        return backend_model.resource_con[node, tech, timestep] <= available_resource


def balance_transmission_constraint_rule(backend_model, carrier, node, tech, timestep):
    """
    Balance carrier production and consumption of transmission technologies

    .. container:: scrolling-wrapper

        .. math::

            -1 * \\boldsymbol{carrier_{con}}(loc_{from}::tech:loc_{to}::carrier, timestep)
            \\times \\eta_{energy}(loc::tech, timestep)
            = \\boldsymbol{carrier_{prod}}(loc_{to}::tech:loc_{from}::carrier, timestep)
            \\quad \\forall loc::tech:loc \\in locs::techs:locs_{transmission},
            \\forall timestep \\in timesteps

    Where a link is the connection between :math:`loc_{from}::tech:loc_{to}`
    and :math:`loc_{to}::tech:loc_{from}` for locations `to` and `from`.

    """
    energy_eff = get_param(backend_model, "energy_eff", (node, tech, timestep))
    remote_tech = backend_model.link_remote_techs[node, tech].value
    remote_node = backend_model.link_remote_nodes[node, tech].value

    return (
        backend_model.carrier_prod[carrier, node, tech, timestep]
        == -1
        * backend_model.carrier_con[carrier, remote_node, remote_tech, timestep]
        * energy_eff
    )


def balance_supply_plus_constraint_rule(backend_model, carrier, node, tech, timestep):
    """
    Balance carrier production and resource consumption of supply_plus technologies
    alongside any use of resource storage.

    .. _system_balance_constraint:

    .. container:: scrolling-wrapper

        .. math::

            \\boldsymbol{storage}(loc::tech, timestep) =
            \\boldsymbol{storage}(loc::tech, timestep_{previous})
            \\times (1 - storage\\_loss(loc::tech, timestep))^{timestep\\_resolution(timestep)} +
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

    run_config = backend_model.__calliope_run_config

    resource_eff = get_param(backend_model, "resource_eff", (node, tech, timestep))
    energy_eff = get_param(backend_model, "energy_eff", (node, tech, timestep))
    parasitic_eff = get_param(backend_model, "parasitic_eff", (node, tech, timestep))
    total_eff = energy_eff * parasitic_eff

    if po.value(total_eff) == 0:
        carrier_prod = 0
    else:
        carrier_prod = (
            backend_model.carrier_prod[carrier, node, tech, timestep] / total_eff
        )

    # A) Case where no storage allowed
    if not backend_model.include_storage[node, tech]:
        return (
            backend_model.resource_con[node, tech, timestep] * resource_eff
            == carrier_prod
        )

    # B) Case where storage is allowed
    else:
        resource = backend_model.resource_con[node, tech, timestep] * resource_eff
        # Pyomo returns the order 1-indexed, but we want 0-indexing
        current_timestep = backend_model.timesteps.ord(timestep) - 1
        if current_timestep == 0 and not run_config["cyclic_storage"]:
            storage_previous_step = (
                get_param(backend_model, "storage_initial", (node, tech))
                * backend_model.storage_cap[node, tech]
            )
        elif (
            hasattr(backend_model, "storage_inter_cluster")
            and backend_model.lookup_cluster_first_timestep[timestep]
        ):
            storage_previous_step = 0
        else:
            if (
                hasattr(backend_model, "clusters")
                and backend_model.lookup_cluster_first_timestep[timestep]
            ):
                previous_step = backend_model.lookup_cluster_last_timestep[
                    timestep
                ].value
            elif current_timestep == 0 and run_config["cyclic_storage"]:
                previous_step = backend_model.timesteps[-1]
            else:
                previous_step = get_previous_timestep(backend_model.timesteps, timestep)
            storage_loss = get_param(backend_model, "storage_loss", (node, tech))
            time_resolution = backend_model.timestep_resolution[previous_step]
            storage_previous_step = (
                (1 - storage_loss) ** time_resolution
            ) * backend_model.storage[node, tech, previous_step]

        return (
            backend_model.storage[node, tech, timestep]
            == storage_previous_step + resource - carrier_prod
        )


def balance_storage_constraint_rule(backend_model, carrier, node, tech, timestep):
    """
    Balance carrier production and consumption of storage technologies,
    alongside any use of the stored volume.

    .. container:: scrolling-wrapper

        .. math::

            \\boldsymbol{storage}(loc::tech, timestep) =
            \\boldsymbol{storage}(loc::tech, timestep_{previous})
            \\times (1 - storage\\_loss(loc::tech, timestep))^{resolution(timestep)}
            - \\boldsymbol{carrier_{con}}(loc::tech::carrier, timestep)
            \\times \\eta_{energy}(loc::tech, timestep)
            - \\frac{\\boldsymbol{carrier_{prod}}(loc::tech::carrier, timestep)}{\\eta_{energy}(loc::tech, timestep)}
            \\quad \\forall loc::tech \\in loc::techs_{storage}, \\forall timestep \\in timesteps
    """
    run_config = backend_model.__calliope_run_config

    energy_eff = get_param(backend_model, "energy_eff", (node, tech, timestep))

    if po.value(energy_eff) == 0:
        carrier_prod = 0
    else:
        carrier_prod = (
            backend_model.carrier_prod[carrier, node, tech, timestep] / energy_eff
        )

    carrier_con = backend_model.carrier_con[carrier, node, tech, timestep] * energy_eff

    # Pyomo returns the order 1-indexed, but we want 0-indexing
    current_timestep = backend_model.timesteps.ord(timestep) - 1
    if current_timestep == 0 and not run_config["cyclic_storage"]:
        storage_previous_step = (
            get_param(backend_model, "storage_initial", (node, tech))
            * backend_model.storage_cap[node, tech]
        )
    elif (
        hasattr(backend_model, "storage_inter_cluster")
        and backend_model.lookup_cluster_first_timestep[timestep]
    ):
        storage_previous_step = 0
    else:
        if (
            hasattr(backend_model, "clusters")
            and backend_model.lookup_cluster_first_timestep[timestep]
        ):
            previous_step = backend_model.lookup_cluster_last_timestep[timestep].value
        elif current_timestep == 0 and run_config["cyclic_storage"]:
            previous_step = backend_model.timesteps[-1]
        else:
            previous_step = get_previous_timestep(backend_model.timesteps, timestep)
        storage_loss = get_param(backend_model, "storage_loss", (node, tech))
        time_resolution = backend_model.timestep_resolution[previous_step]
        storage_previous_step = (
            (1 - storage_loss) ** time_resolution
        ) * backend_model.storage[node, tech, previous_step]

    return (
        backend_model.storage[node, tech, timestep]
        == storage_previous_step - carrier_prod - carrier_con
    )


def balance_storage_inter_constraint_rule(backend_model, node, tech, datestep):
    """
    When clustering days, to reduce the timeseries length, balance the daily stored
    energy across all days of the original timeseries.

    `Ref: DOI 10.1016/j.apenergy.2018.01.023 <https://doi.org/10.1016/j.apenergy.2018.01.023>`_

    .. container:: scrolling-wrapper

        .. math::

            \\boldsymbol{storage_{inter\\_cluster}}(loc::tech, datestep) =
            \\boldsymbol{storage_{inter\\_cluster}}(loc::tech, datestep_{previous})
            \\times (1 - storage\\_loss(loc::tech, timestep))^{24}
            + \\boldsymbol{storage}(loc::tech, timestep_{final, cluster(datestep))})
            \\quad \\forall loc::tech \\in loc::techs_{store}, \\forall datestep \\in datesteps

    Where :math:`timestep_{final, cluster(datestep_{previous}))}` is the final timestep of the
    cluster in the clustered timeseries corresponding to the previous day
    """
    run_config = backend_model.__calliope_run_config
    # Pyomo returns the order 1-indexed, but we want 0-indexing
    current_datestep = backend_model.datesteps.ord(datestep) - 1

    if current_datestep == 0 and not run_config["cyclic_storage"]:
        storage_previous_step = get_param(
            backend_model, "storage_initial", (node, tech)
        )
        storage_intra = 0
    else:
        if current_datestep == 0 and run_config["cyclic_storage"]:
            previous_step = backend_model.datesteps[-1]
        else:
            previous_step = get_previous_timestep(backend_model.datesteps, datestep)
        storage_loss = get_param(backend_model, "storage_loss", (node, tech))
        storage_previous_step = (
            (1 - storage_loss) ** 24
        ) * backend_model.storage_inter_cluster[node, tech, previous_step]
        final_timestep = backend_model.lookup_datestep_last_cluster_timestep[
            previous_step
        ].value
        storage_intra = backend_model.storage[node, tech, final_timestep]
    return (
        backend_model.storage_inter_cluster[node, tech, datestep]
        == storage_previous_step + storage_intra
    )


def storage_initial_constraint_rule(backend_model, node, tech):
    """
    If storage is cyclic, allow an initial storage to still be set. This is
    applied to the storage of the final timestep/datestep of the series as that,
    in cyclic storage, is the 'storage_previous_step' for the first
    timestep/datestep.

    If clustering and ``storage_inter_cluster`` exists:

    .. container:: scrolling-wrapper

        .. math::

            \\boldsymbol{storage_{inter\\_cluster}}(loc::tech, datestep_{final})
            \\times ((1 - storage_loss) ** 24) = storage_{initial}(loc::tech) \\times storage_{cap}(loc::tech)
            \\quad \\forall loc::tech \\in loc::techs_{store}, \\forall datestep \\in datesteps

    Where :math:`datestep_{final}` is the last datestep of the timeseries

    Else:
    .. container:: scrolling-wrapper

        .. math::

            \\boldsymbol{storage}(loc::tech, timestep_{final})
            \\times ((1 - storage_loss) ** 24) = storage_{initial}(loc::tech) \\times storage_{cap}(loc::tech)
            \\quad \\forall loc::tech \\in loc::techs_{store}, \\forall timestep \\in timesteps

    Where :math:`timestep_{final}` is the last timestep of the timeseries
    """

    storage_initial = get_param(backend_model, "storage_initial", (node, tech))

    storage_loss = get_param(backend_model, "storage_loss", (node, tech))
    if hasattr(backend_model, "storage_inter_cluster"):
        storage = backend_model.storage_inter_cluster
        final_step = backend_model.datesteps[-1]
        time_resolution = 24
    else:
        storage = backend_model.storage
        final_step = backend_model.timesteps[-1]
        time_resolution = backend_model.timestep_resolution[final_step]

    return (
        storage[node, tech, final_step] * ((1 - storage_loss) ** time_resolution)
        == storage_initial * backend_model.storage_cap[node, tech]
    )
