"""
Copyright (C) since 2013 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

dispatch.py
~~~~~~~~~~~~~~~~~

Energy dispatch constraints, limiting production/consumption to the capacities
of the technologies

"""

import pyomo.core as po  # pylint: disable=import-error

from calliope.backend.pyomo.util import get_param, get_loc_tech, get_previous_timestep

ORDER = 10  # order in which to invoke constraints relative to other constraint files


def load_constraints(backend_model):
    sets = backend_model.__calliope_model_data["sets"]

    if "loc_tech_carriers_carrier_production_max_constraint" in sets:
        backend_model.carrier_production_max_constraint = po.Constraint(
            backend_model.loc_tech_carriers_carrier_production_max_constraint,
            backend_model.timesteps,
            rule=carrier_production_max_constraint_rule,
        )
    if "loc_tech_carriers_carrier_production_min_constraint" in sets:
        backend_model.carrier_production_min_constraint = po.Constraint(
            backend_model.loc_tech_carriers_carrier_production_min_constraint,
            backend_model.timesteps,
            rule=carrier_production_min_constraint_rule,
        )
    if "loc_tech_carriers_carrier_consumption_max_constraint" in sets:
        backend_model.carrier_consumption_max_constraint = po.Constraint(
            backend_model.loc_tech_carriers_carrier_consumption_max_constraint,
            backend_model.timesteps,
            rule=carrier_consumption_max_constraint_rule,
        )

    if "loc_techs_resource_max_constraint" in sets:
        backend_model.resource_max_constraint = po.Constraint(
            backend_model.loc_techs_resource_max_constraint,
            backend_model.timesteps,
            rule=resource_max_constraint_rule,
        )

    if "loc_techs_storage_max_constraint" in sets:
        backend_model.storage_max_constraint = po.Constraint(
            backend_model.loc_techs_storage_max_constraint,
            backend_model.timesteps,
            rule=storage_max_constraint_rule,
        )

    if "loc_techs_storage_discharge_depth" in sets:
        backend_model.storage_discharge_depth_constraint = po.Constraint(
            backend_model.loc_techs_storage_discharge_depth,
            backend_model.timesteps,
            rule=storage_discharge_depth_constraint_rule,
        )

    if "loc_tech_carriers_ramping_constraint" in sets:
        backend_model.ramping_up_constraint = po.Constraint(
            backend_model.loc_tech_carriers_ramping_constraint,
            backend_model.timesteps,
            rule=ramping_up_constraint_rule,
        )

        backend_model.ramping_down_constraint = po.Constraint(
            backend_model.loc_tech_carriers_ramping_constraint,
            backend_model.timesteps,
            rule=ramping_down_constraint_rule,
        )

    if "loc_techs_storage_intra_max_constraint" in sets:
        backend_model.storage_intra_max_constraint = po.Constraint(
            backend_model.loc_techs_storage_intra_max_constraint,
            backend_model.timesteps,
            rule=storage_intra_max_rule,
        )
    if "loc_techs_storage_intra_min_constraint" in sets:
        backend_model.storage_intra_min_constraint = po.Constraint(
            backend_model.loc_techs_storage_intra_min_constraint,
            backend_model.timesteps,
            rule=storage_intra_min_rule,
        )
    if "loc_techs_storage_inter_max_constraint" in sets:
        backend_model.storage_inter_max_constraint = po.Constraint(
            backend_model.loc_techs_storage_inter_max_constraint,
            backend_model.datesteps,
            rule=storage_inter_max_rule,
        )
    if "loc_techs_storage_inter_min_constraint" in sets:
        backend_model.storage_inter_min_constraint = po.Constraint(
            backend_model.loc_techs_storage_inter_min_constraint,
            backend_model.datesteps,
            rule=storage_inter_min_rule,
        )


def carrier_production_max_constraint_rule(backend_model, loc_tech_carrier, timestep):
    """
    Set maximum carrier production. All technologies.

    .. container:: scrolling-wrapper

        .. math::

            \\boldsymbol{carrier_{prod}}(loc::tech::carrier, timestep) \\leq energy_{cap}(loc::tech)
            \\times timestep\\_resolution(timestep) \\times parasitic\\_eff(loc::tec)

    """
    loc_tech = get_loc_tech(loc_tech_carrier)
    carrier_prod = backend_model.carrier_prod[loc_tech_carrier, timestep]
    timestep_resolution = backend_model.timestep_resolution[timestep]
    parasitic_eff = get_param(backend_model, "parasitic_eff", (loc_tech, timestep))

    return carrier_prod <= (
        backend_model.energy_cap[loc_tech] * timestep_resolution * parasitic_eff
    )


def carrier_production_min_constraint_rule(backend_model, loc_tech_carrier, timestep):
    """
    Set minimum carrier production. All technologies except ``conversion_plus``.

    .. container:: scrolling-wrapper

        .. math::

            \\boldsymbol{carrier_{prod}}(loc::tech::carrier, timestep) \\geq energy_{cap}(loc::tech)
            \\times timestep\\_resolution(timestep) \\times energy_{cap,min\\_use}(loc::tec)

    """
    loc_tech = get_loc_tech(loc_tech_carrier)
    carrier_prod = backend_model.carrier_prod[loc_tech_carrier, timestep]
    timestep_resolution = backend_model.timestep_resolution[timestep]
    min_use = get_param(backend_model, "energy_cap_min_use", (loc_tech, timestep))

    return carrier_prod >= (
        backend_model.energy_cap[loc_tech] * timestep_resolution * min_use
    )


def carrier_consumption_max_constraint_rule(backend_model, loc_tech_carrier, timestep):
    """
    Set maximum carrier consumption for demand, storage, and transmission techs.

    .. container:: scrolling-wrapper

        .. math::

            \\boldsymbol{carrier_{con}}(loc::tech::carrier, timestep) \\geq
            -1 \\times energy_{cap}(loc::tech)
            \\times timestep\\_resolution(timestep)

    """
    loc_tech = get_loc_tech(loc_tech_carrier)
    carrier_con = backend_model.carrier_con[loc_tech_carrier, timestep]
    timestep_resolution = backend_model.timestep_resolution[timestep]

    return carrier_con >= (
        -1 * backend_model.energy_cap[loc_tech] * timestep_resolution
    )


def resource_max_constraint_rule(backend_model, loc_tech, timestep):
    """
    Set maximum resource consumed by supply_plus techs.

    .. container:: scrolling-wrapper

        .. math::

            \\boldsymbol{resource_{con}}(loc::tech, timestep) \\leq
            timestep\\_resolution(timestep) \\times resource_{cap}(loc::tech)

    """
    timestep_resolution = backend_model.timestep_resolution[timestep]

    return backend_model.resource_con[loc_tech, timestep] <= (
        timestep_resolution * backend_model.resource_cap[loc_tech]
    )


def storage_max_constraint_rule(backend_model, loc_tech, timestep):
    """
    Set maximum stored energy. Supply_plus & storage techs only.

    .. container:: scrolling-wrapper

        .. math::

            \\boldsymbol{storage}(loc::tech, timestep) \\leq
            storage_{cap}(loc::tech)

    """
    return (
        backend_model.storage[loc_tech, timestep] <= backend_model.storage_cap[loc_tech]
    )


def storage_discharge_depth_constraint_rule(backend_model, loc_tech, timestep):
    """
    Forces storage state of charge to be greater than the allowed depth of discharge.

    .. container:: scrolling-wrapper

        .. math::

            \\boldsymbol{storage}(loc::tech, timestep) >=
            \\boldsymbol{storage_discharge_depth}\\forall loc::tech \\in loc::techs_{storage}, \\forall timestep \\in timesteps

    """
    storage_discharge_depth = get_param(
        backend_model, "storage_discharge_depth", loc_tech
    )
    return (
        backend_model.storage[loc_tech, timestep]
        >= storage_discharge_depth * backend_model.storage_cap[loc_tech]
    )


def ramping_up_constraint_rule(backend_model, loc_tech_carrier, timestep):
    """
    Ramping up constraint.

    .. container:: scrolling-wrapper

        .. math::

            diff(loc::tech::carrier, timestep) \\leq max\\_ramping\\_rate(loc::tech::carrier, timestep)

    """
    return ramping_constraint(backend_model, loc_tech_carrier, timestep, direction=0)


def ramping_down_constraint_rule(backend_model, loc_tech_carrier, timestep):
    """
    Ramping down constraint.

    .. container:: scrolling-wrapper

        .. math::

            -1 \\times max\\_ramping\\_rate(loc::tech::carrier, timestep) \\leq diff(loc::tech::carrier, timestep)

    """
    return ramping_constraint(backend_model, loc_tech_carrier, timestep, direction=1)


def ramping_constraint(backend_model, loc_tech_carrier, timestep, direction=0):
    """
    Ramping rate constraints.

    Direction: 0 is up, 1 is down.

    .. container:: scrolling-wrapper

        .. math::

            \\boldsymbol{max\\_ramping\\_rate}(loc::tech::carrier, timestep) =
            energy_{ramping}(loc::tech, timestep) \\times energy_{cap}(loc::tech)

            \\boldsymbol{diff}(loc::tech::carrier, timestep) =
            (carrier_{prod}(loc::tech::carrier, timestep) + carrier_{con}(loc::tech::carrier, timestep))
            / timestep\\_resolution(timestep) -
            (carrier_{prod}(loc::tech::carrier, timestep-1) + carrier_{con}(loc::tech::carrier, timestep-1))
            / timestep\\_resolution(timestep-1)

    """

    # No constraint for first timestep
    # Pyomo returns the order 1-indexed, but we want 0-indexing
    if backend_model.timesteps.ord(timestep) - 1 == 0:
        return po.Constraint.NoConstraint
    else:
        previous_step = get_previous_timestep(backend_model.timesteps, timestep)
        time_res = backend_model.timestep_resolution[timestep]
        time_res_prev = backend_model.timestep_resolution[previous_step]
        loc_tech = loc_tech_carrier.rsplit("::", 1)[0]
        # Ramping rate (fraction of installed capacity per hour)
        ramping_rate = get_param(backend_model, "energy_ramping", (loc_tech, timestep))

        try:
            prod_this = backend_model.carrier_prod[loc_tech_carrier, timestep]
            prod_prev = backend_model.carrier_prod[loc_tech_carrier, previous_step]
        except KeyError:
            prod_this = 0
            prod_prev = 0

        try:
            con_this = backend_model.carrier_con[loc_tech_carrier, timestep]
            con_prev = backend_model.carrier_con[loc_tech_carrier, previous_step]
        except KeyError:
            con_this = 0
            con_prev = 0

        diff = (prod_this + con_this) / time_res - (
            prod_prev + con_prev
        ) / time_res_prev

        max_ramping_rate = ramping_rate * backend_model.energy_cap[loc_tech]

        if direction == 0:
            return diff <= max_ramping_rate
        else:
            return -1 * max_ramping_rate <= diff


def storage_intra_max_rule(backend_model, loc_tech, timestep):
    """
    When clustering days, to reduce the timeseries length, set limits on
    intra-cluster auxiliary maximum storage decision variable.
    `Ref: DOI 10.1016/j.apenergy.2018.01.023 <https://doi.org/10.1016/j.apenergy.2018.01.023>`_

    .. container:: scrolling-wrapper

        .. math::

            \\boldsymbol{storage}(loc::tech, timestep) \\leq
            \\boldsymbol{storage_{intra\\_cluster, max}}(loc::tech, cluster(timestep))
            \\quad \\forall loc::tech \\in loc::techs_{store}, \\forall timestep \\in timesteps

    Where :math:`cluster(timestep)` is the cluster number in which the timestep
    is located.
    """
    cluster = backend_model.__calliope_model_data["data"]["timestep_cluster"][timestep]
    return (
        backend_model.storage[loc_tech, timestep]
        <= backend_model.storage_intra_cluster_max[loc_tech, cluster]
    )


def storage_intra_min_rule(backend_model, loc_tech, timestep):
    """
    When clustering days, to reduce the timeseries length, set limits on
    intra-cluster auxiliary minimum storage decision variable.
    `Ref: DOI 10.1016/j.apenergy.2018.01.023 <https://doi.org/10.1016/j.apenergy.2018.01.023>`_

    .. container:: scrolling-wrapper

        .. math::

            \\boldsymbol{storage}(loc::tech, timestep) \\geq
            \\boldsymbol{storage_{intra\\_cluster, min}}(loc::tech, cluster(timestep))
            \\quad \\forall loc::tech \\in loc::techs_{store}, \\forall timestep \\in timesteps

    Where :math:`cluster(timestep)` is the cluster number in which the timestep
    is located.
    """
    cluster = backend_model.__calliope_model_data["data"]["timestep_cluster"][timestep]
    return (
        backend_model.storage[loc_tech, timestep]
        >= backend_model.storage_intra_cluster_min[loc_tech, cluster]
    )


def storage_inter_max_rule(backend_model, loc_tech, datestep):
    """
    When clustering days, to reduce the timeseries length, set maximum limit on
    the intra-cluster and inter-date stored energy.
    intra-cluster = all timesteps in a single cluster
    datesteps = all dates in the unclustered timeseries (each has a corresponding cluster)
    `Ref: DOI 10.1016/j.apenergy.2018.01.023 <https://doi.org/10.1016/j.apenergy.2018.01.023>`_

    .. container:: scrolling-wrapper

        .. math::

            \\boldsymbol{storage_{inter\\_cluster}}(loc::tech, datestep) +
            \\boldsymbol{storage_{intra\\_cluster, max}}(loc::tech, cluster(datestep))
            \\leq \\boldsymbol{storage_{cap}}(loc::tech) \\quad \\forall
            loc::tech \\in loc::techs_{store}, \\forall datestep \\in datesteps

    Where :math:`cluster(datestep)` is the cluster number in which the datestep
    is located.
    """
    cluster = backend_model.__calliope_model_data["data"]["lookup_datestep_cluster"][
        datestep
    ]
    return (
        backend_model.storage_inter_cluster[loc_tech, datestep]
        + backend_model.storage_intra_cluster_max[loc_tech, cluster]
        <= backend_model.storage_cap[loc_tech]
    )


def storage_inter_min_rule(backend_model, loc_tech, datestep):
    """
    When clustering days, to reduce the timeseries length, set minimum limit on
    the intra-cluster and inter-date stored energy.
    intra-cluster = all timesteps in a single cluster
    datesteps = all dates in the unclustered timeseries (each has a corresponding cluster)
    `Ref: DOI 10.1016/j.apenergy.2018.01.023 <https://doi.org/10.1016/j.apenergy.2018.01.023>`_

    .. container:: scrolling-wrapper

        .. math::

            \\boldsymbol{storage_{inter\\_cluster}}(loc::tech, datestep)
            \\times (1 - storage\\_loss(loc::tech, timestep))^{24} +
            \\boldsymbol{storage_{intra\\_cluster, min}}(loc::tech, cluster(datestep))
            \\geq 0 \\quad \\forall loc::tech \\in loc::techs_{store},
            \\forall datestep \\in datesteps

    Where :math:`cluster(datestep)` is the cluster number in which the datestep
    is located.
    """
    cluster = backend_model.__calliope_model_data["data"]["lookup_datestep_cluster"][
        datestep
    ]
    storage_loss = get_param(backend_model, "storage_loss", loc_tech)
    return (
        backend_model.storage_inter_cluster[loc_tech, datestep]
        * ((1 - storage_loss) ** 24)
        + backend_model.storage_intra_cluster_min[loc_tech, cluster]
        >= 0
    )
