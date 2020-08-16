"""
Copyright (C) 2013-2019 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

group.py
~~~~~~~~

Group constraints.

"""

import logging

import numpy as np
import pyomo.core as po  # pylint: disable=import-error

from calliope.backend.pyomo.util import loc_tech_is_in, get_param, check_value

logger = logging.getLogger(__name__)

ORDER = 20  # order in which to invoke constraints relative to other constraint files


def return_noconstraint(*args):
    logger.debug("group constraint returned NoConstraint: {}".format(",".join(args)))
    return po.Constraint.NoConstraint


def load_constraints(backend_model):
    model_data_dict = backend_model.__calliope_model_data["data"]

    for sense in ["min", "max", "equals"]:
        if "group_energy_cap_share_{}".format(sense) in model_data_dict:
            setattr(
                backend_model,
                "group_energy_cap_share_{}_constraint".format(sense),
                po.Constraint(
                    getattr(
                        backend_model, "group_names_energy_cap_share_{}".format(sense)
                    ),
                    [sense],
                    rule=energy_cap_share_constraint_rule,
                ),
            )

        if "group_energy_cap_{}".format(sense) in model_data_dict:
            setattr(
                backend_model,
                "group_energy_cap_{}_constraint".format(sense),
                po.Constraint(
                    getattr(backend_model, "group_names_energy_cap_{}".format(sense)),
                    [sense],
                    rule=energy_cap_constraint_rule,
                ),
            )

        if "group_storage_cap_{}".format(sense) in model_data_dict:
            setattr(
                backend_model,
                "group_storage_cap_{}_constraint".format(sense),
                po.Constraint(
                    getattr(backend_model, "group_names_storage_cap_{}".format(sense)),
                    [sense],
                    rule=storage_cap_constraint_rule,
                ),
            )              
            
        if "group_resource_area_{}".format(sense) in model_data_dict:
            setattr(
                backend_model,
                "group_resource_area_{}_constraint".format(sense),
                po.Constraint(
                    getattr(
                        backend_model, "group_names_resource_area_{}".format(sense)
                    ),
                    [sense],
                    rule=resource_area_constraint_rule,
                ),
            )

        if "group_carrier_prod_{}".format(sense) in model_data_dict:
            setattr(
                backend_model,
                "group_carrier_prod_{}_constraint".format(sense),
                po.Constraint(
                    getattr(backend_model, "group_names_carrier_prod_{}".format(sense)),
                    backend_model.carriers,
                    [sense],
                    rule=carrier_prod_constraint_rule,
                ),
            )

        if "group_demand_share_{}".format(sense) in model_data_dict:
            setattr(
                backend_model,
                "group_demand_share_{}_constraint".format(sense),
                po.Constraint(
                    getattr(backend_model, "group_names_demand_share_{}".format(sense)),
                    backend_model.carriers,
                    [sense],
                    rule=demand_share_constraint_rule,
                ),
            )

        if "group_demand_share_per_timestep_{}".format(sense) in model_data_dict:
            setattr(
                backend_model,
                "group_demand_share_per_timestep_{}_constraint".format(sense),
                po.Constraint(
                    getattr(
                        backend_model,
                        "group_names_demand_share_per_timestep_{}".format(sense),
                    ),
                    backend_model.carriers,
                    backend_model.timesteps,
                    [sense],
                    rule=demand_share_per_timestep_constraint_rule,
                ),
            )

        if "group_carrier_prod_share_{}".format(sense) in model_data_dict:
            setattr(
                backend_model,
                "group_carrier_prod_share_{}_constraint".format(sense),
                po.Constraint(
                    getattr(
                        backend_model, "group_names_carrier_prod_share_{}".format(sense)
                    ),
                    backend_model.carriers,
                    [sense],
                    rule=carrier_prod_share_constraint_rule,
                ),
            )

        if "group_carrier_prod_share_per_timestep_{}".format(sense) in model_data_dict:
            setattr(
                backend_model,
                "group_carrier_prod_share_per_timestep_{}_constraint".format(sense),
                po.Constraint(
                    getattr(
                        backend_model,
                        "group_names_carrier_prod_share_per_timestep_{}".format(sense),
                    ),
                    backend_model.carriers,
                    backend_model.timesteps,
                    [sense],
                    rule=carrier_prod_share_per_timestep_constraint_rule,
                ),
            )

        if "group_net_import_share_{}".format(sense) in model_data_dict:
            setattr(
                backend_model,
                "group_net_import_share_{}_constraint".format(sense),
                po.Constraint(
                    getattr(
                        backend_model, "group_names_net_import_share_{}".format(sense)
                    ),
                    backend_model.carriers,
                    [sense],
                    rule=net_import_share_constraint_rule,
                ),
            )

        if "group_cost_{}".format(sense) in model_data_dict:
            setattr(
                backend_model,
                "group_cost_{}_constraint".format(sense),
                po.Constraint(
                    getattr(backend_model, "group_names_cost_{}".format(sense)),
                    backend_model.costs,
                    [sense],
                    rule=cost_cap_constraint_rule,
                ),
            )
        if "group_cost_var_{}".format(sense) in model_data_dict:
            setattr(
                backend_model,
                "group_cost_var_{}_constraint".format(sense),
                po.Constraint(
                    getattr(backend_model, "group_names_cost_var_{}".format(sense)),
                    backend_model.costs,
                    [sense],
                    rule=cost_var_cap_constraint_rule,
                ),
            )
        if "group_cost_investment_{}".format(sense) in model_data_dict:
            setattr(
                backend_model,
                "group_cost_investment_{}_constraint".format(sense),
                po.Constraint(
                    getattr(
                        backend_model, "group_names_cost_investment_{}".format(sense)
                    ),
                    backend_model.costs,
                    [sense],
                    rule=cost_investment_cap_constraint_rule,
                ),
            )

    if "group_demand_share_per_timestep_decision" in model_data_dict:
        backend_model.group_demand_share_per_timestep_decision_main_constraint = po.Constraint(
            backend_model.group_names_demand_share_per_timestep_decision,
            backend_model.carriers,
            backend_model.techs,
            backend_model.timesteps,
            rule=demand_share_per_timestep_decision_main_constraint_rule,
        )
        backend_model.group_demand_share_per_timestep_decision_sum_constraint = po.Constraint(
            backend_model.group_names_demand_share_per_timestep_decision,
            backend_model.carriers,
            rule=demand_share_per_timestep_decision_sum_constraint_rule,
        )


def equalizer(lhs, rhs, sign):
    if sign == "max":
        return lhs <= rhs
    elif sign == "min":
        return lhs >= rhs
    elif sign == "equals":
        return lhs == rhs
    else:
        raise ValueError("Invalid sign: {}".format(sign))


def get_demand_share_lhs_and_rhs_loc_tech_carriers(backend_model, group_name, carrier):
    """
    Returns
    -------
    (lhs_loc_tech_carriers, rhs_loc_tech_carriers):
        lhs are the supply technologies, rhs are the demand technologies
    """
    lhs_loc_techs = getattr(
        backend_model, "group_constraint_loc_techs_{}".format(group_name)
    )
    lhs_locs = set(loc_tech.split("::")[0] for loc_tech in lhs_loc_techs)
    lhs_loc_tech_carriers = [
        i
        for i in backend_model.loc_tech_carriers_prod
        if i.rsplit("::", 1)[0] in lhs_loc_techs and i.split("::")[-1] == carrier
    ]
    rhs_loc_tech_carriers = [
        i
        for i in backend_model.loc_tech_carriers_demand
        if i.split("::")[0] in lhs_locs and i.split("::")[-1] == carrier
    ]
    return (lhs_loc_tech_carriers, rhs_loc_tech_carriers)


def demand_share_constraint_rule(backend_model, group_name, carrier, what):
    """
    Enforces shares of demand of a carrier to be met by the given groups
    of technologies at the given locations, on average over the entire
    model period. The share is relative to ``demand`` technologies only.

    .. container:: scrolling-wrapper

        .. math::

            \\sum_{loc::tech::carrier \\in given\\_group, timestep \\in timesteps} carrier_{prod}(loc::tech::carrier, timestep) \\leq
            share \\times \\sum_{loc::tech:carrier \\in loc\\_techs\\_demand \\in given\\_locations, timestep\\in timesteps}
            carrier_{con}(loc::tech::carrier, timestep)

    """
    share = get_param(
        backend_model, "group_demand_share_{}".format(what), (carrier, group_name)
    )

    if check_value(share):
        return return_noconstraint("demand_share", group_name)
    else:
        (
            lhs_loc_tech_carriers,
            rhs_loc_tech_carriers,
        ) = get_demand_share_lhs_and_rhs_loc_tech_carriers(
            backend_model, group_name, carrier
        )

        lhs = sum(
            backend_model.carrier_prod[loc_tech_carrier, timestep]
            for loc_tech_carrier in lhs_loc_tech_carriers
            for timestep in backend_model.timesteps
        )

        rhs = (
            share
            * -1
            * sum(
                backend_model.carrier_con[loc_tech_carrier, timestep]
                for loc_tech_carrier in rhs_loc_tech_carriers
                for timestep in backend_model.timesteps
            )
        )

        return equalizer(lhs, rhs, what)


def demand_share_per_timestep_constraint_rule(
    backend_model, group_name, carrier, timestep, what
):
    """
    Enforces shares of demand of a carrier to be met by the given groups
    of technologies at the given locations, in each timestep.
    The share is relative to ``demand`` technologies only.

    .. container:: scrolling-wrapper

        .. math::

            \\sum_{loc::tech::carrier \\in given\\_group} carrier_{prod}(loc::tech::carrier, timestep) \\leq
            share \\times \\sum_{loc::tech:carrier \\in loc\\_techs\\_demand \\in given\\_locations}
            carrier_{con}(loc::tech::carrier, timestep) for timestep \\in timesteps

    """
    share = get_param(
        backend_model,
        "group_demand_share_per_timestep_{}".format(what),
        (carrier, group_name),
    )

    if check_value(share):
        return return_noconstraint("demand_share_per_timestep", group_name)
    else:
        (
            lhs_loc_tech_carriers,
            rhs_loc_tech_carriers,
        ) = get_demand_share_lhs_and_rhs_loc_tech_carriers(
            backend_model, group_name, carrier
        )

        lhs = sum(
            backend_model.carrier_prod[loc_tech_carrier, timestep]
            for loc_tech_carrier in lhs_loc_tech_carriers
        )

        rhs = (
            share
            * -1
            * sum(
                backend_model.carrier_con[loc_tech_carrier, timestep]
                for loc_tech_carrier in rhs_loc_tech_carriers
            )
        )

        return equalizer(lhs, rhs, what)


def demand_share_per_timestep_decision_main_constraint_rule(
    backend_model, group_name, carrier, tech, timestep
):
    """
    Allows the model to decide on how a fraction demand for a carrier is met
    by the given groups, which will all have the same share in each timestep.
    The share is relative to the actual demand from ``demand`` technologies only.

    The main constraint enforces that the shares are the same in each timestep.

    .. container:: scrolling-wrapper

        .. math::
            \\sum_{loc::tech::carrier \\in given\\_group} carrier_{prod}(loc::tech::carrier, timestep)

            =

            \\sum_{loc::tech::carrier \\in given\\_group}
            required\\_resource(loc::tech::carrier, timestep)

            \\times \\sum_{loc::tech::carrier \\in given\\_group}
            demand\\_share\\_per\\_timestep\\_decision(loc::tech::carrier)

            \\forall timestep \\in timesteps

            \\forall tech \\in techs

    """
    share_of_carrier_demand = get_param(
        backend_model, "group_demand_share_per_timestep_decision", (carrier, group_name)
    )

    if check_value(share_of_carrier_demand):
        return return_noconstraint(
            "demand_share_per_timestep_decision_main", group_name
        )
    else:
        # lhs are the supply technologies, rhs are the demand technologies
        (
            lhs_loc_tech_carriers,
            rhs_loc_tech_carriers,
        ) = get_demand_share_lhs_and_rhs_loc_tech_carriers(
            backend_model, group_name, carrier
        )
        # Filter the supply loc_tech_carriers by the current tech
        lhs_loc_tech_carriers = [
            i for i in lhs_loc_tech_carriers if "::{}::".format(tech) in i
        ]

        # Only techs that are in the given group are considered
        if len(lhs_loc_tech_carriers) == 0:
            return return_noconstraint(
                "demand_share_per_timestep_decision_main", group_name
            )

        lhs = sum(
            backend_model.carrier_prod[loc_tech_carrier, timestep]
            for loc_tech_carrier in lhs_loc_tech_carriers
        )

        rhs = (
            -1
            * sum(
                backend_model.required_resource[
                    rhs_loc_tech_carrier.rsplit("::", 1)[0], timestep
                ]
                for rhs_loc_tech_carrier in rhs_loc_tech_carriers
            )
            * sum(
                backend_model.demand_share_per_timestep_decision[lhs_loc_tech_carrier]
                for lhs_loc_tech_carrier in lhs_loc_tech_carriers
            )
        )

        return equalizer(lhs, rhs, "equals")


def demand_share_per_timestep_decision_sum_constraint_rule(
    backend_model, group_name, carrier
):
    """
    Allows the model to decide on how a fraction of demand for a carrier is met
    by the given groups, which will all have the same share in each timestep.
    The share is relative to the actual demand from ``demand`` technologies only.

    The sum constraint ensures that all decision shares add up to the share of
    carrier demand specified in the constraint.

    This constraint is only applied if the share of carrier demand has been
    set to a not-None value.

 .. container:: scrolling-wrapper

        .. math::
            share = \\sum_{loc::tech::carrier \\in given\\_group}
            demand\\_share\\_per\\_timestep\\_decision(loc::tech::carrier)


    """
    share_of_carrier_demand = get_param(
        backend_model, "group_demand_share_per_timestep_decision", (carrier, group_name)
    )

    # If inf was given that means that we don't limit the total share
    if check_value(share_of_carrier_demand) or np.isinf(share_of_carrier_demand):
        return return_noconstraint("demand_share_per_timestep_decision_sum", group_name)
    else:
        lhs_loc_tech_carriers, _ = get_demand_share_lhs_and_rhs_loc_tech_carriers(
            backend_model, group_name, carrier
        )

        return share_of_carrier_demand == sum(
            backend_model.demand_share_per_timestep_decision[loc_tech_carrier]
            for loc_tech_carrier in lhs_loc_tech_carriers
        )


def get_carrier_prod_share_lhs_and_rhs_loc_techs(backend_model, group_name):
    lhs_loc_techs = getattr(
        backend_model, "group_constraint_loc_techs_{}".format(group_name)
    )
    lhs_locs = [loc_tech.split("::")[0] for loc_tech in lhs_loc_techs]
    rhs_loc_techs = [
        i
        for i in backend_model.loc_techs_supply_conversion_all
        if i.split("::")[0] in lhs_locs
    ]
    return (lhs_loc_techs, rhs_loc_techs)


def carrier_prod_share_constraint_rule(backend_model, constraint_group, carrier, what):
    """
    Enforces shares of carrier_prod for groups of technologies and locations,
    on average over the entire model period. The share is relative to
    ``supply`` and ``supply_plus`` technologies only.

    .. container:: scrolling-wrapper

        .. math::

            \\sum_{loc::tech::carrier \\in given\\_group, timestep \\in timesteps} carrier_{prod}(loc::tech::carrier, timestep) \\leq
            share \\times \\sum_{loc::tech:carrier \\in loc\\_tech\\_carriers\\_supply\\_all \\in given\\_locations, timestep\\in timesteps}
            carrier_{prod}(loc::tech::carrier, timestep)

    """
    share = get_param(
        backend_model,
        "group_carrier_prod_share_{}".format(what),
        (carrier, constraint_group),
    )

    if check_value(share):
        return return_noconstraint("supply_share", constraint_group)
    else:
        lhs_loc_techs, rhs_loc_techs = get_carrier_prod_share_lhs_and_rhs_loc_techs(
            backend_model, constraint_group
        )

        lhs = sum(
            backend_model.carrier_prod[loc_tech + "::" + carrier, timestep]
            for loc_tech in lhs_loc_techs
            for timestep in backend_model.timesteps
        )
        rhs = share * sum(
            backend_model.carrier_prod[loc_tech + "::" + carrier, timestep]
            for loc_tech in rhs_loc_techs
            for timestep in backend_model.timesteps
        )

        return equalizer(lhs, rhs, what)


def carrier_prod_share_per_timestep_constraint_rule(
    backend_model, constraint_group, carrier, timestep, what
):
    """
    Enforces shares of carrier_prod for groups of technologies and locations,
    in each timestep. The share is relative to ``supply`` and ``supply_plus``
    technologies only.

    .. container:: scrolling-wrapper

        .. math::

            \\sum_{loc::tech::carrier \\in given\\_group} carrier_{prod}(loc::tech::carrier, timestep) \\leq
            share \\times \\sum_{loc::tech:carrier \\in loc\\_tech\\_carriers\\_supply\\_all \\in given\\_locations}
            carrier_{prod}(loc::tech::carrier, timestep) for timestep \\in timesteps

    """
    share = get_param(
        backend_model,
        "group_carrier_prod_share_per_timestep_{}".format(what),
        (carrier, constraint_group),
    )

    if check_value(share):
        return return_noconstraint("carrier_prod_share_per_timestep", constraint_group)
    else:
        lhs_loc_techs, rhs_loc_techs = get_carrier_prod_share_lhs_and_rhs_loc_techs(
            backend_model, constraint_group
        )

        lhs = sum(
            backend_model.carrier_prod[loc_tech + "::" + carrier, timestep]
            for loc_tech in lhs_loc_techs
        )
        rhs = share * sum(
            backend_model.carrier_prod[loc_tech + "::" + carrier, timestep]
            for loc_tech in rhs_loc_techs
        )

        return equalizer(lhs, rhs, what)


def net_import_share_constraint_rule(backend_model, constraint_group, carrier, what):
    """
    Enforces demand shares of net imports from transmission technologies for groups of locations,
    on average over the entire model period. Transmission within the group are ignored. The share
    is relative to ``demand`` technologies only.

    .. container:: scrolling-wrapper

        .. math::

            \\sum_{loc::tech::carrier \\in loc\\_tech\\_carriers\\_transmission \\in given\\_locations, timestep \\in timesteps} carrier_{prod}(loc::tech::carrier, timestep)
            + \\sum_{loc::tech::carrier \\in loc\\_tech\\_carriers\\_transmission \\in given\\_locations, timestep \\in timesteps} carrier_{con}(loc::tech::carrier, timestep) \\leq
            share \\times \\sum_{loc::tech:carrier \\in loc\\_tech\\_demand \\in given\\_locations, timestep\\in timesteps}
            carrier_{con}(loc::tech::carrier, timestep)

    """
    share = get_param(
        backend_model,
        "group_net_import_share_{}".format(what),
        (carrier, constraint_group),
    )

    if check_value(share):
        return return_noconstraint("net_import_share", constraint_group)
    else:
        trans_loc_tech = getattr(
            backend_model, "group_constraint_loc_techs_{}".format(constraint_group)
        )
        locs = set(loc_tech.split("::")[0] for loc_tech in trans_loc_tech)
        trans_loc_tech = filter(
            lambda loc_tec: loc_tec.split(":")[-1] not in locs, trans_loc_tech
        )
        demand_loc_tech = [
            i
            for i in backend_model.loc_tech_carriers_demand
            if i.split("::")[0] in locs
        ]

        lhs = sum(
            (
                backend_model.carrier_prod[loc_tech + "::" + carrier, timestep]
                + backend_model.carrier_con[loc_tech + "::" + carrier, timestep]
            )
            for loc_tech in trans_loc_tech
            for timestep in backend_model.timesteps
        )
        rhs = -share * sum(
            backend_model.carrier_con[loc_tech, timestep]
            for loc_tech in demand_loc_tech
            for timestep in backend_model.timesteps
        )
        return equalizer(lhs, rhs, what)


def carrier_prod_constraint_rule(backend_model, constraint_group, carrier, what):
    """
    Enforces carrier_prod for groups of technologies and locations,
    as a sum over the entire model period.

    .. container:: scrolling-wrapper

        .. math::

            \\sum_{loc::tech::carrier \\in given\\_group, timestep \\in timesteps} carrier_{prod}(loc::tech::carrier, timestep) \\leq supply_max

    """
    limit = get_param(
        backend_model, "group_carrier_prod_{}".format(what), (carrier, constraint_group)
    )

    if check_value(limit):
        return return_noconstraint("carrier_prod", constraint_group)
    else:
        # We won't actually use the rhs techs
        lhs_loc_techs, rhs_loc_techs = get_carrier_prod_share_lhs_and_rhs_loc_techs(
            backend_model, constraint_group
        )

        lhs = sum(
            backend_model.carrier_prod[loc_tech + "::" + carrier, timestep]
            for loc_tech in lhs_loc_techs
            for timestep in backend_model.timesteps
            if loc_tech + "::" + carrier in backend_model.loc_tech_carriers_prod
        )

        return equalizer(lhs, limit, what)


def energy_cap_share_constraint_rule(backend_model, constraint_group, what):
    """
    Enforces shares of energy_cap for groups of technologies and locations. The
    share is relative to ``supply`` and ``supply_plus`` technologies only.

    .. container:: scrolling-wrapper

        .. math::

            \\sum_{loc::tech \\in given\\_group} energy_{cap}(loc::tech) \\leq
            share \\times \\sum_{loc::tech \\in loc\\_tech\\_supply\\_all \\in given\\_locations} energy_{cap}(loc::tech)
    """
    share = get_param(
        backend_model, "group_energy_cap_share_{}".format(what), (constraint_group)
    )

    if check_value(share):
        return return_noconstraint("energy_cap_share", constraint_group)
    else:
        lhs_loc_techs = getattr(
            backend_model, "group_constraint_loc_techs_{}".format(constraint_group)
        )
        lhs_locs = [loc_tech.split("::")[0] for loc_tech in lhs_loc_techs]
        rhs_loc_techs = [
            i
            for i in backend_model.loc_techs_supply_conversion_all
            if i.split("::")[0] in lhs_locs
        ]

        lhs = sum(backend_model.energy_cap[loc_tech] for loc_tech in lhs_loc_techs)
        rhs = share * sum(
            backend_model.energy_cap[loc_tech] for loc_tech in rhs_loc_techs
        )

        return equalizer(lhs, rhs, what)


def energy_cap_constraint_rule(backend_model, constraint_group, what):
    """
    Enforce upper and lower bounds for energy_cap of energy_cap
    for groups of technologies and locations.

    .. container:: scrolling-wrapper

        .. math::

            \\sum_{loc::tech \\in given\\_group} energy_{cap}(loc::tech) \\leq energy\\_cap\\_max\\\\

            \\sum_{loc::tech \\in given\\_group} energy_{cap}(loc::tech) \\geq energy\\_cap\\_min

    """
    threshold = get_param(
        backend_model, "group_energy_cap_{}".format(what), (constraint_group)
    )

    if check_value(threshold):
        return return_noconstraint("energy_cap", constraint_group)
    else:
        lhs_loc_techs = getattr(
            backend_model, "group_constraint_loc_techs_{}".format(constraint_group)
        )

        # Transmission techs only contribute half their capacity in each direction
        lhs = []
        for loc_tech in lhs_loc_techs:
            if loc_tech_is_in(backend_model, loc_tech, "loc_techs_transmission"):
                weight = 0.5
            else:
                weight = 1

            lhs.append(weight * backend_model.energy_cap[loc_tech])

        rhs = threshold

        return equalizer(sum(lhs), rhs, what)

def storage_cap_constraint_rule(backend_model, constraint_group, what):
    """
    Enforce upper and lower bounds for storage_cap of storage_cap
    for groups of technologies and locations.

    .. container:: scrolling-wrapper

        .. math::

            \\sum_{loc::tech \\in given\\_group} storage_{cap}(loc::tech) \\leq storage\\_cap\\_max\\\\

            \\sum_{loc::tech \\in given\\_group} storage_{cap}(loc::tech) \\geq storage\\_cap\\_min

    """
    threshold = get_param(
        backend_model, "group_storage_cap_{}".format(what), (constraint_group)
    )

    if check_value(threshold):
        return return_noconstraint("storage_cap", constraint_group)
    else:
        lhs_loc_techs = getattr(
            backend_model, "group_constraint_loc_techs_{}".format(constraint_group)
        )

        # Transmission techs only contribute half their capacity in each direction
        lhs = []
        for loc_tech in lhs_loc_techs:
            if loc_tech_is_in(backend_model, loc_tech, "loc_techs_transmission"):
                weight = 0.5
            else:
                weight = 1

            lhs.append(weight * backend_model.storage_cap[loc_tech])

        rhs = threshold

        return equalizer(sum(lhs), rhs, what)
    
def cost_cap_constraint_rule(backend_model, group_name, cost, what):
    """
    Limit cost for a specific cost class to a certain value,
    i.e. Ɛ-constrained costs,
    for groups of technologies and locations.

    .. container:: scrolling-wrapper

        .. math::

            \\sum{loc::tech \\in loc\\_techs_{group\\_name}, timestep \\in timesteps}
            \\boldsymbol{cost}(cost, loc::tech, timestep)
            \\begin{cases}
                \\leq cost\\_max(cost)
                \\geq cost\\_min(cost)
                = cost\\_equals(cost)
            \\end{cases}

    """
    cost_cap = get_param(
        backend_model, "group_cost_{}".format(what), (cost, group_name)
    )

    if check_value(cost_cap):
        return return_noconstraint("cost_cap", group_name)
    else:
        loc_techs = [
            i
            for i in getattr(
                backend_model, "group_constraint_loc_techs_{}".format(group_name)
            )
            if i in backend_model.loc_techs_cost
        ]

        sum_cost = sum(backend_model.cost[cost, loc_tech] for loc_tech in loc_techs)

        return equalizer(sum_cost, cost_cap, what)


def cost_investment_cap_constraint_rule(backend_model, group_name, cost, what):
    """
    Limit investment costs specific to a cost class to a
    certain value, i.e. Ɛ-constrained costs,
    for groups of technologies and locations.

    .. container:: scrolling-wrapper

        .. math::

            \\sum{loc::tech \\in loc\\_techs_{group\\_name}, timestep \\in timesteps}
            \\boldsymbol{cost\\_{investment}}(cost, loc::tech, timestep)
            \\begin{cases}
                \\leq cost\\_investment\\_max(cost)
                \\geq cost\\_investment\\_min(cost)
                = cost\\_investment\\_equals(cost)
            \\end{cases}

    """
    cost_cap = get_param(
        backend_model, "group_cost_investment_{}".format(what), (cost, group_name)
    )

    if check_value(cost_cap):
        return return_noconstraint("cost_investment_cap", group_name)
    else:
        loc_techs = [
            i
            for i in getattr(
                backend_model, "group_constraint_loc_techs_{}".format(group_name)
            )
            if i in backend_model.loc_techs_investment_cost
        ]

        sum_cost = sum(
            backend_model.cost_investment[cost, loc_tech] for loc_tech in loc_techs
        )

        return equalizer(sum_cost, cost_cap, what)


def cost_var_cap_constraint_rule(backend_model, group_name, cost, what):
    """
    Limit variable costs specific to a cost class
    to a certain value, i.e. Ɛ-constrained costs,
    for groups of technologies and locations.

    .. container:: scrolling-wrapper

        .. math::

            \\sum{loc::tech \\in loc\\_techs_{group\\_name}, timestep \\in timesteps}
            \\boldsymbol{cost\\_{var}}(cost, loc::tech, timestep)
            \\begin{cases}
                \\leq cost\\_var\\_max(cost)
                \\geq cost\\_var\\_min(cost)
                = cost\\_var\\_equals(cost)
            \\end{cases}

    """
    cost_cap = get_param(
        backend_model, "group_cost_var_{}".format(what), (cost, group_name)
    )

    if check_value(cost_cap):
        return return_noconstraint("cost_var_cap", group_name)
    else:
        loc_techs = [
            i
            for i in getattr(
                backend_model, "group_constraint_loc_techs_{}".format(group_name)
            )
            if i in backend_model.loc_techs_om_cost
        ]

        sum_cost = sum(
            backend_model.cost_var[cost, loc_tech, timestep]
            for loc_tech in loc_techs
            for timestep in backend_model.timesteps
        )

        return equalizer(sum_cost, cost_cap, what)


def resource_area_constraint_rule(backend_model, constraint_group, what):
    """
    Enforce upper and lower bounds of resource_area for groups of
    technologies and locations.

    .. container:: scrolling-wrapper

        .. math::

            \\boldsymbol{resource_{area}}(loc::tech) \\leq group\\_resource\\_area\\_max\\\\

            \\boldsymbol{resource_{area}}(loc::tech) \\geq group\\_resource\\_area\\_min

    """
    threshold = get_param(
        backend_model, "group_resource_area_{}".format(what), (constraint_group)
    )

    if check_value(threshold):
        return return_noconstraint("resource_area", constraint_group)
    else:
        lhs_loc_techs = getattr(
            backend_model, "group_constraint_loc_techs_{}".format(constraint_group)
        )

        lhs = sum(backend_model.resource_area[loc_tech] for loc_tech in lhs_loc_techs)
        rhs = threshold

        return equalizer(lhs, rhs, what)
