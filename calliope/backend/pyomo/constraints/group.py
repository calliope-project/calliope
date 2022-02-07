"""
Copyright (C) since 2013 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

group.py
~~~~~~~~

Group constraints.

"""

import logging

import numpy as np
import pyomo.core as po  # pylint: disable=import-error

from calliope.backend.pyomo.util import (
    loc_tech_is_in,
    get_param,
    invalid,
    get_timestep_weight,
)

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
                    [sense],
                    rule=carrier_prod_constraint_rule,
                ),
            )

        if "group_carrier_con_{}".format(sense) in model_data_dict:
            setattr(
                backend_model,
                "group_carrier_con_{}_constraint".format(sense),
                po.Constraint(
                    getattr(backend_model, "group_names_carrier_con_{}".format(sense)),
                    [sense],
                    rule=carrier_con_constraint_rule,
                ),
            )

        if "group_demand_share_{}".format(sense) in model_data_dict:
            setattr(
                backend_model,
                "group_demand_share_{}_constraint".format(sense),
                po.Constraint(
                    getattr(backend_model, "group_names_demand_share_{}".format(sense)),
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
        relaxation = backend_model.__calliope_run_config["relax_constraint"][
            "demand_share_per_timestep_decision_main_constraint"
        ]
        if relaxation == 0:
            sense_scale = [("equals", 1)]
        else:
            sense_scale = [("min", 1 - relaxation), ("max", 1 + relaxation)]
        for group_name in backend_model.group_names_demand_share_per_timestep_decision:
            setattr(
                backend_model,
                f"group_demand_share_per_timestep_decision_{group_name}_constraint",
                po.Constraint(
                    [group_name],
                    get_group_lhs_and_rhs_loc_tech_carriers(backend_model, group_name)[
                        0
                    ],
                    backend_model.timesteps,
                    sense_scale,
                    rule=demand_share_per_timestep_decision_main_constraint_rule,
                ),
            )
        backend_model.group_demand_share_per_timestep_decision_sum_constraint = (
            po.Constraint(
                backend_model.group_names_demand_share_per_timestep_decision,
                rule=demand_share_per_timestep_decision_sum_constraint_rule,
            )
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


def get_group_lhs_loc_tech_carriers(backend_model, group_name):
    return getattr(
        backend_model, "group_constraint_loc_tech_carriers_{}".format(group_name)
    )


def get_group_lhs_and_rhs_loc_tech_carriers(backend_model, group_name):
    lhs_loc_tech_carriers = getattr(
        backend_model, "group_constraint_loc_tech_carriers_{}_lhs".format(group_name)
    )
    rhs_loc_tech_carriers = getattr(
        backend_model, "group_constraint_loc_tech_carriers_{}_rhs".format(group_name)
    )
    return lhs_loc_tech_carriers, rhs_loc_tech_carriers


def demand_share_constraint_rule(backend_model, group_name, what):
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
    share = get_param(backend_model, "group_demand_share_{}".format(what), (group_name))

    if invalid(share):
        return return_noconstraint("demand_share", group_name)
    else:
        (
            lhs_loc_tech_carriers,
            rhs_loc_tech_carriers,
        ) = get_group_lhs_and_rhs_loc_tech_carriers(backend_model, group_name)

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
    backend_model, group_name, timestep, what
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
        (group_name),
    )

    if invalid(share):
        return return_noconstraint("demand_share_per_timestep", group_name)
    else:
        (
            lhs_loc_tech_carriers,
            rhs_loc_tech_carriers,
        ) = get_group_lhs_and_rhs_loc_tech_carriers(backend_model, group_name)

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
    backend_model, group_name, loc_tech_carrier, timestep, sense, scale
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
    (
        _,
        rhs_loc_tech_carriers,
    ) = get_group_lhs_and_rhs_loc_tech_carriers(backend_model, group_name)
    lhs = backend_model.carrier_prod[loc_tech_carrier, timestep]
    rhs = (
        -1
        * scale
        * sum(
            backend_model.required_resource[
                rhs_loc_tech_carrier.rsplit("::", 1)[0], timestep
            ]
            for rhs_loc_tech_carrier in rhs_loc_tech_carriers
        )
        * backend_model.demand_share_per_timestep_decision[loc_tech_carrier]
    )

    return equalizer(lhs, rhs, sense)


def demand_share_per_timestep_decision_sum_constraint_rule(backend_model, group_name):
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
        backend_model, "group_demand_share_per_timestep_decision", (group_name)
    )

    # If inf was given that means that we don't limit the total share
    if invalid(share_of_carrier_demand) or np.isinf(po.value(share_of_carrier_demand)):
        return return_noconstraint("demand_share_per_timestep_decision_sum", group_name)
    else:
        lhs_loc_tech_carriers, _ = get_group_lhs_and_rhs_loc_tech_carriers(
            backend_model, group_name
        )

        return share_of_carrier_demand == sum(
            backend_model.demand_share_per_timestep_decision[loc_tech_carrier]
            for loc_tech_carrier in lhs_loc_tech_carriers
        )


def carrier_prod_share_constraint_rule(backend_model, group_name, what):
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
        backend_model, "group_carrier_prod_share_{}".format(what), (group_name)
    )

    if invalid(share):
        return return_noconstraint("carrier_prod_share", group_name)
    else:
        (
            lhs_loc_tech_carriers,
            rhs_loc_tech_carriers,
        ) = get_group_lhs_and_rhs_loc_tech_carriers(backend_model, group_name)

        lhs = sum(
            backend_model.carrier_prod[loc_tech_carrier, timestep]
            for loc_tech_carrier in lhs_loc_tech_carriers
            for timestep in backend_model.timesteps
        )
        rhs = share * sum(
            backend_model.carrier_prod[loc_tech_carrier, timestep]
            for loc_tech_carrier in rhs_loc_tech_carriers
            for timestep in backend_model.timesteps
        )

        return equalizer(lhs, rhs, what)


def carrier_prod_share_per_timestep_constraint_rule(
    backend_model, group_name, timestep, what
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
        (group_name),
    )

    if invalid(share):
        return return_noconstraint("carrier_prod_share_per_timestep", group_name)
    else:
        (
            lhs_loc_tech_carriers,
            rhs_loc_tech_carriers,
        ) = get_group_lhs_and_rhs_loc_tech_carriers(backend_model, group_name)

        lhs = sum(
            backend_model.carrier_prod[loc_tech_carrier, timestep]
            for loc_tech_carrier in lhs_loc_tech_carriers
        )
        rhs = share * sum(
            backend_model.carrier_prod[loc_tech_carrier, timestep]
            for loc_tech_carrier in rhs_loc_tech_carriers
        )

        return equalizer(lhs, rhs, what)


def net_import_share_constraint_rule(backend_model, group_name, what):
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
    share = get_param(backend_model, f"group_net_import_share_{what}", (group_name))

    if invalid(share):
        return return_noconstraint("net_import_share", group_name)
    else:
        (
            lhs_loc_tech_carriers,
            rhs_loc_tech_carriers,
        ) = get_group_lhs_and_rhs_loc_tech_carriers(backend_model, group_name)

        lhs = sum(
            (
                backend_model.carrier_prod[loc_tech_carrier, timestep]
                + backend_model.carrier_con[loc_tech_carrier, timestep]
            )
            for loc_tech_carrier in lhs_loc_tech_carriers
            for timestep in backend_model.timesteps
        )
        rhs = -share * sum(
            backend_model.carrier_con[loc_tech_carrier, timestep]
            for loc_tech_carrier in rhs_loc_tech_carriers
            for timestep in backend_model.timesteps
        )
        return equalizer(lhs, rhs, what)


def carrier_prod_constraint_rule(backend_model, group_name, what):
    """
    Enforces carrier_prod for groups of technologies and locations,
    as a sum over the entire model period.

    .. container:: scrolling-wrapper

        .. math::

            \\sum_{loc::tech::carrier \\in given\\_group, timestep \\in timesteps} carrier_{prod}(loc::tech::carrier, timestep) \\leq carrier_prod_max

    """
    limit = get_param(backend_model, f"group_carrier_prod_{what}", (group_name))

    if invalid(limit):
        return return_noconstraint("carrier_prod", group_name)
    else:
        lhs_loc_tech_carriers = get_group_lhs_loc_tech_carriers(
            backend_model, group_name
        )

        lhs = sum(
            backend_model.carrier_prod[loc_tech_carrier, timestep]
            for loc_tech_carrier in lhs_loc_tech_carriers
            for timestep in backend_model.timesteps
        )
        return equalizer(lhs, limit, what)


def carrier_con_constraint_rule(backend_model, constraint_group, what):
    """
    Enforces carrier_con for groups of technologies and locations,
    as a sum over the entire model period. limits are always negative, so min/max
    is relative to zero (i.e. min = -1 means carrier_con must be -1 or less)

    .. container:: scrolling-wrapper

        .. math::

            \\sum_{loc::tech::carrier \\in given\\_group, timestep \\in timesteps} carrier_{con}(loc::tech::carrier, timestep) \\geq carrier_con_max

    """
    limit = get_param(
        backend_model, "group_carrier_con_{}".format(what), (constraint_group)
    )

    if invalid(limit):
        return return_noconstraint("carrier_con", constraint_group)
    else:
        lhs_loc_tech_carriers = get_group_lhs_loc_tech_carriers(
            backend_model, constraint_group
        )

        lhs = sum(
            backend_model.carrier_con[loc_tech_carrier, timestep]
            for loc_tech_carrier in lhs_loc_tech_carriers
            for timestep in backend_model.timesteps
        )

        return equalizer(limit, lhs, what)


def energy_cap_share_constraint_rule(backend_model, constraint_group, what):
    """
    Enforces shares of energy_cap for groups of technologies and locations. The
    share is relative to ``supply``, ``supply_plus``, ``conversion``, and ``conversion_plus``
    technologies only.

    .. container:: scrolling-wrapper

        .. math::

            \\sum_{loc::tech \\in given\\_group} energy_{cap}(loc::tech) \\leq
            share \\times \\sum_{loc::tech \\in loc\\_tech\\_supply\\_all \\in given\\_locations} energy_{cap}(loc::tech)
    """
    share = get_param(
        backend_model, "group_energy_cap_share_{}".format(what), (constraint_group)
    )

    if invalid(share):
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

    if invalid(threshold):
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
    Enforce upper and lower bounds of storage_cap for groups of technologies and locations.

    .. container:: scrolling-wrapper

        .. math::

            \\sum_{loc::tech \\in given\\_group} storage_{cap}(loc::tech) \\leq storage\\_cap\\_max\\\\

            \\sum_{loc::tech \\in given\\_group} storage_{cap}(loc::tech) \\geq storage\\_cap\\_min

    """
    threshold = get_param(
        backend_model, "group_storage_cap_{}".format(what), (constraint_group)
    )

    if invalid(threshold):
        return return_noconstraint("storage_cap", constraint_group)
    else:
        lhs_loc_techs = getattr(
            backend_model, "group_constraint_loc_techs_{}".format(constraint_group)
        )

        lhs = sum(backend_model.storage_cap[loc_tech] for loc_tech in lhs_loc_techs)
        rhs = threshold

        return equalizer(lhs, rhs, what)


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
    if invalid(cost_cap):
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

    if invalid(cost_cap):
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

    if invalid(cost_cap):
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

    if invalid(threshold):
        return return_noconstraint("resource_area", constraint_group)
    else:
        lhs_loc_techs = getattr(
            backend_model, "group_constraint_loc_techs_{}".format(constraint_group)
        )

        lhs = sum(backend_model.resource_area[loc_tech] for loc_tech in lhs_loc_techs)
        rhs = threshold

        return equalizer(lhs, rhs, what)
