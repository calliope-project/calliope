"""
Copyright (C) since 2013 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

policy.py
~~~~~~~~~

Policy constraints.

"""

import pyomo.core as po  # pylint: disable=import-error

ORDER = 10  # order in which to invoke constraints relative to other constraint files


def load_constraints(backend_model):
    sets = backend_model.__calliope_model_data["sets"]
    run_config = backend_model.__calliope_run_config

    if "techlists_group_share_energy_cap_min_constraint" in sets:
        backend_model.group_share_energy_cap_min_constraint = po.Constraint(
            backend_model.techlists_group_share_energy_cap_min_constraint,
            ["min"],
            rule=group_share_energy_cap_constraint_rule,
        )

    if "techlists_group_share_energy_cap_max_constraint" in sets:
        backend_model.group_share_energy_cap_max_constraint = po.Constraint(
            backend_model.techlists_group_share_energy_cap_max_constraint,
            ["max"],
            rule=group_share_energy_cap_constraint_rule,
        )

    if "techlists_group_share_energy_cap_equals_constraint" in sets:
        backend_model.group_share_energy_cap_equals_constraint = po.Constraint(
            backend_model.techlists_group_share_energy_cap_equals_constraint,
            ["equals"],
            rule=group_share_energy_cap_constraint_rule,
        )

    if "techlists_carrier_group_share_carrier_prod_min_constraint" in sets:
        backend_model.group_share_carrier_prod_min_constraint = po.Constraint(
            backend_model.techlists_carrier_group_share_carrier_prod_min_constraint,
            ["min"],
            rule=group_share_carrier_prod_constraint_rule,
        )

    if "techlists_carrier_group_share_carrier_prod_max_constraint" in sets:
        backend_model.group_share_carrier_prod_max_constraint = po.Constraint(
            backend_model.techlists_carrier_group_share_carrier_prod_max_constraint,
            ["max"],
            rule=group_share_carrier_prod_constraint_rule,
        )

    if "techlists_carrier_group_share_carrier_prod_equals_constraint" in sets:
        backend_model.group_share_carrier_prod_equals_constraint = po.Constraint(
            backend_model.techlists_carrier_group_share_carrier_prod_equals_constraint,
            ["equals"],
            rule=group_share_carrier_prod_constraint_rule,
        )

    if "carriers_reserve_margin_constraint" in sets and run_config["mode"] != "operate":
        backend_model.reserve_margin_constraint = po.Constraint(
            backend_model.carriers_reserve_margin_constraint,
            rule=reserve_margin_constraint_rule,
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


# FIXME: docstring should show all variants: \geq, =, and \leq
def group_share_energy_cap_constraint_rule(backend_model, techlist, what):
    """
    Enforce shares in energy_cap for groups of technologies. Applied
    to ``supply`` and ``supply_plus`` technologies only.

    .. container:: scrolling-wrapper

        .. math::

            \\sum_{loc::tech \\in given\\_group} energy_{cap}(loc::tech) =
            fraction \\times \\sum_{loc::tech \\in loc\\_techs\\_supply \\or loc\\_techs\\_supply\\_plus} energy_{cap}(loc::tech)

    """
    model_data_dict = backend_model.__calliope_model_data["data"]
    sets = backend_model.__calliope_model_data["sets"]
    fraction = model_data_dict["group_share_energy_cap_{}".format(what)][techlist]

    if "loc_techs_supply_plus" in sets:
        rhs_loc_techs = (
            backend_model.loc_techs_supply | backend_model.loc_techs_supply_plus
        )
        lhs_loc_techs = [
            i
            for i in backend_model.loc_techs_supply
            | backend_model.loc_techs_supply_plus
            if i.split("::")[1] in techlist.split(",")
        ]
    else:
        rhs_loc_techs = backend_model.loc_techs_supply
        lhs_loc_techs = [
            i
            for i in backend_model.loc_techs_supply
            if i.split("::")[1] in techlist.split(",")
        ]

    rhs = fraction * sum(
        backend_model.energy_cap[loc_tech] for loc_tech in rhs_loc_techs
    )
    lhs = sum(backend_model.energy_cap[loc_tech] for loc_tech in lhs_loc_techs)

    return equalizer(lhs, rhs, what)


def group_share_carrier_prod_constraint_rule(backend_model, techlist_carrier, what):
    """
    Enforce shares in carrier_prod for groups of technologies. Applied
    to ``loc_tech_carriers_supply_conversion_all``, which includes supply,
    supply_plus, conversion, and conversion_plus.

    .. container:: scrolling-wrapper

        .. math::

            \\sum_{loc::tech::carrier \\in given\\_group, timestep \\in timesteps} carrier_{prod}(loc::tech::carrier, timestep) =
            fraction \\times \\sum_{loc::tech:carrier \\in loc\\_tech\\_carriers\\_supply\\_all, timestep\\in timesteps} carrier_{prod}(loc::tech::carrier, timestep)

    """
    model_data_dict = backend_model.__calliope_model_data["data"]
    techlist, carrier = techlist_carrier.split("::")
    fraction = model_data_dict["group_share_carrier_prod_{}".format(what)][
        (carrier, techlist)
    ]

    rhs_loc_tech_carriers = [
        i
        for i in backend_model.loc_tech_carriers_supply_conversion_all
        if i.split("::")[-1] == carrier
    ]
    lhs_loc_tech_carriers = [
        i
        for i in backend_model.loc_tech_carriers_supply_conversion_all
        if i.split("::")[1] in techlist.split(",") and i.split("::")[-1] == carrier
    ]
    rhs = fraction * sum(
        backend_model.carrier_prod[loc_tech_carrier, timestep]
        for loc_tech_carrier in rhs_loc_tech_carriers
        for timestep in backend_model.timesteps
    )
    lhs = sum(
        backend_model.carrier_prod[loc_tech_carrier, timestep]
        for loc_tech_carrier in lhs_loc_tech_carriers
        for timestep in backend_model.timesteps
    )

    return equalizer(lhs, rhs, what)


def reserve_margin_constraint_rule(backend_model, carrier):
    """
    Enforces a system reserve margin per carrier.

    .. container:: scrolling-wrapper

        .. math::

            \\sum_{loc::tech::carrier \\in loc\\_tech\\_carriers\\_supply\\_all} energy_{cap}(loc::tech::carrier, timestep_{max\\_demand})
            \\geq
            \\sum_{loc::tech::carrier \\in loc\\_tech\\_carriers\\_demand} carrier_{con}(loc::tech::carrier, timestep_{max\\_demand})
            \\times -1 \\times \\frac{1}{time\\_resolution_{max\\_demand}}
            \\times (1 + reserve\\_margin)

    """
    model_data_dict = backend_model.__calliope_model_data["data"]

    reserve_margin = model_data_dict["reserve_margin"][carrier]
    max_demand_timestep = model_data_dict["max_demand_timesteps"][carrier]
    max_demand_time_res = backend_model.timestep_resolution[max_demand_timestep]

    return sum(  # Sum all supply capacity for this carrier
        backend_model.energy_cap[loc_tech_carrier.rsplit("::", 1)[0]]
        for loc_tech_carrier in backend_model.loc_tech_carriers_supply_conversion_all
        if loc_tech_carrier.split("::")[-1] == carrier
    ) >= sum(  # Sum all demand for this carrier and timestep
        backend_model.carrier_con[loc_tech_carrier, max_demand_timestep]
        for loc_tech_carrier in backend_model.loc_tech_carriers_demand
        if loc_tech_carrier.split("::")[-1] == carrier
    ) * -1 * (
        1 / max_demand_time_res
    ) * (
        1 + reserve_margin
    )
