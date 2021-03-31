"""
Copyright (C) since 2013 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

dispatch.py
~~~~~~~~~~~~~~~~~

Energy dispatch constraints, limiting production/consumption to the capacities
of the technologies

"""

import pyomo.core as po  # pylint: disable=import-error

from calliope.backend.pyomo.util import (
    get_param,
    get_loc_tech,
    invalid,
    split_comma_list,
    get_timestep_weight,
)

ORDER = 10  # order in which to invoke constraints relative to other constraint files


def load_constraints(backend_model):
    sets = backend_model.__calliope_model_data["sets"]

    if "loc_tech_carrier_production_max_time_varying_constraint" in sets:
        backend_model.carrier_production_max_time_varying_constraint = po.Constraint(
            backend_model.loc_tech_carrier_production_max_time_varying_constraint,
            backend_model.timesteps,
            rule=carrier_production_max_time_varying_constraint_rule,
        )
    if "loc_techs_chp_extraction_cb_constraint" in sets:
        backend_model.chp_extraction_cb_constraint = po.Constraint(
            backend_model.loc_techs_chp_extraction_cb_constraint,
            backend_model.timesteps,
            rule=chp_extraction_cb_constraint_rule,
        )
    if "loc_techs_chp_extraction_cv_constraint" in sets:
        backend_model.chp_extraction_cv_constraint = po.Constraint(
            backend_model.loc_techs_chp_extraction_cv_constraint,
            backend_model.timesteps,
            rule=chp_extraction_cv_constraint_rule,
        )
    if "loc_techs_chp_extraction_p2h_constraint" in sets:
        backend_model.chp_extraction_p2h_constraint = po.Constraint(
            backend_model.loc_techs_chp_extraction_p2h_constraint,
            backend_model.timesteps,
            rule=chp_extraction_p2h_constraint_rule,
        )
    if "loc_tech_carriers_link_con_to_prod_constraint" in sets:
        backend_model.link_con_to_prod_constraint = po.Constraint(
            backend_model.loc_tech_carriers_link_con_to_prod_constraint,
            backend_model.timesteps,
            rule=link_con_to_prod_constraint_rule,
        )
    if "loc_tech_carriers_capacity_factor_min_constraint" in sets:
        backend_model.capacity_factor_min_constraint = po.Constraint(
            backend_model.loc_tech_carriers_capacity_factor_min_constraint,
            rule=capacity_factor_min_constraint_rule,
        )
    if "loc_tech_carriers_capacity_factor_max_constraint" in sets:
        backend_model.capacity_factor_max_constraint = po.Constraint(
            backend_model.loc_tech_carriers_capacity_factor_max_constraint,
            rule=capacity_factor_max_constraint_rule,
        )
    if "loc_techs_net_transfer_ratio_constraint" in sets:
        backend_model.net_transfer_ratio_constraint = po.Constraint(
            backend_model.loc_techs_net_transfer_ratio_constraint,
            rule=net_transfer_ratio_constraint_rule,
        )
    for sense in ["min", "max", "equals"]:
        if f"loc_tech_carriers_carrier_prod_per_week_{sense}_constraint" in sets:
            setattr(
                backend_model,
                f"carrier_prod_per_week_{sense}_constraint",
                po.Constraint(
                    getattr(
                        backend_model,
                        f"loc_tech_carriers_carrier_prod_per_week_{sense}_constraint",
                    ),
                    backend_model.weeks,
                    [sense],
                    rule=carrier_prod_per_week_constraint_rule,
                ),
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


def carrier_production_max_time_varying_constraint_rule(
    backend_model, loc_tech, timestep
):
    """
    Set maximum carrier production for technologies with time varying maximum capacity
    """
    model_data_dict = backend_model.__calliope_model_data["data"]
    timestep_resolution = backend_model.timestep_resolution[timestep]
    loc_tech_carriers_out = split_comma_list(
        model_data_dict["lookup_loc_techs_conversion_plus"]["out", loc_tech]
    )

    energy_cap_max = backend_model.energy_cap_max_time_varying[loc_tech, timestep]

    carrier_prod = sum(
        backend_model.carrier_prod[loc_tech_carrier, timestep]
        for loc_tech_carrier in loc_tech_carriers_out
    )
    return carrier_prod <= (
        backend_model.energy_cap[loc_tech] * timestep_resolution * energy_cap_max
    )


def chp_extraction_cb_constraint_rule(backend_model, loc_tech, timestep):
    """
    Set backpressure line for CHP plants with extraction/condensing turbine
    """
    model_data_dict = backend_model.__calliope_model_data
    loc_tech_carrier_out = model_data_dict["data"]["lookup_loc_techs_conversion_plus"][
        ("out", loc_tech)
    ]
    loc_tech_carrier_out_2 = model_data_dict["data"][
        "lookup_loc_techs_conversion_plus"
    ][("out_2", loc_tech)]

    power_to_heat_ratio = get_param(backend_model, "cb", (loc_tech))

    return backend_model.carrier_prod[loc_tech_carrier_out, timestep] >= (
        backend_model.carrier_prod[loc_tech_carrier_out_2, timestep]
        * power_to_heat_ratio
    )


def chp_extraction_cv_constraint_rule(backend_model, loc_tech, timestep):
    """
    Set extraction line for CHP plants with extraction/condensing turbine
    """
    model_data_dict = backend_model.__calliope_model_data
    loc_tech_carrier_out = model_data_dict["data"]["lookup_loc_techs_conversion_plus"][
        ("out", loc_tech)
    ]
    loc_tech_carrier_out_2 = model_data_dict["data"][
        "lookup_loc_techs_conversion_plus"
    ][("out_2", loc_tech)]

    power_loss_factor = get_param(backend_model, "cv", (loc_tech))

    return backend_model.carrier_prod[loc_tech_carrier_out, timestep] <= (
        backend_model.energy_cap[loc_tech]
        - backend_model.carrier_prod[loc_tech_carrier_out_2, timestep]
        * power_loss_factor
    )


def chp_extraction_p2h_constraint_rule(backend_model, loc_tech, timestep):
    """
    Set power-to-heat tail for CHPs that allow trading off power output for heat
    """
    model_data_dict = backend_model.__calliope_model_data
    loc_tech_carrier_out = model_data_dict["data"]["lookup_loc_techs_conversion_plus"][
        ("out", loc_tech)
    ]
    loc_tech_carrier_out_2 = model_data_dict["data"][
        "lookup_loc_techs_conversion_plus"
    ][("out_2", loc_tech)]

    power_to_heat_ratio = get_param(backend_model, "cb", loc_tech)
    energy_cap_ratio = get_param(
        backend_model, "energy_cap_ratio", ("out_2", loc_tech_carrier_out_2)
    )
    slope = power_to_heat_ratio / (energy_cap_ratio - 1)
    return backend_model.carrier_prod[loc_tech_carrier_out, timestep] <= (
        slope
        * (
            backend_model.energy_cap[loc_tech] * energy_cap_ratio
            - backend_model.carrier_prod[loc_tech_carrier_out_2, timestep]
        )
    )


def link_con_to_prod_constraint_rule(backend_model, loc_tech_carrier, timestep):
    """
    Force the carrier consumption of a specific sotrage technology to only come
    from production of a specific set of other technologies
    """
    loc, tech, carrier = loc_tech_carrier.split("::")
    loc_tech_carriers_prod = getattr(backend_model, f"link_{loc}_{tech}_to_prod")

    return -1 * backend_model.carrier_con[loc_tech_carrier, timestep] <= sum(
        backend_model.carrier_prod[loc_tech_carrier_prod, timestep]
        for loc_tech_carrier_prod in loc_tech_carriers_prod
    )


def capacity_factor_min_constraint_rule(backend_model, loc_tech_carrier):
    """
    If there is capacity of a technology, force the annual capacity factor to be
    at least a certain amount
    """
    loc_tech = get_loc_tech(loc_tech_carrier)
    capacity_factor = get_param(backend_model, "capacity_factor_min", (loc_tech))
    return (
        sum(
            backend_model.carrier_prod[loc_tech_carrier, timestep]
            * backend_model.timestep_weights[timestep]
            for timestep in backend_model.timesteps
        )
        >= backend_model.energy_cap[loc_tech]
        * capacity_factor
        * get_timestep_weight(backend_model)
        * 8760
    )


def capacity_factor_max_constraint_rule(backend_model, loc_tech_carrier):
    """
    If there is capacity of a technology, force the annual capacity factor to be
    at most a certain amount
    """
    loc_tech = get_loc_tech(loc_tech_carrier)
    capacity_factor = get_param(backend_model, "capacity_factor_max", (loc_tech))
    return (
        sum(
            backend_model.carrier_prod[loc_tech_carrier, timestep]
            * backend_model.timestep_weights[timestep]
            for timestep in backend_model.timesteps
        )
        <= backend_model.energy_cap[loc_tech]
        * capacity_factor
        * get_timestep_weight(backend_model)
        * 8760
    )


def net_transfer_ratio_constraint_rule(backend_model, loc_tech):
    """
    Set the exact net transfer along a link,
    throughout the year to a certain ratio of sum(prod)/sum(con)
    i.e. net transfer of 1 means that sum(prod) == sum(con)
    i.e. net transfer of 0.5 means that sum(prod) == 0.5 * sum(con)
    'consumption' on a line is taken to actually be the 'production' on the line
    in the remote location, i.e. taking account of transmission losses to the remote.
    """

    net_transfer = get_param(backend_model, "net_transfer_ratio", (loc_tech))

    loc_tech_carrier = backend_model.lookup_loc_techs[loc_tech].value
    loc_tech_remote = backend_model.lookup_remotes[loc_tech].value
    loc_tech_carrier_remote = backend_model.lookup_loc_techs[loc_tech_remote].value

    prod = sum(
        backend_model.carrier_prod[loc_tech_carrier, timestep]
        * backend_model.timestep_weights[timestep]
        for timestep in backend_model.timesteps
    )
    prod_remote = sum(
        backend_model.carrier_prod[loc_tech_carrier_remote, timestep]
        * backend_model.timestep_weights[timestep]
        for timestep in backend_model.timesteps
    )

    return prod == prod_remote * net_transfer


def carrier_prod_per_week_constraint_rule(backend_model, loc_tech_carrier, week, what):
    """
    Set the min/max amount of carrier consumption (relative to annual consumption)
    for a specific loc tech that must take place in a given calender week in the model
    """

    prod = backend_model.carrier_prod
    prod_total = sum(
        prod[loc_tech_carrier, timestep] for timestep in backend_model.timesteps
    )
    prod_week = sum(
        prod[loc_tech_carrier, timestep]
        for timestep in backend_model.timesteps
        if backend_model.week_numbers[timestep] == week
    )
    if "timesteps" in [
        i.name
        for i in getattr(
            backend_model, f"carrier_prod_per_week_{what}"
        )._index.subsets()
    ]:
        prod_fraction = sum(
            get_param(
                backend_model,
                f"carrier_prod_per_week_{what}",
                (loc_tech_carrier, timestep),
            )
            for timestep in backend_model.timesteps
            if backend_model.week_numbers[timestep] == week
        )
    else:
        prod_fraction = get_param(
            backend_model, f"carrier_prod_per_week_{what}", (loc_tech_carrier)
        )

    return equalizer(prod_week, prod_total * prod_fraction, what)
