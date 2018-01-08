using JuMP; using NCDatasets; using AxisArrays; using Util;

function load_conversion_plus_constraints(model_dict)
    sets = model_dict["sets"]
    constraint_dict = Dict()

    constraint_dict["balance_conversion_plus_primary_constraint"] = (
        build_constraint(
            ["loc_techs_balance_conversion_plus_primary_constraint",
             "timesteps"], "balance_conversion_plus_primary", model_dict
        )
    )

    constraint_dict["carrier_production_max_conversion_plus_constraint"] = (
        build_constraint(
            ["loc_techs_carrier_production_max_conversion_plus_constraint",
             "timesteps"], "carrier_production_max_conversion_plus", model_dict
        )
    )

    if haskey(sets, "loc_techs_carrier_production_min_conversion_plus_constraint")
        constraint_dict["carrier_production_min_conversion_plus_constraint"] = (
            build_constraint(
                ["loc_techs_carrier_production_min_conversion_plus_constraint",
                "timesteps"], "carrier_production_min_conversion_plus", model_dict
            )
        )
    end

    if haskey(sets, "loc_techs_cost_var_conversion_plus_constraint")
        constraint_dict["cost_var_conversion_plus_constraint"] = (
            build_constraint(
                ["costs", "loc_techs_cost_var_conversion_plus_constraint", "timesteps"],
                "cost_var_conversion_plus", model_dict
            )
        )
    end

    if haskey(sets, "loc_techs_balance_conversion_plus_in_2_constraint")
        constraint_dict["balance_conversion_plus_in_2_constraint"] = (
            build_constraint(
                ["loc_techs_balance_conversion_plus_in_2_constraint", "timesteps"],
                "balance_conversion_plus_tiers", model_dict, tier="in_2"
            )
        )
    end

    if haskey(sets, "loc_techs_balance_conversion_plus_in_3_constraint")
        constraint_dict["balance_conversion_plus_in_3_constraint"] = (
            build_constraint(
                ["loc_techs_balance_conversion_plus_in_3_constraint", "timesteps"],
                "balance_conversion_plus_tiers", model_dict, tier="in_3"
            )
        )
    end

    if haskey(sets, "loc_techs_balance_conversion_plus_out_2_constraint")
        constraint_dict["balance_conversion_plus_out_2_constraint"] = (
            build_constraint(
                ["loc_techs_balance_conversion_plus_out_2_constraint", "timesteps"],
                "balance_conversion_plus_tiers", model_dict, tier="out_2"
            )
        )
    end

    if haskey(sets, "loc_techs_balance_conversion_plus_out_3_constraint")
        constraint_dict["balance_conversion_plus_out_3_constraint"] = (
            build_constraint(
                ["loc_techs_balance_conversion_plus_out_3_constraint", "timesteps"],
                "balance_conversion_plus_tiers", model_dict, tier="out_3"
            )
        )
    end

    return constraint_dict
end


function balance_conversion_plus_primary_constraint_rule(backend_model, set_indices, model_dict)
    """
    Balance energy carrier consumption and production for carrier_in and carrier_out
    """
    loc_tech, timestep = set_indices
    variables = model_dict["variables"]
    parameters = model_dict["parameters"]

    loc_tech_carriers_out = split_comma_list(
        parameters["lookup_loc_techs_conversion_plus"][loc_tech, "out"]
    )
    loc_tech_carriers_in = split_comma_list(
        parameters["lookup_loc_techs_conversion_plus"][loc_tech, "in"]
    )

    energy_eff = get_param(model_dict, "energy_eff", [loc_tech], timestep)

    carrier_prod = sum(
        variables["carrier_prod"][loc_tech_carrier, timestep] /
        get_param(model_dict, "carrier_ratios", [loc_tech_carrier, "out"])
        for loc_tech_carrier in loc_tech_carriers_out
    )
    carrier_con = sum(
        variables["carrier_con"][loc_tech_carrier, timestep] *
        get_param(model_dict, "carrier_ratios", [loc_tech_carrier, "in"])
        for loc_tech_carrier in loc_tech_carriers_in
    )

    return @constraint(backend_model, carrier_prod == -1 * carrier_con * energy_eff)
end


function carrier_production_max_conversion_plus_constraint_rule(backend_model, set_indices, model_dict)
    """
    Set maximum conversion_plus carrier production.
    """
    loc_tech, timestep = set_indices
    variables = model_dict["variables"]
    parameters = model_dict["parameters"]

    timestep_resolution = model_dict["parameters"]["timestep_resolution"][timestep]
    loc_tech_carriers_out = split_comma_list(
        parameters["lookup_loc_techs_conversion_plus"][loc_tech, "out"]
    )

    carrier_prod = sum(variables["carrier_prod"][loc_tech_carrier, timestep]
                 for loc_tech_carrier in loc_tech_carriers_out)

    return @constraint(backend_model, carrier_prod <= timestep_resolution * variables["energy_cap"][loc_tech])
end


function carrier_production_min_conversion_plus_constraint_rule(backend_model, set_indices, model_dict)
    """
    Set minimum conversion_plus carrier production.
    """
    loc_tech, timestep = set_indices
    variables = model_dict["variables"]
    parameters = model_dict["parameters"]

    timestep_resolution = model_dict["parameters"]["timestep_resolution"][timestep]
    min_use = get_param(model_dict, "energy_cap_min_use", [loc_tech], timestep)

    loc_tech_carriers_out = split_comma_list(
        parameters["lookup_loc_techs_conversion_plus"][loc_tech, "out"]
    )

    carrier_prod = sum(variables["carrier_prod"][loc_tech_carrier, timestep]
                       for loc_tech_carrier in loc_tech_carriers_out)

    return @constraint(backend_model, carrier_prod >=
        timestep_resolution * variables["energy_cap"][loc_tech] * min_use
    )
end


function cost_var_conversion_plus_constraint_rule(backend_model, set_indices, model_dict)
    """
    Add time-varying conversion_plus technology costs
    """
    cost, loc_tech, timestep = set_indices
    variables = model_dict["variables"]
    parameters = model_dict["parameters"]
    sets = model_dict["sets"]
    expressions = model_dict["expressions"]

    weight = parameters["timestep_weights"][timestep]

    loc_tech_carrier = parameters["lookup_primary_loc_tech_carriers"][loc_tech]

    var_cost = 0

    if loc_tech_carrier in sets["loc_tech_carriers_prod"]
        cost_om_prod = get_param(model_dict, "cost_om_prod", [loc_tech, cost], timestep)
        if cost_om_prod > 0
            var_cost += (
                cost_om_prod * weight *
                variables["carrier_prod"][loc_tech_carrier, timestep]
            )
        end
    end

    if loc_tech_carrier in sets["loc_tech_carriers_con"]
        cost_om_con = get_param(model_dict, "cost_om_con", [loc_tech, cost], timestep)
        if cost_om_con > 0
            var_cost += (
                cost_om_con * weight *
                variables["carrier_con"][loc_tech_carrier, timestep]
            )
        end
    end

    expressions["cost_var_rhs"][cost, loc_tech, timestep] = var_cost

    return @constraint(backend_model,
        variables["cost_var"][cost, loc_tech, timestep]
        == expressions["cost_var_rhs"][cost, loc_tech, timestep]
    )
end


function balance_conversion_plus_tiers_constraint_rule(backend_model, set_indices, model_dict, tier)
    """
    Force all carrier_in_2/carrier_in_3 and carrier_out_2/carrier_out_3 to follow
    carrier_in and carrier_out (respectively).
    """
    loc_tech, timestep = set_indices
    parameters = model_dict["parameters"]
    primary_tier, decision_variable = get_conversion_plus_io(model_dict, tier)

    loc_tech_carriers_1 = split_comma_list(
        parameters["lookup_loc_techs_conversion_plus"][loc_tech, primary_tier]
    )
    loc_tech_carriers_2 = split_comma_list(
        parameters["lookup_loc_techs_conversion_plus"][loc_tech, tier]
    )

    c_1 = sum(decision_variable[loc_tech_carrier, timestep]
        / get_param(model_dict, "carrier_ratios", [loc_tech_carrier, primary_tier])
        for loc_tech_carrier in loc_tech_carriers_1)
    c_2 = sum(decision_variable[loc_tech_carrier, timestep]
        / get_param(model_dict, "carrier_ratios", [loc_tech_carrier, tier])
        for loc_tech_carrier in loc_tech_carriers_2)
    c_min = parameters["carrier_ratios_min"][loc_tech, tier]

    return @constraint(backend_model, c_1 * c_min == c_2)
end