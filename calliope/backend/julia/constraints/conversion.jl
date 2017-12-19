using JuMP; using NCDatasets; using AxisArrays; using Util;

function load_conversion_constraints(model_dict)
    sets = model_dict["sets"]
    constraint_dict = Dict()

    constraint_dict["balance_conversion_constraint"] = (
        build_constraint(
            ["loc_techs_balance_conversion_constraint", "timesteps"],
            "balance_conversion", model_dict
        )
    )

    if haskey(sets, "loc_techs_cost_var_conversion_constraint")
        constraint_dict["cost_var_conversion_constraint"] = (
            build_constraint(
                ["costs", "loc_techs_cost_var_conversion_constraint", "timesteps"],
                "cost_var_conversion", model_dict
            )
        )
    end

    return constraint_dict
end


function balance_conversion_constraint_rule(backend_model, set_indices, model_dict)
    """
    Balance energy carrier consumption and production
    """
    loc_tech, timestep = set_indices
    parameters = model_dict["parameters"]
    variables = model_dict["variables"]

    loc_tech_carrier_out = parameters["lookup_loc_techs_conversion"][loc_tech, "out"]
    loc_tech_carrier_in = parameters["lookup_loc_techs_conversion"][loc_tech, "in"]

    energy_eff = get_param(model_dict, "energy_eff", [loc_tech], timestep)

    return @constraint(backend_model,
        variables["carrier_prod"][loc_tech_carrier_out, timestep] == -1 *
        variables["carrier_con"][loc_tech_carrier_in, timestep] * energy_eff
    )

end


function cost_var_conversion_constraint_rule(backend_model, set_indices, model_dict)
    """
    Add time-varying conversion technology costs
    """
    cost, loc_tech, timestep = set_indices
    parameters = model_dict["parameters"]
    variables = model_dict["variables"]
    expressions = model_dict["expressions"]

    weight = parameters["timestep_weights"][timestep]

    loc_tech_carrier_in = (
        parameters["lookup_loc_techs_conversion"][loc_tech, "in"]
    )

    loc_tech_carrier_out = (
        parameters["lookup_loc_techs_conversion"][loc_tech, "out"]
    )

    cost_om_prod = get_param(model_dict, "cost_om_prod", [loc_tech, cost], timestep)
    cost_om_con = get_param(model_dict, "cost_om_con", [loc_tech, cost], timestep)

    if cost_om_prod
        cost_prod = (cost_om_prod * weight *
            variables["carrier_prod"][loc_tech_carrier_out, timestep])
    else cost_prod = 0
    end

    if cost_om_con
        cost_con = (cost_om_con * weight *
            variables["carrier_con"][loc_tech_carrier_in, timestep])
    else cost_con = 0
    end

    expressions["cost_var_rhs"][cost, loc_tech, timestep] = cost_prod + cost_con

    return @constraint(backend_model,
        variables["cost_var"][cost, loc_tech, timestep] ==
        expressions["cost_var_rhs"][cost, loc_tech, timestep]
    )
end
