using JuMP; using NCDatasets; using AxisArrays; using Util;

function load_cost_constraints(model_dict)
    sets = model_dict["sets"]
    constraint_dict = Dict()

    if haskey(sets, "loc_techs_cost_constraint")
        constraint_dict["cost_constraint"] = (
            build_constraint(["costs", "loc_techs_cost_constraint"],
            "cost", model_dict)
        )
    end

    if haskey(sets, "loc_techs_cost_investment_constraint")
        model_dict["expressions"]["cost_investment_rhs"] = @expression(
            model_dict["backend_model"], [costs=sets["costs"],
            loc_techs_investment=sets["loc_techs_investment_cost"]],
            0
        )

        constraint_dict["cost_investment_constraint"] = (
            build_constraint(["costs", "loc_techs_cost_investment_constraint"],
            "cost_investment", model_dict)
        )
    end

    if haskey(sets, "loc_techs_cost_var_constraint")
        model_dict["expressions"]["cost_var_rhs"] = @expression(
            model_dict["backend_model"], [costs=sets["costs"],
            loc_techs_om_var=sets["loc_techs_om_cost"],
            timesteps=sets["timesteps"]],
            0
        )

        constraint_dict["cost_var_constraint"] = (
            build_constraint(["costs", "loc_techs_cost_var_constraint", "timesteps"],
            "cost_var", model_dict)
        )

    end

    return constraint_dict
end

function cost_constraint_rule(backend_model, set_indices, model_dict)
    """
    Combine investment and time varying costs into one cost per technology
    """
    cost, loc_tech = set_indices
    variables = model_dict["variables"]
    sets = model_dict["sets"]

    if loc_tech_is_in(sets, loc_tech, "loc_techs_investment_cost")
        cost_investment = variables["cost_investment"][cost, loc_tech]
    else
        cost_investment = 0
    end

    if loc_tech_is_in(sets, loc_tech, "loc_techs_om_cost")
        cost_var = sum(variables["cost_var"][cost, loc_tech, timestep]
                       for timestep in sets["timesteps"])
    else
        cost_var = 0
    end

    return @constraint(backend_model,
        variables["cost"][cost, loc_tech] == cost_investment + cost_var
    )

end

function cost_investment_constraint_rule(backend_model, set_indices, model_dict)
    """
    Calculate costs from capacity decision variables
    """
    cost, loc_tech = set_indices
    variables = model_dict["variables"]
    sets = model_dict["sets"]

    function _get_investment_cost(capacity_decision_variable, calliope_set)
        """
        Conditionally add investment costs, if the relevant set of technologies
        exists. Both inputs are strings.
        """
        if loc_tech_is_in(sets, loc_tech, calliope_set)
            _cost = (
                variables[capacity_decision_variable][loc_tech] *
                get_param(model_dict, string("cost_", capacity_decision_variable),
                          [loc_tech, cost])
            )
            return _cost
        else
            return 0
        end
    end

    cost_energy_cap = (variables["energy_cap"][loc_tech]
        * get_param(model_dict, "cost_energy_cap", [loc_tech, cost]))

    cost_storage_cap = _get_investment_cost("storage_cap", "loc_techs_store")
    cost_resource_cap = _get_investment_cost("resource_cap", "loc_techs_supply_plus")
    cost_resource_area = _get_investment_cost("resource_area", "loc_techs_area")

    cost_om_annual_investment_fraction = get_param(model_dict, "cost_om_annual_investment_fraction", [loc_tech, cost])
    cost_om_annual = get_param(model_dict, "cost_om_annual", [loc_tech, cost])

    ts_weight = get_timestep_weight(model_dict)
    depreciation_rate = model_dict["parameters"]["cost_depreciation_rate"][loc_tech, cost]

    cost_con = (
        depreciation_rate * ts_weight *
        (cost_energy_cap + cost_storage_cap + cost_resource_cap +
         cost_resource_area)
    )

    # Transmission technologies exist at two locations, thus their cost is divided by 2
    if loc_tech_is_in(sets, loc_tech, "loc_techs_transmission")
            cost_con = cost_con / 2
    end
    cost_fractional_om = cost_om_annual_investment_fraction * cost_con
    cost_fixed_om = cost_om_annual * variables["energy_cap"][loc_tech] * ts_weight

    append!(model_dict["expressions"]["cost_investment_rhs"][cost, loc_tech],
        cost_fractional_om + cost_fixed_om + cost_con
    )

    return @constraint(backend_model,
        variables["cost_investment"][cost, loc_tech] ==
        model_dict["expressions"]["cost_investment_rhs"][cost, loc_tech]
    )
end

function cost_var_constraint_rule(backend_model, set_indices, model_dict)
    """
    Calculate costs from time-varying decision variables
    """
    cost, loc_tech, timestep = set_indices
    variables = model_dict["variables"]
    sets = model_dict["sets"]

    cost_om_prod = get_param(model_dict, "cost_om_prod", [loc_tech, cost], timestep)
    cost_om_con = get_param(model_dict, "cost_om_con", [loc_tech, cost], timestep)
    weight = model_dict["parameters"]["timestep_weights"][timestep]

    loc_tech_carrier = model_dict["parameters"]["lookup_loc_techs"][loc_tech]

    if cost_om_prod !== nothing
        cost_prod = cost_om_prod * weight * variables["carrier_prod"][loc_tech_carrier, timestep]
    else
        cost_prod = 0
    end

    if loc_tech_is_in(sets, loc_tech, "loc_techs_supply_plus") && cost_om_con !== nothing
        resource_eff = get_param(model_dict, "resource_eff", [loc_tech], timestep)
        if resource_eff > 0  # In case resource_eff is zero, to avoid an infinite value
            # Dividing by r_eff here so we get the actual r used, not the r
            # moved into storage...
            cost_con = cost_om_con * weight * (variables["resource_con"][loc_tech, timestep] / resource_eff)
        else
            cost_con = 0
        end
    elseif loc_tech_is_in(sets, loc_tech, "loc_techs_supply") && cost_om_con > 0
        energy_eff = get_param(model_dict, "energy_eff", [loc_tech], timestep)
        if energy_eff > 0  # in case energy_eff is zero, to avoid an infinite value
            cost_con = cost_om_con * weight * (
                variables["carrier_prod"][loc_tech_carrier, timestep] / energy_eff
            )
        else
            cost_con = 0
        end
    else
        cost_con = 0
    end

    model_dict["expressions"]["cost_var_rhs"][cost, loc_tech, timestep] = (
        cost_prod + cost_con
    )

    return @constraint(
        backend_model, variables["cost_var"][cost, loc_tech, timestep] ==
        model_dict["expressions"]["cost_var_rhs"][cost, loc_tech, timestep]
    )
end