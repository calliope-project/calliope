
function load_export_constraints(model_dict)
    sets = model_dict["sets"]
    constraint_dict = Dict()
    backend_model = model_dict["backend_model"]

    if haskey(sets, "loc_carriers_update_system_balance_constraint")
        for loc_carrier=sets["loc_carriers_update_system_balance_constraint"], timestep=sets["timesteps"]
            update_system_balance_constraint(model_dict, loc_carrier, timestep)
        end
    end

    if haskey(sets, "loc_tech_carriers_export_balance_constraint")
        constraint_dict["export_balance_constraint"] = (
            build_constraint(["loc_tech_carriers_export_balance_constraint", "timesteps"],
            "export_balance", model_dict
            )
        )
    end

    if haskey(sets, "loc_techs_update_costs_var_constraint")
        for cost=sets["costs"], loc_tech=sets["loc_techs_update_costs_var_constraint"], timestep=sets["timesteps"]
            update_costs_var_constraint(model_dict, cost, loc_tech, timestep)
        end
    end

    if haskey(sets, "loc_tech_carriers_export_max_constraint")
        constraint_dict["export_max_constraint"] = (
            build_constraint(["loc_tech_carriers_export_max_constraint", "timesteps"],
            "export_max", model_dict
            )
        )
    end

    return constraint_dict
end


function update_system_balance_constraint(model_dict, loc_carrier, timestep)
    """
    Update system balance constraint (from energy_balance.py) to include export
    """
    system_balance = model_dict["expressions"]["system_balance"]
    sets = model_dict["sets"]
    prod, con, exporting = get_loc_tech_carriers(model_dict, loc_carrier)

    append!(system_balance[loc_carrier, timestep],
        -1 * sum(model_dict["variables"]["carrier_export"][loc_tech_carrier, timestep]
                 for loc_tech_carrier in exporting)
    )
    loc_carrier_index = findin(sets["loc_carriers"], [loc_carrier])
    timestep_index = findin(sets["loc_carriers"], [timestep])
    model_dict["constraints"]["system_balance_constraint"][loc_carrier_index, timestep_index] = (
        @constraint(model_dict["backend_model"],
            system_balance[loc_carrier, timestep] == 0)
    )
end


function export_balance_constraint_rule(backend_model, set_indices, model_dict)
    """
    Ensure no technology can "pass" its export capability to another technology
    with the same carrier_out, by limiting its export to the capacity of its production
    """
    loc_tech_carrier, timestep = set_indices
    return @constraint(backend_model,
        model_dict["variables"]["carrier_prod"][loc_tech_carrier, timestep] >=
        model_dict["variables"]["carrier_export"][loc_tech_carrier, timestep]
    )

end


function update_costs_var_constraint(model_dict, cost, loc_tech, timestep)
    """
    Update time varying cost constraint (from costs.py) to include export
    """
    parameters = model_dict["parameters"]
    sets = model_dict["sets"]
    cost_var_rhs = model_dict["expressions"]["cost_var_rhs"]
    loc_tech_carrier = parameters["lookup_loc_techs_export"][loc_tech]
    weight = parameters["timestep_weights"][timestep]

    cost_export = (
        get_param(model_dict, "cost_export", [loc_tech, cost], timestep)
        * model_dict["variables"]["carrier_prod"][loc_tech_carrier, timestep]
        * weight
    )

    loc_tech_index = findin(sets["loc_carriers"], [loc_tech])
    cost_index = findin(sets["loc_carriers"], [cost])
    timestep_index = findin(sets["loc_carriers"], [timestep])
    append!(model_dict["expressions"]["cost_var_rhs"][cost, loc_tech, timestep], cost_export)
    model_dict["constraints"]["cost_var_constraint"][cost_index, loc_tech_index, timestep_index] = (
        @constraint(model_dict["backend_model"],
            model_dict["variables"]["cost_var"][cost, loc_tech, timestep] ==
            model_dict["expressions"]["cost_var_rhs"][cost, loc_tech, timestep])
    )
end


function export_max_constraint_rule(backend_model, set_indices, model_dict)
    """
    Set maximum export. All exporting technologies.
    """
    loc_tech_carrier, timestep = set_indices
    loc_tech = get_loc_tech(loc_tech_carrier)
    sets = model_dict["sets"]
    variables = model_dict["variables"]

    if loc_tech_is_in(sets, loc_tech, "loc_techs_milp")
        operating_units = variables["operating_units"][loc_tech, timestep]
    else
        operating_units = 1
    end

    export_cap = get_param(model_dict, "export_cap", [loc_tech])

    return @constraint(backend_model,
        variables["carrier_export"][loc_tech_carrier, timestep] <=
        export_cap * operating_units
    )
end
