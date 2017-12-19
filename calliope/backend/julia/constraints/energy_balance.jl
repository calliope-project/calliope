using JuMP; using NCDatasets; using AxisArrays; using Util;

function load_energy_balance_constraints(model_dict)
    sets = model_dict["sets"]
    constraint_dict = Dict()

    model_dict["expressions"]["system_balance"] = @expression(
        model_dict["backend_model"],
        [loc_carrier=sets["loc_carriers"],
                       timestep=sets["timesteps"]], 0
    )

    constraint_dict["system_balance_constraint"] = (
        build_constraint(
            ["loc_carriers_system_balance_constraint", "timesteps"],
            "system_balance", model_dict
        )
    )


    if haskey(sets, "loc_techs_finite_resource_supply_plus")
        constraint_dict["resource_availability_supply_plus"] = (
            build_constraint(["loc_techs_finite_resource_supply_plus", "timesteps"],
                "resource_availability_supply_plus", model_dict)
        )
    end

    if haskey(sets, "loc_techs_finite_resource_supply")
        constraint_dict["balance_supply_constraint"] = (
            build_constraint(["loc_techs_finite_resource_supply", "timesteps"],
                "balance_supply", model_dict)
        )
    end

    if haskey(sets, "loc_techs_balance_demand_constraint")
        constraint_dict["balance_demand_constraint"] = (
            build_constraint(["loc_techs_balance_demand_constraint", "timesteps"],
                "balance_demand", model_dict)
        )
    end

    if haskey(sets, "loc_techs_balance_transmission_constraint")
        constraint_dict["balance_transmission_constraint"] = (
            build_constraint(["loc_techs_balance_transmission_constraint", "timesteps"],
                "balance_transmission", model_dict)
        )
    end

    if haskey(sets, "loc_techs_balance_supply_plus_constraint")
        constraint_dict["balance_supply_plus_constraint"] = (
            build_constraint(["loc_techs_balance_supply_plus_constraint", "timesteps"],
                "balance_supply_plus", model_dict)
        )
    end

    if haskey(sets, "loc_techs_balance_storage_constraint")
        constraint_dict["balance_storage_constraint"] = (
            build_constraint(["loc_techs_balance_storage_constraint", "timesteps"],
                "balance_storage", model_dict)
        )
    end

    return constraint_dict
end


function system_balance_constraint_rule(backend_model, set_indices, model_dict)
    """
    System balance ensures that, within each location, the production and
    consumption of each carrier is balanced.
    """
    loc_carrier, timestep = set_indices
    _prod, _con, exporting = get_loc_tech_carriers(model_dict, loc_carrier)
    variables = model_dict["variables"]
    expressions = model_dict["expressions"]
    expressions["system_balance"][loc_carrier, timestep] = (
        sum(variables["carrier_prod"][loc_tech_carrier, timestep]
            for loc_tech_carrier in _prod) +
        sum(variables["carrier_con"][loc_tech_carrier, timestep]
            for loc_tech_carrier in _con)
    )

    return @constraint(backend_model,
        expressions["system_balance"][loc_carrier, timestep] == 0
    )
end

# Supply Balance
function balance_supply_constraint_rule(backend_model, set_indices, model_dict)
    loc_tech, timestep = set_indices
    variables = model_dict["variables"]
    sets = model_dict["sets"]
    loc_tech_carrier = model_dict["parameters"]["lookup_loc_techs"][loc_tech]


    energy_eff = get_param(model_dict, "energy_eff", [loc_tech], timestep)

    if energy_eff == 0
        energy_eff = 0
    else
        energy_eff = 1 / get_param(model_dict, "energy_eff", [loc_tech], timestep)
    end

    @expression(backend_model, prod,
        energy_eff * variables["carrier_prod"][loc_tech_carrier, timestep]
    )

    @expression(backend_model, available_resource,
        get_param(model_dict, "resource", [loc_tech], timestep) *
        get_param(model_dict, "resource_scale", [loc_tech], timestep)
    )

    if loc_tech_is_in(sets, loc_tech, "loc_techs_area")
        available_resource *= variables["resource_area"][loc_tech]
    end

    # creating the constraint just requires putting together the affine expressions
    if get_param(model_dict, "force_resource", [loc_tech], timestep) == true
        return @constraint(backend_model, prod == available_resource)
    else
        return @constraint(backend_model, prod <= available_resource)
    end
end


# Demand Balance
function balance_demand_constraint_rule(backend_model, set_indices, model_dict)
    """
    Limit consumption from demand techs to their required resource
    """
    loc_tech, timestep = set_indices
    variables = model_dict["variables"]
    sets = model_dict["sets"]
    loc_tech_carrier = model_dict["parameters"]["lookup_loc_techs"][loc_tech]


    @expression(backend_model, con,
        variables["carrier_con"][loc_tech_carrier, timestep]
        * get_param(model_dict, "energy_eff", [loc_tech], timestep)
    )

    @expression(backend_model, available_resource,
        get_param(model_dict, "resource", [loc_tech], timestep) *
        get_param(model_dict, "resource_scale", [loc_tech], timestep)
    )

    if loc_tech_is_in(sets, loc_tech, "loc_techs_area")
        available_resource *= variables["resource_area"][loc_tech]
    end
    # creating the constraint just requires putting together the affine expressions
    if get_param(model_dict, "force_resource", [loc_tech], timestep) == true
        return @constraint(backend_model, con == available_resource)
    else
        return @constraint(backend_model, con >= available_resource)
    end
end


# Resource availablity
function resource_availability_supply_plus_constraint_rule(backend_model, set_indices, model_dict)
    """
    Limit production from supply_plus techs to their available resource
    """
    loc_tech, timestep = set_indices
    variables = model_dict["variables"]
    sets = model_dict["sets"]

    @expression(backend_model, available_resource,
        get_param(model_dict, "resource", [loc_tech], timestep) *
        get_param(model_dict, "resource_scale", [loc_tech], timestep) *
        get_param(model_dict, "resource_eff", [loc_tech], timestep)
    )

    if loc_tech_is_in(sets, loc_tech, "loc_techs_area")
        available_resource *= variables["resource_area"][loc_tech]
    end

    if get_param(model_dict, "force_resource", [loc_tech], timestep) == true
        return @constraint(backend_model, variables["resource_con"][loc_tech, timestep] == available_resource)
    else
        return @constraint(backend_model, variables["resource_con"][loc_tech, timestep] <= available_resource)
    end
end


# Transmission Balance
function balance_transmission_constraint_rule(backend_model, set_indices, model_dict)
    """
    Balance carrier production and consumption of transmission technologies
    """
    parameters = model_dict["parameters"]
    variables = model_dict["variables"]
    loc_tech, timestep = set_indices

    loc_tech_carrier = parameters["lookup_loc_techs"][loc_tech]
    remote_loc_tech = parameters["lookup_remotes"][loc_tech]
    remote_loc_tech_carrier = parameters["lookup_loc_techs"][remote_loc_tech]

    return @constraint(backend_model, variables["carrier_prod"][loc_tech_carrier, timestep]
        == -1 * variables["carrier_con"][remote_loc_tech_carrier, timestep]
        * get_param(model_dict, "energy_eff", [loc_tech], timestep))
end

# Supply_plus Balance
function balance_supply_plus_constraint_rule(backend_model, set_indices, model_dict)
    """
    Balance carrier production and resource consumption of supply_plus technologies
    alongside any use of resource storage
    """
    loc_tech, timestep = set_indices
    variables = model_dict["variables"]
    sets = model_dict["sets"]
    parameters = model_dict["parameters"]
    loc_tech_carrier = parameters["lookup_loc_techs"][loc_tech]
    total_eff = (get_param(model_dict, "energy_eff", [loc_tech], timestep) *
        get_param(model_dict, "parasitic_eff", [loc_tech], timestep))

    if total_eff == 0
        _prod = 0
    else
        _prod = variables["carrier_prod"][loc_tech_carrier, timestep] / total_eff
    end

    if !loc_tech_is_in(sets, loc_tech, "loc_techs_store")
        return @constraint(backend_model, variables["resource_con"][loc_tech, timestep] == _prod)
    else
        timestep_index = findin(sets["timesteps"], [timestep])[1]
        if timestep_index == 1
            storage_previous_step = get_param(model_dict, "storage_initial", [loc_tech])
        else
            storage_loss = get_param(model_dict, "storage_loss", [loc_tech])
            previous_timestep = model_dict["sets"]["timesteps"][timestep_index - 1]
            time_resolution = model_dict["parameters"]["timestep_resolution"][previous_timestep]
            storage_previous_step = (
                ((1 - storage_loss) ^ time_resolution) *
                variables["storage"][loc_tech, previous_timestep]
            )
        end
        return @constraint(backend_model,
            variables["storage"][loc_tech, timestep] ==
            storage_previous_step + variables["resource_con"][loc_tech, timestep] - _prod
        )
    end
end

# Storage Balance
function balance_storage_constraint_rule(backend_model, set_indices, model_dict)
    loc_tech, timestep = set_indices
    variables = model_dict["variables"]
    sets = model_dict["sets"]
    parameters = model_dict["parameters"]
    loc_tech_carrier = parameters["lookup_loc_techs"][loc_tech]

    energy_eff = get_param(model_dict, "energy_eff", [loc_tech], timestep)

    if energy_eff == 0
        _prod = 0
    else
        _prod = variables["carrier_prod"][loc_tech_carrier, timestep] / energy_eff
    end

    _con = variables["carrier_con"][loc_tech_carrier, timestep] * energy_eff

    timestep_index = findin(sets["timesteps"], [timestep])[1]
    if timestep_index == 1
        storage_previous_step = get_param(model_dict, "storage_initial", [loc_tech])
    else
        storage_loss = get_param(model_dict, "storage_loss", [loc_tech])
        previous_timestep = model_dict["sets"]["timesteps"][timestep_index - 1]
        time_resolution = model_dict["parameters"]["timestep_resolution"][previous_timestep]
        storage_previous_step = (
            ((1 - storage_loss) ^ time_resolution) *
            variables["storage"][loc_tech, previous_timestep]
        )
    end

    return @constraint(backend_model,
        variables["storage"][loc_tech, timestep] == storage_previous_step - _prod - _con
    )

end