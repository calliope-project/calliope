using JuMP; using NCDatasets; using AxisArrays; using Util;

function load_dispatch_constraints(model_dict)

    sets = model_dict["sets"]
    constraint_dict = Dict()

    if haskey(sets, "loc_tech_carriers_carrier_production_max_constraint")
        constraint_dict["carrier_production_max_constraint"] = (
            build_constraint(
                ["loc_tech_carriers_carrier_production_max_constraint", "timesteps"],
                "carrier_production_max", model_dict
            )
        )
    end

    if haskey(sets, "loc_tech_carriers_carrier_production_min_constraint")
        constraint_dict["carrier_production_min_constraint"] = (
            build_constraint(
                ["loc_tech_carriers_carrier_production_min_constraint", "timesteps"],
                "carrier_production_min", model_dict
            )
        )
    end

    if haskey(sets, "loc_tech_carriers_carrier_consumption_max_constraint")
        constraint_dict["carrier_consumption_max_constraint"] = (
            build_constraint(
                ["loc_tech_carriers_carrier_consumption_max_constraint", "timesteps"],
                "carrier_consumption_max", model_dict
            )
        )
    end

    if haskey(sets, "loc_techs_resource_max_constraint")
        constraint_dict["resource_max_constraint"] = (
            build_constraint(
                ["loc_techs_resource_max_constraint", "timesteps"],
                "resource_max", model_dict
            )
        )
    end

    if haskey(sets, "loc_techs_storage_max_constraint")
        constraint_dict["storage_max_constraint"] = (
            build_constraint(
                ["loc_techs_storage_max_constraint", "timesteps"],
                "storage_max", model_dict
            )
        )
    end

    return constraint_dict
end

function carrier_production_max_constraint_rule(backend_model, set_indices, model_dict)
    """
    Set maximum carrier production. All technologies.
    """
    loc_tech_carrier, timestep = set_indices
    variables = model_dict["variables"]
    loc_tech = get_loc_tech(loc_tech_carrier)
    carrier_prod = variables["carrier_prod"][loc_tech_carrier, timestep]
    timestep_resolution = model_dict["parameters"]["timestep_resolution"][timestep]
    parasitic_eff = get_param(model_dict, "parasitic_eff", [loc_tech], timestep)

    return @constraint(backend_model,
        carrier_prod <= variables["energy_cap"][loc_tech] * timestep_resolution * parasitic_eff
    )
end

function carrier_production_min_constraint_rule(backend_model, set_indices, model_dict)
    """
    Set minimum carrier production. All technologies except conversion_plus
    """
    loc_tech_carrier, timestep = set_indices
    variables = model_dict["variables"]
    loc_tech = get_loc_tech(loc_tech_carrier)
    carrier_prod = variables["carrier_prod"][loc_tech_carrier, timestep]
    timestep_resolution = model_dict["parameters"]["timestep_resolution"][timestep]
    min_use = get_param(model_dict, "energy_cap_min_use", [loc_tech], timestep)

    return @constraint(backend_model, (
        carrier_prod >= variables["energy_cap"][loc_tech] * timestep_resolution * min_use
    ))
end

function carrier_consumption_max_constraint_rule(backend_model, set_indices, model_dict)
    """
    Set maximum carrier consumption for demand, storage, and transmission techs
    """
    loc_tech_carrier, timestep = set_indices
    variables = model_dict["variables"]
    loc_tech = get_loc_tech(loc_tech_carrier)
    carrier_con = variables["carrier_con"][loc_tech_carrier, timestep]
    timestep_resolution = model_dict["parameters"]["timestep_resolution"][timestep]

    return @constraint(backend_model, (
        carrier_con >= -1 * variables["energy_cap"][loc_tech] * timestep_resolution
    ))
end

function resource_max_constraint_rule(backend_model, set_indices, model_dict)
    """
    Set maximum resource consumed by supply_plus techs
    """
    loc_tech, timestep = set_indices
    variables = model_dict["variables"]
    timestep_resolution = model_dict["parameters"]["timestep_resolution"][timestep]

    return @constraint(backend_model, (
        variables["resource_con"][loc_tech, timestep] <=
        timestep_resolution * variables["resource_cap"][loc_tech]
    ))
end

function storage_max_constraint_rule(backend_model, set_indices, model_dict)
    """
    Set maximum stored energy. Supply_plus & storage techs only.
    """
    loc_tech, timestep = set_indices
    variables = model_dict["variables"]

    return @constraint(backend_model, (
        variables["storage"][loc_tech, timestep] <= variables["storage_cap"][loc_tech]
    ))
end