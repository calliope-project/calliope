using JuMP; using NCDatasets; using AxisArrays; using Util; using Missings

function load_capacity_constraints(model_dict)
    sets = model_dict["sets"]
    constraint_dict = Dict()

    if haskey(sets, "loc_techs_storage_capacity_constraint")
        constraint_dict["storage_capacity_constraint"] = (
            build_constraint(["loc_techs_storage_capacity_constraint"],
            "storage_capacity", model_dict)
        )
    end

    if haskey(sets, "loc_techs_energy_capacity_storage_constraint")
        constraint_dict["energy_capacity_storage_constraint"] = (
            build_constraint(["loc_techs_energy_capacity_storage_constraint"],
            "energy_capacity_storage", model_dict)
        )
    end

    if haskey(sets, "loc_techs_resource_capacity_constraint")
        constraint_dict["resource_capacity_constraint"] = (
            build_constraint(["loc_techs_resource_capacity_constraint"],
            "resource_capacity", model_dict)
        )
    end

    if haskey(sets, "loc_techs_resource_capacity_equals_energy_capacity_constraint")
        constraint_dict["resource_capacity_equals_energy_capacity_constraint"] = (
            build_constraint(["loc_techs_resource_capacity_equals_energy_capacity_constraint"],
            "resource_capacity_equals_energy_capacity", model_dict)
        )
    end

    if haskey(sets, "loc_techs_resource_area_constraint")
        constraint_dict["resource_area_constraint"] = (
            build_constraint(["loc_techs_resource_area_constraint"],
            "resource_area", model_dict)
        )
    end

    if haskey(sets, "loc_techs_resource_area_per_energy_capacity_constraint")
        constraint_dict["resource_area_per_energy_capacity_constraint"] = (
            build_constraint(["loc_techs_resource_area_per_energy_capacity_constraint"],
            "resource_area_per_energy_capacity", model_dict)
        )
    end

    if haskey(sets, "locs_resource_area_capacity_per_loc_constraint")
        constraint_dict["resource_area_capacity_per_loc_constraint"] = (
            build_constraint(["locs_resource_area_capacity_per_loc_constraint"],
            "resource_area_capacity_per_loc", model_dict)
        )
    end

    if haskey(sets, "loc_techs_energy_capacity_constraint")
        constraint_dict["energy_capacity_constraint"] = (
            build_constraint(["loc_techs_energy_capacity_constraint"],
            "energy_capacity", model_dict)
        )
    end

    return constraint_dict
end


function get_capacity_constraint(model_dict, parameter, loc_tech ;
                    _equals=nothing, _max=nothing, _min=nothing, scale=nothing)
    decision_variable = model_dict["variables"][parameter]
    backend_model = model_dict["backend_model"]

    if _equals == nothing
        _equals = get_param(model_dict, string(parameter, "_equals"), [loc_tech])
    end

    if _max == nothing
        _max = get_param(model_dict, string(parameter, "_max"), [loc_tech])
    end

    if _min == nothing
        _min = get_param(model_dict, string(parameter, "_min"), [loc_tech])
    end

    if scale != nothing
        _equals = scale * _equals
        _min = scale * _min
        _max = scale * _max
    end

    if _equals != false && _equals != nothing
        if isinf(_equals)
            throw("Cannot use inf for $parameter/_equals for loc:tech `$loc_tech`")
        end

        return @constraint(backend_model, decision_variable[loc_tech] == _equals)

    else
        if !isinf(_max) && _max != nothing
            setupperbound(decision_variable[loc_tech], _max)
        end

        if _min != 0 && _min != nothing
            setlowerbound(decision_variable[loc_tech], _min)
        end

        return @constraint(backend_model, 0==0)
    end
end


function storage_capacity_constraint_rule(backend_model, set_indices, model_dict)
    """
    Set maximum storage capacity. Supply_plus & storage techs only
    This can be set by either storage_cap (kWh) or by
    energy_cap (charge/discharge capacity) * charge rate.
    If storage_cap.equals and energy_cap.equals are set for the technology, then
    storage_cap * charge rate = energy_cap must hold. Otherwise, take the lowest capacity
    capacity defined by storage_cap.max or energy_cap.max / charge rate.
    """
    loc_tech = set_indices[1]
    storage_cap_equals = get_param(model_dict, "storage_cap_equals", [loc_tech])
    scale = get_param(model_dict, "energy_cap_scale", [loc_tech])
    energy_cap_equals = scale * get_param(model_dict, "energy_cap_equals", [loc_tech])

    energy_cap_max = get_param(model_dict, "energy_cap_max", [loc_tech])
    storage_cap_max = get_param(model_dict, "storage_cap_max", [loc_tech])

    charge_rate = get_param(model_dict, "charge_rate", [loc_tech])

    # FIXME working out storage_cap_max or storage_cap_equals from e_cap_max/_equals
    # should be done in preprocessing, not here.

    if storage_cap_equals > 0
        return get_capacity_constraint(model_dict, "storage_cap",
                                        loc_tech, _equals=storage_cap_equals)
    elseif energy_cap_equals > 0 && charge_rate > 0
        storage_cap_equals = energy_cap_equals * scale / charge_rate
        return get_capacity_constraint(model_dict, "storage_cap",
                                        loc_tech, _equals=storage_cap_equals)
    elseif storage_cap_max > 0
        return get_capacity_constraint(model_dict, "storage_cap",
                                        loc_tech, _max=storage_cap_max)
    elseif energy_cap_max > 0 && charge_rate > 0
        storage_cap_max = energy_cap_max * scale / charge_rate
        return get_capacity_constraint(model_dict, "storage_cap",
                                        loc_tech, _max=storage_cap_max)
    else
        return @constraint(backend_model, 0==0)
    end
end


function energy_capacity_storage_constraint_rule(backend_model, set_indices, model_dict)
    """
    Set an additional energy capacity constraint on storage technologies,
    based on their use of `charge_rate`
    """
    loc_tech = set_indices[1]
    variables = model_dict["variables"]
    energy_cap_scale = get_param(model_dict, "energy_cap_scale", [loc_tech])
    charge_rate = get_param(model_dict, "charge_rate", [loc_tech])

    return @constraint(backend_model,
        variables["energy_cap"][loc_tech] <=
        variables["storage_cap"][loc_tech] * charge_rate * energy_cap_scale
    )
end


function resource_capacity_constraint_rule(backend_model, set_indices, model_dict)
    """
    Add upper and lower bounds for resource_cap
    """
    loc_tech = set_indices[1]
    return get_capacity_constraint(model_dict, "resource_cap", loc_tech)
end


function resource_capacity_equals_energy_capacity_constraint_rule(backend_model, set_indices, model_dict)
    """
    Add equality constraint for resource_cap to equal energy_cap, for any technologies
    which have defined resource_cap_equals_energy_cap
    """
    loc_tech = set_indices[1]
    variables = model_dict["variables"]
    return @constraint(backend_model,
        variables["resource_cap"][loc_tech] == variables["energy_cap"][loc_tech]
    )
end


function resource_area_constraint_rule(backend_model, set_indices, model_dict)
    """
    Set upper and lower bounds for resource_area.
    """
    loc_tech = set_indices[1]
    variables = model_dict["variables"]
    energy_cap_max = get_param(model_dict, "energy_cap_max", [loc_tech])
    area_per_energy_cap = get_param(model_dict, "resource_area_per_energy_cap", [loc_tech])

    if energy_cap_max == 0
        # If a technology has no energy_cap here, we force resource_area to zero,
        # so as not to accrue spurious costs
        return @constraint(backend_model, variables["resource_area"][loc_tech] == 0)
    else
        return get_capacity_constraint(model_dict, "resource_area", loc_tech)
    end
end


function resource_area_per_energy_capacity_constraint_rule(backend_model, set_indices, model_dict)
    """
    Add equality constraint for resource_area to equal a percentage of energy_cap,
    for any technologies which have defined resource_area_per_energy_cap
    """
    loc_tech = set_indices[1]
    variables = model_dict["variables"]
    area_per_energy_cap = get_param(model_dict, "resource_area_per_energy_cap", [loc_tech])

    return @constraint(backend_model,
        variables["resource_area"][loc_tech] ==
        variables["energy_cap"][loc_tech] * area_per_energy_cap
    )
end


function resource_area_capacity_per_loc_constraint_rule(backend_model, set_indices, model_dict)
    """
    Set upper bound on use of area for all locations which have `available_area`
    constraint set. Does not consider resource_area applied to demand technologies
    """
    loc = set_indices[1]
    variables = model_dict["variables"]
    parameters = model_dict["parameters"]
    available_area = parameters["available_area"][loc]

    loc_techs = split_comma_list(parameters["lookup_loc_techs_area"][loc])

    return @constraint(backend_model,
        sum(variables["resource_area"][loc_tech] for loc_tech in loc_techs)
        <= available_area
    )
end


function energy_capacity_constraint_rule(backend_model, set_indices, model_dict)
    """
    Set upper and lower bounds for energy_cap.
    """
    loc_tech = set_indices[1]
    return get_capacity_constraint(model_dict, "energy_cap", loc_tech)
end
