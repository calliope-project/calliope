using JuMP; using NCDatasets; using AxisArrays; using Util;

function load_milp_constraints(model_dict)
    sets = model_dict["sets"]
    constraint_dict = Dict()
    backend_model = model_dict["backend_model"]

    if haskey(sets, "loc_techs_milp")
        constraint_dict["unit_commitment_constraint"] = build_constraint(
            ["loc_techs_unit_commitment_constraint", "timesteps"],
            "unit_commitment_constraint", model_dict
        )

        constraint_dict["carrier_production_max_milp_constraint"] = build_constraint(
            ["loc_tech_carriers_carrier_production_max_conversion_plus_milp_constraint_rule",
            "timesteps"],"carrier_production_max_milp", model_dict
        )

        constraint_dict["carrier_consumption_max_milp_constraint"] = build_constraint(
            ["loc_tech_carriers_carrier_consumption_max_milp_constraint",
            "timesteps"], "carrier_consumption_max_milp", model_dict
        )
    end

    if haskey(sets, "loc_tech_carriers_carrier_production_min_milp_constraint")
        constraint_dict["carrier_production_min_milp_constraint"] = build_constraint(
            ["loc_tech_carriers_carrier_production_min_milp_constraint",
            "timesteps"], "carrier_production_min_milp", model_dict
        )
    end

    if haskey(sets, "loc_techs_storage_capacity_milp_constraint")
        constraint_dict["storage_capacity_milp_constraint"] = build_constraint(
            ["loc_techs_storage_capacity_milp_constraint"],
            "storage_capacity_milp", model_dict
        )
    end

    if haskey(sets, "loc_techs_energy_capacity_units_constraint")
        constraint_dict["energy_capacity_units_constraint"] = build_constraint(
            ["loc_techs_energy_capacity_units_constraint"],
            "energy_capacity_units", model_dict
        )
    end

    if haskey(sets, "loc_techs_update_costs_investment_units_constraint")
        for loc_tech=sets["loc_techs_update_costs_investment_units_constraint"], cost=sets["costs"]

            update_costs_investment_units_constraint(backend_model, loc_tech, cost)
        end
    end

    if haskey(sets, "loc_techs_update_costs_investment_purchase_constraint")
        for loc_tech=sets["loc_techs_update_costs_investment_purchase_constraint"], cost=sets["costs"]

            update_costs_investment_purchase_constraint(model_dict, loc_tech, cost)
        end
    end

    if haskey(sets, "loc_techs_energy_capacity_max_purchase_constraint")
        constraint_dict["energy_capacity_max_purchase_constraint"] = build_constraint(
            ["loc_techs_energy_capacity_max_purchase_constraint"],
            "energy_capacity_max_purchase", model_dict
        )
    end

    if haskey(sets, "loc_techs_energy_capacity_min_purchase_constraint")
        constraint_dict["energy_capacity_min_purchase_constraint"] = build_constraint(
            ["loc_techs_energy_capacity_min_purchase_constraint"],
            "energy_capacity_min_purchase", model_dict
        )
    end

    return constraint_dict
end

function unit_commitment_constraint_rule(backend_model, set_indices, model_dict)
    """
    Constraining the number of integer units
    """
    loc_tech, timestep = set_indices
    variables = model_dict["variables"]
    return @constraint(backend_model,
        variables["operating_units"][loc_tech, timestep] <= variables["units"][loc_tech]
    )
end

function carrier_production_max_milp_constraint_rule(backend_model, set_indices, model_dict)
    """
    Set maximum carrier production of MILP techs that aren"t conversion plus
    """
    loc_tech_carrier, timestep = set_indices
    variables = model_dict["variables"]
    loc_tech = get_loc_tech(loc_tech_carrier)
    timestep_resolution = model_dict["parameters"]["timestep_resolution"][timestep]
    parasitic_eff = get_param(model_dict, "parasitic_eff", [loc_tech], timestep)
    e_cap = get_param(model_dict, "energy_cap_per_unit", [loc_tech])

    return @constraint(backend_model,
        variables["carrier_prod"][loc_tech_carrier, timestep] <=
        variables["operating_units"][loc_tech, timestep] * timestep_resolution * e_cap
        * parasitic_eff
    )
end


function carrier_production_max_conversion_plus_milp_constraint_rule(backend_model, set_indices, model_dict)
    """
    Set maximum carrier production of conversion_plus MILP techs
    """
    loc_tech, timestep = set_indices
    variables = model_dict["variables"]
    timestep_resolution = model_dict["parameters"]["timestep_resolution"][timestep]
    e_cap = get_param(model_dict, "energy_cap_per_unit", [loc_tech])
    loc_tech_carriers_out = split_comma_list(get_param(
        backend_model, "lookup_loc_techs_conversion_plus", (loc_tech, "out")
    ))

    c_prod = sum(variables["carrier_prod"][loc_tech_carrier, timestep]
                 for loc_tech_carrier in loc_tech_carriers_out)

    return @constraint(backend_model,
        c_prod <= variables["operating_units"][loc_tech, timestep] * timestep_resolution * e_cap
    )
end


function carrier_production_min_milp_constraint_rule(backend_model, set_indices, model_dict)
    """
    Set minimum carrier production of MILP techs that aren"t conversion plus
    """
    loc_tech_carrier, timestep = set_indices
    variables = model_dict["variables"]
    loc_tech = get_loc_tech(loc_tech_carrier)
    timestep_resolution = model_dict["parameters"]["timestep_resolution"][timestep]
    min_use = get_param(model_dict, "energy_cap_min_use", [loc_tech], timestep)
    e_cap = get_param(model_dict, "energy_cap_per_unit", [loc_tech])

    return @constraint(backend_model,
        variables["carrier_prod"][loc_tech_carrier, timestep] >=
        variables["operating_units"][loc_tech, timestep] * timestep_resolution
        * e_cap * min_use
    )
end


function carrier_consumption_max_milp_constraint_rule(backend_model, set_indices, model_dict)
    """
    Set maximum carrier consumption of demand, storage, && tranmission MILP techs
    """
    loc_tech_carrier, timestep = set_indices
    variables = model_dict["variables"]
    loc_tech = get_loc_tech(loc_tech_carrier)
    timestep_resolution = model_dict["parameters"]["timestep_resolution"][timestep]
    e_cap = get_param(model_dict, "energy_cap_per_unit", [loc_tech])

    return @constraint(backend_model,
        variables["carrier_con"][loc_tech_carrier, timestep] >= -1 *
        variables["operating_units"][loc_tech, timestep] * timestep_resolution
        * e_cap
    )
end


function energy_capacity_units_constraint_rule(backend_model, set_indices, model_dict)
    """
    Set energy capacity decision variable as a function of purchased units
    """
    loc_tech = set_indices
    variables = model_dict["variables"]
    return @constraint(backend_model,
        variables["energy_cap"][loc_tech] == variables["units"][loc_tech] *
        get_param(model_dict, "energy_cap_per_unit", [loc_tech])
    )
end


function energy_capacity_max_purchase_constraint_rule(backend_model, set_indices, model_dict)
    """
    Set maximum energy capacity decision variable upper bound as a function of
    binary purchase variable
    """
    loc_tech = set_indices
    variables = model_dict["variables"]
    energy_cap_max = get_param(model_dict, "energy_cap_max", [loc_tech])
    energy_cap_equals = get_param(model_dict, "energy_cap_equals", [loc_tech])
    energy_cap_scale = get_param(model_dict, "energy_cap_scale", [loc_tech])

    if energy_cap_equals > 0
        return @constraint(backend_model,
            variables["energy_cap"][loc_tech] ==
            energy_cap_equals * energy_cap_scale * variables["purchased"][loc_tech]
        )

    elseif isinf(energy_cap_max)
        return

    else
        return @constraint(backend_model,
            variables["energy_cap"][loc_tech] <=
            energy_cap_max * energy_cap_scale * variables["purchased"][loc_tech]
        )
    end
end

function energy_capacity_min_purchase_constraint_rule(backend_model, set_indices, model_dict)
    """
    Set minimum energy capacity decision variable upper bound as a function of
    binary purchase variable
    """
    loc_tech = set_indices
    variables = model_dict["variables"]
    energy_cap_min = get_param(model_dict, "energy_cap_min", [loc_tech])
    energy_cap_equals = get_param(model_dict, "energy_cap_equals", [loc_tech])
    energy_cap_scale = get_param(model_dict, "energy_cap_scale", [loc_tech])

    return @constraint(backend_model,
        variables["energy_cap"][loc_tech] >=
        energy_cap_min * energy_cap_scale * variables["purchased"][loc_tech]
    )
end

function storage_capacity_milp_constraint_rule(backend_model, set_indices, model_dict)
    """
    Set maximum storage capacity. Supply_plus & storage techs only
    This can be set by either storage_cap (kWh) or by
    energy_cap (charge/discharge capacity) * charge rate.
    If storage_cap.equals and energy_cap.equals are set for the technology, then
    storage_cap * charge rate = energy_cap must hold. Otherwise, take the lowest capacity
    capacity defined by storage_cap.max or energy_cap.max / charge rate.
    """
    loc_tech = set_indices
    variables = model_dict["variables"]
    # FIXME? energy_cap_equals could be already dealt with in preprocessing, to
    # either be energy_cap_equals or units_equals * energy_cap_per_unit. Similarly for
    # storage_cap_equals
    units_equals = get_param(model_dict, "units_equals", [loc_tech])
    storage_cap_per_unit = get_param(model_dict, "storage_cap_per_unit", [loc_tech])
    energy_cap_per_unit = get_param(model_dict, "energy_cap_per_unit", [loc_tech])

    scale = get_param(model_dict, "energy_cap_scale", [loc_tech])
    charge_rate = get_param(model_dict, "charge_rate", [loc_tech])

    # First, set the variable with "==" is unit numbers are set in stone
    if units_equals > 0 && storage_cap_per_unit > 0
        return @constraint(backend_model,
            variables["storage_cap"][loc_tech] == storage_cap_per_unit * units_equals
        )

    elseif units_equals > 0 && energy_cap_per_unit > 0 && charge_rate > 0
        return @constraint(backend_model,
            variables["storage_cap"][loc_tech] ==
            energy_cap_per_unit * scale * units_equals / charge_rate
        )

    # If not set in stone, use the variable "units" to set maximum
    elseif storage_cap_per_unit > 0
        s_cap = variables["units"][loc_tech] * storage_cap_per_unit
        return @constraint(backend_model, variables["storage_cap"][loc_tech] <= s_cap)

    elseif energy_cap_per_unit > 0 && charge_rate > 0
        e_cap = variables["units"][loc_tech] * energy_cap_per_unit * scale / charge_rate
        return @constraint(backend_model, variables["storage_cap"][loc_tech] <= e_cap)

    # if insufficient "per_unit" information is given, assume there is no capacity
    else
        return
    end
end

function update_costs_investment_units_constraint(model_dict, loc_tech, cost)
    """
    Add MILP investment costs (cost * number of units purchased)
    """
    variables = model_dict["variables"]
    ts_weight = get_timestep_weight(model_dict)
    sets = model_dict["sets"]
    depreciation_rate = model_dict["parameters"]["cost_depreciation_rate"][loc_tech, cost]

    cost_purchase = get_param(model_dict, "cost_purchase", [cost, loc_tech])
    cost_of_purchase = (
        variables["units"][loc_tech] * cost_purchase * ts_weight * depreciation_rate
    )

    append!(model_dict["expressions"]["investment_rhs"][cost, loc_tech], cost_of_purchase)

    loc_tech_index = findin(sets["loc_carriers"], [loc_tech])
    cost_index = findin(sets["loc_carriers"], [cost])
    model_dict["constraints"]["cost_investment_constraint"][cost_index, loc_tech_index] = (
        @constraint(model_dict["backend_model"],
            variables["cost_investment"][cost, loc_tech] ==
            model_dict["expressions"]["investment_rhs"][cost, loc_tech]
        )
    )
    return
end

function update_costs_investment_purchase_constraint(model_dict, loc_tech, cost)
    """
    Add binary investment costs (cost * binary_purchased_unit)
    """
    variables = model_dict["variables"]
    sets = model_dict["sets"]
    ts_weight = get_timestep_weight(model_dict)
    depreciation_rate = model_dict["parameters"]["cost_depreciation_rate"][loc_tech, cost]

    cost_purchase = get_param(model_dict, "cost_purchase", [cost, loc_tech])
    cost_of_purchase = (
        variables["purchased"][loc_tech] * cost_purchase * ts_weight * depreciation_rate
    )

    append!(backend_model.investment_rhs[cost, loc_tech], cost_of_purchase)

    loc_tech_index = findin(sets["loc_carriers"], [loc_tech])
    cost_index = findin(sets["loc_carriers"], [cost])
    model_dict["constraints"]["cost_investment_constraint"][cost_index, loc_tech_index] = (
        @constraint(model_dict["backend_model"],
            variables["cost_investment"][cost, loc_tech] ==
            model_dict["expressions"]["investment_rhs"][cost, loc_tech]
        )
    )
    return
end
