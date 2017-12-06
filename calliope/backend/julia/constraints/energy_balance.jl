using JuMP; using NCDatasets; using AxisArrays; using Util;

function load_energy_balance_constraints(dataset, sets, parameters, backend_model)

    constraint_dict = Dict()

    if haskey(sets, "loc_techs_finite_resource_supply_plus")
        constraint_dict["resource_availability_constraint"] = (
            build_constraint(backend_model, ["loc_techs_finite_resource_supply_plus", "timesteps"],
                "resource_availability", sets, parameters)
        )
    end

    if haskey(sets, "loc_techs_finite_resource_supply")
        constraint_dict["balance_supply_constraint"] = (
            build_constraint(backend_model, ["loc_techs_finite_resource_supply", "timesteps"],
                "balance_supply", sets, parameters)
        )
    end

    if haskey(sets, "loc_techs_finite_resource_demand")
        constraint_dict["balance_demand_constraint"] = (
            build_constraint(backend_model, ["loc_techs_finite_resource_demand", "timesteps"],
                "balance_demand", sets, parameters)
        )
    end

    if haskey(sets, "loc_techs_transmission")
        constraint_dict["balance_transmission_constraint"] = (
            build_constraint(backend_model, ["loc_techs_transmission", "timesteps"],
                "balance_transmission", sets, parameters)
        )
    end

    if haskey(sets, "loc_techs_conversion")
        constraint_dict["balance_conversion_constraint"] = (
            build_constraint(backend_model, ["loc_techs_conversion", "timesteps"],
                "balance_conversion", sets, parameters)
        )
    end

    if haskey(sets, "loc_techs_conversion_plus")
        constraint_dict["balance_conversion_plus_primary_constraint"] = (
            build_constraint(backend_model, ["loc_techs_conversion_plus", "timesteps"],
                "balance_conversion_plus_primary", sets, parameters)
        )
    end

    if haskey(sets, "loc_techs_out_2")
        constraint_dict["balance_conversion_plus_out_2_constraint"] = (
            build_constraint(backend_model, ["loc_techs_out_2", "timesteps"],
                "balance_conversion_plus_out_2", sets, parameters)
        )
    end

    if haskey(sets, "loc_techs_out_3")
        constraint_dict["balance_conversion_plus_out_3_constraint"] = (
            build_constraint(backend_model, ["loc_techs_out_3", "timesteps"],
                "balance_conversion_plus_out_3", sets, parameters)
        )
    end

    if haskey(sets, "loc_techs_in_2")
        constraint_dict["balance_conversion_plus_in_2_constraint"] = (
            build_constraint(backend_model, ["loc_techs_in_2", "timesteps"],
                "balance_conversion_plus_in_2", sets, parameters)
        )
    end

    if haskey(sets, "loc_techs_in_3")
        constraint_dict["balance_conversion_plus_in_3_constraint"] = (
            build_constraint(backend_model, ["loc_techs_in_3", "timesteps"],
                "balance_conversion_plus_in_3", sets, parameters)
        )
    end

    if haskey(sets, "loc_techs_supply_plus")
        constraint_dict["balance_supply_plus_constraint"] = (
            build_constraint(backend_model, ["loc_techs_supply_plus", "timesteps"],
                "balance_supply_plus", sets, parameters)
        )
    end

    if haskey(sets, "loc_techs_storage")
        constraint_dict["balance_storage_constraint"] = (
            build_constraint(backend_model, ["loc_techs_storage", "timesteps"],
                "balance_storage", sets, parameters)
        )
    end
end
# Resource availablity
function resource_availability_constraint_rule(backend_model, set_indices, sets, parameters)
    print(set_indices)
    loc_tech, timestep = set_indices

    @expression(backend_model, available_resource,
        parameters["resource"][loc_tech, timestep] *
        parameters["resource_scale"][loc_tech] *
        timesliced_variable(dataset, "resource_eff", timestep)[loc_tech]
    )

    if loc_tech in sets["loc_techs_area"]
        m.available_resource *= resource_area[loc_tech]
    end

    if timesliced_variable(dataset, "force_resource", timestep)[loc_tech] == true
        @constraint(backend_model, resource[loc_tech, timestep] == available_resource)
    else
        @constraint(backend_model, resource[loc_tech, timestep] <= available_resource)
    end
end

# Supply Balance
function balance_supply_constraint_rule(backend_model, set_indices, sets, parameters)
    loc_tech, timestep = set_indices
    loc_tech_carrier = parameters["lookup_loc_tech_carriers"][loc_tech]
    # initialise the affine expression `prod` in model `m` for supply techs
    energy_eff = timesliced_variable(dataset, "energy_eff", timestep)[loc_tech]

    if energy_eff == 0
        energy_eff = 0
    else
        energy_eff = 1 / timesliced_variable(dataset, "energy_eff", timestep)[loc_tech]
    end

    @expression(backend_model, prod,
        energy_eff * carrier_prod[loc_tech_carrier, timestep]
    )
    @expression(backend_model, available_resource,
        parameters["resource"][loc_tech, timestep] *
        parameters["resource_scale"][loc_tech]
    )
    if loc_tech in sets["loc_techs_area"]
        available_resource *= resource_area[loc_tech]
    end
    # creating the constraint just requires putting together the affine expressions
    if timesliced_variable(dataset, "force_resource", timestep)[loc_tech] == true
        @constraint(backend_model, prod == available_resource)
    else
        @constraint(backend_model, prod <= available_resource)
    end
end

# Demand Balance
function balance_demand_constraint_rule(backend_model, set_indices, sets, parameters)
    loc_tech, timestep = set_indices
    loc_tech_carrier = parameters["lookup_loc_tech_carriers"][loc_tech]
    @expression(backend_model, con,
        carrier_con[loc_tech_carrier, timestep]
        * timesliced_variable(dataset, "energy_eff", timestep)[loc_tech]
    )

    @expression(backend_model, available_resource,
        parameters["resource"][loc_tech, timestep] *
        parameters["resource_scale"][loc_tech]
    )
    if loc_tech in sets["loc_techs_area"]
        available_resource *= resource_area[loc_tech]
    end
    # creating the constraint just requires putting together the affine expressions
    if timesliced_variable(dataset, "force_resource", timestep)[loc_tech] == true
        @constraint(backend_model, con == available_resource)
    else
        @constraint(backend_model, con >= available_resource)
    end
end

# Transmission Balance
function balance_transmission_constraint_rule(backend_model, set_indices, sets, parameters)
    loc_tech, timestep = set_indices
    loc_tech_carrier = parameters["lookup_loc_tech_carriers"][loc_tech]
    loc_tech_remote = parameters["lookup_remotes"][loc_tech]
    loc_tech_carrier_remote = parameters["lookup_loc_tech_carriers"][loc_tech_remote]

    @constraint(backend_model, carrier_prod[carrier_carrier, timestep]
        == -1 * carrier_con[loc_tech_carrier_remote, timestep]
        * timesliced_variable(dataset, "energy_eff", timestep)[loc_tech])
end

# Conversion Balance
function balance_conversion_constraint_rule(backend_model, set_indices, sets, parameters)
    loc_tech, timestep = set_indices
    loc_tech_carrier_out = parameters["lookup_carriers_conversion"][loc_tech, "out"]
    loc_tech_carrier_in = parameters["lookup_carriers_conversion"][loc_tech, "in"]
    @constraint(backend_model, carrier_prod[loc_tech_carrier_out, timestep]
                == -1 * carrier_con[loc_tech_carrier_in, timestep] *
                timesliced_variable(dataset, "energy_eff", timestep)[loc_tech])
end

# Conversion_plus primary carrier balance
function balance_conversion_plus_primary_constraint_rule(backend_model, set_indices)
    loc_tech, timestep = set_indices
    carriers = parameters["lookup_carriers_conversion_plus"][loc_tech]
    loc_tech_carriers_out = carriers[Axis{:carrier_tiers}("out")]
    loc_tech_carriers_in = carriers[Axis{:carrier_tiers}("in")]

    carrier_ratios_out = parameters["carrier_ratios"][
        Axis{:carrier_tiers}("out"),
        Axis{:loc_tech_carriers_conversion_plus}(loc_tech_carriers_out)]
    @constraint(backend_model,
        sum(carrier_prod[loc_tech_carrier, timestep] / carrier_out[carrier]
            for carrier in keys(carrier_out) if carrier_out[carrier] != 0)
        == -1 *
        sum(carrier_con[loc_tech, carrier, timestep] * carrier_in[carrier]
            for carrier in keys(carrier_in) if carrier_in[carrier] != 0)
        * timesliced_variable(dataset, "energy_eff", timestep)[loc_tech]
    )
end

# Conversion_plus carrier_out_2 balancee
function balance_conversion_plus_out_2_constraint_rule(backend_model, set_indices, sets, parameters)
    loc_tech, timestep = set_indices
    carrier_out = get_carrier(loc_tech, "out")
    carrier_out_2 = get_carrier(loc_tech, "out_2")
    carriers = sets["carriers"]
    c_min = minimum([c for c in carrier_out_2[loc_tech, :] if c != 0])

    @constraint(backend_model,
        sum(carrier_prod[loc_tech, c, timestep] / carrier_out[loc_tech, c]
            for c in carriers if carrier_out[loc_tech, c] != 0)
        * c_min ==
        sum(carrier_prod[loc_tech, c, timestep] / (carrier_out_2[loc_tech, c] / c_min)
            for c in carriers if carrier_out_2[loc_tech, c] != 0)
    )
end

# Conversion_plus carrier_out_3 balance
function balance_conversion_plus_out_3_constraint_rule(backend_model, set_indices, sets, parameters)
    loc_tech, timestep = set_indices
    carrier_out = get_carrier(loc_tech, "out")
    carrier_out_3 = get_carrier(loc_tech, "out_3")
    carriers = sets["carriers"]
    c_min = minimum([c for c in carrier_out_3[loc_tech, :] if c != 0])

    @constraint(backend_model,
        sum(carrier_prod[loc_tech, c, timestep] / carrier_out[loc_tech, c]
            for c in carriers if carrier_out[loc_tech, c] != 0)
        * c_min ==
        sum(carrier_prod[loc_tech, c, timestep] / (carrier_out_3[loc_tech, c] / c_min)
            for c in carriers if carrier_out_3[loc_tech, c] != 0)
    )
end

# Conversion_plus carrier_in_2 balance
function balance_conversion_plus_in_2_constraint_rule(backend_model, set_indices, sets, parameters)
    loc_tech, timestep = set_indices
    carrier_in = get_carrier(loc_tech, "in")
    carrier_in_2 = get_carrier(loc_tech, "in_2")
    carriers = sets["carriers"]
    c_min = minimum([c for c in carrier_in_2[loc_tech, :] if c != 0])

    @constraint(backend_model,
        sum(carrier_con[loc_tech, c, timestep] / carrier_in[loc_tech, c]
            for c in carriers if carrier_in[loc_tech, c] != 0)
        * c_min ==
        sum(carrier_con[loc_tech, c, timestep] / (carrier_in_2[loc_tech, c] / c_min)
            for c in carriers if carrier_in_2[loc_tech, c] != 0)
    )
end

# Conversion_plus carrier_in_3 balance
function balance_conversion_plus_in_3_constraint_rule(backend_model, set_indices, sets, parameters)
    loc_tech, timestep = set_indices
    carrier_in = get_carrier(loc_tech, "in")
    carrier_in_3 = get_carrier(loc_tech, "in_3")
    carriers = sets["carriers"]
    c_min = minimum([c for c in carrier_in_3[loc_tech, :] if c != 0])

    @constraint(backend_model,
        sum(carrier_con[c, loc_tech, timestep] / carrier_in[loc_tech, c]
            for c in carriers if carrier_in[loc_tech, c] != 0)
        * c_min ==
        sum(carrier_con[c, loc_tech, timestep] / (carrier_in_3[loc_tech, c] / c_min)
            for c in carriers if carrier_in_3[loc_tech, c] != 0)
    )
end

# Supply_plus Balance
function balance_supply_plus_constraint_rule(backend_model, set_indices, sets, parameters)
    loc_tech, timestep = set_indices
    loc_tech_carrier = parameters["lookup_loc_tech_carriers"][loc_tech]
    total_eff = (timesliced_variable(dataset, "energy_eff", timestep)[loc_tech] *
                 timesliced_variable(dataset, "parasitic_eff", timestep)[loc_tech])

    if total_eff == 0
        _c_prod = 0
    else
        _c_prod = carrier_prod[loc_tech_carrier timestep] / total_eff
    end

    if loc_tech not in sets["loc_techs_store"]
        @constraint(backend_model, resource[loc_tech, timestep] == _c_prod)
    else
        if findin(sets["timesteps"], timestep) == 1
            try
                s_minus_one = parameters["storage_initial"][loc_tech]
            catch
                s_minus_one = 0
            end
        else
            s_minus_one = (
                ((1 - parameters["storage_loss"][loc_tech])
                    ^ parameters["time_resolution"][timestep - 1])
                * storage[loc_tech, timestep-1]
            )
        end
        @constraint(backend_model,
            storage[loc_tech, timestep] ==
            s_minus_one + resource[loc_tech, timestep] - _c_prod
        )
    end
end

# Storage Balance
function balance_storage_constraint_rule(backend_model, set_indices, sets, parameters)
    loc_tech, timestep = set_indices
    loc_tech_carrier = parameters["lookup_loc_tech_carriers"][loc_tech]
    energy_eff = timesliced_variable(dataset, "energy_eff", timestep)[loc_tech]

    if energy_eff == 0
        _c_prod = 0
    else
        _c_prod = carrier_prod[loc_tech_carrier, timestep] / total_eff
    end
    _c_con = carrier_con[loc_tech_carrier, timestep] * total_eff

    if findin(sets["timesteps"], timestep) == 1
        try
            s_minus_one = parameters["storage_initial"][loc_tech]
        catch
            s_minus_one = 0
        end
    else
        previous_timestep = sets["timesteps"][findin(sets["timesteps"], timestep) - 1]
        s_minus_one = (
            ((1 - parameters["storage_loss"][loc_tech])
                ^ parameters["time_resolution"][previous_timestep])
            * storage[loc_tech, previous_timestep]
        )
    end

    @constraint(backend_model,
        storage[loc_tech, timestep] == s_minus_one - _c_prod - _c_con
    )

end