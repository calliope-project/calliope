module Util
export get_variable, get_param, build_constraint, get_previous_timestep,
get_loc_tech_carriers, get_loc_tech, get_timestep_weight, split_comma_list,
get_conversion_plus_io, loc_tech_is_in

using AxisArrays; using NCDatasets; using JuMP; using CalliopeBackend; using Missings


function get_variable(dataset, var::String)
    """
    returns an AxisArray of the given variable, indexed over its relevant
    dimensions

    Parameters
    ----------
    var : string
        From the variables available in the loaded Dataset
    Returns
    -------
    AxisArray
    """
    var_dimnames = NCDatasets.dimnames(dataset[var])
    var_out = AxisArray(dataset[var][:],
        Tuple([AxisArrays.Axis{Symbol(i)}(dataset[i][:])
            for i in var_dimnames]))

    return var_out
end


function get_param(model_dict, param, non_time_indices, timestep=nothing)
    """
    returns an AxisArray of the given variable, indexed over all dimensions
    except timesteps, in order to remove timesteps from a possible timeseries
    variable (which may also be a static variable, we just don"t know!)

    Parameters
    ----------
    param : string
        From the variables available in the loaded Dataset
    non_time_indices : string or tuple
        List of all search indices that aren"t the timesteps. Should be ordered
        as reversed alphabetical (e.g. [loc_techs, costs])
    timestep : nothing or string, default = nothing
        one value from from sets["timesteps"]
    Returns
    -------
    AxisArray without the time axis
    """
    sets = model_dict["sets"]
    dataset = model_dict["dataset"]
    param_index = []

    if !haskey(dataset, param)
        return_val = model_dict["defaults"][param]
    else
        axes = NCDatasets.dimnames(dataset[param])
        for i in 1:length(non_time_indices)
            if "timesteps" in axes
                j = i + 1
            else
                j = i
            end
            if non_time_indices[i] in sets[axes[j]]
                push!(param_index, findin(sets[axes[j]], [non_time_indices[i]])[1])
            else
                return_val = model_dict["defaults"][param]
                break
            end
        end

        if length(param_index) == length(non_time_indices)
            if axes[1] == "timesteps"
                unshift!(param_index, findin(sets["timesteps"], [timestep])[1])
            end
            return_val = getindex(dataset[param], param_index...)
        end
    end

    if ismissing(return_val)
        return_val = model_dict["defaults"][param]
    end

    if return_val == nothing
        return 0
    else
        return return_val
    end
end


function build_constraint(constraint_sets, constraint_name, model_dict; tier=nothing)
    """
    create a constraint object, indexed over the given sets.

    Parameters
    ----------
    constraint_sets : array of strings
        names of the sets over which the constraint will be indexed
    constraint_name : string
        Name of constraint obect to build. Will also be used to refer to the
        correct constraint rule, which will be constraint_name + "_constraint_rule"
    tier : string
        For us in conversion_plus.jl `balance_conversion_plus_tiers_constraint_rule`
        whereby the same rule is used for multiple carrier tiers
    Return
    ------
    No direct return, the constraint Dictionary constraint_dict will contain an
    additional entry with the key `constraint_name`
    """

    sets = model_dict["sets"]
    backend_model = model_dict["backend_model"]

    # create a tuple of integer lists
    indexed_sets = [1:length(sets[i]) for i in constraint_sets]

    # Use the Julia @constraintref macro to build an empty constraint object
    temporary_constraint = Array{ConstraintRef}((length(i) for i in indexed_sets)...)
    # get the relevant function from this module
    temporary_constraint_rule = getfield(
        CalliopeBackend.load_constraints,
        Symbol(string(constraint_name, "_constraint_rule"))
    )

    # Loop through to populate the previously built empty constraint object
    for i in CartesianRange(size(temporary_constraint))
        idx = [item for item in i.I]
        if tier != nothing
            temporary_constraint[i] = temporary_constraint_rule(backend_model,
                [sets[constraint_sets[constr]][idx[constr]]
                for constr in 1:length(constraint_sets)],
                model_dict, tier
            )
        else
            temporary_constraint[i] = temporary_constraint_rule(backend_model,
                [sets[constraint_sets[constr]][idx[constr]]
                for constr in 1:length(constraint_sets)],
                model_dict
            )
        end
    end

    return temporary_constraint
end


function get_previous_timestep(model_dict, timestep)
    timestep_idx = findfirst(model_dict["sets"]["timesteps"][DateTime(timestep)])

    return model_dict["sets"]["timesteps"][timestep_idx - 1]
end


function get_loc_tech_carriers(model_dict, loc_carrier)

    lookup = model_dict["parameters"]["lookup_loc_carriers"]
    sets = model_dict["sets"]
    loc_tech_carriers = split_comma_list(lookup[loc_carrier])

    loc_tech_carriers_prod = [
        i for i in loc_tech_carriers if i in sets["loc_tech_carriers_prod"]
    ]
    loc_tech_carriers_con = [
        i for i in loc_tech_carriers if i in sets["loc_tech_carriers_con"]
    ]

    if haskey(sets, "loc_tech_carriers_export")
        loc_tech_carriers_export = [
            i for i in loc_tech_carriers if i in sets["loc_tech_carriers_export"]
        ]
    else
        loc_tech_carriers_export = []
    end

    return (
        loc_tech_carriers_prod,
        loc_tech_carriers_con,
        loc_tech_carriers_export
    )
end


function get_loc_tech(loc_tech_carrier)
    return rsplit(loc_tech_carrier, "::", limit=2)[1]
end


function get_timestep_weight(model_dict)
    time_res = model_dict["parameters"]["timestep_resolution"].data
    weights = model_dict["parameters"]["timestep_weights"].data

    return sum(time_res .* weights) / 8760
end


function split_comma_list(comma_list)
    """
    Take a comma deliminated string and split it into a list of strings
    """
    return split(comma_list, ",")
end


function get_conversion_plus_io(model_dict, tier)
    """
    from a carrier_tier, return the primary tier (of `in`, `out`) and
    corresponding decision variable (`carrier_con` and `carrier_prod`, respectively)
    """
    if contains(tier, "out")
        return "out", model_dict["variables"]["carrier_prod"]
    elseif contains(tier, "in")
        return "in", model_dict["variables"]["carrier_con"]
    end
end


function loc_tech_is_in(sets, loc_tech, model_set)
    """
    Check if set exists and if loc_tech is in the set

    Parameters
    ----------
    loc_tech : string
    model_set : string
    """

    if haskey(sets, model_set) && loc_tech in sets[model_set]
        return true
    else
        return false
    end
end

end