using JuMP; using NCDatasets; using AxisArrays; using Util;

function load_network_constraints(model_dict)
    sets = model_dict["sets"]
    constraint_dict = Dict()

    if haskey(sets, "loc_techs_symmetric_transmission_constraint")
        constraint_dict["symmetric_transmission_constraint"] = (
            build_constraint(
                ["loc_techs_symmetric_transmission_constraint"],
                "symmetric_transmission", model_dict
            )
        )
    end

    return constraint_dict
end


function symmetric_transmission_constraint_rule(backend_model, set_indices, model_dict)
    """
    Constrain e_cap symmetrically for transmission nodes. Transmission techs only.
    """
    loc_tech = set_indices[1]
    variables = model_dict["variables"]

    loc_tech_remote = model_dict["parameters"]["lookup_remotes"][loc_tech]

    return @constraint(backend_model,
        variables["energy_cap"][loc_tech] == variables["energy_cap"][loc_tech_remote]
    )
end