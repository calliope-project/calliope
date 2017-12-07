module CalliopeBackend

export build_julia_model

# Import external modules
using NCDatasets; using JuMP; using AxisArrays;

# Import internal modules
using Util, Variables

module load_constraints

    export
        load_capacity_constraints, load_costs_constraints,
        load_dispatch_constraints, load_energy_balance_constraints,
        load_milp_constraints, load_network_constraints

    include(joinpath("constraints", "capacity.jl"))
    include(joinpath("constraints", "costs.jl"))
    include(joinpath("constraints", "dispatch.jl"))
    include(joinpath("constraints", "energy_balance.jl"))
    include(joinpath("constraints", "milp.jl"))
    include(joinpath("constraints", "network.jl"))

end

function build_julia_model(path_to_dataset)

    # Bring in Dataset
    dataset = NCDatasets.Dataset(path_to_dataset);

    # Create JuMP model
    backend_model = JuMP.Model();

    #
    ## Sets
    #
    sets = Dict()
    dimensions = [NCDatasets.nc_inq_dimname(dataset.ncid, i)
                  for i in NCDatasets.nc_inq_dimids(dataset.ncid, false)]
    for var in keys(dataset)
        if var in dimensions
            sets[var] = dataset[var][:]
        end
    end
    model_dict = Dict("backend_model"=>backend_model,
                      "dataset"=>dataset,
                      "defaults"=>JSON.parse(dataset.attrib["defaults"]),
                      "sets"=>sets)

    #
    ## Variables
    #
    initialize_decision_variables(model_dict);

    #
    ## Constraints
    #

    constraints = merge(
        load_constraints.load_capacity_constraints(model_dict),
        load_constraints.load_costs_constraints(model_dict),
        load_constraints.load_dispatch_constraints(model_dict),
        load_constraints.load_energy_balance_constraints(model_dict),
        load_constraints.load_milp_constraints(model_dict),
        load_constraints.load_network_constraints(model_dict)
    )

    return backend_model
end
end