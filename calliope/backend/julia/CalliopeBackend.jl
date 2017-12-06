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
    ## Sets and Parameters
    #
    global sets, parameters
    sets = Dict()
    parameters = Dict()
    dimensions = [NCDatasets.nc_inq_dimname(dataset.ncid, i)
                  for i in NCDatasets.nc_inq_dimids(dataset.ncid, false)]
    for var in keys(dataset)
        if var in dimensions
            sets[var] = dataset[var][:]
        else
            parameters[var] = get_variable(dataset, var)
        end
    end

    #
    ## Variables
    #
    initialize_decision_variables(backend_model, sets);

    #
    ## Constraints
    #

    constraints = merge(
        load_constraints.load_capacity_constraints(dataset, sets, parameters, backend_model),
        load_constraints.load_costs_constraints(dataset, sets, parameters, backend_model),
        load_constraints.load_dispatch_constraints(dataset, sets, parameters, backend_model),
        load_constraints.load_energy_balance_constraints(dataset, sets, parameters, backend_model),
        load_constraints.load_milp_constraints(dataset, sets, parameters, backend_model),
        load_constraints.load_network_constraints(dataset, sets, parameters, backend_model)
    )
end
end