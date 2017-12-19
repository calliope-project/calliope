module CalliopeBackend

export build_julia_model, model_dict

# Import external modules
using NCDatasets; using JuMP; using AxisArrays; using YAML;

using CPLEX; using GLPKMathProgInterface; using Gurobi

# Import internal modules
using Util, Variables, Objective

module load_constraints

    export
        load_capacity_constraints, load_costs_constraints,
        load_dispatch_constraints, load_energy_balance_constraints,
        load_milp_constraints, load_network_constraints,
        load_conversion_constraints, load_conversion_plus_constraints,
        load_export_constraints

    include(joinpath("constraints", "capacity.jl"))
    include(joinpath("constraints", "costs.jl"))
    include(joinpath("constraints", "dispatch.jl"))
    include(joinpath("constraints", "energy_balance.jl"))
    include(joinpath("constraints", "milp.jl"))
    include(joinpath("constraints", "network.jl"))
    include(joinpath("constraints", "conversion.jl"))
    include(joinpath("constraints", "conversion_plus.jl"))
    include(joinpath("constraints", "export.jl"))

end

function build_julia_model(path_to_dataset)

    # Bring in Dataset
    dataset = NCDatasets.Dataset(path_to_dataset);

    #
    ## Sets
    #
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

    # Create JuMP model, including assigning a solver
    solver_dict = Dict(
        "cplex"=>CplexSolver,
        "gurobi"=>GurobiSolver
    )

    if haskey(sets, "loc_techs_milp") || haskey(sets, "loc_techs_purchase")
        solver_dict["glpk"] = GLPKSolverMIP
    else
        solver_dict["glpk"] = GLPKSolverLP
    end

    backend_model = JuMP.Model(solver = solver_dict[dataset.attrib["run.solver"]]());

    model_dict = Dict("backend_model"=>backend_model,
                      "parameters"=>parameters,
                      "defaults"=>YAML.load(dataset.attrib["defaults"]),
                      "sets"=>sets,
                      "dataset"=>dataset)

    #
    ## Variables
    #
    model_dict["variables"] = initialize_decision_variables(model_dict);

    #
    ## Expressions
    #
    model_dict["expressions"] = Dict()

    #
    ## Constraints
    #

    model_dict["constraints"] = merge(
        load_constraints.load_capacity_constraints(model_dict),
        load_constraints.load_cost_constraints(model_dict),
        load_constraints.load_dispatch_constraints(model_dict),
        load_constraints.load_energy_balance_constraints(model_dict),
        load_constraints.load_network_constraints(model_dict)
    )

    if haskey(sets, "loc_techs_conversion")
        model_dict["constraints"] = merge(
            model_dict["constraints"],
            load_constraints.load_conversion_constraints(model_dict)
        )
    end

    if haskey(sets, "loc_techs_conversion_plus")
        model_dict["constraints"] = merge(
            model_dict["constraints"],
            load_constraints.load_conversion_plus_constraints(model_dict)
        )
    end

    if haskey(sets, "loc_techs_milp") || haskey(sets, "loc_techs_purchase")
        model_dict["constraints"] = merge(
            model_dict["constraints"],
            load_constraints.load_milp_constraints(model_dict)
        )
    end

    if haskey(sets, "loc_techs_export")
        model_dict["constraints"] = merge(
            model_dict["constraints"],
            load_constraints.load_export_constraints(model_dict)
        )
    end

    #
    ## Objective
    #
    cost_minimization(model_dict)

    solve(model_dict["backend_model"])
    return model_dict

end
end