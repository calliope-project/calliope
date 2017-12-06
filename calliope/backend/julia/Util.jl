function get_variable(var)
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
    var_out = AxisArray(dataset[var][:],
        Tuple([AxisArrays.Axis{Symbol(i)}(dataset[i][:])
            for i in NCDatasets.dimnames(dataset[var])]))

    return var_out
end


function timesliced_variable(var, timestep)
    """
    returns an AxisArray of the given variable, indexed over all dimensions
    except timesteps, in order to remove timesteps from a possible timeseries
    variable (which may also be a static variable, we just don't know!)

    Parameters
    ----------
    var : string
        From the variables available in the loaded Dataset
    timestep : DateTime
        one value from from sets["timesteps"]
    Returns
    -------
    AxisArray without the time axis
    """

    if "timesteps" in NCDatasets.dimnames(dataset[var])
        return parameters[var][Axis{:timesteps}(timestep)]
    else
        return var
    end
end


function get_carrier(loc_tech, carrier_tier)
    """
    returns the carrier in/out of the given loc_tech as a string, if a
    conversion_plus technology, it might return a Dict() including the
    carrier_ratios

    Parameters
    ----------
    loc_tech : string
        From the variables available in the loaded Dataset
    carrier_tier : one of ["in", "out", "in_2", "out_2", "in_3", "out_3"]
        these refer to the suffix of the carrier (i.e. "carrier_" + carrier_tier)
    Returns
    -------
    string or Dict()
    """
    tech = split(loc_tech, ":")[2]
    carriers = sets["resources"][
        find(parameters["loc_tech_carriers"][
            Axis{:carrier_tiers}(carrier_tier), Axis{:techs}(tech)]
            )
        ]
    if loc_tech in sets["loc_techs_conversion_plus"]
        carrier_dict = Dict(i =>
            parameters["carrier_ratios"][carrier_tier, i, loc_tech]
            for i in carriers)

        return carrier_dict
    else
        return carriers[1] # Julia is 1 indexed
    end
end


function build_constraint(constraint_sets, constraint_name)
    """
    create a constraint object, indexed over the given sets.

    Parameters
    ----------
    constraint_sets : array of strings
        names of the sets over which the constraint will be indexed
    constraint_name : string
        Name of constraint obect to build. Will also be used to refer to the
        correct constraint rule, which will be constraint_name + "_constraint_rule"

    Return
    ------
    No direct return, the constraint Dictionary constraint_dict will contain an
    additional entry with the key `constraint_name`
    """

    # create a tuple of integer lists
    indexed_sets = [1:length(sets[i]) for i in constraint_sets]

    # Use the Julia @constraintref macro to build an empty constraint object
    temporary_constraint = Array{ConstraintRef}((length(i) for i in indexed_sets)...)
    # get the relevant function from this module
    temporary_constraint_rule = getfield(
        Main, Symbol(string(constraint_name, "_constraint_rule"))
    )

    # Loop through to populate the previously built empty constraint object
    for i in CartesianRange(size(temporary_constraint))
        idx = [item for item in i.I]
        temporary_constraint[i] = temporary_constraint_rule(m,
            (sets[constraint_sets[constr]][idx[constr]] for constr in 1:length(constraint_sets))
        )
    end
    return temporary_constraint
end