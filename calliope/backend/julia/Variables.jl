module Variables

function initialize_decision_variables(backend_model, sets)
    """
    Decision variables are undefined at model initialization, the purpose of the
    optimisation is to find the values that the decision variables should take,
    based on a combination of those decision variables being minimized/maximized.

    The decision variables of interest are:

    energy_cap : Capacity (in kW) of a technology at a given location,
        limiting produced energy
    carrier_prod : Produced energy at every timestep (in kWh) from a technology
        at a given location
    carrier_con : Consumed energy at every timestep (in kWh) from a technology
        at a given location
    cost : Cost associated with a technology (sum of investment and operational)
    resource_area : available area (m2) over which to capture an energy resource
    storage_cap : Stored volume capacity (in kWh) of a technology at a given
        location, limiting stored energy.
    storage : Stored volume at every timestep (in kWh) for a technology at a
        given location
    resource_cap : Resource capacity (in kW or kW/m2 is resource_area used) of
        a technology at a given location, limiting available energy to consume
    resource : Resource consumed at every timestep (in kWh) from a technology
        at a given location
    carrier_export : Energy exported out of the system at every timestep (i.e.
        energy is produced but not consumed via another technology's energy_con,
        instead it is sent outside the system)
    cost_om : Operational and maintenance costs associated with a technology at
        a given location, in every timestep
    cost_investment : Fixed investment costs associated with a technology at a
        given location
    purchased : Binary variable stating whether or not a technology has a
        non-zero capacity in the system at a given location
    units : Integer variable stating how many of a technology (with fixed
        capacity) exist in the system at a given location
    operating_units : Integer variable stating how many integer units of a
        technology are committed in a given timestep, at a given location

    """
    #
    ## Variables which are always assigned
    #

    # Capacity
    @variable(backend_model, energy_cap[loc_techs=sets["loc_techs"]] >= 0);

    # Dispatch
    @variable(backend_model, carrier_prod[loc_techs=sets["loc_techs_carrier_prod"],
        carriers=sets["carriers"], timesteps=sets["timesteps"]] >= 0);
    @variable(backend_model, carrier_con[loc_techs=sets["loc_techs_carrier_con"],
        carriers=sets["carriers"], timesteps=sets["timesteps"]] <= 0);

    # Costs
    @variable(backend_model, cost[loc_techs=sets["loc_techs_cost"],
        costs=sets["costs"]]);

    #
    ## Conditionally assigned variables
    #

    if haskey(sets, "loc_techs_area")
        # Capacity
        @variable(backend_model,
            resource_area[loc_techs=sets["loc_techs_area"]] >= 0
        );
    end

    if haskey(sets, "loc_techs_store")
        # Capacity
        @variable(backend_model,
            storage_cap[loc_techs=sets["loc_techs_store"]] >= 0
        );
        # Dispatch
        @variable(backend_model,
            storage[loc_techs=sets["loc_techs_store"],
                timesteps=sets["timesteps"]
            ] >= 0
        );
    end

    if haskey(sets, "loc_techs_finite_resource")
        # Capacity
        @variable(backend_model,
            resource_cap[loc_techs=sets["loc_techs_finite_resource"]] >= 0
        );
        # Dispatch
        @variable(backend_model,
            resource[loc_techs=sets["loc_techs_finite_resource"],
                timesteps=sets["timesteps"]
            ]
        );
    end

    if haskey(sets, "loc_techs_export")
        # Dispatch
        @variable(backend_model,
            carrier_export[loc_techs=["loc_techs_export"],
                timesteps=sets["timesteps"]
            ] >= 0
        );
    end

    if haskey(sets, "loc_techs_variable_costs")
        # Costs
        @variable(backend_model,
            cost_om[loc_techs=sets["loc_techs_variable_costs"],
                timesteps=sets["timesteps"],
                costs=sets["costs"]
            ]
        );
    end
    if haskey(sets, "loc_techs_investment_costs")
        # Costs
        @variable(backend_model,
            cost_investment[loc_techs=sets["loc_techs_investment_costs"],
                costs=sets["costs"]
            ]
        );
    end

    # Binary/Integer variables
    if haskey(sets, "loc_techs_purchase")
        # Capacity
        @variable(backend_model,
            purchased[loc_techs=sets["loc_techs_purchase"]], Bin
        );
    end

    if haskey(sets, "loc_techs_milp")
        # Capacity
        @variable(backend_model,
            units[loc_techs=sets["loc_techs_milp"]] >=0, Int
        );
        # Dispatch
        @variable(backend_model,
            operating_units[loc_techs=sets["loc_techs_milp"],
                timesteps=sets["timesteps"]
            ] >=0, Int
        );
    end
end

end