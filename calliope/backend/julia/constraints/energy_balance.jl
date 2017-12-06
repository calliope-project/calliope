function load_energy_balance_constraints(dataset, sets, parameters, backend_model)

end

# -------- Resource availablity -------->
function resource_availability_constraint_rule(m, loc_tech, timestep)
    # not tested this to see if it warns as expected

    if loc_tech in [sets["loc_techs_supply"]; sets["loc_techs_demand"]]
        # initialise the affine expression `prod` in model `m` for supply techs
        e_eff = 1 / timesliced_variable("energy_eff", timestep)[loc_tech] if \
            timesliced_variable("energy_eff", timestep)[loc_tech] != 0 else 0
        @expression(m, prod,
            e_eff * sum(carrier_prod[carrier, loc_tech, timestep]
                        for carrier in sets["carriers"])
        )
        # demand techs
        @expression(m, con, sum(c_con[carrier, loc_tech, timestep]
            for carrier in sets["carriers"]) *
            timesliced_variable("energy_eff", timestep)[loc_tech]
        )

        @expression(m, available_resource,
            parameters["resource"][loc_tech, timestep] *
            parameters["resource_scale"][loc_tech]
        )

        if loc_tech in sets["loc_techs_area"]
            available_resource *= resource_area[loc_tech]
        end

        # creating the constraint just requires putting together the affine expressions
        if timesliced_variable("force_resource", timestep)[loc_tech] == true
            @constraint(m, prod + con == available_resource)
        else:
            @constraint(m, prod - con <= available_resource)
        end

    elseif loc_tech in sets["loc_techs_supply_plus"]
        @expression(m, available_resource,
            parameters["resource"][loc_tech, timestep] *
            parameters["resource_scale"][loc_tech] *
            timesliced_variable("resource_eff", timestep)[loc_tech]
        )

        if loc_tech in sets["loc_techs_area"]
            m.available_resource *= resource_area[loc_tech]
        end

        if timesliced_variable("force_resource", timestep)[loc_tech] == true
            @constraint(m, resource[loc_tech, timestep] == available_resource)
        else:
            @constraint(m, resource[loc_tech, timestep] <= available_resource)
        end
    end
end