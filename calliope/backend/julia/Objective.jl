module Objective

export cost_minimization

using JuMP

function cost_minimization(model_dict)
    """
    Minimizes total system monetary cost.

    """
    backend_model= model_dict["backend_model"]
    cost = model_dict["variables"]["cost"]
    loc_techs_cost = model_dict["sets"]["loc_techs_cost"]

    @objective(backend_model, :Min,
        sum(cost["monetary", loc_tech] for loc_tech in loc_techs_cost)
    )

end
end