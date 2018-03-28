"""
Copyright (C) 2013-2018 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

uncertainty.py
~~~~~~~~~

Constraints specific to robust optimisation, which deals with uncertainty between
porbabilistic scenarios.

"""
import pyomo.core as po


def load_constraints(backend_model):
    backend_model.CVaR_constraint = po.Constraint(
        backend_model.scenarios,
        rule=CVaR_constraint_rule
    )


def CVaR_constraint_rule(backend_model, scenario):
    """
    This constraint fits the auxiliary variables into the backend_model

    .. math::

        \\sum_{loc::techs_{cost}} \\boldsymbol{cost}('monetary', loc::tech, scenario)
        - \\boldsymbol{\\xi} \\leq \\boldsymbol{\\eta}(scenario)
        \\quad \\forall scenario \\in scenarios

    """

    if hasattr(backend_model, 'unmet_demand'):
        unmet_demand = sum(
            backend_model.unmet_demand[loc_carrier, scenario, timestep]
            for loc_carrier in backend_model.loc_carriers
            for timestep in backend_model.timesteps
        ) * backend_model.bigM
    else:
        unmet_demand = 0

    cost_sum = sum(
        backend_model.cost['monetary', loc_tech, scenario]
        for loc_tech in backend_model.loc_techs_cost
    ) + unmet_demand

    return cost_sum - backend_model.xi <= backend_model.eta[scenario]
