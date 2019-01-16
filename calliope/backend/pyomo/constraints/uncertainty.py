"""
Copyright (C) 2013-2018 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

uncertainty.py
~~~~~~~~~

Constraints to handle uncertainty, currently specific to probabilistic scenarios

"""
import pyomo.core as po
from calliope.backend.pyomo.util import get_param

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
    cost_class = backend_model.__calliope_model_data__['attrs'].get(
        'run.objective_options.CVaR_cost_class',
        backend_model.__calliope_model_data__['attrs']['run.objective_options.cost_class']
    )
    tech = backend_model.__calliope_model_data__['attrs'].get(
        'run.objective_options.CVaR_tech', None
    )

    if cost_class == 'cost_of_carbon':
        cost_of_carbon = backend_model.__calliope_model_data__['attrs'].get(
            'run.objective_options.map_2_to_1', 1
        )
        cost_class = 'carbon'
    else:
        cost_of_carbon = 1

    if hasattr(backend_model, 'unmet_demand'):
        unmet_demand = sum(
            (backend_model.unmet_demand[loc_carrier, scenario, timestep] -
             backend_model.excess_supply[loc_carrier, scenario, timestep]) *
            get_param(backend_model, 'timestep_weights', timesteps=timestep, scenarios=scenario)
            for loc_carrier in backend_model.loc_carriers
            for timestep in backend_model.timesteps
        ) * backend_model.bigM
    else:
        unmet_demand = 0

    if cost_class == 'unmet_demand':
        cost_sum = unmet_demand
    else:
        if tech is not None:
            loc_techs = [i for i in backend_model.loc_techs_cost if i.split('::')[-1] == tech]
        else:
            loc_techs = backend_model.loc_techs_cost
        cost_sum = cost_of_carbon * sum(
            backend_model.cost[cost_class, loc_tech, scenario]
            for loc_tech in loc_techs
        ) + unmet_demand

    return cost_sum - backend_model.xi <= backend_model.eta[scenario]
