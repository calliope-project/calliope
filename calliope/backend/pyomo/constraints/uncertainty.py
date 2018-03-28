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

    def _cost_equation(backend_model, scenario):
        return sum(
        backend_model.cost['monetary', loc_tech, scenario]
        for loc_tech in backend_model.loc_techs_cost
    )

    cost_sum = _cost_equation(backend_model, scenario)

    return cost_sum - backend_model.xi <= backend_model.eta[scenario]
