"""
Copyright (C) 2013-2018 Calliope contributors listed in AUTHORS.

Licensed under the Apache 2.0 License (see LICENSE file).

objective.py
~~~~~~~~~~~~

Objective functions.

"""

import pyomo.core as po  # pylint: disable=import-error
from calliope.core.util.tools import load_function
from calliope.backend.pyomo.util import get_param

def minmax_cost_optimization(backend_model, cost_class, sense, **kwargs):
    """
    Minimize or maximise total system cost for specified cost class.

    If unmet_demand is in use, then the calculated cost of unmet_demand is
    added or subtracted from the total cost in the opposite sense to the
    objective.

    .. container:: scrolling-wrapper

        .. math::

            min: z = \sum_{loc::tech_{cost}} cost(loc::tech, cost=cost_{k})) + \sum_{loc::carrier,timestep} unmet\_demand(loc::carrier, timestep) \\times bigM
            max: z = \sum_{loc::tech_{cost}} cost(loc::tech, cost=cost_{k})) - \sum_{loc::carrier,timestep} unmet\_demand(loc::carrier, timestep) \\times bigM

    """
    def obj_rule(backend_model):
        if hasattr(backend_model, 'unmet_demand'):
            unmet_demand = sum(
                (backend_model.unmet_demand[loc_carrier, scenario, timestep] -
                backend_model.excess_supply[loc_carrier, scenario, timestep]) *
                get_param(backend_model, 'timestep_weights', timesteps=timestep, scenarios=scenario)
                for loc_carrier in backend_model.loc_carriers
                for scenario in backend_model.scenarios
                for timestep in backend_model.timesteps
            ) * backend_model.bigM
            if sense == 'maximize':
                unmet_demand *= -1
        else:
            unmet_demand = 0

        return (
            sum(
                backend_model.cost[cost_class, loc_tech, scenario]
                for loc_tech in backend_model.loc_techs_cost
                for scenario in backend_model.scenarios
            ) + unmet_demand
        )

    backend_model.obj = po.Objective(sense=load_function('pyomo.core.' + sense),
                                     rule=obj_rule)
    backend_model.obj.domain = po.Reals


def minmax_map_costs(backend_model, cost_class_1, cost_class_2, sense, map_2_to_1, **kwargs):
    """
    Minimize or maximise total system cost for specified cost class. This function
    considers a secondary cost class, which maps to the primary cost class using
    a mapping value (`map_2_to_1`). This is a generalisation of the use of a cost
    of carbon to map 'carbon' (`cost_class_2`) to 'monetary' (`cost_class_1`) cost.

    If unmet_demand is in use, then the calculated cost of unmet_demand is
    added or subtracted from the total cost in the opposite sense to the
    objective.

    .. container:: scrolling-wrapper

        .. math::

            min: z = \\sum_{loc::tech_{cost}} cost(loc::tech, cost=cost_{k_1})) +
                cost(loc::tech, cost=cost_{k_2})) \\times mapping_{k_1/k_2} +
                \\sum_{loc::carrier,timestep} unmet\_demand(loc::carrier, timestep) \\times bigM
            max: z = \\sum_{loc::tech_{cost}} cost(loc::tech, cost=cost_{k_1})) +
                cost(loc::tech, cost=cost_{k_2})) \\times mapping_{k_1/k_2} -
                \\sum_{loc::carrier,timestep} unmet\_demand(loc::carrier, timestep) \\times bigM

    """
    def obj_rule(backend_model):
        if hasattr(backend_model, 'unmet_demand'):
            unmet_demand = sum(
                (backend_model.unmet_demand[loc_carrier, scenario, timestep] -
                backend_model.excess_supply[loc_carrier, scenario, timestep]) *
                get_param(backend_model, 'timestep_weights', timesteps=timestep, scenarios=scenario)
                for loc_carrier in backend_model.loc_carriers
                for scenario in backend_model.scenarios
                for timestep in backend_model.timesteps
            ) * backend_model.bigM
            if sense == 'maximize':
                unmet_demand *= -1
        else:
            unmet_demand = 0

        return (
            sum(
                backend_model.cost[cost_class_1, loc_tech, scenario] +
                backend_model.cost[cost_class_2, loc_tech, scenario] * map_2_to_1
                for loc_tech in backend_model.loc_techs_cost
                for scenario in backend_model.scenarios
            ) + unmet_demand
        )

    backend_model.obj = po.Objective(sense=load_function('pyomo.core.' + sense),
                                     rule=obj_rule)
    backend_model.obj.domain = po.Reals


def check_feasibility(backend_model, **kwargs):
    """
    Dummy objective, to check that there are no conflicting constraints.

    .. container:: scrolling-wrapper

        .. math::

            min: z = 1

    """
    def obj_rule(backend_model):
        return 1

    backend_model.obj = po.Objective(sense=po.minimize, rule=obj_rule)
    backend_model.obj.domain = po.Reals


def risk_aware_cost_minimization(backend_model, cost_class, sense, **kwargs):
    """
    Minimizes total system monetary cost and the sum of conditional value at risk.
    Used as a default if a model does not specify another objective.

    .. math::

        min: z = \\sum_{scenarios} (
            probability(scenario) \\times
            \\sum_{loc::techs_{cost}} \\boldsymbol{cost}('monetary', loc::tech, scenario)
        ) + beta \\times (
            \\boldsymbol{\\xi} + \\frac{1}{1 - alpha} \\times
            \\sum_{scenarios} (
                probability(scenario) \\times \\boldsymbol{\\eta}(scenario)
            )
        )

    """

    # beta is the degree of risk aversion, which can be anything from 0 (no
    # risk aversion) to infinity (infinite risk aversion)
    beta = backend_model.beta
    # alpha is the percentile of the cost distribution associated with Value at
    # Risk (VaR)
    alpha = backend_model.alpha

    if ('cost_class_2' in kwargs.keys() and
            kwargs['cost_class_2'] != cost_class and 'map_2_to_1' in kwargs.keys()):
        cost_classes = {cost_class: 1, kwargs['cost_class_2']: kwargs['map_2_to_1']}
    else:
        cost_classes = {cost_class: 1}

    def get_unmet_demand(backend_model, scenario):
        if hasattr(backend_model, 'unmet_demand'):
            unmet_demand = sum(
                (backend_model.unmet_demand[loc_carrier, scenario, timestep] -
                 backend_model.excess_supply[loc_carrier, scenario, timestep]) *
                get_param(backend_model, 'timestep_weights', timesteps=timestep, scenarios=scenario)
                for loc_carrier in backend_model.loc_carriers
                for timestep in backend_model.timesteps
            ) * backend_model.bigM
            if sense == 'maximize':
                unmet_demand *= -1
        else:
            unmet_demand = 0

        return unmet_demand

    def cost_equation(backend_model, cost_class, map_to_cost_class_1, scenario):
        return sum(
            backend_model.cost[cost_class, loc_tech, scenario]
            for loc_tech in backend_model.loc_techs_cost
        ) * map_to_cost_class_1

    def obj_rule(backend_model):
        return (
            sum(
                backend_model.probability[scenario] *
                (cost_equation(backend_model, k, v, scenario))
                for scenario in backend_model.scenarios
                for k, v in cost_classes.items()
            ) + sum(
                backend_model.probability[scenario] *
                get_unmet_demand(backend_model, scenario)
                for scenario in backend_model.scenarios
            ) + beta * (
                backend_model.xi + 1 / (1 - alpha) *
                sum(
                    backend_model.probability[scenario] *
                    backend_model.eta[scenario]
                    for scenario in backend_model.scenarios
                )
            )
        )

    backend_model.obj = po.Objective(sense=load_function('pyomo.core.' + sense),
                                     rule=obj_rule)
    backend_model.obj.domain = po.Reals
