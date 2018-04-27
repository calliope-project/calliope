"""
Copyright (C) 2013-2018 Calliope contributors listed in AUTHORS.

Licensed under the Apache 2.0 License (see LICENSE file).

objective.py
~~~~~~~~~~~~

Objective functions.

"""

import pyomo.core as po  # pylint: disable=import-error
from calliope.core.util.tools import load_function


def minmax_cost_optimization(backend_model, cost_class, sense):
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
                backend_model.unmet_demand[loc_carrier, timestep]
                for loc_carrier in backend_model.loc_carriers
                for timestep in backend_model.timesteps
            ) * backend_model.bigM
            if sense == 'maximize':
                unmet_demand *= -1
        else:
            unmet_demand = 0

        return (
            sum(
                backend_model.cost[cost_class, loc_tech]
                for loc_tech in backend_model.loc_techs_cost
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
