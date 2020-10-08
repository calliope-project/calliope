"""
Copyright (C) since 2013 Calliope contributors listed in AUTHORS.

Licensed under the Apache 2.0 License (see LICENSE file).

objective.py
~~~~~~~~~~~~

Objective functions.

"""

import pyomo.core as po  # pylint: disable=import-error
from calliope.core.util.tools import load_function


def minmax_cost_optimization(backend_model):
    """
    Minimize or maximise total system cost for specified cost class or a set of cost classes.
    cost_class is a string or dictionary. If a string, it is automatically converted to a
    dictionary with a single key:value pair where value == 1. The dictionary provides a weight
    for each cost class of interest: {cost_1: weight_1, cost_2: weight_2, etc.}.

    If unmet_demand is in use, then the calculated cost of unmet_demand is
    added or subtracted from the total cost in the opposite sense to the
    objective.

    .. container:: scrolling-wrapper

        .. math::

            min: z = \\sum_{loc::tech_{cost},k} (cost(loc::tech, cost=cost_{k}) \\times weight_{k}) +
             \\sum_{loc::carrier,timestep} (unmet\\_demand(loc::carrier, timestep) \\times bigM)

            max: z = \\sum_{loc::tech_{cost},k} (cost(loc::tech, cost=cost_{k}) \\times weight_{k}) -
             \\sum_{loc::carrier,timestep} (unmet\\_demand(loc::carrier, timestep) \\times bigM)

    """

    def obj_rule(backend_model):
        if backend_model.__calliope_run_config.get("ensure_feasibility", False):
            unmet_demand = (
                sum(
                    (
                        backend_model.unmet_demand[loc_carrier, timestep]
                        - backend_model.unused_supply[loc_carrier, timestep]
                    )
                    * backend_model.timestep_weights[timestep]
                    for loc_carrier in backend_model.loc_carriers
                    for timestep in backend_model.timesteps
                )
                * backend_model.bigM
            )
            if backend_model.objective_sense == "maximize":
                unmet_demand *= -1
        else:
            unmet_demand = 0

        return (
            sum(
                backend_model.cost[k, loc_tech] * v
                for loc_tech in backend_model.loc_techs_cost
                for k, v in backend_model.objective_cost_class.items()
            )
            + unmet_demand
        )

    backend_model.obj = po.Objective(
        sense=load_function("pyomo.core." + backend_model.objective_sense),
        rule=obj_rule,
    )
    backend_model.obj.domain = po.Reals


def check_feasibility(backend_model):
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
