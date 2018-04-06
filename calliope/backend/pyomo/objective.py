"""
Copyright (C) 2013-2018 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

objective.py
~~~~~~~~~~~~

Objective functions.

"""

import pyomo.core as po  # pylint: disable=import-error


def cost_minimization(backend_model):
    """
    Minimizes total system monetary cost.

    .. container:: scrolling-wrapper

        .. math::

            min: z = \sum_{loc::tech_{cost}} cost(loc::tech, cost=cost_{monetary})) + \sum_{loc::carrier,timestep} unmet\_demand(loc::carrier, timestep) \\times bigM

    """
    def obj_rule(backend_model):
        if hasattr(backend_model, 'unmet_demand'):
            unmet_demand = sum(
                backend_model.unmet_demand[loc_carrier, timestep]
                for loc_carrier in backend_model.loc_carriers
                for timestep in backend_model.timesteps
            ) * backend_model.bigM
        else:
            unmet_demand = 0

        return (
            sum(
                backend_model.cost['monetary', loc_tech]
                for loc_tech in backend_model.loc_techs_cost
            ) + unmet_demand
        )

    backend_model.obj = po.Objective(sense=po.minimize, rule=obj_rule)
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
