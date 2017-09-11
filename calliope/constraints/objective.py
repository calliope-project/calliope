"""
Copyright (C) 2013-2017 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

objective.py
~~~~~~~~~~~~

Objective functions.

"""

import pyomo.core as po  # pylint: disable=import-error

def objective_cost_minimization(model):
    """
    Minimizes total system monetary cost.
    Used as a default if a model does not specify another objective.

    """
    m = model.m

    def get_y(loc_tech):
        return loc_tech.split(":", 1)[1]

    def obj_rule(m):
        return sum(model.get_option(get_y(loc_tech) + '.weight') *
                       m.cost[loc_tech, 'monetary'] for loc_tech in m.loc_tech)

    m.obj = po.Objective(sense=po.minimize, rule=obj_rule)
    m.obj.domain = po.Reals
