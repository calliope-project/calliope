"""
Copyright (C) 2013-2016 Stefan Pfenninger.
Licensed under the Apache 2.0 License (see LICENSE file).

objective.py
~~~~~~~~~~~~

Objective functions.

"""

import pyomo.core as po


def objective_cost_minimization(model):
    """
    Minimizes total system monetary cost. Used as a default if
    a model does not specify another objective.

    """
    m = model.m

    def obj_rule(m):
        return sum(model.get_option(y + '.weight') *
                   sum(m.cost[y, x, 'monetary'] for x in m.x)
                   for y in m.y)

    m.obj = po.Objective(sense=po.minimize, rule=obj_rule)
    m.obj.domain = po.NonNegativeReals
