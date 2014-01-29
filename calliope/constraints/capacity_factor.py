"""
Copyright (C) 2013 Stefan Pfenninger.
Licensed under the Apache 2.0 License (see LICENSE file).

capacity_factor.py
~~~~~~~~~~~~~~~~~~

Capacity factor constraints.

"""

from __future__ import print_function
from __future__ import division

import coopr.pyomo as cp


def capacity_factor(model):
    """Depends on: node_energy_balance, node_constraints_build"""
    m = model.m

    # Variables
    m.cf_prod = cp.Var(m.c, m.y, m.x, within=cp.NonNegativeReals)

    # Constraint rules
    def c_cf_prod_rule(m, c, y, x):
        return m.cf_prod[c, y, x] == (sum(m.e_prod[c, y, x, t] for t in m.t)
                                      / (m.e_cap[y, x]
                                         * sum(m.time_res[t] for t in m.t)))

    def c_cf_prod_max_rule(m, c, y, x):
        if model.get_option(y + '.constraints.cf_max'):
            return (m.cf_prod[c, y, x]
                    <= model.get_option(y + '.constraints.cf_max'))
        else:
            return cp.Constraint.NoConstraint

    # Constraints
    m.c_cf_prod = cp.Constraint(m.c, m.y, m.x)
    m.c_cf_prod_max = cp.Constraint(m.c, m.y, m.x)