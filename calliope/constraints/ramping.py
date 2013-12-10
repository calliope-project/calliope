from __future__ import print_function
from __future__ import division

import coopr.pyomo as cp


def ramping_rate(model):
    """Depends on: node_energy_balance, node_constraints_build"""
    m = model.m

    # Constraint rules
    def c_ramping_rule(m, y, x, t):
        try:
            # e_ramping: Ramping rate [fraction of installed capacity per hour]
            ramping_rate = model.get_option(y + '.constraints.e_ramping')
        except KeyError:
            # If the technology defines no `e_ramping`, we don't build a
            # ramping constraint for it!
            return cp.Constraint.NoConstraint
        # If there was no KeyError, we build and return a constraint
        if m.t.order_dict[t] > 1:
            diff = m.e[y, x, t] - m.e[y, x, model.prev(t)]
        else:
            diff = 0
        return diff <= ramping_rate * m.time_res[t] * m.e_cap[y, x]

    # Constraints
    m.c_ramping = cp.Constraint(m.y, m.x, m.t)
