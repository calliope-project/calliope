from __future__ import print_function
from __future__ import division

import coopr.pyomo as cp


def ramping_rate(model):
    """Depends on: node_energy_balance, node_constraints_build"""
    m = model.m

    # Constraint rules
    def _ramping_rule(m, y, x, t, direction):
        try:
            # e_ramping: Ramping rate [fraction of installed capacity per hour]
            ramping_rate = model.get_option(y + '.constraints.e_ramping')
        except KeyError:
            # If the technology defines no `e_ramping`, we don't build a
            # ramping constraint for it!
            return cp.Constraint.NoConstraint
        # If there was no KeyError, we build and return a constraint
        if m.t.order_dict[t] <= 1:
            return cp.Constraint.NoConstraint
        else:
            carrier = model.get_option(y + '.carrier')
            diff = m.e[carrier, y, x, t] - m.e[carrier, y, x, model.prev(t)]
            max_ramping_rate = ramping_rate * m.time_res[t] * m.e_cap[y, x]
            if direction == 'up':
                return diff <= max_ramping_rate
            else:
                return -1 * max_ramping_rate <= diff

    def c_ramping_up_rule(m, y, x, t):
        return _ramping_rule(m, y, x, t, direction='up')

    def c_ramping_down_rule(m, y, x, t):
        return _ramping_rule(m, y, x, t, direction='down')

    # Constraints
    m.c_ramping_up = cp.Constraint(m.y, m.x, m.t)
    m.c_ramping_down = cp.Constraint(m.y, m.x, m.t)
