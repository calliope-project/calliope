"""
Copyright (C) 2013-2015 Stefan Pfenninger.
Licensed under the Apache 2.0 License (see LICENSE file).

ramping.py
~~~~~~~~~~

Ramping constraints.

"""

import pyomo.core as po


def ramping_rate(model):
    """Depends on: node_energy_balance, node_constraints_build"""
    m = model.m
    time_res = model.data.time_res_series

    # Constraint rules
    def _ramping_rule(m, y, x, t, direction):
        # e_ramping: Ramping rate [fraction of installed capacity per hour]
        ramping_rate = model.get_option(y + '.constraints.e_ramping')
        if ramping_rate is False:
            # If the technology defines no `e_ramping`, we don't build a
            # ramping constraint for it!
            return po.Constraint.NoConstraint
        else:
            # No constraint for first timestep
            # NB: From Pyomo 3.5 to 3.6, order_dict became zero-indexed
            if m.t.order_dict[t] == 0:
                return po.Constraint.NoConstraint
            else:
                carrier = model.get_option(y + '.carrier')
                diff = ((m.es_prod[carrier, y, x, t]
                         + m.es_con[carrier, y, x, t]) / time_res.at[t]
                        - (m.es_prod[carrier, y, x, model.prev(t)]
                           + m.es_con[carrier, y, x, model.prev(t)])
                        / time_res.at[model.prev(t)])
                max_ramping_rate = ramping_rate * m.e_cap[y, x]
                if direction == 'up':
                    return diff <= max_ramping_rate
                else:
                    return -1 * max_ramping_rate <= diff

    def c_ramping_up_rule(m, y, x, t):
        return _ramping_rule(m, y, x, t, direction='up')

    def c_ramping_down_rule(m, y, x, t):
        return _ramping_rule(m, y, x, t, direction='down')

    # Constraints
    m.c_ramping_up = po.Constraint(m.y, m.x, m.t, rule=c_ramping_up_rule)
    m.c_ramping_down = po.Constraint(m.y, m.x, m.t, rule=c_ramping_down_rule)
