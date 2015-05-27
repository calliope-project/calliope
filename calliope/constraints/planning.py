"""
Copyright (C) 2013-2015 Stefan Pfenninger.
Licensed under the Apache 2.0 License (see LICENSE file).

planning.py
~~~~~~~~~~~

Planning constraints.

"""

import numpy as np
import pyomo.core as po


def node_constraints_build_total(model):
    m = model.m

    # Constraint rules
    def c_e_cap_total_systemwide_rule(m, y):
        e_cap_max_total = model.get_option(y + '.constraints.'
                                           'e_cap_max_total')
        e_cap_max_total_force = model.get_option(y + '.constraints.'
                                                 'e_cap_max_total_force')
        e_cap_max_scale = model.get_option(y + '.constraints.e_cap_max_scale')

        if np.isinf(e_cap_max_total):
            return po.Constraint.NoConstraint
        else:
            sum_expr = sum(m.e_cap[y, x] for x in m.x)
            total_expr = e_cap_max_total * e_cap_max_scale
            if e_cap_max_total_force:
                return sum_expr == total_expr
            elif model.mode == 'plan':
                return sum_expr <= total_expr

    # Constraints
    m.c_e_cap_total_systemwide = \
        po.Constraint(m.y, rule=c_e_cap_total_systemwide_rule)


def system_margin(model):
    """

    """
    m = model.m
    time_res = model.data.time_res_series

    def carrier(y):
        return model.get_option(y + '.carrier')

    # Constraint rules
    def c_system_margin_rule(m, c):
        # If no margin defined for a carrier, use 0 (i.e. no margin)
        margin = model.config_model.system_margin.get_key(c, default=0)
        if margin:
            t = model.t_max_demand[c]
            return (sum(m.es_prod[c, y, x, t] for y in m.y for x in m.x)
                    * (1 + margin)
                    <= time_res.at[t]
                    * sum((m.e_cap[y, x] / model.get_eff_ref('e', y, x))
                          for y in m.y if carrier(y) == c
                          for x in m.x))
        else:
            return po.Constraint.NoConstraint

    # Constraints
    m.c_system_margin = po.Constraint(m.c, rule=c_system_margin_rule)
