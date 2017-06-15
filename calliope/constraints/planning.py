"""
Copyright (C) 2013-2017 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

planning.py
~~~~~~~~~~~

Planning constraints.

"""

import numpy as np
import pyomo.core as po  # pylint: disable=import-error
from . import base


def node_constraints_build_total(model):
    """

    """
    m = model.m

    # Constraint rules
    def c_e_cap_total_systemwide_rule(m, y):
        total_max = model.get_option(y + '.constraints.e_cap.total_max')
        total_equals = model.get_option(y + '.constraints.e_cap.total_equals')
        scale = model.get_option(y + '.constraints.e_cap_scale')

        if np.isinf(total_max) and not total_equals:
            return po.Constraint.NoConstraint

        sum_expr = sum(m.e_cap[y, x] for x in m.x)
        total_expr = total_equals * scale if total_equals else total_max * scale

        if total_equals:
            return sum_expr == total_expr
        else:
            return sum_expr <= total_expr

    # Constraints
    m.c_e_cap_total_systemwide = \
        po.Constraint(m.y, rule=c_e_cap_total_systemwide_rule)


def system_margin(model):
    """

    """
    m = model.m
    time_res = model.data['_time_res'].to_series()

    def carrier(y, x):
        if y in m.y_conversion_plus:
            return model.get_cp_carriers(y, x)[0]
        else:
            return model.get_option(y + '.carrier', default=y + '.carrier_out')

    # Constraint rules
    def c_system_margin_rule(m, c):
        # If no margin defined for a carrier, use 0 (i.e. no margin)
        margin = model.config_model.system_margin.get_key(c, default=0)
        if margin:
            t = model.t_max_demand[c]
            return (sum(m.c_prod[c, y, x, t] for y in m.y if y not in m.y_demand
                        for x in m.x) * (1 + margin)
                    <= time_res.at[t] *
                    sum(
                        (m.e_cap[y, x] / base.get_constraint_param(model, 'e_eff', y, x, t))
                        for y in m.y if y not in m.y_demand
                        for x in m.x if carrier(y, x) == c)
                    )
        else:
            return po.Constraint.NoConstraint

    # Constraints
    m.c_system_margin = po.Constraint(m.c, rule=c_system_margin_rule)
