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

    def carrier(loc_tech):
        x, y = get_y_x(loc_tech)
        return model.get_carrier(y, 'out', x=x, primary=True)

    def get_y_x(loc_tech):
        return loc_tech.split(":", 1)
    # Constraint rules
    def c_system_margin_rule(m, c):
        # If no margin defined for a carrier, use 0 (i.e. no margin)
        margin = model.config_model.system_margin.get_key(c, default=0)
        if margin:
            t = model.t_max_demand[c]
            return (sum(m.c_prod[c, loc_tech, t] for loc_tech in m.loc_tech if
                        loc_tech not in m.loc_tech_demand) * (1 + margin)
                    <= time_res.at[t] *
                    sum(
                        (m.e_cap[loc_tech] /
                         base.get_constraint_param(model, 'e_eff',
                            get_y_x(loc_tech)[1], get_y_x(loc_tech)[0], t))
                         for loc_tech in m.loc_tech if loc_tech not in
                         m.loc_tech_demand and carrier(loc_tech) == c)
                    )
        else:
            return po.Constraint.NoConstraint

    # Constraints
    m.c_system_margin = po.Constraint(m.c, rule=c_system_margin_rule)
