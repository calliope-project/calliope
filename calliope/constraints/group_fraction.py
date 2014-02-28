"""
Copyright (C) 2013 Stefan Pfenninger.
Licensed under the Apache 2.0 License (see LICENSE file).

group_fraction.py
~~~~~~~~~~~~~~~~~

Constrain groups of technologies to reach given fractions of e_prod.

"""

from __future__ import print_function
from __future__ import division

import coopr.pyomo as cp


def group_fraction(model):
    """
    Depends on:
    """
    m = model.m
    o = model.config_model

    def fraction(group):
        return model.config_model.group_fraction.get_key(group)

    supply_techs = (model.get_group_members('supply') +
                    model.get_group_members('conversion'))

    # Sets
    m.group = cp.Set(initialize=model.config_model.group_fraction.keys())

    # Constraint rules
    def c_group_fraction_rule(m, c, group):
        rhs = (fraction(group)
               * sum(m.e_prod[c, y, x, t] for y in supply_techs
                     for x in m.x for t in m.t))
        lhs = sum(m.e_prod[c, y, x, t]
                  for y in model.get_group_members(group) for x in m.x
                  for t in m.t)
        if o.group_fraction_mode == 'lesser_or_equal':
            return lhs <= rhs
        elif o.group_fraction_mode == 'greater_or_equal':
            return lhs >= rhs
        elif o.group_fraction_mode == 'equal':
            return lhs == rhs

    # Constraints
    m.c_group_fraction = cp.Constraint(m.c, m.group)
