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

    def sign_fraction(group, group_type):
        o = model.config_model
        sign, fraction = o.group_fraction[group_type].get_key(group)
        return sign, fraction

    def group_set(group_type):
        try:
            group = model.config_model.group_fraction[group_type].keys()
        except KeyError:
            group = []
        return cp.Set(initialize=group)

    def equalizer(lhs, rhs, sign):
        if sign == '<=':
            return lhs <= rhs
        elif sign == '>=':
            return lhs >= rhs
        elif sign == '==':
            return lhs == rhs

    supply_techs = (model.get_group_members('supply') +
                    model.get_group_members('conversion'))

    # Sets
    m.output_group = group_set('output')
    m.capacity_group = group_set('capacity')

    # Constraint rules
    def c_group_fraction_output_rule(m, c, output_group):
        sign, fraction = sign_fraction(output_group, 'output')
        rhs = (fraction
               * sum(m.es_prod[c, y, x, t] for y in supply_techs
                     for x in m.x for t in m.t))
        lhs = sum(m.es_prod[c, y, x, t]
                  for y in model.get_group_members(output_group) for x in m.x
                  for t in m.t)
        return equalizer(lhs, rhs, sign)

    def c_group_fraction_capacity_rule(m, c, capacity_group):
        sign, fraction = sign_fraction(capacity_group, 'capacity')
        rhs = (fraction
               * sum(m.e_cap[y, x] for y in supply_techs for x in m.x))
        lhs = sum(m.e_cap[y, x] for y in model.get_group_members(capacity_group)
                  for x in m.x)
        return equalizer(lhs, rhs, sign)

    # Constraints
    m.c_group_fraction_output = cp.Constraint(m.c, m.output_group)
    m.c_group_fraction_capacity = cp.Constraint(m.c, m.capacity_group)
