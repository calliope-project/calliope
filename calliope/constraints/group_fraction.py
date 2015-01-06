"""
Copyright (C) 2013-2015 Stefan Pfenninger.
Licensed under the Apache 2.0 License (see LICENSE file).

group_fraction.py
~~~~~~~~~~~~~~~~~

Constrain groups of technologies to reach given fractions of e_prod.

"""

import pyomo.core as po


def group_fraction(model):
    """

    """
    m = model.m

    def sign_fraction(group, group_type):
        o = model.config_model
        sign, fraction = o.group_fraction[group_type].get_key(group)
        return sign, fraction

    def group_set(group_type):
        try:
            group = model.config_model.group_fraction[group_type].keys()
        except (TypeError, KeyError):
            group = []
        return po.Set(initialize=group)

    def techs_to_consider(supply_techs, group_type):
        # Remove ignored techs if any defined
        gfc = model.config_model.group_fraction
        if 'ignored_techs' in gfc and group_type in gfc.ignored_techs:
            return [i for i in supply_techs
                    if i not in gfc.ignored_techs[group_type]]
        else:
            return supply_techs

    def equalizer(lhs, rhs, sign):
        if sign == '<=':
            return lhs <= rhs
        elif sign == '>=':
            return lhs >= rhs
        elif sign == '==':
            return lhs == rhs
        else:
            raise ValueError('Invalid sign: {}'.format(sign))

    supply_techs = (model.get_group_members('supply') +
                    model.get_group_members('conversion'))

    # Sets
    m.output_group = group_set('output')
    m.capacity_group = group_set('capacity')
    m.demand_power_peak_group = group_set('demand_power_peak')

    # Constraint rules
    def c_group_fraction_output_rule(m, c, output_group):
        sign, fraction = sign_fraction(output_group, 'output')
        techs = techs_to_consider(supply_techs, 'output')
        rhs = (fraction
               * sum(m.es_prod[c, y, x, t] for y in techs
                     for x in m.x for t in m.t))
        lhs = sum(m.es_prod[c, y, x, t]
                  for y in model.get_group_members(output_group) for x in m.x
                  for t in m.t)
        return equalizer(lhs, rhs, sign)

    def c_group_fraction_capacity_rule(m, c, capacity_group):
        sign, fraction = sign_fraction(capacity_group, 'capacity')
        techs = techs_to_consider(supply_techs, 'capacity')
        rhs = (fraction
               * sum(m.e_cap[y, x] for y in techs for x in m.x))
        lhs = sum(m.e_cap[y, x] for y in model.get_group_members(capacity_group)
                  for x in m.x)
        return equalizer(lhs, rhs, sign)

    def c_group_fraction_demand_power_peak_rule(m, c, demand_power_peak_group):
        sign, fraction = sign_fraction(demand_power_peak_group,
                                       'demand_power_peak')
        margin = model.config_model.system_margin.get_key(c, default=0)
        peak_timestep = model.t_max_demand.power
        y = 'demand_power'
        # Calculate demand peak taking into account both r_scale and time_res
        peak = (float(sum(model.m.r[y, x, peak_timestep]
                      * model.get_option(y + '.constraints.r_scale', x=x)
                      for x in model.m.x))
                / model.data.time_res_series.at[peak_timestep])
        rhs = fraction * (-1 - margin) * peak
        lhs = sum(m.e_cap[y, x]
                  for y in model.get_group_members(demand_power_peak_group)
                  for x in m.x)
        return equalizer(lhs, rhs, sign)

    # Constraints
    m.c_group_fraction_output = \
        po.Constraint(m.c, m.output_group, rule=c_group_fraction_output_rule)
    m.c_group_fraction_capacity = \
        po.Constraint(m.c, m.capacity_group, rule=c_group_fraction_capacity_rule)
    grp = m.demand_power_peak_group
    m.c_group_fraction_demand_power_peak = \
        po.Constraint(m.c, grp, rule=c_group_fraction_demand_power_peak_rule)
