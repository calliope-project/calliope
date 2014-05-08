"""
Copyright (C) 2013 Stefan Pfenninger.
Licensed under the Apache 2.0 License (see LICENSE file).

base.py
~~~~~~~

Basic model constraints.

"""

from __future__ import print_function
from __future__ import division

import coopr.pyomo as cp
import numpy as np

from .. import transmission
from .. import utils


def node_resource(model):
    """
    Defines variables:

    * rs: resource <-> storage (+ production, - consumption)
    * r_area: resource collector area

    """
    m = model.m

    def availability(y, x, t):
        if model.config_model.availability and y in model.data._y_def_r:
            availability = m.a[y, x, t]
        else:
            availability = 1.0
        return availability

    # Variables
    m.rs = cp.Var(m.y, m.x, m.t, within=cp.Reals)
    m.r_area = cp.Var(m.y_def_r, m.x, within=cp.NonNegativeReals)

    # Constraint rules
    def c_rs_rule(m, y, x, t):
        r_avail = (m.r[y, x, t]
                   * model.get_option(y + '.constraints.r_scale', x=x)
                   * m.r_area[y, x]
                   * model.get_option(y + '.constraints.r_eff'))
        if model.get_option(y + '.constraints.force_r'):
            return m.rs[y, x, t] == r_avail
        # TODO reformulate conditionally once Pyomo supports that:
        # had to remove the following formulation because it is not
        # re-evaluated on model re-construction -- we now check for
        # demand/supply tech instead, which means that `r` can only
        # be ALL negative or ALL positive for a given tech!
        # elif cp.value(m.r[y, x, t]) > 0:
        elif (y in model.get_group_members('supply') or
              y in model.get_group_members('unmet_demand')):
            # Supply technologies make use of availability
            return m.rs[y, x, t] <= r_avail * availability(y, x, t)
        elif y in model.get_group_members('demand'):
            return m.rs[y, x, t] >= r_avail
        elif y in model.get_group_members('storage'):
            return m.rs[y, x, t] == 0

    # Constraints
    m.c_rs = cp.Constraint(m.y_def_r, m.x, m.t)


def node_energy_balance(model):
    """
    Defines variables:

    * s: storage level
    * rs_: secondary resource -> storage (+ production)
    * es_prod: storage -> carrier (+ production)
    * es_con: storage <- carrier (- consumption)

    """
    m = model.m
    d = model.data

    def get_e_eff(m, y, x, t):
        if y in m.y_def_e_eff:
            return m.e_eff[y, x, t]
        else:
            return d.e_eff[y][x].iat[0]  # Just get 0-th entry in DataFrame

    def get_e_eff_per_distance(model, y, x):
        try:
            e_loss = model.get_option(y + '.constraints_per_distance.e_loss')
            per_distance = model.get_option(y + '.per_distance')
            distance = model.get_option(y + '.distance', x=x)
            return 1 - (e_loss * (distance / per_distance))
        except KeyError:
            return 1.0

    # Variables
    m.s = cp.Var(m.y_pc, m.x, m.t, within=cp.NonNegativeReals)
    #m.rs_ = cp.Var(m.y, m.x, m.t, within=cp.NonNegativeReals)
    m.es_prod = cp.Var(m.c, m.y, m.x, m.t, within=cp.NonNegativeReals)
    m.es_con = cp.Var(m.c, m.y, m.x, m.t, within=cp.NegativeReals)

    # Constraint rules
    def c_s_balance_transmission_rule(m, y, x, t):
        y_remote, x_remote = transmission.get_remotes(y, x)
        if y_remote in m.y_trans:
            c = model.get_option(y + '.carrier')
            return (m.es_prod[c, y, x, t]
                    == -1 * m.es_con[c, y_remote, x_remote, t]
                    * get_e_eff(m, y, x, t)
                    * get_e_eff_per_distance(model, y, x))
        else:
            return cp.Constraint.NoConstraint

    def c_s_balance_conversion_rule(m, y, x, t):
        c_prod = model.get_option(y + '.carrier')
        c_source = model.get_option(y + '.source_carrier')
        return (m.es_prod[c_prod, y, x, t]
                == -1 * m.es_con[c_source, y, x, t] * get_e_eff(m, y, x, t))

    def c_s_balance_pc_rule(m, y, x, t):
        e_eff = get_e_eff(m, y, x, t)
        # FIXME this doesn't update on param update!
        if cp.value(e_eff) == 0:
            e_prod = 0
        else:
            e_prod = sum(m.es_prod[c, y, x, t] for c in m.c) / e_eff
        e_con = sum(m.es_con[c, y, x, t] for c in m.c) * e_eff

        # A) Case where no storage allowed
        if model.get_option(y + '.constraints.s_cap_max', x=x) == 0:
            return m.rs[y, x, t] == e_prod + e_con  # - m.rs_[y, x, t]

        # B) Case where storage is allowed
        else:
            # set up s_minus_one
            if m.t.order_dict[t] == 1:  # order_dict starts at 1
                s_minus_one = m.s_init[y, x]
            else:
                s_loss = model.get_option(y + '.constraints.s_loss')
                s_minus_one = (((1 - s_loss)
                                ** model.data.time_res_series.at[model.prev(t)])
                               * m.s[y, x, model.prev(t)])
            return (m.s[y, x, t] == s_minus_one + m.rs[y, x, t]
                    # + m.rs_[y, x, t]
                    - e_prod - e_con)

    # Constraints
    m.c_s_balance_transmission = cp.Constraint(m.y_trans, m.x, m.t)
    m.c_s_balance_conversion = cp.Constraint(m.y_conv, m.x, m.t)
    m.c_s_balance_pc = cp.Constraint(m.y_pc, m.x, m.t)


def node_constraints_build(model):
    """
    Defines variables:

    * s_cap: installed storage capacity
    * r_cap: installed resource <-> storage conversion capacity
    * e_cap: installed storage <-> grid conversion capacity

    """
    m = model.m
    d = model.data

    # Variables
    m.s_cap = cp.Var(m.y_pc, m.x, within=cp.NonNegativeReals)
    m.r_cap = cp.Var(m.y_def_r, m.x, within=cp.NonNegativeReals)
    m.e_cap = cp.Var(m.y, m.x, within=cp.NonNegativeReals)

    # Constraint rules
    def c_s_cap_rule(m, y, x):
        max_force = model.get_option(y + '.constraints.s_cap_max_force', x=x)
        # Get s_cap_max
        if model.get_option(y + '.constraints.use_s_time', x=x):
            s_time_max = model.get_option(y + '.constraints.s_time_max', x=x)
            e_cap_max = model.get_option(y + '.constraints.e_cap_max', x=x)
            e_eff_ref = model.get_eff_ref('e', y)
            s_cap_max = s_time_max * e_cap_max / e_eff_ref
        else:
            s_cap_max = model.get_option(y + '.constraints.s_cap_max', x=x)
        # Apply constraint
        if max_force or model.mode == 'operate':
            return m.s_cap[y, x] == s_cap_max
        elif model.mode == 'plan':
            return m.s_cap[y, x] <= s_cap_max

    def c_r_cap_rule(m, y, x):
        r_cap_max = model.get_option(y + '.constraints.r_cap_max', x=x)
        if np.isinf(r_cap_max):
            return cp.Constraint.NoConstraint
        elif model.mode == 'plan':
            return m.r_cap[y, x] <= r_cap_max
        elif model.mode == 'operate':
            return m.r_cap[y, x] == r_cap_max

    def c_r_area_rule(m, y, x):
        area_per_cap = model.get_option(y + '.constraints.r_area_per_e_cap')
        if area_per_cap:
            return m.r_area[y, x] == m.e_cap[y, x] * area_per_cap
        else:
            r_area_max = model.get_option(y + '.constraints.r_area_max', x=x)
            if r_area_max is False:
                return m.r_area[y, x] == 1.0
            elif model.mode == 'plan':
                return m.r_area[y, x] <= r_area_max
            elif model.mode == 'operate':
                return m.r_area[y, x] == r_area_max

    def c_e_cap_rule(m, y, x):
        e_cap_max = model.get_option(y + '.constraints.e_cap_max', x=x)
        e_cap_max_scale = model.get_option(y + '.constraints.e_cap_max_scale',
                                           x=x)
        e_cap_max_force = model.get_option(y + '.constraints.e_cap_max_force',
                                           x=x)
        # First check whether this tech is allowed at this location
        if not d.locations.ix[x, y] == 1:
            return m.e_cap[y, x] == 0
        elif np.isinf(e_cap_max):
            return cp.Constraint.NoConstraint
        elif e_cap_max_force or model.mode == 'operate':
            return m.e_cap[y, x] == e_cap_max * e_cap_max_scale
        elif model.mode == 'plan':
            return m.e_cap[y, x] <= e_cap_max * e_cap_max_scale

    # Constraints
    m.c_s_cap = cp.Constraint(m.y_pc, m.x)
    m.c_r_cap = cp.Constraint(m.y_def_r, m.x)
    m.c_r_area = cp.Constraint(m.y_def_r, m.x)
    m.c_e_cap = cp.Constraint(m.y, m.x)


def node_constraints_operational(model):
    m = model.m
    time_res = model.data.time_res_series

    # Constraint rules
    def c_rs_max_upper_rule(m, y, x, t):
        return (m.rs[y, x, t] <=
                time_res.at[t]
                * (m.r_cap[y, x] / model.get_option(y + '.constraints.r_eff')))

    def c_rs_max_lower_rule(m, y, x, t):
        return (m.rs[y, x, t] >=
                -1 * time_res.at[t]
                * (m.r_cap[y, x] / model.get_option(y + '.constraints.r_eff')))

    def c_es_prod_max_rule(m, c, y, x, t):
        if c == model.get_option(y + '.carrier'):
            return m.es_prod[c, y, x, t] <= time_res.at[t] * m.e_cap[y, x]
        else:
            return m.es_prod[c, y, x, t] == 0

    def c_es_prod_min_rule(m, c, y, x, t):
        min_use = model.get_option(y + '.constraints.e_cap_min_use')
        if (min_use and c == model.get_option(y + '.carrier')):
            return (m.es_prod[c, y, x, t]
                    >= time_res.at[t] * m.e_cap[y, x] * min_use)
        else:
            return cp.Constraint.NoConstraint

    def c_es_con_max_rule(m, c, y, x, t):
        if (model.get_option(y + '.constraints.e_can_be_negative') is True and
                c == model.get_option(y + '.carrier')):
            return m.es_con[c, y, x, t] >= -1 * time_res.at[t] * m.e_cap[y, x]
        else:
            return m.es_con[c, y, x, t] == 0

    def c_s_max_rule(m, y, x, t):
        return m.s[y, x, t] <= m.s_cap[y, x]

    # def c_rs__rule(m, y, x, t):
    #     # rs_ (secondary resource) is allowed only during
    #     # the hours within startup_time
    #     # and only if the technology allows this
    #     if (model.get_option(y + '.constraints.allow_rs_')
    #             and t < model.data.startup_time_bounds):
    #         try:
    #             return m.rsecs[y, x, t] <= (model.data.time_res_series.at[t]
    #                                         * m.e_cap[y, x]) / m.e_eff[y, x, t]
    #         except ZeroDivisionError:
    #             return m.rs_[y, x, t] == 0
    #     else:
    #         return m.rs_[y, x, t] == 0

    # Constraints
    m.c_rs_max_upper = cp.Constraint(m.y_def_r, m.x, m.t)
    m.c_rs_max_lower = cp.Constraint(m.y_def_r, m.x, m.t)
    m.c_es_prod_max = cp.Constraint(m.c, m.y, m.x, m.t)
    m.c_es_prod_min = cp.Constraint(m.c, m.y, m.x, m.t)
    m.c_es_con_max = cp.Constraint(m.c, m.y, m.x, m.t)
    m.c_s_max = cp.Constraint(m.y_pc, m.x, m.t)
    # m.c_rs_ = cp.Constraint(m.y, m.x, m.t)


def transmission_constraints(model):
    """
    Constrains e_cap symmetrically for transmission nodes.

    """
    m = model.m

    # Constraint rules
    def c_transmission_capacity_rule(m, y, x):
        y_remote, x_remote = transmission.get_remotes(y, x)
        if y_remote in m.y_trans:
            return m.e_cap[y, x] == m.e_cap[y_remote, x_remote]
        else:
            return cp.Constraint.NoConstraint

    # Constraints
    m.c_transmission_capacity = cp.Constraint(m.y_trans, m.x)


def node_costs(model):
    """
    Defines variables:

    * cost: total costs
    * cost_con: construction costs
    * cost_op: operation costs

    """
    m = model.m

    @utils.memoize
    def _depreciation_rate(y, k):
        interest = model.get_option(y + '.depreciation.interest.' + k,
                                    default=y + '.depreciation.interest.default')
        plant_life = model.get_option(y + '.depreciation.plant_life')
        if interest == 0:
            dep = 1 / plant_life
        else:
            dep = ((interest * (1 + interest) ** plant_life)
                   / (((1 + interest) ** plant_life) - 1))
        return dep

    @utils.memoize
    def _cost(cost, y, k):
        return model.get_option(y + '.costs.' + k + '.' + cost,
                                default=y + '.costs.default.' + cost)

    @utils.memoize
    def _cost_per_distance(cost, y, k, x):
        try:
            cost = model.get_option(y + '.costs_per_distanc.e' + k + '.' + cost)
            per_distance = model.get_option(y + '.per_distance')
            distance = model.get_option(y + '.distance', x=x)
            distance_cost = cost * (distance / per_distance)
        except KeyError:
            distance_cost = 0
        return distance_cost

    # Variables
    m.cost = cp.Var(m.y, m.x, m.k, within=cp.NonNegativeReals)
    m.cost_con = cp.Var(m.y, m.x, m.k, within=cp.NonNegativeReals)
    m.cost_op = cp.Var(m.y, m.x, m.k, within=cp.NonNegativeReals)

    # Constraint rules
    def c_cost_rule(m, y, x, k):
        return m.cost[y, x, k] == m.cost_con[y, x, k] + m.cost_op[y, x, k]

    def c_cost_con_rule(m, y, x, k):
        if y in m.y_pc:
            cost_s_cap = _cost('s_cap', y, k) * m.s_cap[y, x]
        else:
            cost_s_cap = 0

        if y in m.y_def_r:
            cost_r_cap = _cost('r_cap', y, k) * m.r_cap[y, x]
            cost_r_area = _cost('r_area', y, k) * m.r_area[y, x]
        else:
            cost_r_cap = 0
            cost_r_area = 0

        if y in m.y_trans:
            cost_e_cap = (_cost('e_cap', y, k)
                          + _cost_per_distance('e_cap', y, k, x))
        else:
            cost_e_cap = _cost('e_cap', y, k)

        return (m.cost_con[y, x, k] == _depreciation_rate(y, k)
                * (sum(model.data.time_res_series) / 8760)
                * (cost_s_cap + cost_r_cap + cost_r_area
                   + cost_e_cap * m.e_cap[y, x]))

    def c_cost_op_rule(m, y, x, k):
        # TODO currently only counting es_prod for op costs, makes sense?
        if y in m.y:
            carrier = model.get_option(y + '.carrier')
            return (m.cost_op[y, x, k] ==
                    # FIXME this is incorrect in cases where sum(t) is < 8760
                    _cost('om_frac', y, k) * m.cost_con[y, x, k]
                    + (_cost('om_fixed', y, k) * m.e_cap[y, x] *
                       (sum(model.data.time_res_series) / 8760))
                    + _cost('om_var', y, k) * sum(m.es_prod[carrier, y, x, t]
                                                  for t in m.t)
                    + _cost('om_fuel', y, k) * sum(m.rs[y, x, t] for t in m.t))
        else:
            return m.cost_op[y, x, k] == 0

    # Constraints
    m.c_cost = cp.Constraint(m.y, m.x, m.k)
    m.c_cost_con = cp.Constraint(m.y, m.x, m.k)
    m.c_cost_op = cp.Constraint(m.y, m.x, m.k)


def model_constraints(model):
    m = model.m

    @utils.memoize
    def get_parents(level):
        locations = model.data.locations
        return list(locations[locations._level == level].index)

    @utils.memoize
    def get_children(parent):
        locations = model.data.locations
        return list(locations[locations._within == parent].index)

    # Constraint rules
    def c_system_balance_rule(m, c, x, t):
        # TODO for now, hardcoding level 1, so can only use levels 0 and 1
        parents = get_parents(1)
        if x not in parents:
            return cp.Constraint.NoConstraint
        else:
            fam = get_children(x) + [x]  # list of children + parent
            if c == 'power':
                return (sum(m.es_prod[c, y, xs, t]
                            for xs in fam for y in m.y)
                        + sum(m.es_con[c, y, xs, t]
                              for xs in fam for y in m.y)
                        == 0)
            else:  # e.g. for heat
                return (sum(m.es_prod[c, y, xs, t]
                            for xs in fam for y in m.y)
                        + sum(m.es_con[c, y, xs, t]
                              for xs in fam for y in m.y)
                        >= 0)

    # Constraints
    m.c_system_balance = cp.Constraint(m.c, m.x, m.t)


def model_objective(model):
    m = model.m

    # Count monetary costs only
    def obj_rule(m):
        return (sum(model.get_option(y + '.weight')
                    * sum(m.cost[y, x, 'monetary'] for x in m.x) for y in m.y))

    m.obj = cp.Objective(sense=cp.minimize)
    #m.obj.domain = cp.NonNegativeReals
