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


def node_energy_balance(model):
    """
    Defines variables:

    * s: storage level
    * rs: resource <-> storage
    * rsecs: secondary resource <-> storage
    * e: carrier <-> grid (positive: to grid, negative: from grid)
    * e_prod: carrier -> grid (always positive)
    * e_con: carrier <- grid (always negative)
    * es_prod: storage -> carrier (always positive)
    * es_con: storage <- carrier (always negative)
    * os: storage <-> overflow
    * r_area: resource collector area

    """
    m = model.m
    d = model.data

    # Variables
    m.s = cp.Var(m.y, m.x, m.t, within=cp.NonNegativeReals)
    m.rs = cp.Var(m.y, m.x, m.t, within=cp.Reals)
    m.rsecs = cp.Var(m.y, m.x, m.t, within=cp.NonNegativeReals)
    m.e = cp.Var(m.c, m.y, m.x, m.t, within=cp.Reals)
    m.e_prod = cp.Var(m.c, m.y, m.x, m.t, within=cp.NonNegativeReals)
    m.e_con = cp.Var(m.c, m.y, m.x, m.t, within=cp.NegativeReals)
    m.es_prod = cp.Var(m.c, m.y, m.x, m.t, within=cp.NonNegativeReals)
    m.es_con = cp.Var(m.c, m.y, m.x, m.t, within=cp.NegativeReals)
    m.os = cp.Var(m.y, m.x, m.t, within=cp.NonNegativeReals)
    m.r_area = cp.Var(m.y, m.x, within=cp.NonNegativeReals)

    # Constraint rules
    def c_e_rule(m, c, y, x, t):
        return m.e[c, y, x, t] == m.e_prod[c, y, x, t] + m.e_con[c, y, x, t]

    def c_e_prod_rule(m, c, y, x, t):
        return m.e_prod[c, y, x, t] == m.es_prod[c, y, x, t] * m.e_eff[y, x, t]

    def c_e_con_rule(m, c, y, x, t):
        # Nodes with a source_carrier have an efficiency of 1.0 in consuming it
        # since emitting their consumed energy via their primary carrier
        # would otherwise apply efficiency losses twice
        if c == model.get_option(y + '.source_carrier'):
            eff = 1.0
        else:
            try:
                eff = 1 / m.e_eff[y, x, t]
            except ZeroDivisionError:
                eff = 0
        return m.e_con[c, y, x, t] == m.es_con[c, y, x, t] * eff

    def c_rs_rule(m, y, x, t):
        this_r = m.r[y, x, t]
        #
        # If `r` is set to `inf`, it is interpreted as unconstrained `r`/`rs`
        #
        if this_r == float('inf'):
            return cp.Constraint.NoConstraint
        #
        # Otherwise, set up a context-dependent `rs` constraint
        #
        elif (d.locations.ix[x, y] != 1) or (this_r == 0):
            # `rs` is forced to 0 if technology not allowed at this location,
            # and also if `r` is 0
            return m.rs[y, x, t] == 0
        else:
            r_avail = (this_r
                       * model.get_option(y + '.constraints.r_scale')
                       * m.r_area[y, x]
                       * model.get_option(y + '.constraints.r_eff'))
            if model.get_option(y + '.constraints.force_r'):
                return m.rs[y, x, t] == r_avail
            elif this_r > 0:
                return m.rs[y, x, t] <= r_avail
            else:
                return m.rs[y, x, t] >= r_avail

    def c_s_balance_rule(m, y, x, t):
        # A) Special case for transmission technologies
        if y in model.data.transmission_y:
            y_remote, x_remote = transmission.get_remotes(y, x)
            if y_remote in model.data.transmission_y:
                carrier = model.get_option(y + '.carrier')
                # Divide by efficiency to balance the fact that we
                # multiply by efficiency twice (at each x)
                return (m.es_prod[carrier, y, x, t]
                        == -1 * m.es_con[carrier, y_remote, x_remote, t]
                        / m.e_eff[y, x, t])
            else:
                return cp.Constraint.NoConstraint
        # B) All other nodes have the same balancing rule
        # Define s_minus_one differently for cases:
        #   1. no storage allowed
        #   2. storage allowed and time step is not the first timestep
        #   3. storage allowed and initializing an iteration of operation mode
        #   4. storage allowed and initializing the first timestep
        elif model.get_option(y + '.constraints.s_cap_max') == 0:  # 1st case
            s_minus_one = 0
        elif m.t.order_dict[t] > 1:  # 2nd case
            s_minus_one = (((1 - model.get_option(y + '.constraints.s_loss'))
                            ** m.time_res[model.prev(t)])
                           * m.s[y, x, model.prev(t)])
        elif model.mode == 'operate' and 's_init' in model.data:  # 3rd case
            s_minus_one = model.data.s_init.at[x, y]
        else:  # 4th case
            s_minus_one = model.get_option(y + '.constraints.s_init', x=x)
        return (m.s[y, x, t] == s_minus_one + m.rs[y, x, t] + m.rsecs[y, x, t]
                - sum(m.es_prod[c, y, x, t] for c in m.c)
                - sum(m.es_con[c, y, x, t] for c in m.c)
                - m.os[y, x, t])

    # Constraints
    m.c_e = cp.Constraint(m.c, m.y, m.x, m.t)
    m.c_e_prod = cp.Constraint(m.c, m.y, m.x, m.t)
    m.c_e_con = cp.Constraint(m.c, m.y, m.x, m.t)
    m.c_rs = cp.Constraint(m.y, m.x, m.t)
    m.c_s_balance = cp.Constraint(m.y, m.x, m.t)


def node_constraints_build(model):
    """Depends on: node_energy_balance

    Defines variables:

    * s_cap: installed storage capacity
    * r_cap: installed resource <-> storage conversion capacity
    * e_cap: installed storage <-> electricity conversion capacity

    """
    m = model.m
    d = model.data

    # Variables
    m.s_cap = cp.Var(m.y, m.x, within=cp.NonNegativeReals)
    m.r_cap = cp.Var(m.y, m.x, within=cp.NonNegativeReals)
    m.e_cap = cp.Var(m.y, m.x, within=cp.NonNegativeReals)

    # Constraint rules
    def c_s_cap_rule(m, y, x):
        s_cap_max = model.get_option(y + '.constraints.s_cap_max', x=x)
        if model.mode == 'plan':
            return m.s_cap[y, x] <= s_cap_max
        elif model.mode == 'operate':
            return m.s_cap[y, x] == s_cap_max

    def c_r_cap_rule(m, y, x):
        r_cap_max = model.get_option(y + '.constraints.r_cap_max', x=x)
        if model.mode == 'plan' or np.isinf(r_cap_max):
            # We take this constraint even in operate mode, if r_cap_max
            # is set to infinite!
            return m.r_cap[y, x] <= r_cap_max
        elif model.mode == 'operate':
            return m.r_cap[y, x] == r_cap_max

    def c_r_area_rule(m, y, x):
        r_area_max = model.get_option(y + '.constraints.r_area_max', x=x)
        if r_area_max is False:
            return m.r_area[y, x] == 1.0
        elif model.mode == 'plan':
            return m.r_area[y, x] <= r_area_max
        elif model.mode == 'operate':
            return m.r_area[y, x] == r_area_max

    def c_e_cap_rule(m, y, x):
        e_cap_max = model.get_option(y + '.constraints.e_cap_max', x=x)
        # First check whether this tech is allowed at this location
        if not d.locations.ix[x, y] == 1:
            return m.e_cap[y, x] == 0
        elif model.mode == 'plan' or np.isinf(e_cap_max):
            # We take this constraint even in operate mode, if e_cap_max
            # is set to infinite!
            return m.e_cap[y, x] <= e_cap_max
        elif model.mode == 'operate':
            return m.e_cap[y, x] == e_cap_max

    # Constraints
    m.c_s_cap = cp.Constraint(m.y, m.x)
    m.c_r_cap = cp.Constraint(m.y, m.x)
    m.c_r_area = cp.Constraint(m.y, m.x)
    m.c_e_cap = cp.Constraint(m.y, m.x)


def node_constraints_operational(model):
    """Depends on: node_energy_balance, node_constraints_build"""
    m = model.m

    # Constraint rules
    def c_rs_max_rule(m, y, x, t):
        return (m.rs[y, x, t] <=
                m.time_res[t]
                * (m.r_cap[y, x] / model.get_option(y + '.constraints.r_eff')))

    def c_rs_min_rule(m, y, x, t):
        return (m.rs[y, x, t] >=
                -1 * m.time_res[t]
                * (m.r_cap[y, x] / model.get_option(y + '.constraints.r_eff')))

    # def c_e_prod_max_rule(m, c, y, x, t):
    #     if c == model.get_option(y + '.carrier'):
    #         return m.e_prod[c, y, x, t] <= m.time_res[t] * m.e_cap[y, x]
    #     else:
    #         return m.e_prod[c, y, x, t] == 0

    # def c_e_con_max_rule(m, c, y, x, t):
    #     if c == model.get_option(y + '.carrier'):
    #         if model.get_option(y + '.constraints.e_can_be_negative') is False:
    #             return m.e_con[c, y, x, t] == 0
    #         else:
    #             return (m.e_con[c, y, x, t] >=
    #                     -1 * m.time_res[t] * m.e_cap[y, x])

    def c_es_prod_max_rule(m, c, y, x, t):
        if c == model.get_option(y + '.carrier'):
            return (m.es_prod[c, y, x, t] <=
                    m.time_res[t]
                    * (m.e_cap[y, x] / model.get_eff_ref('e', y, x)))
        else:
            return m.es_prod[c, y, x, t] == 0

    def c_es_con_max_rule(m, c, y, x, t):
        if c == model.get_option(y + '.carrier'):
            if model.get_option(y + '.constraints.e_can_be_negative') is False:
                return m.es_con[c, y, x, t] == 0
            else:
                return (m.es_con[c, y, x, t] >= -1 * m.time_res[t]
                        * (m.e_cap[y, x] / model.get_eff_ref('e', y, x)))
        elif c == model.get_option(y + '.source_carrier'):
            # Special case for conversion technologies,
            # defining consumption for the source carrier
            # TODO if implement more generic secondary carriers can move this
            # into balancing equation together with carrier-specific
            # efficiencies or conversion rates, which already is
            # implemented in a basic way in `c_e_con_rule`
            carrier = model.get_option(y + '.carrier')
            return (m.es_con[c, y, x, t] == -1 * m.es_prod[carrier, y, x, t])
        else:
            return m.es_con[c, y, x, t] == 0

    def c_s_max_rule(m, y, x, t):
        return m.s[y, x, t] <= m.s_cap[y, x]

    def c_rsecs_rule(m, y, x, t):
        # rsec (secondary resource) is allowed only during
        # the hours within startup_time
        # and only if the technology allows this
        if (model.get_option(y + '.constraints.allow_rsec')
                and t < model.data.startup_time_bounds):
            try:
                return m.rsecs[y, x, t] <= (m.time_res[t]
                                            * m.e_cap[y, x]) / m.e_eff[y, x, t]
            except ZeroDivisionError:
                return m.rsecs[y, x, t] == 0
        else:
            return m.rsecs[y, x, t] == 0

    # Constraints
    m.c_rs_max = cp.Constraint(m.y, m.x, m.t)
    m.c_rs_min = cp.Constraint(m.y, m.x, m.t)
    # m.c_e_prod_max = cp.Constraint(m.c, m.y, m.x, m.t)
    # m.c_e_con_max = cp.Constraint(m.c, m.y, m.x, m.t)
    m.c_es_prod_max = cp.Constraint(m.c, m.y, m.x, m.t)
    m.c_es_con_max = cp.Constraint(m.c, m.y, m.x, m.t)
    m.c_s_max = cp.Constraint(m.y, m.x, m.t)
    m.c_rsecs = cp.Constraint(m.y, m.x, m.t)


def transmission_constraints(model):
    """Depends on: node_constraints_build

    Constrains e_cap symmetrically for transmission nodes.

    """
    m = model.m

    # Constraint rules
    def c_transmission_capacity_rule(m, y, x):
        if y in model.data.transmission_y:
            y_remote, x_remote = transmission.get_remotes(y, x)
            if y_remote in model.data.transmission_y:
                return m.e_cap[y, x] == m.e_cap[y_remote, x_remote]
            else:
                return cp.Constraint.NoConstraint
        else:
            return cp.Constraint.NoConstraint

    # Constraints
    m.c_transmission_capacity = cp.Constraint(m.y, m.x)


def node_costs(model):
    """
    Depends on: node_energy_balance, node_constraints_build

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

    # Variables
    m.cost = cp.Var(m.y, m.x, m.k, within=cp.NonNegativeReals)
    m.cost_con = cp.Var(m.y, m.x, m.k, within=cp.NonNegativeReals)
    m.cost_op = cp.Var(m.y, m.x, m.k, within=cp.NonNegativeReals)

    # Constraint rules
    def c_cost_rule(m, y, x, k):
        return m.cost[y, x, k] == m.cost_con[y, x, k] + m.cost_op[y, x, k]

    def c_cost_con_rule(m, y, x, k):
        return (m.cost_con[y, x, k] == _depreciation_rate(y, k)
                * (sum(m.time_res[t] for t in m.t) / 8760)
                * (_cost('s_cap', y, k) * m.s_cap[y, x]
                   + _cost('r_cap', y, k) * m.r_cap[y, x]
                   + _cost('r_area', y, k) * m.r_area[y, x]
                   + _cost('e_cap', y, k) * m.e_cap[y, x]))

    def c_cost_op_rule(m, y, x, k):
        # TODO currently only counting e_prod for op costs, makes sense?
        carrier = model.get_option(y + '.carrier')
        return (m.cost_op[y, x, k] ==
                _cost('om_frac', y, k) * m.cost_con[y, x, k]
                + _cost('om_var', y, k) * sum(m.e_prod[carrier, y, x, t]
                                              for t in m.t)
                + _cost('om_fuel', y, k) * sum(m.rs[y, x, t] for t in m.t))

    # Constraints
    m.c_cost = cp.Constraint(m.y, m.x, m.k)
    m.c_cost_con = cp.Constraint(m.y, m.x, m.k)
    m.c_cost_op = cp.Constraint(m.y, m.x, m.k)


def model_constraints(model):
    """Depends on: node_energy_balance"""
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
            family = get_children(x) + [x]  # list of children + parent
            if c == 'power':
                return (sum(m.e_prod[c, y, xs, t] for xs in family for y in m.y)
                        + sum(m.e_con[c, y, xs, t] for xs in family for y in m.y)
                        == 0)
            elif c == 'heat':
                return (sum(m.e_prod[c, y, xs, t] for xs in family for y in m.y)
                        + sum(m.e_con[c, y, xs, t] for xs in family for y in m.y)
                        >= 0)
            else:
                return cp.Constraint.NoConstraint

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
