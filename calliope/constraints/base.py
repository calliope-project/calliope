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
    * rs: energy resource <-> storage
    * bs: backup resource <-> storage
    * e: storage <-> electricity (positive: to grid, negative: from grid)
    * e_prod: electricity -> grid (always positive)
    * e_con: electricity <- grid (always negative)
    * es_prod: storage -> electricity (always positive)
    * es_con: storage <- electricity (always negative)
    * os: storage <-> overflow
    * e: node <-> grid
    * r_area: installed collector area

    """
    m = model.m
    d = model.data

    # Variables
    m.s = cp.Var(m.y, m.x, m.t, within=cp.NonNegativeReals)
    m.rs = cp.Var(m.y, m.x, m.t, within=cp.Reals)
    m.bs = cp.Var(m.y, m.x, m.t, within=cp.NonNegativeReals)
    m.e = cp.Var(m.y, m.x, m.t, within=cp.Reals)
    m.e_prod = cp.Var(m.y, m.x, m.t, within=cp.NonNegativeReals)
    m.e_con = cp.Var(m.y, m.x, m.t, within=cp.NegativeReals)
    m.es_prod = cp.Var(m.y, m.x, m.t, within=cp.NonNegativeReals)
    m.es_con = cp.Var(m.y, m.x, m.t, within=cp.NegativeReals)
    m.os = cp.Var(m.y, m.x, m.t, within=cp.NonNegativeReals)
    m.r_area = cp.Var(m.y, m.x, within=cp.NonNegativeReals)

    # Constraint rules
    def c_e_rule(m, y, x, t):
        return m.e[y, x, t] == m.e_prod[y, x, t] + m.e_con[y, x, t]

    def c_e_prod_rule(m, y, x, t):
        return m.e_prod[y, x, t] == m.es_prod[y, x, t] * m.e_eff[y, x, t]

    def c_e_con_rule(m, y, x, t):
        try:
            return m.e_con[y, x, t] == (m.es_con[y, x, t]
                                        * (1/m.e_eff[y, x, t]))
        except ZeroDivisionError:
            return m.e_con[y, x, t] == 0

    def c_rs_rule(m, y, x, t):
        this_r = m.r[y, x, t]
        #
        # If `r` is set to `inf`, it is interpreted as unconstrained `r`/`rs`
        #
        if this_r == float('inf'):
            # This also applies to transmission technologies which should
            # always have their `r` set to `inf`
            return cp.Constraint.NoConstraint
        #
        # Otherwise, set up a context-dependent `rs` constraint
        #
        elif (d.nodes.ix[x, y] != 1) or (this_r == 0):
            # `rs` is forced to 0 if technology not allowed at this location,
            # and also if `r` is 0
            return m.rs[y, x, t] == 0
        else:
            r_avail = (this_r
                       * model.get_option(y + '.constraints.r_scale')
                       * m.r_area[y, x] * m.r_eff[y, x, t])
            if model.get_option(y + '.constraints.force_r'):
                return m.rs[y, x, t] == r_avail
            elif this_r > 0:
                return m.rs[y, x, t] <= r_avail
            else:
                return m.rs[y, x, t] >= r_avail

    def c_s_balance_rule(m, y, x, t):
        if y in model.data.transmission_y:
            # Transmission technologies need a different energy balance rule
            y_remote, x_remote = transmission.get_remotes(y, x)
            if y_remote in model.data.transmission_y:
                return (m.es_prod[y, x, t]
                        == -1 * m.es_con[y_remote, x_remote, t]
                        / m.e_eff[y, x, t])
            else:
                return cp.Constraint.NoConstraint
        elif model.get_option(y + '.constraints.s_cap_max') == 0:
            s_minus_one = 0
        elif m.t.order_dict[t] > 1:
            s_minus_one = (((1 - model.get_option(y + '.constraints.s_loss'))
                            ** m.time_res[model.prev(t)])
                           * m.s[y, x, model.prev(t)])
        elif model.mode == 'operate' and 's_init' in model.data:
            s_minus_one = model.data.s_init.at[x, y]
        else:
            s_minus_one = model.get_option(y + '.constraints.s_init', x=x)
        return (m.s[y, x, t] == s_minus_one + m.rs[y, x, t] + m.bs[y, x, t]
                - m.es_prod[y, x, t] - m.es_con[y, x, t] - m.os[y, x, t])

    # Constraints
    m.c_e = cp.Constraint(m.y, m.x, m.t)
    m.c_e_prod = cp.Constraint(m.y, m.x, m.t)
    m.c_e_con = cp.Constraint(m.y, m.x, m.t)
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
        # First check whether this tech is allowed at this node
        if not d.nodes.ix[x, y] == 1:
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

    def eff_ref(var, y, x):
        """Get reference efficiency, falling back to efficiency if no
        reference efficiency has been set."""
        base = y + '.constraints.' + var
        eff_ref = model.get_option(base + '_eff_ref', x=x)
        if eff_ref is False:
            eff_ref = model.get_option(base + '_eff', x=x)
        # NOTE: Will cause errors in the case where (1) eff_ref is not defined
        # and (2) eff is set to "file". That is ok however because in this edge
        # case eff_ref should be manually set as there is no straightforward
        # way to derive it from the time series file.
        return eff_ref

    # Constraint rules
    def c_rs_max_rule(m, y, x, t):
        return m.rs[y, x, t] <= m.time_res[t] * (m.r_cap[y, x]
                                                 / eff_ref('r', y, x))

    def c_rs_min_rule(m, y, x, t):
        return m.rs[y, x, t] >= -1 * m.time_res[t] * (m.r_cap[y, x]
                                                      / eff_ref('r', y, x))

    def c_e_max_rule(m, y, x, t):
        return m.e_prod[y, x, t] <= m.time_res[t] * m.e_cap[y, x]

    def c_e_min_rule(m, y, x, t):
        if model.get_option(y + '.constraints.e_can_be_negative') is False:
            return m.e_con[y, x, t] == 0
        else:
            return m.e_con[y, x, t] >= -1 * m.time_res[t] * m.e_cap[y, x]

    def c_es_max_rule(m, y, x, t):
        return m.es_prod[y, x, t] <= m.time_res[t] * (m.e_cap[y, x]
                                                      / eff_ref('e', y, x))

    def c_es_min_rule(m, y, x, t):
        if model.get_option(y + '.constraints.e_can_be_negative') is False:
            return m.es_con[y, x, t] == 0
        else:
            return m.es_con[y, x, t] >= (-1 * m.time_res[t]
                                         * (m.e_cap[y, x] / eff_ref('e', y, x)))

    def c_s_max_rule(m, y, x, t):
        return m.s[y, x, t] <= m.s_cap[y, x]

    def c_bs_rule(m, y, x, t):
        # bs (backup resource) is allowed only during
        # the hours within startup_time
        # TODO this entire thing is a hack right now
        if y == 'csp' and t < model.data.startup_time_bounds:
            try:
                return m.bs[y, x, t] <= (m.time_res[t]
                                         * m.e_cap[y, x]) / m.e_eff[y, x, t]
            except ZeroDivisionError:
                return m.bs[y, x, t] == 0
        else:
            return m.bs[y, x, t] == 0

    # Constraints
    m.c_rs_max = cp.Constraint(m.y, m.x, m.t)
    m.c_rs_min = cp.Constraint(m.y, m.x, m.t)
    m.c_e_max = cp.Constraint(m.y, m.x, m.t)
    m.c_e_min = cp.Constraint(m.y, m.x, m.t)
    m.c_es_max = cp.Constraint(m.y, m.x, m.t)
    m.c_es_min = cp.Constraint(m.y, m.x, m.t)
    m.c_s_max = cp.Constraint(m.y, m.x, m.t)
    m.c_bs = cp.Constraint(m.y, m.x, m.t)


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
    def _depreciation_rate(y):
        interest = model.get_option(y + '.depreciation.interest')
        plant_life = model.get_option(y + '.depreciation.plant_life')
        dep = ((interest * (1 + interest) ** plant_life)
               / (((1 + interest) ** plant_life) - 1))
        return dep

    # Variables
    m.cost = cp.Var(m.y, m.x, within=cp.NonNegativeReals)
    m.cost_con = cp.Var(m.y, m.x, within=cp.NonNegativeReals)
    m.cost_op = cp.Var(m.y, m.x, within=cp.NonNegativeReals)

    # Constraint rules
    def c_cost_rule(m, y, x):
        return m.cost[y, x] == m.cost_con[y, x] + m.cost_op[y, x]

    def c_cost_con_rule(m, y, x):
        return (m.cost_con[y, x] == _depreciation_rate(y)
                * (sum(m.time_res[t] for t in m.t) / 8760)
                * (model.get_option(y + '.costs.s_cap') * m.s_cap[y, x]
                   + model.get_option(y + '.costs.r_cap') * m.r_cap[y, x]
                   + model.get_option(y + '.costs.r_area') * m.r_area[y, x]
                   + model.get_option(y + '.costs.e_cap') * m.e_cap[y, x]))

    def c_cost_op_rule(m, y, x):
        # TODO currently only counting e_prod for op costs, makes sense?
        return (m.cost_op[y, x] ==
                model.get_option(y + '.costs.om_frac') * m.cost_con[y, x]
                + (model.get_option(y + '.costs.om_var')
                   * sum(m.e_prod[y, x, t] for t in m.t))
                + (model.get_option(y + '.costs.om_fuel')
                   * sum(m.rs[y, x, t] for t in m.t)))

    # Constraints
    m.c_cost = cp.Constraint(m.y, m.x)
    m.c_cost_con = cp.Constraint(m.y, m.x)
    m.c_cost_op = cp.Constraint(m.y, m.x)


def model_constraints(model):
    """Depends on: node_energy_balance"""
    m = model.m

    @utils.memoize
    def get_parents(level):
        nodes = model.data.nodes
        return list(nodes[nodes._level == level].index)

    @utils.memoize
    def get_children(parent):
        nodes = model.data.nodes
        return list(nodes[nodes._within == parent].index)

    # Constraint rules
    def c_system_balance_rule(m, x, t):
        # TODO for now, hardcoding level 1, so can only use levels 0 and 1
        parents = get_parents(1)
        if x not in parents:
            return cp.Constraint.NoConstraint
        else:
            family = get_children(x) + [x]  # list of children + parent
            return (sum(m.e_prod[y, xs, t] for xs in family for y in m.y)
                    + sum(m.e_con[y, xs, t] for xs in family for y in m.y)
                    == 0)

    # Constraints
    m.c_system_balance = cp.Constraint(m.x, m.t)


def model_objective(model):
    m = model.m

    def obj_rule(m):
        return (sum(model.get_option(y + '.weight') * sum(m.cost[y, x]
                for x in m.x) for y in m.y))

    m.obj = cp.Objective(sense=cp.minimize)
    #m.obj.domain = cp.NonNegativeReals
