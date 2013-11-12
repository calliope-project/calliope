from __future__ import print_function
from __future__ import division

import coopr.pyomo as cp


def node_energy_balance(m, o, d, model):
    """
    Defines variables:

    * s: storage level
    * rs: energy resource <> storage
    * bs: backup resource <> storage
    * es: storage <> electricity
    * os: storage <> overflow
    * e: node <> grid

    """
    # Variables
    m.s = cp.Var(m.y, m.x, m.t, within=cp.NonNegativeReals)  # was E_stor
    m.rs = cp.Var(m.y, m.x, m.t, within=cp.NonNegativeReals)  # was Q_sf
    m.bs = cp.Var(m.y, m.x, m.t, within=cp.NonNegativeReals)  # was Q_bak
    m.es = cp.Var(m.y, m.x, m.t, within=cp.NonNegativeReals)  # was Q_gen
    m.os = cp.Var(m.y, m.x, m.t, within=cp.NonNegativeReals)  # was Q_diss
    m.e = cp.Var(m.y, m.x, m.t, within=cp.NonNegativeReals)  # was P

    # Constraint rules
    def c_e_rule(m, y, x, t):  # was c_P_rule
        return m.e[y, x, t] == m.es[y, x, y] * m.e_eff[y, x, t]

    def c_rs_rule(m, y, x, t):  # was c_Q_sf_rule
        return (m.rs[y, x, t] == m.r[y, x, t]
                * 0.001 * m.r_area[y, x] * m.r_eff[y, x, t])
        # TODO remove 0.001 and instead scale the dni input files!

    def c_s_balance_rule(m, y, x, t):  # was c_Q_balance_rule
        if m.t.order_dict[t] > 1:
            s_minus_one = (((1 - m.s_loss[y])
                            ** m.time_res[model.prev(t)])
                           * m.s[y, x, model.prev(t)])
        else:
            s_minus_one = d.s_init[y]
        return (m.s[y, x, t] == s_minus_one
                + m.r_eff[y] * m.rs[y, x, t] + m.bs[y, x, t]
                - m.es[y, x, t] - m.os[y, x, t])

    def c_bs_rule(m, y, x, t):  # was c_Q_bak_rule
        # bs (backup resource) is allowed only during
        # the hours within startup_time
        if t < d.startup_time_bounds:
            return m.bs[y, x, t] <= (m.time_res[t]
                                     * m.e_cap[y, x]) / m.e_eff[y, x, t]
        else:
            return m.bs[y, x, t] == 0

    # Constraints
    m.c_e = cp.Constraint(m.y, m.x, m.t)
    m.c_rs = cp.Constraint(m.y, m.x, m.t)
    m.c_s_balance = cp.Constraint(m.y, m.x, m.t)
    m.c_bs = cp.Constraint(m.y, m.x, m.t)


def node_costs(m, o, d, model):
    """
    Depends on: node_energy_balance, node_constraints_build

    Defines variables:

    * c: total costs
    * c_con: construction costs
    * c_op: operation costs

    """
    # Variables
    m.c = cp.Var(m.y, m.x, within=cp.NonNegativeReals)
    m.c_con = cp.Var(m.y, m.x, within=cp.NonNegativeReals)
    m.c_op = cp.Var(m.y, m.x, within=cp.NonNegativeReals)

    # Constraint rules
    def c_c_rule(m, y, x):
        return m.c[y, x] == m.c_con[y, x] + m.c_op[y, x]

    def c_c_con_rule(m, y, x):
        return (m.c_con[y, x] == d.depreciation[y]
                * (sum(m.time_res[t] for t in m.t) / 8760)
                * (model.get_option('costs', y, 's') * m.s_cap[y, x]
                   + model.get_option('costs', y, 'r') * m.r_cap[y, x]
                   + model.get_option('costs', y, 'r_area') * m.r_area[y, x]
                   + model.get_option('costs', y, 'e') * m.e_cap[y, x]))

    def c_c_op_rule(m, y, x):
        return (m.c_op[y, x] ==
                model.get_option('costs', y, 'om_frac') * m.c_con[y, x]
                + (model.get_option('costs', y, 'om_var')
                   * sum(m.e[y, x, t] for t in m.t))
                + (model.get_option('costs', y, 'om_fuel')
                   * sum(m.rs[y, x, t] for t in m.t)))

    # Constraints
    m.c_c = cp.Constraint(m.i)
    m.c_c_con = cp.Constraint(m.i)
    m.c_c_op = cp.Constraint(m.i)


def node_constraints_operational(m, o, d):
    """Depends on: node_energy_balance, node_constraints_build"""
    # Constraint rules
    def c_e_max_rule(m, y, x, t):
        return m.e[y, x, t] <= m.time_res[t] * m.e_cap[y, x]

    def c_s_max_rule(m, y, x, t):
        return m.s[y, x, t] <= m.s_cap[y, x]

    # Constraints
    m.c_e_max = cp.Constraint(m.y, m.x, m.t)
    m.c_s_max = cp.Constraint(m.y, m.x, m.t)


def node_constraints_build(m, o, d, model):
    """Depends on: node_energy_balance

    Defines variables:

    * s_cap: installed storage capacity
    * r_cap: installed resource <> storage conversion capacity
    * e_cap: installed storage <> electricity conversion capacity

    """
    # Variables
    m.s_cap = cp.Var(m.y, m.x, within=cp.NonNegativeReals)  # was E_built
    m.r_cap = cp.Var(m.y, m.x, within=cp.NonNegativeReals)
    m.r_area = cp.Var(m.y, m.x, within=cp.NonNegativeReals)  # was sf_built
    m.e_cap = cp.Var(m.y, m.x, within=cp.NonNegativeReals)  # was P_built

    # Constraint rules
    def c_s_cap_rule(m, y, x):
        if model.mode == 'plan':
            return m.s_cap[y, x] <= model.get_option('constraints',
                                                     y, 's_cap_max')
        elif model.mode == 'operate':
            # TODO need a better way to load an existing system's built
            # capacities with spatial differentiation!
            return m.s_cap[y, x] == model.get_option('constraints',
                                                     y, 's_cap_max')

    def c_r_cap_rule(m, y, x):
        if model.mode == 'plan':
            return m.r_area[y, x] <= model.get_option('constraints', y,
                                                      'r_cap_max')
        elif model.mode == 'operate':
            return m.r_area[y, x] == model.get_option('constraints', y,
                                                      'r_cap_max')

    def c_r_area_rule(m, y, x):
        if model.mode == 'plan':
            return m.r_area[y, x] <= model.get_option('constraints', y,
                                                      'r_area_max')
        elif model.mode == 'operate':
            return m.r_area[y, x] == model.get_option('constraints', y,
                                                      'r_area_max')

    def c_e_cap_rule(m, y, x):
        if model.mode == 'plan':
            return m.e_cap[y, x] <= model.get_option('constraints', y,
                                                     'e_cap_max')
        elif model.mode == 'operate':
            return m.e_cap[y, x] == model.get_option('constraints', y,
                                                     'e_cap_max')

    # Constraints
    m.c_s_cap = cp.Constraint(m.y, m.x)
    m.c_r_cap = cp.Constraint(m.y, m.x)
    m.c_r_area = cp.Constraint(m.y, m.x)
    m.c_e_cap = cp.Constraint(m.y, m.x)


def model_constraints(m, o, d):
    """Depends on: node_energy_balance"""
    # Constraint rules
    def c_system_balance_rule(m, t):
        return (sum(m.e[y, x, t] for x in m.x for y in m.y)
                + m.slack[t] >= m.D[t])

    # Constraints
    m.c_system_balance = cp.Constraint(m.t)


def model_slack(m, o, d):
    """Defines variables:

    * slack
    * cost_slack

    """
    # Variables
    m.slack = cp.Var(m.t, within=cp.NonNegativeReals)
    m.c_slack = cp.Var(within=cp.NonNegativeReals)

    # Constraint rules
    def c_c_slack_rule(m):
        return m.cost_slack == sum(m.slack[t] for t in m.t)

    # Constraints
    m.c_c_slack = cp.Constraint()


def model_objective(m, o, d, model):
    def weight(y):
        return model.get_option('tech_weights', y)

    def obj_rule(m):
        return (sum(weight(y) * sum(m.cost[y, x] for x in m.x) for y in m.y)
                + o.slack_weight * m.cost_slack)

    m.obj = cp.Objective(sense=cp.minimize)
    #m.obj.domain = cp.NonNegativeReals
