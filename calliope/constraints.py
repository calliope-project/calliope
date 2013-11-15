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
    * r_area: installed collector area

    """
    # Variables
    m.s = cp.Var(m.y, m.x, m.t, within=cp.NonNegativeReals)
    m.rs = cp.Var(m.y, m.x, m.t, within=cp.Reals)
    m.bs = cp.Var(m.y, m.x, m.t, within=cp.NonNegativeReals)
    m.es = cp.Var(m.y, m.x, m.t, within=cp.Reals)
    m.os = cp.Var(m.y, m.x, m.t, within=cp.NonNegativeReals)
    m.e = cp.Var(m.y, m.x, m.t, within=cp.Reals)
    m.r_area = cp.Var(m.y, m.x, within=cp.NonNegativeReals)

    # Constraint rules
    def c_e_rule(m, y, x, t):
        return m.e[y, x, t] == m.es[y, x, t] * m.e_eff[y, x, t]

    def c_rs_rule(m, y, x, t):
        # `rs` is forced to 0 if technology not allowed at this location
        if d.nodes.ix[x, y] == 0:
            return m.rs[y, x, t] == 0
        elif model.get_option('constraints', y, 'r_unlimited'):
            return m.rs[y, x, t] >= 0
        else:
            return (m.rs[y, x, t] == m.r[y, x, t]
                    * model.get_option('constraints', y, 'r_scale')
                    * m.r_area[y, x] * m.r_eff[y, x, t])
        # TODO remove 0.001 and instead scale the dni input files!

    def c_s_balance_rule(m, y, x, t):
        # TODO add another check whether s_cap is 0, and if yes,
        # simply set s_minus_one=0 to make model setup a bit faster?
        if m.t.order_dict[t] > 1:
            s_minus_one = (((1 - model.get_option('constraints', y, 's_loss'))
                            ** m.time_res[model.prev(t)])
                           * m.s[y, x, model.prev(t)])
        else:
            s_minus_one = model.get_option('constraints', y, 's_init')
        return (m.s[y, x, t] == s_minus_one + m.rs[y, x, t] + m.bs[y, x, t]
                - m.es[y, x, t] - m.os[y, x, t])

    # Constraints
    m.c_e = cp.Constraint(m.y, m.x, m.t)
    m.c_rs = cp.Constraint(m.y, m.x, m.t)
    m.c_s_balance = cp.Constraint(m.y, m.x, m.t)


def node_constraints_build(m, o, d, model):
    """Depends on: node_energy_balance

    Defines variables:

    * s_cap: installed storage capacity
    * r_cap: installed resource <> storage conversion capacity
    * e_cap: installed storage <> electricity conversion capacity

    """
    d = model.data

    # Variables
    m.s_cap = cp.Var(m.y, m.x, within=cp.NonNegativeReals)
    m.r_cap = cp.Var(m.y, m.x, within=cp.NonNegativeReals)
    m.e_cap = cp.Var(m.y, m.x, within=cp.NonNegativeReals)

    # Constraint rules
    def c_s_cap_rule(m, y, x):
        if model.mode == 'plan':
            return m.s_cap[y, x] <= model.get_option('constraints',
                                                     y, 's_cap_max')
        elif model.mode == 'operate':
            # TODO need a better way to load an existing system's built
            # capacities with spatial differentiation!
            # TODO also need a flexible approach to disable one of the max
            # constraints e.g. by letting get_option parse 'infinity'
            # and either return a really high number or some sort of
            # infinity Pyomo object
            return m.s_cap[y, x] == model.get_option('constraints',
                                                     y, 's_cap_max')

    def c_r_cap_rule(m, y, x):
        if model.mode == 'plan':
            return m.r_cap[y, x] <= model.get_option('constraints', y,
                                                     'r_cap_max')
        elif model.mode == 'operate':
            return m.r_cap[y, x] == model.get_option('constraints', y,
                                                     'r_cap_max')

    def c_r_area_rule(m, y, x):
        if model.get_option('constraints', y, 'r_area_max') is False:
            return m.r_area[y, x] == 1.0
        elif model.mode == 'plan':
            return m.r_area[y, x] <= model.get_option('constraints', y,
                                                      'r_area_max')
        elif model.mode == 'operate':
            return m.r_area[y, x] == model.get_option('constraints', y,
                                                      'r_area_max')

    def c_e_cap_rule(m, y, x):
        # First check whether this tech is allowed at this node
        if not d.nodes.ix[x, y] == 1:
            return m.e_cap[y, x] == 0
        elif model.mode == 'plan':
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


def node_constraints_operational(m, o, d, model):
    """Depends on: node_energy_balance, node_constraints_build"""
    # Constraint rules
    def c_e_max_rule(m, y, x, t):
        return m.e[y, x, t] <= m.time_res[t] * m.e_cap[y, x]

    def c_e_min_rule(m, y, x, t):
        if model.get_option('constraints', y, 'e_can_be_negative'):
            # return cp.Constraint.Skip
            return m.e[y, x, t] >= -1 * m.time_res[t] * m.e_cap[y, x]
        else:
            return m.e[y, x, t] >= 0

    def c_s_max_rule(m, y, x, t):
        return m.s[y, x, t] <= m.s_cap[y, x]

    def c_bs_rule(m, y, x, t):
        # bs (backup resource) is allowed only during
        # the hours within startup_time
        # TODO this entire thing is a hack right now
        if y == 'csp' and t < d.startup_time_bounds:
            try:
                return m.bs[y, x, t] <= (m.time_res[t]
                                         * m.e_cap[y, x]) / m.e_eff[y, x, t]
            except ZeroDivisionError:
                return m.bs[y, x, t] == 0
        else:
            return m.bs[y, x, t] == 0

    # Constraints
    m.c_e_max = cp.Constraint(m.y, m.x, m.t)
    m.c_e_min = cp.Constraint(m.y, m.x, m.t)
    m.c_s_max = cp.Constraint(m.y, m.x, m.t)
    m.c_bs = cp.Constraint(m.y, m.x, m.t)


def node_costs(m, o, d, model):
    """
    Depends on: node_energy_balance, node_constraints_build

    Defines variables:

    * cost: total costs
    * cost_con: construction costs
    * cost_op: operation costs

    """
    # Variables
    m.cost = cp.Var(m.y, m.x, within=cp.NonNegativeReals)
    m.cost_con = cp.Var(m.y, m.x, within=cp.NonNegativeReals)
    m.cost_op = cp.Var(m.y, m.x, within=cp.NonNegativeReals)

    # Constraint rules
    def c_cost_rule(m, y, x):
        return m.cost[y, x] == m.cost_con[y, x] + m.cost_op[y, x]

    def c_cost_con_rule(m, y, x):
        return (m.cost_con[y, x] == d.depreciation[y]
                * (sum(m.time_res[t] for t in m.t) / 8760)
                * (model.get_option('costs', y, 's_cap') * m.s_cap[y, x]
                   + model.get_option('costs', y, 'r_cap') * m.r_cap[y, x]
                   + model.get_option('costs', y, 'r_area') * m.r_area[y, x]
                   + model.get_option('costs', y, 'e_cap') * m.e_cap[y, x]))

    def c_cost_op_rule(m, y, x):
        return (m.cost_op[y, x] ==
                model.get_option('costs', y, 'om_frac') * m.cost_con[y, x]
                + (model.get_option('costs', y, 'om_var')
                   * sum(m.e[y, x, t] for t in m.t))
                + (model.get_option('costs', y, 'om_fuel')
                   * sum(m.rs[y, x, t] for t in m.t)))

    # Constraints
    m.c_cost = cp.Constraint(m.y, m.x)
    m.c_cost_con = cp.Constraint(m.y, m.x)
    m.c_cost_op = cp.Constraint(m.y, m.x)


def model_slack(m, o, d):
    """Defines variables:

    * slack
    * c_slack

    """
    # Variables
    m.slack = cp.Var(m.t, within=cp.NonNegativeReals)
    m.cost_slack = cp.Var(within=cp.NonNegativeReals)

    # Constraint rules
    def c_cost_slack_rule(m):
        return m.cost_slack == sum(m.slack[t] for t in m.t)

    # Constraints
    m.c_cost_slack = cp.Constraint()


def model_constraints(m, o, d):
    """Depends on: node_energy_balance, model_slack"""
    # Constraint rules
    def c_system_balance_rule(m, t):
        return (sum(m.e[y, x, t] for x in m.x for y in m.y)
                + m.slack[t] == 0)

    # Constraints
    m.c_system_balance = cp.Constraint(m.t)


def model_objective(m, o, d, model):
    def weight(y):
        return model.get_option('tech_weights', y)

    def obj_rule(m):
        return (sum(weight(y) * sum(m.cost[y, x] for x in m.x) for y in m.y)
                + o.slack_weight * m.cost_slack)

    m.obj = cp.Objective(sense=cp.minimize)
    #m.obj.domain = cp.NonNegativeReals
