"""
Copyright (C) 2013-2015 Stefan Pfenninger.
Licensed under the Apache 2.0 License (see LICENSE file).

base.py
~~~~~~~

Basic model constraints.

"""

import pyomo.core as po
import numpy as np

from .. import exceptions
from .. import transmission
from .. import utils


def node_resource(model):
    """
    Defines variables:

    * rs: resource <-> storage (+ production, - consumption)
    * r_area: resource collector area
    * rbs: secondary resource -> storage (+ production)

    """
    m = model.m

    def availability(y, x, t):
        if model.config_model.availability and y in model.data._y_def_r:
            availability = m.a[y, x, t]
        else:
            availability = 1.0
        return availability

    # Variables
    m.rs = po.Var(m.y, m.x, m.t, within=po.Reals)
    m.r_area = po.Var(m.y_def_r, m.x, within=po.NonNegativeReals)
    m.rbs = po.Var(m.y_rb, m.x, m.t, within=po.NonNegativeReals)

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
        # elif po.value(m.r[y, x, t]) > 0:
        elif (y in model.get_group_members('supply') or
              y in model.get_group_members('unmet_demand')):
            # Supply technologies make use of availability
            return m.rs[y, x, t] <= r_avail * availability(y, x, t)
        elif y in model.get_group_members('demand'):
            return m.rs[y, x, t] >= r_avail
        elif y in model.get_group_members('storage'):
            return m.rs[y, x, t] == 0

    # Constraints
    m.c_rs = po.Constraint(m.y_def_r, m.x, m.t, rule=c_rs_rule)


def node_energy_balance(model):
    """
    Defines variables:

    * s: storage level
    * es_prod: storage -> carrier (+ production)
    * es_con: storage <- carrier (- consumption)

    """
    m = model.m
    d = model.data

    def get_e_eff(m, y, x, t):
        if y in m.y_def_e_eff:
            e_eff = m.e_eff[y, x, t]
        else:
            e_eff = d.e_eff[y][x].iat[0]  # Just get 0-th entry in DataFrame
        return e_eff

    def get_e_eff_per_distance(model, y, x):
        try:
            e_loss = model.get_option(y + '.constraints_per_distance.e_loss')
            per_distance = model.get_option(y + '.per_distance')
            distance = model.get_option(y + '.distance', x=x)
            return 1 - (e_loss * (distance / per_distance))
        except exceptions.OptionNotSetError:
            return 1.0

    # Variables
    m.s = po.Var(m.y_pc, m.x, m.t, within=po.NonNegativeReals)
    m.es_prod = po.Var(m.c, m.y, m.x, m.t, within=po.NonNegativeReals)
    m.es_con = po.Var(m.c, m.y, m.x, m.t, within=po.NegativeReals)

    # Constraint rules
    def transmission_rule(m, y, x, t):
        y_remote, x_remote = transmission.get_remotes(y, x)
        if y_remote in m.y_trans:
            c = model.get_option(y + '.carrier')
            return (m.es_prod[c, y, x, t]
                    == -1 * m.es_con[c, y_remote, x_remote, t]
                    * get_e_eff(m, y, x, t)
                    * get_e_eff_per_distance(model, y, x))
        else:
            return po.Constraint.NoConstraint

    def conversion_rule(m, y, x, t):
        c_prod = model.get_option(y + '.carrier')
        c_source = model.get_option(y + '.source_carrier')
        return (m.es_prod[c_prod, y, x, t]
                == -1 * m.es_con[c_source, y, x, t] * get_e_eff(m, y, x, t))

    def pc_rule(m, y, x, t):
        e_eff = get_e_eff(m, y, x, t)
        # TODO once Pyomo supports it,
        # let this update conditionally on param update!
        if po.value(e_eff) == 0:
            e_prod = 0
        else:
            e_prod = sum(m.es_prod[c, y, x, t] for c in m.c) / e_eff
        e_con = sum(m.es_con[c, y, x, t] for c in m.c) * e_eff

        # If this tech is in the set of techs allowing rb, include it
        if y in m.y_rb:
            rbs = m.rbs[y, x, t]
        else:
            rbs = 0

        # A) Case where no storage allowed
        if (model.get_option(y + '.constraints.s_cap_max', x=x) == 0 and
                not model.get_option(y + '.constraints.use_s_time', x=x)):
            return m.rs[y, x, t] == e_prod + e_con - rbs

        # B) Case where storage is allowed
        else:
            # set up s_minus_one
            # NB: From Pyomo 3.5 to 3.6, order_dict became zero-indexed
            if m.t.order_dict[t] == 0:
                s_minus_one = m.s_init[y, x]
            else:
                s_loss = model.get_option(y + '.constraints.s_loss')
                s_minus_one = (((1 - s_loss)
                                ** d.time_res_series.at[model.prev(t)])
                               * m.s[y, x, model.prev(t)])
            return (m.s[y, x, t] == s_minus_one + m.rs[y, x, t]
                    + rbs - e_prod - e_con)

    # Constraints
    m.c_s_balance_transmission = po.Constraint(m.y_trans, m.x, m.t,
                                               rule=transmission_rule)
    m.c_s_balance_conversion = po.Constraint(m.y_conv, m.x, m.t,
                                             rule=conversion_rule)
    m.c_s_balance_pc = po.Constraint(m.y_pc, m.x, m.t, rule=pc_rule)


def node_constraints_build(model):
    """
    Defines variables:

    * s_cap: installed storage capacity
    * r_cap: installed resource <-> storage conversion capacity
    * e_cap: installed storage <-> grid conversion capacity (gross)
    * e_cap_net: installed storage <-> grid conversion capacity (net)
    * rb_cap: installed secondary resource conversion capacity

    """
    m = model.m
    d = model.data

    # Variables
    m.s_cap = po.Var(m.y_pc, m.x, within=po.NonNegativeReals)
    m.r_cap = po.Var(m.y_def_r, m.x, within=po.NonNegativeReals)
    m.e_cap = po.Var(m.y, m.x, within=po.NonNegativeReals)
    m.e_cap_net = po.Var(m.y, m.x, within=po.NonNegativeReals)
    m.rb_cap = po.Var(m.y_rb, m.x, within=po.NonNegativeReals)

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
            return po.Constraint.NoConstraint
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
        if not d.locations.at[x, y] == 1:
            return m.e_cap[y, x] == 0
        elif np.isinf(e_cap_max):
            return po.Constraint.NoConstraint
        elif e_cap_max_force or model.mode == 'operate':
            return m.e_cap[y, x] == e_cap_max * e_cap_max_scale
        elif model.mode == 'plan':
            return m.e_cap[y, x] <= e_cap_max * e_cap_max_scale

    def c_e_cap_gross_net_rule(m, y, x):
        c_eff = model.get_option(y + '.constraints.c_eff', x=x)
        return m.e_cap[y, x] * c_eff == m.e_cap_net[y, x]

    def c_rb_cap_rule(m, y, x):
        follows = model.get_option(y + '.constraints.rb_cap_follows', x=x)
        force = model.get_option(y + '.constraints.rb_cap_max_force', x=x)
        # First, determine what the maximum is
        if follows == 'r_cap':
            rb_cap_max = m.r_cap[y, x]
        elif follows == 'e_cap':
            rb_cap_max = m.e_cap[y, x]
        elif follows is not False:
            # Raise an error to make sure follows isn't accidentally set to
            # something invalid
            e = exceptions.ModelError
            raise e('rb_cab_follows set to invalid value at '
                    '({}, {}): {}'.format(y, x, follows))
        else:
            # If nothing else set up, simply read rb_cap_max from settings
            rb_cap_max = model.get_option(y + '.constraints.rb_cap_max', x=x)
        # Then return the appropriate constraint
        if np.isinf(rb_cap_max):
            return po.Constraint.NoConstraint
        elif force or model.mode == 'operate':
            return m.rb_cap[y, x] == rb_cap_max
        elif model.mode == 'plan':
            return m.rb_cap[y, x] <= rb_cap_max

    # Constraints
    m.c_s_cap = po.Constraint(m.y_pc, m.x, rule=c_s_cap_rule)
    m.c_r_cap = po.Constraint(m.y_def_r, m.x, rule=c_r_cap_rule)
    m.c_r_area = po.Constraint(m.y_def_r, m.x, rule=c_r_area_rule)
    m.c_e_cap = po.Constraint(m.y, m.x, rule=c_e_cap_rule)
    m.c_e_cap_gross_net = po.Constraint(m.y, m.x, rule=c_e_cap_gross_net_rule)
    m.c_rb_cap = po.Constraint(m.y_rb, m.x, rule=c_rb_cap_rule)


def node_constraints_operational(model):
    m = model.m
    time_res = model.data.time_res_series

    # Constraint rules
    def c_rs_max_upper_rule(m, y, x, t):
        return m.rs[y, x, t] <= time_res.at[t] * m.r_cap[y, x]

    def c_rs_max_lower_rule(m, y, x, t):
        return m.rs[y, x, t] >= -1 * time_res.at[t] * m.r_cap[y, x]

    def c_es_prod_max_rule(m, c, y, x, t):
        if (model.get_option(y + '.constraints.e_prod') is True and
                c == model.get_option(y + '.carrier')):
            return m.es_prod[c, y, x, t] <= time_res.at[t] * m.e_cap[y, x]
        else:
            return m.es_prod[c, y, x, t] == 0

    def c_es_prod_min_rule(m, c, y, x, t):
        min_use = model.get_option(y + '.constraints.e_cap_min_use')
        if (min_use and c == model.get_option(y + '.carrier')):
            return (m.es_prod[c, y, x, t]
                    >= time_res.at[t] * m.e_cap[y, x] * min_use)
        else:
            return po.Constraint.NoConstraint

    def c_es_con_max_rule(m, c, y, x, t):
        if y in m.y_conv:
            carrier = '.source_carrier'
        else:
            carrier = '.carrier'
        if (model.get_option(y + '.constraints.e_con') is True and
                c == model.get_option(y + carrier)):
            return m.es_con[c, y, x, t] >= (-1 * time_res.at[t]
                                            * m.e_cap[y, x])
        else:
            return m.es_con[c, y, x, t] == 0

    def c_s_max_rule(m, y, x, t):
        return m.s[y, x, t] <= m.s_cap[y, x]

    def c_rbs_max_rule(m, y, x, t):
        if (model.get_option(y + '.constraints.rb_startup_only')
                and t >= model.data.startup_time_bounds):
            return m.rbs[y, x, t] == 0
        else:
            return m.rbs[y, x, t] <= (model.data.time_res_series.at[t]
                                      * m.rb_cap[y, x])

    # Constraints
    m.c_rs_max_upper = po.Constraint(m.y_def_r, m.x, m.t,
                                     rule=c_rs_max_upper_rule)
    m.c_rs_max_lower = po.Constraint(m.y_def_r, m.x, m.t,
                                     rule=c_rs_max_lower_rule)
    m.c_es_prod_max = po.Constraint(m.c, m.y, m.x, m.t,
                                    rule=c_es_prod_max_rule)
    m.c_es_prod_min = po.Constraint(m.c, m.y, m.x, m.t,
                                    rule=c_es_prod_min_rule)
    m.c_es_con_max = po.Constraint(m.c, m.y, m.x, m.t,
                                   rule=c_es_con_max_rule)
    m.c_s_max = po.Constraint(m.y_pc, m.x, m.t,
                              rule=c_s_max_rule)
    m.c_rbs_max = po.Constraint(m.y_rb, m.x, m.t,
                                rule=c_rbs_max_rule)


def node_constraints_transmission(model):
    """
    Constrains e_cap symmetrically for transmission nodes.

    """
    m = model.m

    # Constraint rules
    def c_trans_rule(m, y, x):
        y_remote, x_remote = transmission.get_remotes(y, x)
        if y_remote in m.y_trans:
            return m.e_cap[y, x] == m.e_cap[y_remote, x_remote]
        else:
            return po.Constraint.NoConstraint

    # Constraints
    m.c_transmission_capacity = po.Constraint(m.y_trans, m.x,
                                              rule=c_trans_rule)


def node_parasitics(model):
    """
    Additional variables and constraints for plants with internal parasitics.

    Defines variables:

    * ec_prod: storage -> carrier after parasitics (+ production)
    * ec_con: storage <- carrier after parasitics (- consumption)

    """
    m = model.m

    # Variables
    m.ec_prod = po.Var(m.c, m.y_p, m.x, m.t, within=po.NonNegativeReals)
    m.ec_con = po.Var(m.c, m.y_p, m.x, m.t, within=po.NegativeReals)

    # Constraint rules
    def c_ec_prod_rule(m, c, y, x, t):
        return (m.ec_prod[c, y, x, t]
                == m.es_prod[c, y, x, t]
                * model.get_option(y + '.constraints.c_eff', x=x))

    def c_ec_con_rule(m, c, y, x, t):
        if y in m.y_trans or y in m.y_conv:
            # Ensure that transmission and conversion technologies
            # do not double count c_eff
            c_eff = 1.0
        else:
            c_eff = model.get_option(y + '.constraints.c_eff', x=x)
        return (m.ec_con[c, y, x, t]
                == m.es_con[c, y, x, t]
                / c_eff)

    # Constraints
    m.c_ec_prod = po.Constraint(m.c, m.y_p, m.x, m.t, rule=c_ec_prod_rule)
    m.c_ec_con = po.Constraint(m.c, m.y_p, m.x, m.t, rule=c_ec_con_rule)


def node_costs(model):
    """
    Defines variables:

    * cost: total costs
    * cost_con: construction costs
    * cost_op_fixed: fixed operation costs
    * cost_op_var: variable operation costs
    * cost_op_fuel: primary resource fuel costs
    * cost_op_rb: secondary resource fuel costs

    """
    m = model.m

    cost_getter = utils.cost_getter(model.get_option)
    depreciation_getter = utils.depreciation_getter(model.get_option)
    cost_per_distance_getter = utils.cost_per_distance_getter(model.get_option)

    @utils.memoize
    def _depreciation_rate(y, k):
        return depreciation_getter(y, k)

    @utils.memoize
    def _cost(cost, y, k):
        return cost_getter(cost, y, k)

    @utils.memoize
    def _cost_per_distance(cost, y, k, x):
        return cost_per_distance_getter(cost, y, k, x)

    # Variables
    m.cost = po.Var(m.y, m.x, m.k, within=po.NonNegativeReals)
    m.cost_con = po.Var(m.y, m.x, m.k, within=po.NonNegativeReals)
    m.cost_op_fixed = po.Var(m.y, m.x, m.k, within=po.NonNegativeReals)
    m.cost_op_var = po.Var(m.y, m.x, m.k, within=po.NonNegativeReals)
    m.cost_op_fuel = po.Var(m.y, m.x, m.k, within=po.NonNegativeReals)
    m.cost_op_rb = po.Var(m.y, m.x, m.k, within=po.NonNegativeReals)

    # Constraint rules
    def c_cost_rule(m, y, x, k):
        return (m.cost[y, x, k] == m.cost_con[y, x, k]
                + m.cost_op_fixed[y, x, k] + m.cost_op_var[y, x, k]
                + m.cost_op_fuel[y, x, k] + m.cost_op_rb[y, x, k])

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
            # Divided by 2 for transmission techs because construction costs
            # are counted at both ends
            cost_e_cap = (_cost('e_cap', y, k)
                          + _cost_per_distance('e_cap', y, k, x)) / 2
        else:
            cost_e_cap = _cost('e_cap', y, k)

        if y in m.y_rb:
            cost_rb_cap = _cost('rb_cap', y, k) * m.rb_cap[y, x]
        else:
            cost_rb_cap = 0

        return (m.cost_con[y, x, k] == _depreciation_rate(y, k)
                * (sum(model.data.time_res_series) / 8760)
                * (cost_s_cap + cost_r_cap + cost_r_area + cost_rb_cap
                   + cost_e_cap * m.e_cap[y, x]))

    def c_cost_op_fixed_rule(m, y, x, k):
        if y in m.y:
            return (m.cost_op_fixed[y, x, k] ==
                    _cost('om_frac', y, k) * m.cost_con[y, x, k]
                    + (_cost('om_fixed', y, k) * m.e_cap[y, x] *
                       (sum(model.data.time_res_series) / 8760)))
        else:
            return m.cost_op_fixed[y, x, k] == 0

    def c_cost_op_var_rule(m, y, x, k):
        # Note: only counting es_prod for operational costs.
        # This should generally be a reasonable assumption to make.
        if y in m.y:
            carrier = model.get_option(y + '.carrier')
            return (m.cost_op_var[y, x, k] ==
                    _cost('om_var', y, k) * sum(m.es_prod[carrier, y, x, t]
                                                for t in m.t))
        else:
            return m.cost_op_var[y, x, k] == 0

    def c_cost_op_fuel_rule(m, y, x, k):
        if y in m.y:
            # Dividing by r_eff here so we get the actual r used, not the rs
            # moved into storage...
            return (m.cost_op_fuel[y, x, k] ==
                    _cost('om_fuel', y, k) * sum(m.rs[y, x, t]
                    / model.get_option(y + '.constraints.r_eff', x=x)
                    for t in m.t))
        else:
            return m.cost_op_fuel[y, x, k] == 0

    def c_cost_op_rb_rule(m, y, x, k):
        if y in m.y_rb:
            return (m.cost_op_rb[y, x, k] ==
                    _cost('om_rb', y, k) * sum(m.rbs[y, x, t]
                    / model.get_option(y + '.constraints.rb_eff', x=x)
                    for t in m.t))
        else:
            return m.cost_op_rb[y, x, k] == 0

    # Constraints
    m.c_cost = po.Constraint(m.y, m.x, m.k, rule=c_cost_rule)
    m.c_cost_con = po.Constraint(m.y, m.x, m.k, rule=c_cost_con_rule)
    m.c_cost_op_fixed = po.Constraint(m.y, m.x, m.k, rule=c_cost_op_fixed_rule)
    m.c_cost_op_var = po.Constraint(m.y, m.x, m.k, rule=c_cost_op_var_rule)
    m.c_cost_op_fuel = po.Constraint(m.y, m.x, m.k, rule=c_cost_op_fuel_rule)
    m.c_cost_op_rb = po.Constraint(m.y, m.x, m.k, rule=c_cost_op_rb_rule)


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
        # Hardcoded: balancing takes place between locations on level 1 only
        parents = get_parents(1)
        if x not in parents:
            return po.Constraint.NoConstraint
        else:
            fam = get_children(x) + [x]  # list of children + parent
            balance = (sum(m.es_prod[c, y, xs, t]
                           for xs in fam for y in m.y_np)
                       + sum(m.ec_prod[c, y, xs, t]
                             for xs in fam for y in m.y_p)
                       + sum(m.es_con[c, y, xs, t]
                             for xs in fam for y in m.y_np)
                       + sum(m.ec_con[c, y, xs, t]
                             for xs in fam for y in m.y_p))
            if c == 'power':
                return balance == 0
            else:  # e.g. for heat
                return balance >= 0

    # Constraints
    m.c_system_balance = po.Constraint(m.c, m.x, m.t,
                                       rule=c_system_balance_rule)
