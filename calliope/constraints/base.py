"""
Copyright (C) 2013-2016 Stefan Pfenninger.
Licensed under the Apache 2.0 License (see LICENSE file).

base_planning.py
~~~~~~~

Basic model constraints.

"""

import pyomo.core as po
import numpy as np

from .. import exceptions
from .. import transmission
from .. import utils

def get_constraint_param(model, param_string, y, x, t):
    """
    Function to get values for constraints which can optionally be
    loaded from file (so may have time dependency).

    model = calliope model
    param_string = constraint as string
    y = technology
    x = location
    t = timestep
    """

    if param_string in model.data:
        return getattr(model.m, param_string)[y, x, t]
    else:
        return model.get_option(y + '.constraints.' + param_string, x=x)

def get_cost_param(model, cost, k, y, x, t, cost_type='cost'):
    """
    Function to get values for constraints which can optionally be
    loaded from file (so may have time dependency).

    model = calliope model
    cost = cost name, e.g. 'om_fuel'
    k = cost type, e.g. 'monetary'
    y = technology
    x = location
    t = timestep
    """

    cost_getter = utils.cost_getter(model.get_option)

    @utils.memoize
    def _cost(cost, y, k, x=None):
        return cost_getter(cost, y, k, x=x)

    @utils.memoize
    def _revenue(cost, y, k, x=None):
        return cost_getter(cost, y, k, x=x, costs_type='revenue')

    if cost_type == 'cost':
        param_string = 'costs_' + k + '_' + cost #format stored in model.data
        if param_string in model.data:
            return getattr(model.m, param_string)[y, x, t]
        else:
            return _cost(cost, y, k, x)
    elif cost_type == 'revenue':
        param_string = 'revenue_' + k + '_' + cost #format stored in model.data
        if param_string in model.data:
            return getattr(model.m, param_string)[y, x, t]
        else:
            return _revenue(cost, y, k, x)

def generate_variables(model):
    """
    Defines variables:

    * rs: resource <-> storage (+ production, - consumption)
    * r_area: resource collector area
    * rbs: secondary resource -> storage (+ production)
    * es_prod: storage -> carrier (+ production)
    * es_con: storage <- carrier (- consumption)
    * s_cap: installed storage capacity
    * r_cap: installed resource <-> storage conversion capacity
    * e_cap: installed storage <-> grid conversion capacity (gross)
    * r2_cap: installed secondary resource conversion capacity

    * cost: total costs
    * cost_con: construction costs
    * cost_op_fixed: fixed operation costs
    * cost_op_var: variable operation costs
    * cost_op_fuel: primary resource fuel costs
    * cost_op_r2: secondary resource fuel costs
    * revenue_var: variable revenue (operation + fuel)
    * revenue_fixed: fixed revenue
    * revenue: total revenue

    """

    m = model.m

    # Capacity
    m.r_area = po.Var(m.y_r_area, m.x_r, within=po.NonNegativeReals)
    m.s_cap = po.Var(m.y_store, m.x_store, within=po.NonNegativeReals)
    m.r_cap = po.Var(m.y_supply_plus, m.x_r, within=po.NonNegativeReals) # maybe should be y_finite_r?
    m.e_cap = po.Var(m.y_all, m.x, within=po.NonNegativeReals)
    m.r2_cap = po.Var(m.y_sp_r2, m.x_r, within=po.NonNegativeReals)

    # Unit commitment
    m.r = po.Var(m.y_sp_finite_r, m.x_r, m.t, within=po.Reals)
    m.r2 = po.Var(m.y_sp_r2, m.x_r, m.t, within=po.NonNegativeReals)
    m.s = po.Var(m.y_store, m.x_store, m.t, within=po.NonNegativeReals)
    m.c_prod = po.Var(m.c, m.y_all, m.x, m.t, within=po.NonNegativeReals)
    m.c_con = po.Var(m.c, m.y_all, m.x, m.t, within=po.NegativeReals)

    # Costs/revenue
    m.cost_var = po.Var(m.y_cost_var, m.x, m.t, m.kc, within=po.NonNegativeReals)
    m.cost_fixed = po.Var(m.y_cost_fixed, m.x, m.kc, within=po.NonNegativeReals)
    m.cost = po.Var(m.y_all, m.x, m.kc, within=po.NonNegativeReals)
    m.revenue_var = po.Var(m.y_all, m.x, m.t, m.kr, within=po.NonNegativeReals)
    m.revenue_fixed = po.Var(m.y_all, m.x, m.kr, within=po.NonNegativeReals)
    m.revenue = po.Var(m.y_all, m.x, m.kr, within=po.NonNegativeReals)

def node_resource(model):

    m = model.m

    # Constraint rules
    # TODO reformulate c_r_rule conditionally once Pyomo supports that.
    # had to remove the following formulation because it is not
    # re-evaluated on model re-construction -- we now check for
    # demand/supply tech instead, which means that `r` can only
    # be ALL negative or ALL positive for a given tech!
    # Ideally we have `elif po.value(m.r[y, x, t]) > 0:` instead of
    # `elif y in m.y_supply or y in m.y_unmet_demand:` and `elif y in m.y_demand:`
    def r_available_rule(m, y, x, t):
        r_scale = model.get_option(y + '.constraints.r_scale', x=x)
        force_r = get_constraint_param(model, 'force_r', y, x, t)

        if y in m.y_sd:
            e_eff = get_constraint_param(model, 'e_eff', y, x, t)
            if po.value(e_eff) == 0:
                c_prod = 0
            else:
                c_prod = sum(m.c_prod[c, y, x, t] for c in m.c) / e_eff
            c_con = sum(m.c_con[c, y, x, t] for c in m.c) * e_eff

            if y in m.y_sd_r_area:
                r_area = model.get_option(y + '.constraints.r_area_per_r', x=x)
            else:
                r_area = 1.0
            r_avail = m.r_param[y, x, t] * r_scale * r_area

            if force_r:
                return (c_prod * model.get_option(y + 'constraints.e_prod') +
                  c_con * model.get_option(y + 'constraints.e_con') == r_avail)
            elif y in m.y_supply or y in m.y_unmet:
                return c_prod <= r_avail
            elif y in m.y_demand:
                return c_con >= r_avail

        elif y in m.y_supply_plus:
            r_eff = get_constraint_param(model, 'r_eff', y, x, t)

            if y in m.y_sp_r_area:
                r_avail = m.r_param[y, x, t] * r_scale * m.r_area[y, x] * r_eff
            else:
                r_avail = m.r_param[y, x, t] * r_scale * r_eff

            if force_r:
                return m.r[y, x, t] == r_avail
            else:
                return m.r[y, x, t] <= r_avail

    # Constraints
    m.c_r_available = po.Constraint(m.y_finite_r, m.x_r, m.t,
                                    rule=r_available_rule)


def node_energy_balance(model):

    m = model.m
    d = model.data
    time_res = model.data['_time_res'].to_series()

    def get_e_eff_per_distance(model, y, x):
        try:
            e_loss = model.get_option(y + '.constraints_per_distance.e_loss', x=x)
            per_distance = model.get_option(y + '.per_distance')
            distance = model.get_option(y + '.distance')
            return 1 - (e_loss * (distance / per_distance))
        except exceptions.OptionNotSetError:
            return 1.0

    def get_conversion_out(c_1, c_2, m, y, x, t):
        if isinstance(c_1, dict):
            c_prod1 = sum([m.c_prod[c, y, x, t] * c_1[c] for c in c_1.keys()])
        else:
            c_prod1 = m.c_prod[c_1, y, x, t]
        if isinstance(c_2, dict):
            c_prod2 = sum([m.c_prod[c, y, x, t] * c_2[c] for c in c_2.keys()])
        else:
            c_prod2 = m.c_prod[c_2, y, x, t]
        return c_prod1 == c_prod2

    def get_conversion_in(c_1, c_2, m, y, x, t):
        if isinstance(c_1, dict):
            c_con1 = sum([m.c_con[c, y, x, t] * c_1[c] for c in c_1.keys()])
        else:
            c_con1 = m.c_con[c_out, y, x, t]
        if isinstance(c_2, dict):
            c_con2 = sum([m.c_con[c, y, x, t] * c_2[c] for c in c_2.keys()])
        else:
            c_con2 = m.c_con[c_2, y, x, t]
        return c_con1 == c_con2

    # Constraint rules
    def transmission_rule(m, y, x, t):
        y_remote, x_remote = transmission.get_remotes(y, x)
        e_eff = get_constraint_param(model, 'e_eff', y, x, t)
        if y_remote in m.y_transmission:
            c = model.get_option(y + '.carrier')

            return (m.c_prod[c, y, x, t]
                    == -1 * m.c_con[c, y_remote, x_remote, t]
                    * e_eff
                    * get_e_eff_per_distance(model, y, x))
        else:
            return po.Constraint.NoConstraint

    def conversion_rule(m, y, x, t):
        c_out = model.get_option(y + '.carrier_out')
        c_in = model.get_option(y + '.carrier_in')
        e_eff = get_constraint_param(model, 'e_eff', y, x, t)
        return (m.c_prod[c_out, y, x, t]
                == -1 * m.c_con[c_in, y, x, t] * e_eff)

    def conversion_plus_primary_rule(m, y, x, t):
        c_out = model.get_option(y + '.carrier_out')
        c_in = model.get_option(y + '.carrier_in')
        e_eff = get_constraint_param(model, 'e_eff', y, x, t)
        if isinstance(c_out, dict):
            c_prod = sum([m.c_prod[c, y, x, t] * c_out[c] for c in c_out.keys()])
        else:
            c_prod = m.c_prod[c_out, y, x, t]
        if isinstance(c_in, dict):
            c_con = sum([m.c_con[c, y, x, t] * c_in[c] for c in c_in.keys()])
        else:
            c_con = m.c_con[c_in, y, x, t]
        return c_prod == -1 * c_con * e_eff

    def conversion_plus_secondary_out_rule(m, y, x, t):
        c_1 = model.get_option(y + '.carrier_out')
        c_2 = model.get_option(y + '.carrier_out_2')
        return get_conversion_out(c_1, c_2, m, y, x, t)

    def conversion_plus_tertiary_out_rule(m, y, x, t):
        c_1 = model.get_option(y + '.carrier_out')
        c_3 = model.get_option(y + '.carrier_out_3')
        return get_conversion_out(c_1, c_3, m, y, x, t)

    def conversion_plus_secondary_in_rule(m, y, x, t):
        c_1 = model.get_option(y + '.carrier_in')
        c_2 = model.get_option(y + '.carrier_in_2')
        return get_conversion_in(c_1, c_2, m, y, x, t)

    def conversion_plus_tertiary_in_rule(m, y, x, t):
        c_1 = model.get_option(y + '.carrier_in')
        c_3 = model.get_option(y + '.carrier_in_3')
        return get_conversion_in(c_1, c_3, m, y, x, t)

    def supply_plus_rule(m, y, x, t):
        e_eff = get_constraint_param(model, 'e_eff', y, x, t)
        p_eff = model.get_option(y + '.constraints.p_eff', x=x)
        total_eff = e_eff * p_eff

        # TODO once Pyomo supports it,
        # let this update conditionally on param update!
        if po.value(e_eff) == 0:
            c_prod = 0
        else:
            c_prod = sum(m.c_prod[c, y, x, t] for c in m.c) / total_eff
        c_con = sum(m.c_con[c, y, x, t] for c in m.c) * total_eff

        # If this tech is in the set of techs allowing rb, include it
        if y in m.y_r2:
            r2 = m.r2[y, x, t]
        else:
            r2 = 0

        # A) Case where no storage allowed
        if y not in m.y_store:
            return m.r[y, x, t] == c_prod + c_con - r2

        # B) Case where storage is allowed
        s_cap_max = model.get_option(y + '.constraints.s_cap.max', x=x)
        use_s_time = get_constraint_param(model, 'use_s_time', y, x, t)
        # Ensure that storage-only techs have no r_2_s
        r = m.r[y, x, t]
        if m.t.order_dict[t] == 0:
            s_minus_one = m.s_init[y, x]
        else:
            s_loss = get_constraint_param(model, 's_loss', y, x, t)
            s_minus_one = (((1 - s_loss)
                            ** time_res.at[model.prev_t(t)])
                           * m.s[y, x, model.prev_t(t)])
        return (m.s[y, x, t] == s_minus_one + r + r2 - c_prod - c_con)

    def storage_rule(m, y, x, t):
        e_eff = get_constraint_param(model, 'e_eff', y, x, t)
        p_eff = model.get_option(y + '.constraints.p_eff', x=x)
        total_eff = e_eff * p_eff

        # TODO once Pyomo supports it,
        # let this update conditionally on param update!
        if po.value(e_eff) == 0:
            c_prod = 0
        else:
            c_prod = sum(m.c_prod[c, y, x, t] for c in m.c) / total_eff
        c_con = sum(m.c_con[c, y, x, t] for c in m.c) * total_eff

        s_cap_max = model.get_option(y + '.constraints.s_cap.max', x=x)

        if m.t.order_dict[t] == 0:
            s_minus_one = m.s_init[y, x]
        else:
            s_loss = get_constraint_param(model, 's_loss', y, x, t)
            s_minus_one = (((1 - s_loss)
                            ** time_res.at[model.prev_t(t)])
                           * m.s[y, x, model.prev_t(t)])
        return (m.s[y, x, t] == s_minus_one - c_prod - c_con)

    # Constraints
    m.c_balance_transmission = po.Constraint(m.y_transmission, m.x, m.t,
                                            rule=transmission_rule)
    m.c_balance_conversion = po.Constraint(m.y_conv, m.x, m.t,
                                            rule=conversion_rule)
    m.c_balance_conversion_plus = po.Constraint(m.y_conversion_plus, m.x, m.t,
                                            rule=conversion_plus_primary_rule)
    m.c_balance_conversion_plus_secondary_out = po.Constraint(m.y_cp_2out, m.x, m.t,
                                            rule=conversion_plus_secondary_out_rule)
    m.c_balance_conversion_plus_tertiary_out = po.Constraint(m.y_cp_3out, m.x, m.t,
                                            rule=conversion_plus_tertiary_out_rule)
    m.c_balance_conversion_plus_secondary_in = po.Constraint(m.y_cp_2in, m.x, m.t,
                                            rule=conversion_plus_secondary_in_rule)
    m.c_balance_conversion_plus_tertiary_in = po.Constraint(m.y_cp_3in, m.x, m.t,
                                            rule=conversion_plus_tertiary_in_rule)
    m.c_balance_supply_plus = po.Constraint(m.y_supply_plus, m.x, m.t,
                                            rule=supply_plus_rule)
    m.c_balance_storage = po.Constraint(m.y_storage, m.x, m.t,
                                            rule=storage_rule)

def node_constraints_build(model):

    m = model.m
    d = model.data

    def get_var_constraint(model_var, y, var, x,
                           _equals=None, _max=None, _min=None,
                           scale=None):

        if not _equals:
            _equals = model.get_option(y + '.constraints.'
                                       + var + '.equals', x=x)
        if not _max:
            _max = model.get_option(y + '.constraints.' + var + '.max', x=x)
        if not _min:
            _min = model.get_option(y + '.constraints.' + var + '.min', x=x)
        if scale:
            _equals = scale * _equals
            _min = scale * _min
            _max = scale * _max
        if _equals:
            if np.isinf(_equals):
                e = exceptions.ModelError
                raise e('Cannot use inf in operational mode, for value of '
                        '{}.{}.equals.{}'.format(y, var, x))
            return model_var == _equals
        elif model.mode == 'operate':
            # Operational mode but 'equals' constraint not set, we use 'max'
            # instead
            # FIXME this should be logged
            if np.isinf(_max):
                return po.Constraint.NoConstraint
            else:
                return model_var == _max
        else:
            if np.isinf(_max):
                _max = None  # to disable upper bound
            if _min == 0 and _max is None:
                return po.Constraint.NoConstraint
            else:
                return (_min, model_var, _max)

    # Constraint rules
    def c_s_cap_rule(m, y, x):
        """
        Set maximum storage capacity. Supply_plus & storage techs only
        This can be set by either s_cap (kWh) or by
        e_cap (charge/discharge capacity) * charge rate.
        If s_cap.equals and e_cap.equals are set for the technology, then
        s_cap * charge rate = e_cap must hold. Otherwise, take the lowest capacity
        capacity defined by s_cap.max or e_cap.max / charge rate.
        """
        s_cap = model.get_option(y + '.constraints.s_cap.equals', x=x)
        scale = model.get_option(y + '.constraints.e_cap_scale', x=x)
        e_cap = model.get_option(y + '.constraints.s_cap.equals', x=x) * scale
        charge_rate = model.get_option(y + '.constraints.c_rate', x=x)
        if e_cap and s_cap and s_cap * charge_rate != e_cap:
            raise exceptions.ModelError('e_cap.equals and s_cap.equals must '
                        'be equivalent when considering charge rate for {}:{}'
                        .format(y, x))
        if not s_cap:
            s_cap = model.get_option(y + '.constraints.s_cap.max', x=x)
        if not e_cap:
            e_cap = model.get_option(y + '.constraints.e_cap.max', x=x) * scale
        if e_cap and s_cap:
            s_cap_max = min(s_cap, e_cap / charge_rate)
        else:
            s_cap_max = max(s_cap, e_cap / charge_rate)

        return get_var_constraint(m.s_cap[y, x], y, 's_cap', x, _max=s_cap_max)

    def c_r_cap_rule(m, y, x):
        if model.get_option(y + '.constraints.r_cap_equals_e_cap', x=x):
            return m.r_cap[y, x] == m.e_cap[y, x]
        else:
            return get_var_constraint(m.r_cap[y, x], y, 'r_cap', x)

    def c_r_area_rule(m, y, x):
        """
        Set maximum r_area. Supply_plus techs only.
        """
        area_per_cap = model.get_option(y + '.constraints.r_area_per_e_cap', x=x)
        if area_per_cap:
            return m.r_area[y, x] == m.e_cap[y, x] * area_per_cap
        else:
            e_cap_max = model.get_option(y + '.constraints.e_cap.max', x=x)
            if e_cap_max == 0:
                # If a technology has no e_cap here, we force r_area to zero,
                # so as not to accrue spurious costs
                return m.r_area[y, x] == 0
            elif model.get_option(y + '.constraints.r_area.max', x=x) is False:
                return m.r_area[y, x] == 1
            else:
                return get_var_constraint(m.r_area[y, x], y, 'r_area', x)

    def c_e_cap_rule(m, y, x):
        """
        Set maximum e_cap. All technologies.
        """
        # First check whether this tech is allowed at this location
        if not model._locations.at[x, y] == 1:
            return m.e_cap[y, x] == 0
        e_cap_scale = model.get_option(y + '.constraints.e_cap_scale', x=x)
        if y in m.y_store:
            charge_rate = model.get_option(y + '.constraints.c_rate', x=x)
            return m.e_cap[y, x] * e_cap_scale == m.s_cap[x, y] * charge_rate
        else:
            return get_var_constraint(m.e_cap[y, x], y, 'e_cap', x,
                                      scale=e_cap_scale)

    def c_r2_cap_rule(m, y, x):
        """
        Set secondary resource capacity. Supply_plus techs only.
        """
        follow = model.get_option(y + '.constraints.r2_cap_follow', x=x)
        mode = model.get_option(y + '.constraints.r2_cap_follow_mode', x=x)

        # First deal with the special case of ``r2_cap_follow`` being set
        if follow:
            if follow == 'r_cap':
                r2_cap_val = m.r_cap[y, x]
            elif follow == 'e_cap':
                r2_cap_val = m.e_cap[y, x]
            elif follow is not False:
                # Raise an error to make sure follows isn't accidentally set to
                # something invalid
                e = exceptions.ModelError
                raise e('r2_cab_follow set to invalid value at '
                        '({}, {}): {}'.format(y, x, follow))

            if mode == 'max':
                return m.r2_cap[y, x] <= r2_cap_val
            elif mode == 'equals':
                return m.r2_cap[y, x] == r2_cap_val

        else:  # If ``r2_cap_follow`` not set, set up standard constraints
            return get_var_constraint(m.r2_cap[y, x], y, 'r2_cap', x)

    # Constraints
    m.c_s_cap = po.Constraint(m.y_store, m.x, rule=c_s_cap_rule)
    m.c_r_cap = po.Constraint(m.y_sp_finite_r, m.x_r, rule=c_r_cap_rule)
    m.c_r_area = po.Constraint(m.y_sp_r_area, m.x, rule=c_r_area_rule)
    m.c_e_cap = po.Constraint(m.y, m.x, rule=c_e_cap_rule)
    m.c_r2_cap = po.Constraint(m.y_sp_r2, m.x, rule=c_r2_cap_rule)


def node_constraints_operational(model):
    m = model.m
    time_res = model.data['_time_res'].to_series()

    # Constraint rules
    def r_max_upper_rule(m, y, x, t):
        """
        set maximum resource supply. Supply_plus techs only.
        """
        return m.r[y, x, t] <= time_res.at[t] * m.r_cap[y, x]

    def r_max_lower_rule(m, y, x, t):
        """
        set maximum resource consumption. Demand techs only.
        """
        return m.r[y, x, t] >= -1 * time_res.at[t] * m.r_cap[y, x]


    def c_prod_max_rule(m, c, y, x, t):
        """
        Set maximum carrier production. All technologies.
        """
        allow_c_prod = get_constraint_param(model, 'e_prod', y, x, t)
        p_eff = model.get_option(y + '.constraints.p_eff', x=x)
        if y in m.y_conversion or y in m.y_conversion_plus: # conversion techs with 2 output carriers
            c_out = model.get_option(y + '.carrier_out', x=x)
            c_out_2 = model.get_option(y + '.carrier_out_2', x=x)
            c_out_3 = model.get_option(y + '.carrier_out_3', x=x)
            c_prod = 0
            if isinstance(c_out, dict): # conversion_plus tech
                if c_out.get(c, None):
                    c_prod = time_res.at[t] * m.e_cap[y, x] * p_eff * c_out[c]
            elif c == c_out:
                c_prod = time_res.at[t] * m.e_cap[y, x] * p_eff
            if isinstance(c_out_2, dict): # conversion_plus tech
                if c_out_2.get(c, None):
                    return po.Constraint.NoConstraint
            elif isinstance(c_out_3, dict): # conversion_plus tech
                if c_out_3.get(c, None):
                    return po.Constraint.NoConstraint
            if allow_c_prod is True:
                return m.c_prod[c, y, x, t] <= c_prod
            else:
                return m.c_prod[c, y, x, t] == 0
        if (allow_c_prod is True and
                c == model.get_option(y + '.carrier')):
            return m.c_prod[c, y, x, t] <= time_res.at[t] * m.e_cap[y, x] * p_eff
        else:
            return m.c_prod[c, y, x, t] == 0

    def c_prod_min_rule(m, c, y, x, t):
        """
        Set minimum carrier production. All technologies.
        """
        min_use = get_constraint_param(model, 'e_cap_min_use', y, x, t)
        if (min_use and c == model.get_option(y + '.carrier')):
            return (m.c_prod[c, y, x, t]
                    >= time_res.at[t] * m.e_cap[y, x] * min_use)
        else:
            return po.Constraint.NoConstraint

    def c_con_max_rule(m, c, y, x, t):
        """
        Set maximum carrier consumption. All technologies.
        """
        c_con = get_constraint_param(model, 'e_con', y, x, t)
        p_eff = model.get_option(y + '.constraints.p_eff', x=x)
        if y in m.y_conversion or y in m.y_conversion_plus:
            return po.Constraint.Skip
        else:
            carrier = '.carrier'
        if (c_con is True and
                c == model.get_option(y + carrier)):
            return m.c_con[c, y, x, t] >= (-1 * time_res.at[t]
                                            * m.e_cap[y, x] * p_eff)
        else:
            return m.c_con[c, y, x, t] == 0

    def s_max_rule(m, y, x, t):
        """
        Set maximum stored energy. Supply_plus & storage techs only.
        """
        return m.s[y, x, t] <= m.s_cap[y, x]

    def r2_max_rule(m, y, x, t):
        """
        Set maximum secondary resource supply. Supply_plus techs only.
        """
        r2_startup = get_constraint_param(model, 'r2_startup_only', y, x, t)
        if (r2_startup
                and t >= model.data.startup_time_bounds):
            return m.r2[y, x, t] == 0
        else:
            return m.r2[y, x, t] <= time_res.at[t] * m.r2_cap[y, x]

    # Constraints
    m.c_r_max_upper = po.Constraint(m.y_sp_finite_r, m.x_r, m.t,
                                     rule=r_max_upper_rule)
    m.c_r_max_lower = po.Constraint(m.y_sp_finite_r, m.x_r, m.t,
                                     rule=r_max_lower_rule)
    m.c_prod_max = po.Constraint(m.c, m.y, m.x, m.t,
                                    rule=c_prod_max_rule)
    m.c_prod_min = po.Constraint(m.c, m.y, m.x, m.t,
                                    rule=c_prod_min_rule)
    m.c_con_max = po.Constraint(m.c, m.y, m.x, m.t,
                                   rule=c_con_max_rule)
    m.c_s_max = po.Constraint(m.y_store, m.x_store, m.t,
                              rule=s_max_rule)
    m.c_r2_max = po.Constraint(m.y_sp_r2, m.x_r, m.t,
                                rule=r2_max_rule)


def node_constraints_transmission(model):
    """
    Constrain e_cap symmetrically for transmission nodes. Transmission techs only.
    """
    m = model.m

    # Constraint rules
    def c_trans_rule(m, y, x):
        y_remote, x_remote = transmission.get_remotes(y, x)
        if y_remote in m.y_transmission:
            return m.e_cap[y, x] == m.e_cap[y_remote, x_remote]
        else:
            return po.Constraint.NoConstraint

    # Constraints
    m.c_transmission_capacity = po.Constraint(m.y_transmission, m.x,
                                              rule=c_trans_rule)

def node_costs(model):

    m = model.m
    time_res = model.data['_time_res'].to_series()
    weights = model.data['_weights'].to_series()

    cost_getter = utils.cost_getter(model.get_option)
    depreciation_getter = utils.depreciation_getter(model.get_option)
    cost_per_distance_getter = utils.cost_per_distance_getter(model.get_option)

    @utils.memoize
    def _depreciation_rate(y, k):
        return depreciation_getter(y, k)

    @utils.memoize
    def _cost(cost, y, k, x=None):
        return cost_getter(cost, y, k, x=x)

    @utils.memoize
    def _cost_per_distance(cost, y, k, x):
        return cost_per_distance_getter(cost, y, k, x)

    # Variables
    m.cost = po.Var(m.y, m.x, m.k, within=po.NonNegativeReals)
    m.cost_con = po.Var(m.y, m.x, m.k, within=po.NonNegativeReals)
    m.cost_op_fixed = po.Var(m.y, m.x, m.k, within=po.NonNegativeReals)
    m.cost_op_variable = po.Var(m.y, m.x, m.k, within=po.NonNegativeReals)
    m.cost_op_var = po.Var(m.y, m.x, m.t, m.k, within=po.NonNegativeReals)
    m.cost_op_fuel = po.Var(m.y, m.x, m.t, m.k, within=po.NonNegativeReals)
    m.cost_op_rb = po.Var(m.y, m.x, m.t, m.k, within=po.NonNegativeReals)
    # Constraint rules
    ## removed this function as it consolidated everything a bit too much
    ## Instead, cost has been split into fixed and variable cases
    #def c_cost_rule(m, y, x, k):
    #    carrier = model.get_option(y + '.carrier')
    #    if y in m.y_pc:
    #        cost_s_cap = _cost('s_cap', y, k, x) * m.s_cap[y, x]
    #    else:
    #        cost_s_cap = 0
#
    #    if y in m.y_def_r:
    #        cost_r_cap = _cost('r_cap', y, k, x) * m.r_cap[y, x]
    #        cost_r_area = _cost('r_area', y, k, x) * m.r_area[y, x]
    #    else:
    #        cost_r_cap = 0
    #        cost_r_area = 0
#
    #    if y in m.y_transmission:
    #        # Divided by 2 for transmission techs because construction costs
    #        # are counted at both ends
    #        cost_e_cap = (_cost('e_cap', y, k, x)
    #                      + _cost_per_distance('e_cap', y, k, x)) / 2
    #    else:
    #        cost_e_cap = _cost('e_cap', y, k, x)
#
    #    if y in m.y_rb:
    #        cost_rb_cap = _cost('rb_cap', y, k, x) * m.rb_cap[y, x]
    #        cost_op_rb = sum(get_cost_param(model,'om_rb', k, y, x, t) *
    #                         weights.loc[t] * (m.rbs[y, x, t]
    #                         / get_constraint_param(model, 'rb_eff', y, x, t))
    #                         for t in m.t)
    #        # Remove infinite results which happen when rb_eff = 0
    #        cost_op_rb[cost_op_rb == np.inf] == 0
    #    else:
    #        cost_rb_cap = 0
    #        cost_op_rb = 0
#
    #    cost_con = _depreciation_rate(y, k) *
    #        (sum(time_res * weights) / 8760) *
    #        (cost_s_cap + cost_r_cap + cost_r_area + cost_rb_cap +
    #         cost_e_cap * m.e_cap[y, x])
#
    #    cost_op_fixed = _cost('om_frac', y, k, x) * cost_con
    #                + (_cost('om_fixed', y, k, x) * m.e_cap[y, x] *
    #                   (sum(time_res * weights) / 8760))
#
#
    #    cost_op_var = sum(get_cost_param(model,'om_var', k, y, x, t) *
    #                        weights.loc[t] * m.c_prod[carrier, y, x, t]
    #                        for t in m.t)
#
    #    # Dividing by r_eff here so we get the actual r used, not the rs
    #    # moved into storage...
    #    cost_op_fuel = sum(get_cost_param(model,'om_fuel', k, y, x, t) *
    #                         weights.loc[t] * (m.rs[y, x, t] /
    #                         get_constraint_param(model, 'r_eff', y, x, t))
    #                         for t in m.t)
    #    # Remove infinite results which happen when r_eff = 0
    #    cost_op_fuel[cost_op_fuel == np.inf] == 0
#
    #    cost_op_variable = cost_op_var + cost_op_fuel + cost_op_rb
#
    #    return (
    #        m.cost[y, x, k] == cost_con + cost_op_fixed + cost_op_variable
    #    )
#

    def cost_fixed_rule(m, y, x, k):
        if y in m.y_store:
            cost_s_cap = _cost('s_cap', y, k, x) * m.s_cap[y, x]
        else:
            cost_s_cap = 0

        if y in m.y_sp_r_finite:
            cost_r_cap = _cost('r_cap', y, k, x) * m.r_cap[y, x]
        else:
            cost_r_cap = 0

        if y in m.y_r_area:
            cost_r_area = _cost('r_area', y, k, x) * m.r_area[y, x]
        else:
            cost_r_area = 0

        if y in m.y_transmission:
            # Divided by 2 for transmission techs because construction costs
            # are counted at both ends
            cost_e_cap = (_cost('e_cap', y, k, x)
                          + _cost_per_distance('e_cap', y, k, x)) / 2
        else:
            cost_e_cap = _cost('e_cap', y, k, x)

        if y in m.y_sp_r2:
            cost_r2_cap = _cost('r2_cap', y, k, x) * m.r2_cap[y, x]
        else:
            cost_r2_cap = 0

        cost_con = (_depreciation_rate(y, k) *
            (sum(time_res * weights) / 8760) *
            (cost_s_cap + cost_r_cap + cost_r_area + cost_r2_cap +
             cost_e_cap * m.e_cap[y, x]))

        return (m.cost_fixed[y, x, k] ==
                    _cost('om_frac', y, k, x) * cost_con
                    + (_cost('om_fixed', y, k, x) * m.e_cap[y, x] *
                       (sum(time_res * weights) / 8760)) + cost_con)

    def cost_var_rule(m, y, x, t, k):
        om_var = get_cost_param(model,'om_var', k, y, x, t)
        carrier = model.get_option(y + '.carrier')
        # Note: only counting c_prod for operational costs.
        # This should generally be a reasonable assumption to make.
        # It might be necessary to remove parasitic losses for this
        # i.e. c_prod --> es_prod.

        cost_op_var = om_var * weights.loc[t] * m.c_prod[carrier, y, x, t]

        # in case r_eff is zero, to avoid an infinite value
        if y in m.y_pc:
            r_eff = get_constraint_param(model, 'r_eff', y, x, t)
            om_fuel = get_cost_param(model,'om_fuel', k, y, x, t)
            if po.value(r_eff) > 0:
                # Dividing by r_eff here so we get the actual r used, not the rs
                # moved into storage...
                cost_op_fuel = (om_fuel * weights.loc[t] * (m.rs[y, x, t] / r_eff))
            else:
                cost_op_fuel = 0
        else:
            cost_op_fuel = 0

        # in case r2_eff is zero, to avoid an infinite value
        if y in m.y_r2:
            r2_eff = get_constraint_param(model, 'r2_eff', y, x, t)
            if po.value(r2_eff) > 0:
                om_r2 = get_cost_param(model, 'om_r2', k, y, x, t)
                cost_op_r2 = (om_r2 * weights.loc[t] * (m.rbs[y, x, t] / r2_eff))
            else:
                cost_op_r2 = 0
        else: #in case r2_eff is zero, to avoid an infinite value
            cost_op_r2 = 0

        return (m.cost_var[y, x, t, k] == cost_op_var + cost_op_fuel
                                                    + cost_op_r2)

    def cost_rule(m, y, x, k):
        return (
            m.cost[y, x, k] ==
            m.cost_fixed[y, x, k] +
            sum(m.cost_var[y, x, t, k] for t in m.t)
        )

    def revenue_fixed_rule(m, y, x, k):
        revenue = (sum(time_res * weights) / 8760 *
            (_revenue('sub_cap', y, k, x)
            * _depreciation_rate(y, k)
            + _revenue('sub_annual', y, k, x)))
        if y in m.y_demand and revenue > 0:
            e = exceptions.ModelError
            raise e('Cannot receive fixed revenue at a demand node, i.e. '
                    '{}'.format(y))
        else:
            return (m.revenue_fixed[y, x, k] ==
             revenue * m.e_cap[y, x])

    def revenue_var_rule(m, y, x, t, k):
        carrier = model.get_option(y + '.carrier')
        sub_var = get_cost_param(model, 'sub_var', k, y, x, t,
                                 cost_type='revenue')
        if y in m.y_demand:
            return (m.revenue_var[y, x, t, k] ==
                sub_var * weights.loc[t]
                * -m.c_con[carrier, y, x, t])
        else:
            return (m.revenue_var[y, x, t, k] ==
                sub_var * weights.loc[t]
                * m.c_prod[carrier, y, x, t])

    def c_revenue_rule(m, y, x, k):
        return (m.revenue[y, x, k] == m.revenue_fixed[y, x, k] +
            sum(m.revenue_var[y, x, t, k] for t in m.t))

    # Constraints
    m.c_cost_fixed = po.Constraint(m.y, m.x, m.kc, rule=cost_fixed_rule)
    m.c_cost_var = po.Constraint(m.y, m.x, m.t, m.kc, rule=cost_var_rule)
    m.c_cost = po.Constraint(m.y, m.x, m.kc, rule=cost_rule)

    m.c_revenue_fixed = po.Constraint(m.y, m.x, m.kr, rule=revenue_fixed_rule)
    m.c_revenue_var = po.Constraint(m.y, m.x, m.t, m.kr, rule=revenue_var_rule)
    m.c_revenue = po.Constraint(m.y, m.x, m.kr, rule=c_revenue_rule)


def model_constraints(model):
    m = model.m

    @utils.memoize
    def get_parents(level):
        return list(model._locations[model._locations._level == level].index)

    @utils.memoize
    def get_children(parent, childless_only=True):
        """
        If childless_only is True, only children that have no children
        themselves are returned.

        """
        locations = model._locations
        children = list(locations[locations._within == parent].index)
        if childless_only:  # FIXME childless_only param needs tests
            children = [i for i in children if len(get_children(i)) == 0]
        return children

    # Constraint rules
    def c_system_balance_rule(m, c, x, t):
        # Balacing takes place at top-most (level 0) locations, as well
        # as within any lower-level locations that contain children
        if (model._locations.at[x, '_level'] == 0
                or len(get_children(x)) > 0):
            family = get_children(x) + [x]  # list of children + parent
            balance = (sum(m.c_prod[c, y, xs, t]
                           for xs in family for y in m.y)
                       + sum(m.c_con[c, y, xs, t]
                             for xs in family for y in m.y))
            if c == 'power':
                return balance == 0
            else:
                # e.g. for heat. should probably limit the maximum
                # inbalance allowed for these energy types.
                return balance >= 0
        else:
            return po.Constraint.NoConstraint

    # Constraints
    m.c_system_balance = po.Constraint(m.c, m.x, m.t,
                                       rule=c_system_balance_rule)
