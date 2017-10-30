"""
Copyright (C) 2013-2017 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

base_planning.py
~~~~~~~

Basic model constraints.

"""

import warnings

import pyomo.core as po  # pylint: disable=import-error
import numpy as np
import xarray as xr

from .. import exceptions
from .. import transmission
from .. import utils

formatwarning_orig = warnings.formatwarning
def _formatwarning(message, category, filename, lineno, line=None):
    """Formats ModelWarnings as "Warning: message" without extra crud"""
    if category == exceptions.ModelWarning:
        return 'Warning: ' + str(message) + '\n'
    else:
        return formatwarning_orig(message, category, filename, lineno, line)

warnings.formatwarning = _formatwarning

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

    if param_string in model.data and y in model._sets['y_' + param_string + '_timeseries']:
        return getattr(model.m, param_string + '_param')[y, x, t]
    else:
        return model.get_option(y + '.constraints.' + param_string, x=x)


def get_cost_param(model, param_string, k, y, x, t):
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

    if param_string in model.data and y in model._sets['y_' + param_string + '_timeseries']:
        return getattr(model.m, param_string + '_param')[y, x, t, k]
    else:  # Search in model.config_model
        return _cost(param_string, y, k, x=x)

@utils.memoize
def get_y_x(loc_tech):
    x, y = loc_tech.split(":", 1)
    return y, x

def generate_variables(model):
    """
    Defines variables:

    * r: resource -> tech (+ production
    * r_area: resource collector area
    * r2: secondary resource -> storage (+ production)
    * c_prod: tech -> carrier (+ production)
    * c_con: tech <- carrier (- consumption)
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

    """

    m = model.m

    # Capacity
    m.r_area = po.Var(m.loc_tech_area, within=po.NonNegativeReals)
    m.s_cap = po.Var(m.loc_tech_store, within=po.NonNegativeReals)
    m.r_cap = po.Var(m.loc_tech_supply_plus_finite_r, within=po.NonNegativeReals)  # FIXME maybe should be y_finite_r?
    m.e_cap = po.Var(m.loc_tech, within=po.NonNegativeReals)
    m.r2_cap = po.Var(m.loc_tech_r2, within=po.NonNegativeReals)

    # Unit commitment
    m.r = po.Var(m.loc_tech_supply_plus_finite_r, m.t, within=po.Reals)
    m.r2 = po.Var(m.loc_tech_r2, m.t, within=po.NonNegativeReals)
    m.s = po.Var(m.loc_tech_store, m.t, within=po.NonNegativeReals)
    m.c_prod = po.Var(m.c, m.loc_tech, m.t, within=po.NonNegativeReals)
    m.c_con = po.Var(m.c, m.loc_tech, m.t, within=po.NegativeReals)
    m.export = po.Var(m.loc_tech_export, m.t, within=po.NonNegativeReals)

    # Costs
    m.cost_var = po.Var(m.loc_tech, m.t, m.k, within=po.Reals)
    m.cost_fixed = po.Var(m.loc_tech, m.k, within=po.Reals)
    m.cost = po.Var(m.loc_tech, m.k, within=po.Reals)

    # Binary/Integer variables
    m.purchased = po.Var(m.loc_tech_purchase, within=po.Binary)
    m.units = po.Var(m.loc_tech_milp, within=po.NonNegativeIntegers)
    m.operating_units = po.Var(m.loc_tech_milp, m.t, within=po.NonNegativeIntegers)


def node_resource(model):
    m = model.m

    # TODO reformulate c_r_rule conditionally once Pyomo supports that.
    # had to remove the following formulation because it is not
    # re-evaluated on model re-construction -- we now check for
    # demand/supply tech instead, which means that `r` can only
    # be ALL negative or ALL positive for a given tech!
    # Ideally we have `elif po.value(m.r[y, x, t]) > 0:` instead of
    # `elif y in m.y_supply or y in m.y_unmet:` and `elif y in m.y_demand:`

    def r_available_rule(m, loc_tech, t):
        y, x = get_y_x(loc_tech)
        r_scale = model.get_option(y + '.constraints.r_scale', x=x)
        force_r = get_constraint_param(model, 'force_r', y, x, t)
        resource = get_constraint_param(model, 'r', y, x, t)
        if resource == np.inf and force_r is True:
            w = exceptions.ModelWarning
            message = ('`force_r: True` and `r: inf` are incompatible for {}:{}'
                ' at {}. r_available constraint has been ignored. To avoid this'
                ' message in future, set timeseries `r` at technology definition,'
                ' even if you use location override.'.format(y, x, t))
            warnings.warn(message, w)
            return po.Constraint.Skip

        if loc_tech in m.loc_tech_supply or loc_tech in m.loc_tech_demand:
            e_eff = get_constraint_param(model, 'e_eff', y, x, t)
            if po.value(e_eff) == 0:
                c_prod = 0
            else:
                c_prod = sum(m.c_prod[c, loc_tech, t] for c in m.c) / e_eff # supply techs
            c_con = sum(m.c_con[c, loc_tech, t] for c in m.c) * e_eff # demand techs

            if loc_tech in m.loc_tech_area:
                r_avail = resource * r_scale * m.r_area[loc_tech]
            else:
                r_avail = get_constraint_param(model, 'r', y, x, t) * r_scale

            if force_r:
                return c_prod + c_con == r_avail
            else:
                return c_prod - c_con <= r_avail

        elif loc_tech in m.loc_tech_supply_plus:
            r_eff = get_constraint_param(model, 'r_eff', y, x, t)

            if loc_tech in m.loc_tech_area:
                r_avail = resource * r_scale * m.r_area[loc_tech] * r_eff
            else:
                r_avail = resource * r_scale * r_eff

            if force_r:
                return m.r[loc_tech, t] == r_avail
            else:
                return m.r[loc_tech, t] <= r_avail

    # Constraints
    m.c_r_available = po.Constraint(m.loc_tech_finite_r, m.t,
                                    rule=r_available_rule)

def unit_commitment(model):
    m = model.m

    def c_unit_commitment_rule(m, loc_tech, t):
        # operating_units
        # ^^^^^^^^^^^^^^^
        #
        # Constraining the number of integer units
        # :math:`operating_units(y, x, t)` of a technology which
        # can operate in a given timestep, based on maximum purchased units
        # :math:`units(y, x)`
        #
        # .. math::
        #
        #   $operating\_units(y, x, t) \leq units(y, x)
        ##

        return m.operating_units[loc_tech, t] <= m.units[loc_tech]

    m.c_unit_commitment = po.Constraint(m.loc_tech_milp, m.t,
                                        rule=c_unit_commitment_rule)

def node_energy_balance(model):
    m = model.m
    time_res = model.data['_time_res'].to_series()
    e_eff_per_distance_getter = utils.e_eff_per_distance_getter(model.config_model)
    @utils.memoize
    def _e_eff_per_distance(y, x):
        return e_eff_per_distance_getter(y, x)

    def get_conversion_out(c_1, c_2, m, loc_tech, t):
        if isinstance(c_1, dict):
            c_prod1 = sum([m.c_prod[c, loc_tech, t] / c_1[c] for c in c_1.keys()])
        else:
            c_prod1 = m.c_prod[c_1, loc_tech, t]
        if isinstance(c_2, dict):
            c_min = min(c_2.values())
            c_prod2 = sum([m.c_prod[c, loc_tech, t] / (c_2[c] / c_min) for c in c_2.keys()])
        else:
            c_min = 1
            c_prod2 = m.c_prod[c_2, loc_tech, t]
        return c_prod1 * c_min == c_prod2

    def get_conversion_in(c_1, c_2, m, loc_tech, t):
        if isinstance(c_1, dict):
            c_con1 = sum([m.c_con[c, loc_tech, t] / c_1[c] for c in c_1.keys()])
        else:
            c_con1 = m.c_con[c_1, loc_tech, t]
        if isinstance(c_2, dict):
            c_min = min(c_2.values())
            c_con2 = sum([m.c_con[c, loc_tech, t] / (c_2[c] / c_min) for c in c_2.keys()])
        else:
            c_min = 1
            c_con2 = m.c_con[c_2, loc_tech, t]
        return c_con1 * c_min == c_con2

    # Constraint rules
    def transmission_rule(m, loc_tech, t):
        y, x = get_y_x(loc_tech)
        y_remote, x_remote = transmission.get_remotes(y, x)
        loc_tech_remote = ":".join([x_remote, y_remote])
        e_eff = get_constraint_param(model, 'e_eff', y, x, t)
        if loc_tech_remote in m.loc_tech_transmission:
            c = model.get_option(y + '.carrier')
            return (m.c_prod[c, loc_tech, t]
                    == -1 * m.c_con[c, loc_tech_remote, t]
                    * e_eff
                    * _e_eff_per_distance(y, x))
        else:
            return po.Constraint.NoConstraint

    def conversion_rule(m, loc_tech, t):
        y, x = get_y_x(loc_tech)
        c_out = model.get_option(y + '.carrier_out')
        c_in = model.get_option(y + '.carrier_in')
        e_eff = get_constraint_param(model, 'e_eff', y, x, t)
        return (m.c_prod[c_out, loc_tech, t]
                == -1 * m.c_con[c_in, loc_tech, t] * e_eff)

    def conversion_plus_primary_rule(m, loc_tech, t):
        y, x = get_y_x(loc_tech)
        c_out = model.get_option(y + '.carrier_out')
        c_in = model.get_option(y + '.carrier_in')
        e_eff = get_constraint_param(model, 'e_eff', y, x, t)
        if isinstance(c_out, dict):
            c_prod = sum(m.c_prod[c, loc_tech, t] / c_out[c] for c in c_out.keys())
        else:
            c_prod = m.c_prod[c_out, loc_tech, t]
        if isinstance(c_in, dict):
            c_con = sum([m.c_con[c, loc_tech, t] * c_in[c] for c in c_in.keys()])
        else:
            c_con = m.c_con[c_in, loc_tech, t]
        return c_prod == -1 * c_con * e_eff

    def conversion_plus_secondary_out_rule(m, loc_tech, t):
        y, x = get_y_x(loc_tech)
        c_1 = model.get_option(y + '.carrier_out')
        c_2 = model.get_option(y + '.carrier_out_2')
        return get_conversion_out(c_1, c_2, m, loc_tech, t)

    def conversion_plus_tertiary_out_rule(m, loc_tech, t):
        y, x = get_y_x(loc_tech)
        c_1 = model.get_option(y + '.carrier_out')
        c_3 = model.get_option(y + '.carrier_out_3')
        return get_conversion_out(c_1, c_3, m, loc_tech, t)

    def conversion_plus_secondary_in_rule(m, loc_tech, t):
        y, x = get_y_x(loc_tech)
        c_1 = model.get_option(y + '.carrier_in')
        c_2 = model.get_option(y + '.carrier_in_2')
        return get_conversion_in(c_1, c_2, m, loc_tech, t)

    def conversion_plus_tertiary_in_rule(m, loc_tech, t):
        y, x = get_y_x(loc_tech)
        c_1 = model.get_option(y + '.carrier_in')
        c_3 = model.get_option(y + '.carrier_in_3')
        return get_conversion_in(c_1, c_3, m, loc_tech, t)

    def supply_plus_rule(m, loc_tech, t):
        y, x = get_y_x(loc_tech)
        e_eff = get_constraint_param(model, 'e_eff', y, x, t)
        p_eff = model.get_option(y + '.constraints.p_eff', x=x)
        total_eff = e_eff * p_eff
        # TODO once Pyomo supports it,
        # let this update conditionally on param update!
        if po.value(total_eff) == 0:
            c_prod = 0
        else:
            c_prod = (sum(m.c_prod[c, loc_tech, t] for c in m.c)) / total_eff

        # If this tech is in the set of techs allowing r2, include it
        if loc_tech in m.loc_tech_r2:
            r2 = m.r2[loc_tech, t]
        else:
            r2 = 0

        # A) Case where no storage allowed
        if loc_tech not in m.loc_tech_store:
            return m.r[loc_tech, t] == c_prod - r2

        # B) Case where storage is allowed
        r = m.r[loc_tech, t]
        if m.t.order_dict[t] == 0:
            s_minus_one = m.s_init[y, x]
        else:
            s_loss = get_constraint_param(model, 's_loss', y, x, t)
            s_minus_one = (((1 - s_loss)
                            ** time_res.at[model.prev_t(t)])
                           * m.s[loc_tech, model.prev_t(t)])
        return (m.s[loc_tech, t] == s_minus_one + r + r2 - c_prod)

    def storage_rule(m, loc_tech, t):
        y, x = get_y_x(loc_tech)
        e_eff = get_constraint_param(model, 'e_eff', y, x, t)
        p_eff = model.get_option(y + '.constraints.p_eff', x=x)
        total_eff = e_eff * p_eff

        # TODO once Pyomo supports it,
        # let this update conditionally on param update!
        if po.value(total_eff) == 0:
            c_prod = 0
        else:
            c_prod = sum(m.c_prod[c, loc_tech, t] for c in m.c) / total_eff
        c_con = sum(m.c_con[c, loc_tech, t] for c in m.c) * total_eff

        if m.t.order_dict[t] == 0:
            s_minus_one = m.s_init[y, x]
        else:
            s_loss = get_constraint_param(model, 's_loss', y, x, t)
            s_minus_one = (((1 - s_loss)
                            ** time_res.at[model.prev_t(t)])
                           * m.s[loc_tech, model.prev_t(t)])
        return (m.s[loc_tech, t] == s_minus_one - c_prod - c_con)

    # Constraints
    m.c_balance_transmission = po.Constraint(m.loc_tech_transmission, m.t,
                                            rule=transmission_rule)
    m.c_balance_conversion = po.Constraint(m.loc_tech_conversion, m.t,
                                            rule=conversion_rule)
    m.c_balance_conversion_plus = po.Constraint(m.loc_tech_conversion_plus, m.t,
                                            rule=conversion_plus_primary_rule)
    m.c_balance_conversion_plus_secondary_out = po.Constraint(m.loc_tech_2out, m.t,
                                            rule=conversion_plus_secondary_out_rule)
    m.c_balance_conversion_plus_tertiary_out = po.Constraint(m.loc_tech_3out, m.t,
                                            rule=conversion_plus_tertiary_out_rule)
    m.c_balance_conversion_plus_secondary_in = po.Constraint(m.loc_tech_2in, m.t,
                                            rule=conversion_plus_secondary_in_rule)
    m.c_balance_conversion_plus_tertiary_in = po.Constraint(m.loc_tech_3in, m.t,
                                            rule=conversion_plus_tertiary_in_rule)
    m.c_balance_supply_plus = po.Constraint(m.loc_tech_supply_plus, m.t,
                                            rule=supply_plus_rule)
    m.c_balance_storage = po.Constraint(m.loc_tech_storage, m.t,
                                            rule=storage_rule)


def node_constraints_build(model):

    m = model.m

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
        if _equals is not False and _equals is not None:
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

    def get_s_cap(y, x, loc_tech, e_cap, c_rate):
        """
        Get s_cap.max from s_time.max, where applicable. If s_time.max is used,
        the maximum storage possible is the minimum value of storage possible for
        any time length which meets the s_time.max value.
        """
        # TODO:
        # incorporate timeseries resolution. Currently assumes that each
        # timestep is worth one unit of time.
        s_time_max = model.get_option(y + '.constraints.s_time.max', x=x)
        if loc_tech in m.loc_tech_milp:
            units = model.get_option(y + '.constraints.units.max', x=x)
            s_cap_max = (units *
                model.get_option(y + '.constraints.s_cap_per_unit', x=x))
        else:
            s_cap_max = model.get_option(y + '.constraints.s_cap.max', x=x)
        if not s_cap_max:
            s_cap_max = np.inf
        if not s_time_max:
            return s_cap_max
        if not e_cap and not c_rate:
            return 0
        elif not e_cap:
            e_cap = s_cap_max * c_rate
        elif e_cap and c_rate:
            e_cap = min(e_cap, s_cap_max * c_rate)

        s_loss = model.data.loc[dict(y=y, x=x)].get('s_loss',
                                model.get_option(y+ '.constraints.s_loss', x=x))
        e_eff = model.data.loc[dict(y=y, x=x)].get('e_eff',
                                model.get_option(y+ '.constraints.e_eff', x=x))
        try:
            leakage = 1 / (1 - s_loss)  # if s_loss is timeseries dependant, will be a DataArray
            discharge = e_cap / e_eff  # if e_eff is timeseries dependant, will be a DataArray
        except ZeroDivisionError:  # if s_loss = 1 or e_eff = 0
            return np.inf  # i.e. no upper limit on storage
        exponents = [i for i in range(s_time_max)]
        if isinstance(leakage, xr.DataArray) or isinstance(discharge, xr.DataArray):
            roll = model.data.t.rolling(t=s_time_max)  # create arrays of all rolling horizons
            S = {s_cap_max}
            for label, arr_window in roll:
                if len(arr_window) == s_time_max:  # only consider arrays of the maximum time length
                    S.add(sum(discharge * np.power(leakage, exponents)))
            return min(S)  # smallest value of storage is the maximum allowed
        else:  # no need to loop through rolling horizons if all values are static in time
            return min(s_cap_max, sum(discharge * np.power(leakage, exponents)))

    # Constraint rules
    def c_s_cap_rule(m, loc_tech):
        """
        Set maximum storage capacity. Supply_plus & storage techs only
        This can be set by either s_cap (kWh) or by
        e_cap (charge/discharge capacity) * charge rate.
        If s_cap.equals and e_cap.equals are set for the technology, then
        s_cap * charge rate = e_cap must hold. Otherwise, take the lowest capacity
        capacity defined by s_cap.max or e_cap.max / charge rate.
        """
        y, x = get_y_x(loc_tech)
        if loc_tech in m.loc_tech_milp:
            units_equals = model.get_option(y + '.constraints.units.equals', x=x)
            s_cap_equals = (units_equals *
                model.get_option(y + '.constraints.s_cap_per_unit', x=x))
            e_cap = (units_equals *
                model.get_option(y + '.constraints.e_cap_per_unit', x=x))
        else:
            s_cap_equals = model.get_option(y + '.constraints.s_cap.equals', x=x)
            scale = model.get_option(y + '.constraints.e_cap_scale', x=x)
            e_cap = model.get_option(y + '.constraints.e_cap.equals', x=x) * scale
        charge_rate = model.get_option(y + '.constraints.c_rate', x=x)
        if e_cap and s_cap_equals and charge_rate and s_cap_equals * charge_rate != e_cap:
            raise exceptions.ModelError(
                'e_cap.equals and s_cap.equals must '
                'be equivalent when considering charge rate for {}:{}'
                .format(y, x)
            )
        if not e_cap:
            if loc_tech in m.loc_tech_milp:
                e_cap = (model.get_option(y + '.constraints.units.max', x=x)
                    * model.get_option(y + '.constraints.e_cap_per_unit', x=x))
            else:
                e_cap = model.get_option(y + '.constraints.e_cap.max', x=x) * scale
        if not s_cap_equals:
            s_cap_max = get_s_cap(y, x, loc_tech, e_cap, charge_rate)
            return get_var_constraint(m.s_cap[loc_tech], y, 's_cap', x, _max=s_cap_max)
        else:
            return get_var_constraint(m.s_cap[loc_tech], y, 's_cap', x, _equals=s_cap_equals)

    def c_r_cap_rule(m, loc_tech):
        y, x = get_y_x(loc_tech)
        return get_var_constraint(m.r_cap[loc_tech], y, 'r_cap', x)

    def c_r_cap_equals_e_cap_rule(m, loc_tech):
        y, x = get_y_x(loc_tech)
        if model.get_option(y + '.constraints.r_cap_equals_e_cap', x=x):
            return m.r_cap[loc_tech] == m.e_cap[loc_tech]
        else:
            return po.Constraint.Skip

    def c_r_area_rule(m, loc_tech):
        """
        Set maximum r_area. Supply_plus techs only.
        """
        y, x = get_y_x(loc_tech)
        e_cap_max = model.get_option(y + '.constraints.e_cap.max', x=x)
        if e_cap_max == 0:
            # If a technology has no e_cap here, we force r_area to zero,
            # so as not to accrue spurious costs
            return m.r_area[loc_tech] == 0
        else:
            return get_var_constraint(m.r_area[loc_tech], y, 'r_area', x)

    def c_r_area_per_e_cap_rule(m, loc_tech):
        y, x = get_y_x(loc_tech)
        area_per_cap = model.get_option(y + '.constraints.r_area_per_e_cap', x=x)
        if area_per_cap:
            return m.r_area[loc_tech] == m.e_cap[loc_tech] * area_per_cap
        else:
            return po.Constraint.Skip

    def c_e_cap_rule(m, loc_tech):
        """
        Set maximum e_cap. All technologies.
        """
        y, x = get_y_x(loc_tech)
        # First check whether this tech is allowed at this location
        if not model._locations.at[x, y] == 1:
            return m.e_cap[loc_tech] == 0

        # Addition of binary variable describing whether a technology has been
        # purchased or not
        if loc_tech in m.loc_tech_purchase:
            purchased = m.purchased[loc_tech]
        else:
            purchased = 1

        # Addition of integer variable describing how many units of a technology
        # have been purchased
        if loc_tech in m.loc_tech_milp:
            return m.e_cap[loc_tech] == (m.units[loc_tech]
                    * model.get_option(y + '.constraints.e_cap_per_unit', x=x))

        # parameters for transmission technologies are stored in a different
        # location to other technologies
        e_cap_max = model.get_option(y + '.constraints.e_cap.max', x=x)
        e_cap_equals = model.get_option(y + '.constraints.e_cap.equals', x=x)
        if loc_tech in m.loc_tech_transmission:
            e_cap_equals_override = model._locations.get(
                '_override.{}.constraints.e_cap.equals'.format(y), {x: None})[x]
            e_cap_max_override = model._locations.get(
                '_override.{}.constraints.e_cap.max'.format(y), {x: None})[x]
            if e_cap_equals_override is not None and not np.isnan(e_cap_equals_override):
                e_cap_equals = e_cap_equals_override
            if  e_cap_max_override is not None and not np.isnan(e_cap_max_override):
                e_cap_max = e_cap_max_override

        e_cap_scale = model.get_option(y + '.constraints.e_cap_scale', x=x)

        # e_cap.equals forces an equality constraint, which can't be infinite
        if e_cap_equals:
            if e_cap_equals is None or np.isinf(e_cap_equals) or np.isnan(e_cap_equals):
                e = exceptions.ModelError
                raise e('Cannot use inf, NaN, or None for equality constraint: '
                        '{}.e_cap.equals.{}'.format(y, x))
            else:
                return m.e_cap[loc_tech] == e_cap_equals * e_cap_scale * purchased

        # In operation mode, e_cap is forced to an equality constraint, even if
        # e_cap.max is defined.
        if (model.mode == 'operate' and loc_tech not in m.loc_tech_demand
            and loc_tech not in m.loc_tech_unmet):
            if e_cap_max is None or np.isinf(e_cap_max) or np.isnan(e_cap_max):
                e = exceptions.ModelError
                raise e('Cannot use inf, NaN, or None in operational mode, '
                        'for value of {}.e_cap.max.{}'.format(y, x))
            return m.e_cap[loc_tech] == e_cap_max * e_cap_scale * purchased

        # Infinite or undefined e_cap.max leads to an ignored constraint
        elif e_cap_max is None or np.isnan(e_cap_max) or np.isinf(e_cap_max):
            return po.Constraint.Skip
        else:
            return m.e_cap[loc_tech] <= e_cap_max * e_cap_scale * purchased

    def c_e_cap_min_rule(m, loc_tech):
        """
        Set minimum e_cap. All technologies.
        """
        y, x = get_y_x(loc_tech)
        # First check whether this tech is allowed at this location
        if not model._locations.at[x, y] == 1:
            return po.Constraint.Skip

        # Addition of binary variable describing whether a technology has been
        # purchased or not
        if loc_tech in m.loc_tech_purchase:
            purchased = m.purchased[loc_tech]
        else:
            purchased = 1

        # Addition of integer variable describing how many units of a technology
        # have been purchased
        if loc_tech in m.loc_tech_milp:
            return po.Constraint.Skip

        # parameters for transmission technologies are stored in a different
        # location to other technologies
        if loc_tech in m.loc_tech_transmission:
            e_cap_equals = model._locations.get(
                '_override.{}.constraints.e_cap.equals'.format(y), {x: None})[x]
            e_cap_min = model._locations.get(
                '_override.{}.constraints.e_cap.min'.format(y), {x: None})[x]
        else:
            e_cap_min = model.get_option(y + '.constraints.e_cap.min', x=x)
            e_cap_equals = model.get_option(y + '.constraints.e_cap.equals', x=x)
        if e_cap_equals or model.mode == 'operate' or not e_cap_min:
            return po.Constraint.Skip
        else:
            e_cap_scale = model.get_option(y + '.constraints.e_cap_scale', x=x)
            return m.e_cap[loc_tech] >= e_cap_min * e_cap_scale * purchased


    def c_e_cap_storage_rule(m, loc_tech):
        """
        Set maximum e_cap. All storage technologies.
        """
        y, x = get_y_x(loc_tech)
        e_cap_scale = model.get_option(y + '.constraints.e_cap_scale', x=x)
        charge_rate = model.get_option(y + '.constraints.c_rate', x=x)
        if charge_rate:
            e_cap_max = m.s_cap[loc_tech] * charge_rate * e_cap_scale
            return m.e_cap[loc_tech] <= e_cap_max
        else:
            return po.Constraint.Skip

    def c_r2_cap_rule(m, loc_tech):
        """
        Set secondary resource capacity. Supply_plus techs only.
        """
        y, x = get_y_x(loc_tech)
        follow = model.get_option(y + '.constraints.r2_cap_follow', x=x)
        mode = model.get_option(y + '.constraints.r2_cap_follow_mode', x=x)

        # First deal with the special case of ``r2_cap_follow`` being set
        if follow:
            try:
                r2_cap_val = getattr(m, follow)[y, x]
            except AttributeError:
                # Raise an error to make sure follows isn't accidentally set to
                # something invalid
                e = exceptions.ModelError
                raise e('r2_cap_follow set to invalid value at '
                        '({}, {}): {}'.format(y, x, follow))

            if mode == 'max':
                return m.r2_cap[loc_tech] <= r2_cap_val
            elif mode == 'equals':
                return m.r2_cap[loc_tech] == r2_cap_val

        else:  # If ``r2_cap_follow`` not set, set up standard constraints
            return get_var_constraint(m.r2_cap[loc_tech], y, 'r2_cap', x)

    def c_units_rule(m, loc_tech):
        # units
        # ^^^^^^^^
        #
        # Constraining the number of integer purchased units of a technology
        # :math:`units(y, x)` to be between the user-defined maximum
        # :math:`units.max(y, x)` and minimum :math:`units.min(y, x)`, or
        # equal to :math:`units.equals(y, x)`.
        #
        # .. math::
        #
        #   $if units.equals(y, x) then units(y, x) = units.equals(y, x) else
        #   $units.min(y, x) \leq units(y, x) \leq units.min(y, x)
        ##
        y, x = get_y_x(loc_tech)

        return get_var_constraint(m.units[loc_tech], y, 'units', x)

    # Constraints
    m.c_s_cap = po.Constraint(m.loc_tech_store, rule=c_s_cap_rule)
    m.c_r_cap = po.Constraint(m.loc_tech_supply_plus_finite_r, rule=c_r_cap_rule)
    m.c_r_cap_equals_e_cap = po.Constraint(m.loc_tech_supply_plus_finite_r,
        rule=c_r_cap_equals_e_cap_rule)
    m.c_r_area = po.Constraint(m.loc_tech_area, rule=c_r_area_rule)
    m.c_r_area_per_e_cap = po.Constraint(m.loc_tech_area, rule=c_r_area_per_e_cap_rule)
    m.c_e_cap = po.Constraint(m.loc_tech, rule=c_e_cap_rule)
    m.c_e_cap_min = po.Constraint(m.loc_tech, rule=c_e_cap_min_rule)
    m.c_e_cap_storage = po.Constraint(m.loc_tech_store, rule=c_e_cap_storage_rule)
    m.c_r2_cap = po.Constraint(m.loc_tech_r2, rule=c_r2_cap_rule)
    m.c_units = po.Constraint(m.loc_tech_milp, rule=c_units_rule)

def node_constraints_operational(model):
    m = model.m
    time_res = model.data['_time_res'].to_series()

    # Constraint rules
    def r_max_upper_rule(m, loc_tech, t):
        """
        set maximum resource supply. Supply_plus techs only.
        """
        return m.r[loc_tech, t] <= time_res.at[t] * m.r_cap[loc_tech]

    def c_prod_max_rule(m, c, loc_tech, t):
        """
        Set maximum carrier production. All technologies.
        """
        y, x = get_y_x(loc_tech)
        allow_c_prod = get_constraint_param(model, 'e_prod', y, x, t)
        c_prod = m.c_prod[c, loc_tech, t]
        if loc_tech in m.loc_tech_conversion_plus:  # Conversion techs with 2 or more output carriers
            carriers_out = model.get_carrier(y, 'out', all_carriers=True)
            if isinstance(carriers_out, str):
                carriers_out = tuple([carriers_out])
            if (c not in carriers_out) or (c in carriers_out and
                                           model._locations.at[x, y] == 0):
                return c_prod == 0
            else:
                return po.Constraint.Skip
        if not allow_c_prod:
            return c_prod == 0
        p_eff = model.get_option(y + '.constraints.p_eff', x=x)
        if c == model.get_option(y + '.carrier', default=y + '.carrier_out'):
            if loc_tech in m.loc_tech_milp:
                e_cap = model.get_option(y + '.constraints.e_cap_per_unit', x=x)
                return c_prod <= (time_res.at[t] * m.operating_units[loc_tech, t]
                                  * e_cap * p_eff)
            return c_prod <= time_res.at[t] * m.e_cap[loc_tech] * p_eff
        else:
            return m.c_prod[c, loc_tech, t] == 0

    def c_prod_min_rule(m, c, loc_tech, t):
        """
        Set minimum carrier production. All technologies except conversion_plus
        """
        y, x = get_y_x(loc_tech)
        min_use = get_constraint_param(model, 'e_cap_min_use', y, x, t)
        allow_c_prod = get_constraint_param(model, 'e_prod', y, x, t)
        if not min_use or not allow_c_prod:
            return po.Constraint.NoConstraint
        if loc_tech in m.loc_tech_conversion_plus:  # Conversion techs with 2 output carriers
            return po.Constraint.Skip
        elif c == model.get_option(y + '.carrier', default=y + '.carrier_out'):
            if loc_tech in m.loc_tech_milp:
                e_cap = model.get_option(y + '.constraints.e_cap_per_unit', x=x)
                return (m.c_prod[c, loc_tech, t] >= time_res.at[t] *
                    m.operating_units[loc_tech, t] * e_cap * min_use)
            return (m.c_prod[c, loc_tech, t]
                    >= time_res.at[t] * m.e_cap[loc_tech] * min_use)
        else:
            return po.Constraint.Skip

    def c_conversion_plus_prod_max_rule(m, loc_tech, t):
        """
        Set maximum carrier production. Conversion_plus technologies.
        """
        y, x = get_y_x(loc_tech)
        allow_c_prod = get_constraint_param(model, 'e_prod', y, x, t)
        c_out = model.get_option(y + '.carrier_out')
        e_eff = get_constraint_param(model, 'e_eff', y, x, t)
        if isinstance(c_out, dict):
            c_prod = sum(m.c_prod[c, loc_tech, t] for c in c_out.keys())
        else:
            c_prod = m.c_prod[c_out, loc_tech, t]
        if not allow_c_prod:
            return c_prod == 0
        else:
            if loc_tech in m.loc_tech_milp:
                e_cap = model.get_option(y + '.constraints.e_cap_per_unit', x=x)
                return c_prod <= (time_res.at[t] * m.operating_units[loc_tech, t]
                                  * e_cap)
            return c_prod <= time_res.at[t] * m.e_cap[loc_tech]

    def c_conversion_plus_prod_min_rule(m, loc_tech, t):
        """
        Set minimum carrier production. Conversion_plus technologies.
        """
        y, x = get_y_x(loc_tech)
        min_use = get_constraint_param(model, 'e_cap_min_use', y, x, t)
        allow_c_prod = get_constraint_param(model, 'e_prod', y, x, t)
        c_out = model.get_option(y + '.carrier_out')
        if not min_use or not allow_c_prod:
            return po.Constraint.NoConstraint
        else:
            c_prod_min = time_res.at[t] * m.e_cap[loc_tech] * min_use
            if isinstance(c_out, dict):
                c_prod = sum(m.c_prod[c, loc_tech, t] for c in c_out.keys())
            else:
                c_prod = m.c_prod[c_out, loc_tech, t]
            if loc_tech in m.loc_tech_milp:
                e_cap = model.get_option(y + '.constraints.e_cap_per_unit', x=x)
                return c_prod >= (time_res.at[t] * m.operating_units[loc_tech, t]
                                  * e_cap * min_use)
            return c_prod >= c_prod_min

    def c_con_max_rule(m, c, loc_tech, t):
        """
        Set maximum carrier consumption. All technologies.
        """
        y, x = get_y_x(loc_tech)
        allow_c_con = get_constraint_param(model, 'e_con', y, x, t)
        p_eff = model.get_option(y + '.constraints.p_eff', x=x)
        if loc_tech in m.loc_tech_conversion or loc_tech in m.loc_tech_conversion_plus:
            carriers = model.get_cp_carriers(y, x, direction='in')[1]
            if c not in carriers:
                return m.c_con[c, loc_tech, t] == 0
            else:
                return po.Constraint.Skip
        if (allow_c_con is True and
                c == model.get_option(y + '.carrier', default=y + '.carrier_in')):
            if loc_tech in m.loc_tech_milp:
                e_cap = model.get_option(y + '.constraints.e_cap_per_unit', x=x)
                return m.c_con[c, loc_tech, t] >= (time_res.at[t] * e_cap * p_eff
                                               * m.operating_units[loc_tech, t] * -1)
            return m.c_con[c, loc_tech, t] >= (-1 * time_res.at[t]
                                            * m.e_cap[loc_tech] * p_eff)
        else:
            return m.c_con[c, loc_tech, t] == 0

    def s_max_rule(m, loc_tech, t):
        """
        Set maximum stored energy. Supply_plus & storage techs only.
        """
        return m.s[loc_tech, t] <= m.s_cap[loc_tech]

    def r2_max_rule(m, loc_tech, t):
        """
        Set maximum secondary resource supply. Supply_plus techs only.
        """
        y, x = get_y_x(loc_tech)
        r2_startup = get_constraint_param(model, 'r2_startup_only', y, x, t)
        if (r2_startup and t >= model.data.startup_time_bounds):
            return m.r2[loc_tech, t] == 0
        else:
            return m.r2[loc_tech, t] <= time_res.at[t] * m.r2_cap[loc_tech]

    def c_export_max_rule(m, loc_tech, t):
        """
        Set maximum export. All exporting technologies.
        """
        y, x = get_y_x(loc_tech)
        if loc_tech in m.loc_tech_milp:
            operating_units = m.operating_units[loc_tech, t]
        else:
            operating_units = 1
        if get_constraint_param(model, 'export_cap', y, x, t):
            return (m.export[loc_tech, t] <=
                    get_constraint_param(model, 'export_cap', y, x, t)
                    * operating_units)
        else:
            return po.Constraint.Skip

    # Constraints
    m.c_r_max_upper = po.Constraint(
        m.loc_tech_supply_plus_finite_r, m.t, rule=r_max_upper_rule)
    m.c_prod_max = po.Constraint(
        m.c, m.loc_tech, m.t, rule=c_prod_max_rule)
    m.c_prod_min = po.Constraint(
        m.c, m.loc_tech, m.t, rule=c_prod_min_rule)
    m.c_conversion_plus_prod_max = po.Constraint(
        m.loc_tech_conversion_plus, m.t, rule=c_conversion_plus_prod_max_rule)
    m.c_conversion_plus_prod_min = po.Constraint(
        m.loc_tech_conversion_plus, m.t, rule=c_conversion_plus_prod_min_rule)
    m.c_con_max = po.Constraint(
        m.c, m.loc_tech, m.t, rule=c_con_max_rule)
    m.c_s_max = po.Constraint(
        m.loc_tech_store, m.t, rule=s_max_rule)
    m.c_r2_max = po.Constraint(
        m.loc_tech_r2, m.t, rule=r2_max_rule)
    m.c_export_max = po.Constraint(
        m.loc_tech_export, m.t, rule=c_export_max_rule)


def node_constraints_transmission(model):
    """
    Constrain e_cap symmetrically for transmission nodes. Transmission techs only.
    """
    m = model.m

    # Constraint rules
    def c_trans_rule(m, loc_tech):
        y, x = get_y_x(loc_tech)
        y_remote, x_remote = transmission.get_remotes(y, x)
        loc_tech_remote = ":".join([x_remote, y_remote])
        if loc_tech_remote in m.loc_tech_transmission:
            return m.e_cap[loc_tech] == m.e_cap[loc_tech_remote]
        else:
            return po.Constraint.NoConstraint

    # Constraints
    m.c_transmission_capacity = po.Constraint(m.loc_tech_transmission,
                                              rule=c_trans_rule)

def node_costs(model):

    m = model.m
    time_res = model.data['_time_res'].to_series()
    weights = model.data['_weights'].to_series()

    cost_getter = utils.cost_getter(model.get_option)
    depreciation_getter = utils.depreciation_getter(model.get_option)
    cost_per_distance_getter = utils.cost_per_distance_getter(model.config_model)

    @utils.memoize
    def _depreciation_rate(y, x, k):
        return depreciation_getter(y, x, k)

    @utils.memoize
    def _cost(cost, y, k, x=None):
        return cost_getter(cost, y, k, x=x)

    @utils.memoize
    def _cost_per_distance(cost, y, k, x):
        return cost_per_distance_getter(cost, y, k, x)

    def _check_and_set(cost, loc_tech, y, x, k):
        """
        Ensure that sufficient constraints have been set to allow negative
        costs, where applicable.
        Returns cost if bounds are set, raises error if unset
        """
        e = exceptions.OptionNotSetError
        if loc_tech in m.loc_tech_transmission:
            # Divided by 2 for transmission techs because construction costs
            # are counted at both ends
            cost_parameter = (_cost(cost, y, k, x) +
                                _cost_per_distance(cost, y, k, x)) / 2
        else:
            cost_parameter = _cost(cost, y, k, x)

        # Search the Pyomo constraint for this technology and location, see if it's set

        if loc_tech in getattr(m, 'c_' + cost).keys() or cost_parameter >= 0:
            return cost_parameter * getattr(m, cost)[loc_tech]
        elif cost_parameter < 0:
            raise e(cost + '.max must be defined for {}:{} '
                    'as cost is negative'.format(y, x))

    def cost_fixed_rule(m, loc_tech, k):
        y, x = get_y_x(loc_tech)

        if loc_tech in m.loc_tech_store:
            cost_s_cap = _check_and_set('s_cap', loc_tech, y, x, k)
        else:
            cost_s_cap = 0

        if loc_tech in m.loc_tech_supply_plus_finite_r:
            cost_r_cap = _check_and_set('r_cap', loc_tech, y, x, k)
        else:
            cost_r_cap = 0

        if loc_tech in m.loc_tech_area:
            cost_r_area = _check_and_set('r_area', loc_tech, y, x, k)
        else:
            cost_r_area = 0

        cost_e_cap = _check_and_set('e_cap', loc_tech, y, x, k)

        if loc_tech in m.loc_tech_r2:
            cost_r2_cap = _check_and_set('r2_cap', loc_tech, y, x, k)
        else:
            cost_r2_cap = 0

        if loc_tech in m.loc_tech_purchase:
            if loc_tech in m.loc_tech_transmission:
                cost_purchase = m.purchased[loc_tech] * (_cost('purchase', y, k, x)
                                + _cost_per_distance('purchase', y, k, x)) / 2
            else:
                cost_purchase = _cost('purchase', y, k, x) * m.purchased[loc_tech]
        elif loc_tech in m.loc_tech_milp:
            if loc_tech in m.loc_tech_transmission:
                cost_purchase = m.units[loc_tech] * (_cost('purchase', y, k, x)
                                + _cost_per_distance('purchase', y, k, x)) / 2
            else:
                cost_purchase = _cost('purchase', y, k, x) * m.units[loc_tech]
        else:
            cost_purchase = 0

        cost_con = (_depreciation_rate(y, x, k) *
            (sum(time_res * weights) / 8760) *
            (cost_s_cap + cost_r_cap + cost_r_area + cost_r2_cap +
             cost_e_cap + cost_purchase)
        )
        if _cost('om_fixed', y, k, x) < 0 and loc_tech not in m.c_e_cap.keys():
            raise exceptions.OptionNotSetError(
                'e_cap.max must be defined '
                'for {}:{} as `om_fixed` cost is negative'.format(y, x))

        return (m.cost_fixed[loc_tech, k] ==
                    _cost('om_frac', y, k, x) * cost_con
                    + (_cost('om_fixed', y, k, x) * m.e_cap[loc_tech] *
                       (sum(time_res * weights) / 8760)) + cost_con)

    def cost_var_rule(m, loc_tech, t, k):
        y, x = get_y_x(loc_tech)
        om_var = get_cost_param(model, 'om_var', k, y, x, t)
        om_fuel = get_cost_param(model, 'om_fuel', k, y, x, t)
        if loc_tech in m.loc_tech_export:
            export = m.export[loc_tech, t]
            cost_export = get_cost_param(model, 'export', k, y, x, t) * export
        else:
            export = 0
            cost_export = 0
        carrier = model.get_option(y + '.carrier', default=y + '.carrier_out')
        if isinstance(carrier, dict):  # Conversion_plus with multiple primary output carriers
            carrier = model.get_option(y + '.primary_carrier')
            if not carrier:
                raise exceptions.OptionNotSetError(
                    'No specific carrier set for attributing variable costs '
                    'to conversion+ tech {} at {}'.format(y, x))
        # Note: only counting c_prod for operational costs.
        # This should generally be a reasonable assumption to make.
        # It might be necessary to remove parasitic losses for this
        # i.e. c_prod --> es_prod.
        if om_var:
            cost_op_var = om_var * weights.loc[t] * m.c_prod[carrier, loc_tech, t]
        else:
            cost_op_var = 0

        cost_op_fuel = 0
        if loc_tech in m.loc_tech_supply_plus and om_fuel:
            r_eff = get_constraint_param(model, 'r_eff', y, x, t)
            if po.value(r_eff) > 0:  # In case r_eff is zero, to avoid an infinite value
                # Dividing by r_eff here so we get the actual r used, not the r
                # moved into storage...
                cost_op_fuel = (om_fuel * weights.loc[t] * (m.r[loc_tech, t] / r_eff))
        elif loc_tech in m.loc_tech_supply and om_fuel:  # m.r == m.c_prod/e_eff
            e_eff = get_constraint_param(model, 'e_eff', y, x, t)
            if po.value(e_eff) > 0:  # in case e_eff is zero, to avoid an infinite value
                cost_op_fuel = (om_fuel * weights.loc[t] *
                                (m.c_prod[carrier, loc_tech, t] / e_eff))

        cost_op_r2 = 0
        if loc_tech in m.loc_tech_r2:
            r2_eff = get_constraint_param(model, 'r2_eff', y, x, t)
            if po.value(r2_eff) > 0:  # in case r2_eff is zero, to avoid an infinite value
                om_r2 = get_cost_param(model, 'om_r2', k, y, x, t)
                cost_op_r2 = (om_r2 * weights.loc[t] * (m.r2[loc_tech, t] / r2_eff))

        return (m.cost_var[loc_tech, t, k] ==
                cost_op_var + cost_op_fuel + cost_op_r2 + cost_export)

    def cost_rule(m, loc_tech, k):
        return (
            m.cost[loc_tech, k] ==
            m.cost_fixed[loc_tech, k] +
            sum(m.cost_var[loc_tech, t, k] for t in m.t)
        )

    # Constraints
    m.c_cost_fixed = po.Constraint(m.loc_tech, m.k, rule=cost_fixed_rule)
    m.c_cost_var = po.Constraint(m.loc_tech, m.t, m.k, rule=cost_var_rule)
    m.c_cost = po.Constraint(m.loc_tech, m.k, rule=cost_rule)


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
            loc_techs = [i for i in m.loc_tech if i.split(":", 1)[0] in family]
            _loc_techs_export = list(set(loc_techs).intersection(m.loc_tech_export))
            loc_techs_export = [i for i in _loc_techs_export if
                model.get_option(get_y_x(i)[0] + '.export', x=get_y_x(i)[1]) == c]
            balance = (sum(m.c_prod[c, loc_tech, t] for loc_tech in loc_techs)
                       + sum(m.c_con[c, loc_tech, t] for loc_tech in loc_techs)
                       - sum(m.export[loc_tech, t] for loc_tech in loc_techs_export))

            return balance == 0
        else:
            return po.Constraint.NoConstraint

    # Constraints
    m.c_system_balance = po.Constraint(m.c, m.x, m.t,
                                       rule=c_system_balance_rule)

    def c_export_balance_rule(m, c, loc_tech, t):
        y, x = get_y_x(loc_tech)
        # Ensuring no technology can 'pass' its export capability to another
        # technology with the same carrier_out,
        # by limiting its export to the capacity of its production
        if c == model.get_option(y + '.export', x=x):
            return m.c_prod[c, loc_tech, t] >= m.export[loc_tech, t]
        else:
            return po.Constraint.Skip

    # Constraints
    m.c_export_balance = po.Constraint(m.c, m.loc_tech_export, m.t,
                                       rule=c_export_balance_rule)
