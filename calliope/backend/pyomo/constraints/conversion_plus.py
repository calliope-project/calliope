"""
Copyright (C) 2013-2018 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

conversion_plus.py
~~~~~~~~~~~~~~~~~~

Conversion plus technology constraints.

"""

import pyomo.core as po  # pylint: disable=import-error

from calliope.backend.pyomo.util import \
    get_param, \
    split_comma_list, \
    get_conversion_plus_io


def load_constraints(backend_model):
    sets = backend_model.__calliope_model_data__['sets']

    backend_model.balance_conversion_plus_primary_constraint = po.Constraint(
        backend_model.loc_techs_balance_conversion_plus_primary_constraint,
        backend_model.timesteps,
        rule=balance_conversion_plus_primary_constraint_rule
    )

    if 'loc_techs_carrier_production_max_conversion_plus_constraint' in sets:
        backend_model.carrier_production_max_conversion_plus_constraint = po.Constraint(
            backend_model.loc_techs_carrier_production_max_conversion_plus_constraint,
            backend_model.timesteps,
            rule=carrier_production_max_conversion_plus_constraint_rule
        )

    if 'loc_techs_carrier_production_min_conversion_plus_constraint' in sets:
        backend_model.carrier_production_min_conversion_plus_constraint = po.Constraint(
            backend_model.loc_techs_carrier_production_min_conversion_plus_constraint,
            backend_model.timesteps,
            rule=carrier_production_min_conversion_plus_constraint_rule
        )

    if 'loc_techs_cost_var_conversion_plus_constraint' in sets:
        backend_model.cost_var_conversion_plus_constraint = po.Constraint(
            backend_model.costs, backend_model.loc_techs_cost_var_conversion_plus_constraint,
            backend_model.timesteps,
            rule=cost_var_conversion_plus_constraint_rule
        )

    if 'loc_techs_balance_conversion_plus_in_2_constraint' in sets:
        backend_model.balance_conversion_plus_in_2_constraint = po.Constraint(
            ['in_2'], backend_model.loc_techs_balance_conversion_plus_in_2_constraint,
            backend_model.timesteps,
            rule=balance_conversion_plus_tiers_constraint_rule
        )

    if 'loc_techs_balance_conversion_plus_in_3_constraint' in sets:
        backend_model.balance_conversion_plus_in_3_constraint = po.Constraint(
            ['in_3'], backend_model.loc_techs_balance_conversion_plus_in_3_constraint,
            backend_model.timesteps,
            rule=balance_conversion_plus_tiers_constraint_rule
        )

    if 'loc_techs_balance_conversion_plus_out_2_constraint' in sets:
        backend_model.balance_conversion_plus_out_2_constraint = po.Constraint(
            ['out_2'], backend_model.loc_techs_balance_conversion_plus_out_2_constraint,
            backend_model.timesteps,
            rule=balance_conversion_plus_tiers_constraint_rule
        )

    if 'loc_techs_balance_conversion_plus_out_3_constraint' in sets:
        backend_model.balance_conversion_plus_out_3_constraint = po.Constraint(
            ['out_3'], backend_model.loc_techs_balance_conversion_plus_out_3_constraint,
            backend_model.timesteps,
            rule=balance_conversion_plus_tiers_constraint_rule
        )


def balance_conversion_plus_primary_constraint_rule(backend_model, loc_tech, timestep):
    """
    Balance energy carrier consumption and production for carrier_in and carrier_out

    .. container:: scrolling-wrapper

        .. math::

            \\sum_{loc::tech::carrier \\in loc::tech::carriers_{out}}
            \\frac{\\boldsymbol{carrier_{prod}}(loc::tech::carrier, timestep)}{
                carrier\_ratio(loc::tech::carrier, `out')} =
            -1 * \\sum_{loc::tech::carrier \\in loc::tech::carriers_{in}} (
            \\boldsymbol{carrier_{con}}(loc::tech::carrier, timestep)
            * carrier\_ratio(loc::tech::carrier, `in') * \\eta_{energy}(loc::tech, timestep))
            \\quad \\forall loc::tech \\in loc::techs_{conversion^{+}}, \\forall timestep \\in timesteps
    """
    model_data_dict = backend_model.__calliope_model_data__['data']

    loc_tech_carriers_out = split_comma_list(
        model_data_dict['lookup_loc_techs_conversion_plus']['out', loc_tech]
    )
    loc_tech_carriers_in = split_comma_list(
        model_data_dict['lookup_loc_techs_conversion_plus']['in', loc_tech]
    )

    energy_eff = get_param(backend_model, 'energy_eff', (loc_tech, timestep))

    carrier_prod = sum(backend_model.carrier_prod[loc_tech_carrier, timestep]
                 / get_param(backend_model, 'carrier_ratios', ('out', loc_tech_carrier))
                 for loc_tech_carrier in loc_tech_carriers_out)
    carrier_con = sum(backend_model.carrier_con[loc_tech_carrier, timestep]
                 * get_param(backend_model, 'carrier_ratios', ('in', loc_tech_carrier))
                 for loc_tech_carrier in loc_tech_carriers_in)

    return carrier_prod == -1 * carrier_con * energy_eff


def carrier_production_max_conversion_plus_constraint_rule(backend_model, loc_tech, timestep):
    """
    Set maximum conversion_plus carrier production.

    .. container:: scrolling-wrapper

        .. math::

            \sum_{loc::tech::carrier \\in loc::tech::carriers_{out}}
            \\boldsymbol{carrier_{prod}}(loc::tech::carrier, timestep)
            \\leq \\boldsymbol{energy_{cap}}(loc::tech) \\times timestep\_resolution(timestep)
            \\quad \\forall loc::tech \\in loc::techs_{conversion^{+}},
            \\forall timestep \\in timesteps
    """
    model_data_dict = backend_model.__calliope_model_data__['data']

    timestep_resolution = backend_model.timestep_resolution[timestep]
    loc_tech_carriers_out = split_comma_list(
        model_data_dict['lookup_loc_techs_conversion_plus']['out', loc_tech]
    )

    carrier_prod = sum(backend_model.carrier_prod[loc_tech_carrier, timestep]
                 for loc_tech_carrier in loc_tech_carriers_out)

    return carrier_prod <= timestep_resolution * backend_model.energy_cap[loc_tech]


def carrier_production_min_conversion_plus_constraint_rule(backend_model, loc_tech, timestep):
    """
    Set minimum conversion_plus carrier production.

    .. container:: scrolling-wrapper

        .. math::

            \sum_{loc::tech::carrier \\in loc::tech::carriers_{out}}
            \\boldsymbol{carrier_{prod}}(loc::tech::carrier, timestep)
            \\leq \\boldsymbol{energy_{cap}}(loc::tech) \\times timestep\_resolution(timestep)
            \\times energy_{cap, min use}(loc::tech)
            \\quad \\forall loc::tech \\in loc::techs_{conversion^{+}},
            \\forall timestep \\in timesteps
    """
    model_data_dict = backend_model.__calliope_model_data__['data']

    timestep_resolution = backend_model.timestep_resolution[timestep]
    min_use = get_param(backend_model, 'energy_cap_min_use', (loc_tech, timestep))

    loc_tech_carriers_out = split_comma_list(
        model_data_dict['lookup_loc_techs_conversion_plus']['out', loc_tech]
    )

    carrier_prod = sum(backend_model.carrier_prod[loc_tech_carrier, timestep]
                for loc_tech_carrier in loc_tech_carriers_out)

    return carrier_prod >= (
        timestep_resolution * backend_model.energy_cap[loc_tech] * min_use
    )


def cost_var_conversion_plus_constraint_rule(backend_model, cost, loc_tech, timestep):
    """
    Add time-varying conversion_plus technology costs

    .. container:: scrolling-wrapper

        .. math::

            \\boldsymbol{cost_{var}}(loc::tech, cost, timestep) =
            \\boldsymbol{carrier_{prod}}(loc::tech::carrier_{primary}, timestep)
            \\times timestep_{weight}(timestep) \\times cost_{om, prod}(loc::tech, cost, timestep)
            +
            \\boldsymbol{carrier_{con}}(loc::tech::carrier_{primary}, timestep)
            \\times timestep_{weight}(timestep) \\times cost_{om, con}(loc::tech, cost, timestep)
            \\quad \\forall loc::tech \\in loc::techs_{cost_{var}, conversion^{+}}
    """
    model_data_dict = backend_model.__calliope_model_data__['data']
    weight = backend_model.timestep_weights[timestep]

    loc_tech_carrier = (
        model_data_dict['lookup_primary_loc_tech_carriers'][loc_tech]
    )

    var_cost = 0

    if loc_tech_carrier in backend_model.loc_tech_carriers_prod:
        cost_om_prod = get_param(backend_model, 'cost_om_prod',
                                 (cost, loc_tech, timestep))
        if cost_om_prod:
            var_cost += (
                cost_om_prod * weight *
                backend_model.carrier_prod[loc_tech_carrier, timestep]
            )

    if loc_tech_carrier in backend_model.loc_tech_carriers_con:
        cost_om_con = get_param(backend_model, 'cost_om_con',
                                (cost, loc_tech, timestep))
        if cost_om_con:
            var_cost += (
                cost_om_con * weight *
                backend_model.carrier_con[loc_tech_carrier, timestep]
            )

    backend_model.cost_var_rhs[cost, loc_tech, timestep] = var_cost
    return (backend_model.cost_var[cost, loc_tech, timestep] ==
            backend_model.cost_var_rhs[cost, loc_tech, timestep])


def balance_conversion_plus_tiers_constraint_rule(backend_model, tier, loc_tech, timestep):
    """
    Force all carrier_in_2/carrier_in_3 and carrier_out_2/carrier_out_3 to follow
    carrier_in and carrier_out (respectively).

    If `tier` in ['out_2', 'out_3']:

    .. container:: scrolling-wrapper

        .. math::

            \\sum_{loc::tech::carrier \\in loc::tech::carriers_{out}} (
            \\frac{\\boldsymbol{carrier_{prod}}(loc::tech::carrier, timestep)}{
                carrier\_ratio(loc::tech::carrier, `out')} =
            \\sum_{loc::tech::carrier \\in loc::tech::carriers_{tier}} (
            \\frac{\\boldsymbol{carrier_{prod}}(loc::tech::carrier, timestep)}{
                carrier\_ratio(loc::tech::carrier, tier)}
            \\quad \\forall \\text { tier } \\in [`out_2', `out_3'], \\forall loc::tech
                \\in loc::techs_{conversion^{+}}, \\forall timestep \\in timesteps

    If `tier` in ['in_2', 'in_3']:

    .. container:: scrolling-wrapper

        .. math::

            \\sum_{loc::tech::carrier \\in loc::tech::carriers_{in}}
            \\frac{\\boldsymbol{carrier_{con}}(loc::tech::carrier, timestep)}{
                carrier\_ratio(loc::tech::carrier, `in')} =
            \\sum_{loc::tech::carrier \\in loc::tech::carriers_{tier}}
            \\frac{\\boldsymbol{carrier_{prod}}(loc::tech::carrier, timestep)}{
                carrier\_ratio(loc::tech::carrier, tier)}
            \\quad \\forall \\text{ tier } \\in [`in_2', `in_3'], \\forall loc::tech
                \\in loc::techs_{conversion^{+}}, \\forall timestep \\in timesteps
    """
    primary_tier, decision_variable = get_conversion_plus_io(backend_model, tier)
    model_data_dict = backend_model.__calliope_model_data__['data']

    loc_tech_carriers_1 = split_comma_list(
        model_data_dict['lookup_loc_techs_conversion_plus'][primary_tier, loc_tech]
    )
    loc_tech_carriers_2 = split_comma_list(
        model_data_dict['lookup_loc_techs_conversion_plus'][tier, loc_tech]
    )

    c_1 = sum(decision_variable[loc_tech_carrier, timestep]
        / get_param(backend_model, 'carrier_ratios', (primary_tier, loc_tech_carrier))
        for loc_tech_carrier in loc_tech_carriers_1)
    c_2 = sum(decision_variable[loc_tech_carrier, timestep]
        / get_param(backend_model, 'carrier_ratios', (tier, loc_tech_carrier))
        for loc_tech_carrier in loc_tech_carriers_2)

    return c_1 == c_2
