"""
Copyright (C) since 2013 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

conversion_plus.py
~~~~~~~~~~~~~~~~~~

Conversion plus technology constraints.

"""

import pyomo.core as po

from calliope.backend.pyomo.util import (
    get_param,
    get_conversion_plus_io,
)


def balance_conversion_plus_primary_constraint_rule(
    backend_model, node, tech, timestep
):
    """
    Balance energy carrier consumption and production for carrier_in and carrier_out

    .. container:: scrolling-wrapper

        .. math::

            \\sum_{loc::tech::carrier \\in loc::tech::carriers_{out}}
            \\frac{\\boldsymbol{carrier_{prod}}(loc::tech::carrier, timestep)}{
                carrier\\_ratio(loc::tech::carrier, `out')} =
            -1 * \\sum_{loc::tech::carrier \\in loc::tech::carriers_{in}} (
            \\boldsymbol{carrier_{con}}(loc::tech::carrier, timestep)
            * carrier\\_ratio(loc::tech::carrier, `in') * \\eta_{energy}(loc::tech, timestep))
            \\quad \\forall loc::tech \\in loc::techs_{conversion^{+}}, \\forall timestep \\in timesteps
    """

    carriers_out = backend_model.carrier["out", :, tech].index()
    carriers_in = backend_model.carrier["in", :, tech].index()

    energy_eff = get_param(backend_model, "energy_eff", (node, tech, timestep))

    carrier_prod = []
    for idx in carriers_out:
        carrier = idx[1]
        carrier_ratio = get_param(
            backend_model, "carrier_ratios", ("out", carrier, node, tech, timestep)
        )
        if po.value(carrier_ratio) != 0:
            carrier_prod.append(
                backend_model.carrier_prod[carrier, node, tech, timestep]
                / carrier_ratio
            )

    carrier_con = po.quicksum(
        backend_model.carrier_con[idx[1], node, tech, timestep]
        * get_param(
            backend_model, "carrier_ratios", ("in", idx[1], node, tech, timestep)
        )
        for idx in carriers_in
    )

    return po.quicksum(carrier_prod) == -1 * carrier_con * energy_eff


def carrier_production_max_conversion_plus_constraint_rule(
    backend_model, node, tech, timestep
):
    """
    Set maximum conversion_plus carrier production.

    .. container:: scrolling-wrapper

        .. math::

            \\sum_{loc::tech::carrier \\in loc::tech::carriers_{out}}
            \\boldsymbol{carrier_{prod}}(loc::tech::carrier, timestep)
            \\leq \\boldsymbol{energy_{cap}}(loc::tech) \\times timestep\\_resolution(timestep)
            \\quad \\forall loc::tech \\in loc::techs_{conversion^{+}},
            \\forall timestep \\in timesteps
    """

    timestep_resolution = backend_model.timestep_resolution[timestep]
    carriers_out = backend_model.carrier["out", :, tech].index()

    carrier_prod = po.quicksum(
        backend_model.carrier_prod[idx[1], node, tech, timestep] for idx in carriers_out
    )

    return carrier_prod <= timestep_resolution * backend_model.energy_cap[node, tech]


def carrier_production_min_conversion_plus_constraint_rule(
    backend_model, node, tech, timestep
):
    """
    Set minimum conversion_plus carrier production.

    .. container:: scrolling-wrapper

        .. math::

            \\sum_{loc::tech::carrier \\in loc::tech::carriers_{out}}
            \\boldsymbol{carrier_{prod}}(loc::tech::carrier, timestep)
            \\leq \\boldsymbol{energy_{cap}}(loc::tech) \\times timestep\\_resolution(timestep)
            \\times energy_{cap, min use}(loc::tech)
            \\quad \\forall loc::tech \\in loc::techs_{conversion^{+}},
            \\forall timestep \\in timesteps
    """

    timestep_resolution = backend_model.timestep_resolution[timestep]
    min_use = get_param(backend_model, "energy_cap_min_use", (node, tech, timestep))

    carriers_out = backend_model.carrier["out", :, tech].index()

    carrier_prod = po.quicksum(
        backend_model.carrier_prod[idx[1], node, tech, timestep] for idx in carriers_out
    )

    return carrier_prod >= (
        timestep_resolution * backend_model.energy_cap[node, tech] * min_use
    )


def balance_conversion_plus_non_primary_constraint_rule(
    backend_model, tier, node, tech, timestep
):
    """
    Force all carrier_in_2/carrier_in_3 and carrier_out_2/carrier_out_3 to follow
    carrier_in and carrier_out (respectively).

    If `tier` in ['out_2', 'out_3']:

    .. container:: scrolling-wrapper

        .. math::

            \\sum_{loc::tech::carrier \\in loc::tech::carriers_{out}} (
            \\frac{\\boldsymbol{carrier_{prod}}(loc::tech::carrier, timestep)}{
                carrier\\_ratio(loc::tech::carrier, `out')} =
            \\sum_{loc::tech::carrier \\in loc::tech::carriers_{tier}} (
            \\frac{\\boldsymbol{carrier_{prod}}(loc::tech::carrier, timestep)}{
                carrier\\_ratio(loc::tech::carrier, tier)}
            \\quad \\forall \\text { tier } \\in [`out_2', `out_3'], \\forall loc::tech
                \\in loc::techs_{conversion^{+}}, \\forall timestep \\in timesteps

    If `tier` in ['in_2', 'in_3']:

    .. container:: scrolling-wrapper

        .. math::

            \\sum_{loc::tech::carrier \\in loc::tech::carriers_{in}}
            \\frac{\\boldsymbol{carrier_{con}}(loc::tech::carrier, timestep)}{
                carrier\\_ratio(loc::tech::carrier, `in')} =
            \\sum_{loc::tech::carrier \\in loc::tech::carriers_{tier}}
            \\frac{\\boldsymbol{carrier_{prod}}(loc::tech::carrier, timestep)}{
                carrier\\_ratio(loc::tech::carrier, tier)}
            \\quad \\forall \\text{ tier } \\in [`in_2', `in_3'], \\forall loc::tech
                \\in loc::techs_{conversion^{+}}, \\forall timestep \\in timesteps
    """
    primary_tier, decision_variable = get_conversion_plus_io(backend_model, tier)

    carriers_1 = backend_model.carrier[primary_tier, :, tech].index()
    carriers_2 = backend_model.carrier[tier, :, tech].index()

    c_1 = []
    c_2 = []
    for idx in carriers_1:
        carrier_ratio_1 = get_param(
            backend_model,
            "carrier_ratios",
            (primary_tier, idx[1], node, tech, timestep),
        )
        if po.value(carrier_ratio_1) != 0:
            c_1.append(
                decision_variable[idx[1], node, tech, timestep] / carrier_ratio_1
            )
    for idx in carriers_2:
        carrier_ratio_2 = get_param(
            backend_model, "carrier_ratios", (tier, idx[1], node, tech, timestep)
        )
        if po.value(carrier_ratio_2) != 0:
            c_2.append(
                decision_variable[idx[1], node, tech, timestep] / carrier_ratio_2
            )
    if len(c_2) == 0:
        return po.Constraint.Skip
    else:
        return po.quicksum(c_1) == po.quicksum(c_2)


def conversion_plus_prod_con_to_zero_constraint_rule(
    backend_model, tier, carrier, node, tech, timestep
):
    """
    Force any carrier production or consumption for a conversion plus technology to
    zero in timesteps where its carrier_ratio is zero
    """
    primary_tier, decision_variable = get_conversion_plus_io(backend_model, tier)

    carrier_ratio = get_param(
        backend_model, "carrier_ratios", (tier, carrier, node, tech, timestep)
    )
    if po.value(carrier_ratio) == 0:
        return decision_variable[carrier, node, tech, timestep] == 0
    else:
        return po.Constraint.Skip
