"""
Copyright (C) 2013-2019 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

conversion.py
~~~~~~~~~~~~~~~~~

Conversion technology constraints.

"""

import pyomo.core as po  # pylint: disable=import-error

from calliope.backend.pyomo.util import get_param


def balance_conversion_constraint_rule(backend_model, node, tech, timestep):
    """
    Balance energy carrier consumption and production

    .. container:: scrolling-wrapper

        .. math::

            -1 * \\boldsymbol{carrier_{con}}(loc::tech::carrier, timestep)
            \\times \\eta_{energy}(loc::tech, timestep)
            = \\boldsymbol{carrier_{prod}}(loc::tech::carrier, timestep)
            \\times \\eta_{energy}(loc::tech, timestep)
            \\quad \\forall loc::tech \\in locs::techs_{conversion},
            \\forall timestep \\in timesteps
    """

    carrier_out = backend_model.carrier["out", :, tech].index()[0][1]
    carrier_in = backend_model.carrier["in", :, tech].index()[0][1]

    energy_eff = get_param(backend_model, "energy_eff", (node, tech, timestep))

    return (
        backend_model.carrier_prod[carrier_out, node, tech, timestep]
        == -1 * backend_model.carrier_con[carrier_in, node, tech, timestep] * energy_eff
    )


def cost_var_conversion_constraint_rule(backend_model, cost, node, tech, timestep):
    """
    Add time-varying conversion technology costs

    .. container:: scrolling-wrapper

        .. math::

            \\boldsymbol{cost_{var}}(loc::tech, cost, timestep) =
            \\boldsymbol{carrier_{prod}}(loc::tech::carrier, timestep)
            \\times timestep_{weight}(timestep) \\times cost_{om, prod}(loc::tech, cost, timestep)
            +
            \\boldsymbol{carrier_{con}}(loc::tech::carrier, timestep)
            \\times timestep_{weight}(timestep) \\times cost_{om, con}(loc::tech, cost, timestep)
            \\quad \\forall loc::tech \\in loc::techs_{cost_{var}, conversion}
    """
    weight = backend_model.timestep_weights[timestep]

    carrier_out = backend_model.carrier["out", :, tech].index()[0][1]
    carrier_in = backend_model.carrier["in", :, tech].index()[0][1]

    cost_om_prod = get_param(
        backend_model, "cost_om_prod", (cost, node, tech, timestep)
    )
    cost_om_con = get_param(backend_model, "cost_om_con", (cost, node, tech, timestep))
    if po.value(cost_om_prod):
        cost_prod = (
            cost_om_prod
            * weight
            * backend_model.carrier_prod[carrier_out, node, tech, timestep]
        )
    else:
        cost_prod = 0

    if po.value(cost_om_con):
        cost_con = (
            cost_om_con
            * weight
            * -1
            * backend_model.carrier_con[carrier_in, node, tech, timestep]
        )
    else:
        cost_con = 0

    backend_model.cost_var_rhs[cost, node, tech, timestep] = cost_prod + cost_con

    return (
        backend_model.cost_var[cost, node, tech, timestep]
        == backend_model.cost_var_rhs[cost, node, tech, timestep]
    )
