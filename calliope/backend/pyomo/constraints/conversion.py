"""
Copyright (C) since 2013 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

conversion.py
~~~~~~~~~~~~~~~~~

Conversion technology constraints.

"""

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
