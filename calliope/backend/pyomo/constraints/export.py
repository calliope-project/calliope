"""
Copyright (C) since 2013 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

Export.py
~~~~~~~~~~~~~~~~~

Energy export constraints.

"""

from calliope.backend.pyomo.util import (
    get_param,
    loc_tech_is_in,
)


def export_balance_constraint_rule(backend_model, carrier, node, tech, timestep):
    """
    Ensure no technology can 'pass' its export capability to another technology
    with the same carrier_out, by limiting its export to the capacity of its production


    .. container:: scrolling-wrapper

        .. math::

            \\boldsymbol{carrier_{prod}}(loc::tech::carrier, timestep)
            \\geq \\boldsymbol{carrier_{export}}(loc::tech::carrier, timestep)
            \\quad \\forall loc::tech::carrier \\in locs::tech::carriers_{export},
            \\forall timestep \\in timesteps
    """

    return (
        backend_model.carrier_prod[carrier, node, tech, timestep]
        >= backend_model.carrier_export[carrier, node, tech, timestep]
    )


def export_max_constraint_rule(backend_model, carrier, node, tech, timestep):
    """
    Set maximum export. All exporting technologies.

    .. container:: scrolling-wrapper

        .. math::

            \\boldsymbol{carrier_{export}}(loc::tech::carrier, timestep)
            \\leq export_{cap}(loc::tech)
            \\quad \\forall loc::tech::carrier \\in locs::tech::carriers_{export},
            \\forall timestep \\in timesteps

    If the technology is defined by integer units, not a continuous capacity,
    this constraint becomes:

    .. container:: scrolling-wrapper

        .. math::

            \\boldsymbol{carrier_{export}}(loc::tech::carrier, timestep)
            \\leq export_{cap}(loc::tech) \\times
            \\boldsymbol{operating_{units}}(loc::tech, timestep)

    """

    if loc_tech_is_in(backend_model, (node, tech), "operating_units_index"):
        operating_units = backend_model.operating_units[node, tech, timestep]
    else:
        operating_units = 1

    export_max = get_param(backend_model, "export_max", (node, tech))
    return (
        backend_model.carrier_export[carrier, node, tech, timestep]
        <= export_max * operating_units
    )
