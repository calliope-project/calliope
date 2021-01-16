"""
Copyright (C) since 2013 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

policy.py
~~~~~~~~~

Policy constraints.

"""
import pyomo.core as po

from calliope.backend.pyomo.util import get_param


def reserve_margin_constraint_rule(backend_model, carrier):
    """
    Enforces a system reserve margin per carrier.

    .. container:: scrolling-wrapper

        .. math::

            \\sum_{loc::tech::carrier \\in loc\\_tech\\_carriers\\_supply\\_all} energy_{cap}(loc::tech::carrier, timestep_{max\\_demand})
            \\geq
            \\sum_{loc::tech::carrier \\in loc\\_tech\\_carriers\\_demand} carrier_{con}(loc::tech::carrier, timestep_{max\\_demand})
            \\times -1 \\times \\frac{1}{time\\_resolution_{max\\_demand}}
            \\times (1 + reserve\\_margin)

    """

    reserve_margin = get_param(backend_model, "reserve_margin", carrier)
    max_demand_timestep = backend_model.max_demand_timesteps[carrier]
    max_demand_time_res = backend_model.timestep_resolution[max_demand_timestep]
    techs = [idx[-1] for idx in backend_model.carrier["out", carrier, :].index()]
    return po.quicksum(  # Sum all supply capacity for this carrier
        backend_model.energy_cap[node, tech]
        for node, tech in backend_model.energy_cap._index
        if tech in techs
    ) >= po.quicksum(  # Sum all demand for this carrier and timestep
        backend_model.carrier_con[carrier, node, tech, max_demand_timestep]
        for tech in techs
        for node in backend_model.nodes
        if [carrier, node, tech, max_demand_timestep]
        in backend_model.carrier_con._index
    ) * -1 * (
        1 / max_demand_time_res
    ) * (
        1 + reserve_margin
    )
