"""
Copyright (C) since 2013 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

network.py
~~~~~~~~~~

Transmission/distribution network constraints.

"""


def symmetric_transmission_constraint_rule(backend_model, node, tech):
    """
    Constrain e_cap symmetrically for transmission nodes. Transmission techs only.

    .. container:: scrolling-wrapper

        .. math::

            energy_{cap}(loc1::tech:loc2) = energy_{cap}(loc2::tech:loc1)

    """
    remote_tech = backend_model.link_remote_techs[node, tech].value
    remote_node = backend_model.link_remote_nodes[node, tech].value
    return (
        backend_model.energy_cap[node, tech]
        == backend_model.energy_cap[remote_node, remote_tech]
    )
