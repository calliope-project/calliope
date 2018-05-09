"""
Copyright (C) 2013-2018 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

network.py
~~~~~~~~~~

Transmission/distribution network constraints.

"""

import pyomo.core as po  # pylint: disable=import-error


def load_constraints(backend_model):
    sets = backend_model.__calliope_model_data__['sets']

    if 'loc_techs_symmetric_transmission_constraint' in sets and backend_model.mode != 'operate':
        backend_model.symmetric_transmission_constraint = po.Constraint(
            backend_model.loc_techs_symmetric_transmission_constraint,
            rule=symmetric_transmission_constraint_rule
        )


def symmetric_transmission_constraint_rule(backend_model, loc_tech):
    """
    Constrain e_cap symmetrically for transmission nodes. Transmission techs only.

    .. container:: scrolling-wrapper

        .. math::

            energy_{cap}(loc1::tech:loc2) = energy_{cap}(loc2::tech:loc1)

    """
    lookup = backend_model.__calliope_model_data__['data']['lookup_remotes']
    loc_tech_remote = lookup[loc_tech]

    return backend_model.energy_cap[loc_tech] == backend_model.energy_cap[loc_tech_remote]
