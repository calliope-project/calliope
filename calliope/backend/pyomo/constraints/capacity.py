"""
Copyright (C) since 2013 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

capacity.py
~~~~~~~~~~~~~~~~~

Capacity constraints for technologies (output, resource, area, and storage).

"""

import pyomo.core as po

from calliope.backend.pyomo.util import get_param, invalid


def get_capacity_bounds(bounds):
    def _get_bounds(backend_model, *idx):
        def _get_bound(bound):
            if bounds.get(bound) is not None:
                return get_param(backend_model, bounds.get(bound), idx)
            else:
                return None

        scale = _get_bound("scale")
        _equals = _get_bound("equals")
        _min = _get_bound("min")
        _max = _get_bound("max")

        if not invalid(_equals):
            if not invalid(scale):
                _equals *= scale
            bound_tuple = (_equals, _equals)
        else:
            if invalid(_min):
                _min = None
            if invalid(_max):
                _max = None
            bound_tuple = (_min, _max)

        if not invalid(scale):
            bound_tuple = tuple(i * scale for i in bound_tuple)

        return bound_tuple

    return _get_bounds


def energy_capacity_per_storage_capacity_min_constraint_rule(backend_model, node, tech):
    """
    Limit energy capacities of storage technologies based on their storage capacities.

    .. container:: scrolling-wrapper

        .. math::

            \\boldsymbol{energy_{cap}}(loc::tech)
                \\geq \\boldsymbol{storage_{cap}}(loc::tech) \\times energy\\_cap\\_per\\_storage\\_cap\\_min(loc::tech)\\\\
            \\forall loc::tech \\in loc::techs_{store}

    """
    return backend_model.energy_cap[node, tech] >= (
        backend_model.storage_cap[node, tech]
        * get_param(backend_model, "energy_cap_per_storage_cap_min", (node, tech))
    )


def energy_capacity_per_storage_capacity_max_constraint_rule(backend_model, node, tech):
    """
    Limit energy capacities of storage technologies based on their storage capacities.

    .. container:: scrolling-wrapper

        .. math::

            \\boldsymbol{energy_{cap}}(loc::tech)
                \\leq \\boldsymbol{storage_{cap}}(loc::tech) \\times energy\\_cap\\_per\\_storage\\_cap\\_max(loc::tech)\\\\
            \\forall loc::tech \\in loc::techs_{store}

    """
    return backend_model.energy_cap[node, tech] <= (
        backend_model.storage_cap[node, tech]
        * get_param(backend_model, "energy_cap_per_storage_cap_max", (node, tech))
    )


def energy_capacity_per_storage_capacity_equals_constraint_rule(
    backend_model, node, tech
):
    """
    Limit energy capacities of storage technologies based on their storage capacities.

    .. container:: scrolling-wrapper

        .. math::

            \\boldsymbol{energy_{cap}}(loc::tech)
                = \\boldsymbol{storage_{cap}}(loc::tech) \\times energy\\_cap\\_per\\_storage\\_cap\\_equals(loc::tech)
            \\forall loc::tech \\in loc::techs_{store}

    """
    return backend_model.energy_cap[node, tech] == (
        backend_model.storage_cap[node, tech]
        * get_param(backend_model, "energy_cap_per_storage_cap_equals", (node, tech))
    )


def resource_capacity_equals_energy_capacity_constraint_rule(backend_model, node, tech):
    """
    Add equality constraint for resource_cap to equal energy_cap, for any technologies
    which have defined resource_cap_equals_energy_cap.

    .. container:: scrolling-wrapper

        .. math::

            \\boldsymbol{resource_{cap}}(loc::tech) = \\boldsymbol{energy_{cap}}(loc::tech)
            \\quad \\forall loc::tech \\in loc::techs_{finite\\_resource\\_supply\\_plus}
            \\text{ if } resource\\_cap\\_equals\\_energy\\_cap = \\text{True}
    """
    return (
        backend_model.resource_cap[node, tech] == backend_model.energy_cap[node, tech]
    )


# TODO: reintroduce a constraint to set resource_area to zero when energy_cap_max is zero


def force_zero_resource_area_constraint_rule(backend_model, node, tech):
    """
    Set resource_area to zero if energy_cap_max is zero
    (i.e. there can be no energy_cap, so similarly there can be no resource_area)

    """

    return backend_model.resource_area[node, tech] == 0


def resource_area_per_energy_capacity_constraint_rule(backend_model, node, tech):
    """
    Add equality constraint for resource_area to equal a percentage of energy_cap,
    for any technologies which have defined resource_area_per_energy_cap

    .. container:: scrolling-wrapper

        .. math::

            \\boldsymbol{resource_{area}}(loc::tech) =
            \\boldsymbol{energy_{cap}}(loc::tech) \\times area\\_per\\_energy\\_cap(loc::tech)
            \\quad \\forall loc::tech \\in locs::techs_{area} \\text{ if } area\\_per\\_energy\\_cap(loc::tech)
    """
    area_per_energy_cap = get_param(
        backend_model, "resource_area_per_energy_cap", (node, tech)
    )

    return (
        backend_model.resource_area[node, tech]
        == backend_model.energy_cap[node, tech] * area_per_energy_cap
    )


def resource_area_capacity_per_loc_constraint_rule(backend_model, node):
    """
    Set upper bound on use of area for all locations which have `available_area`
    constraint set. Does not consider resource_area applied to demand technologies

    .. container:: scrolling-wrapper

        .. math::

            \\sum_{tech} \\boldsymbol{resource_{area}}(loc::tech) \\leq available\\_area
            \\quad \\forall loc \\in locs \\text{ if } available\\_area(loc)
    """
    available_area = backend_model.available_area[node]

    return (
        po.quicksum(
            backend_model.resource_area[node, tech]
            for tech in backend_model.techs
            if [node, tech] in backend_model.resource_area._index
        )
        <= available_area
    )


def energy_capacity_systemwide_constraint_rule(backend_model, tech):
    """
    Set constraints to limit the capacity of a single technology type across all locations in the model.

    The first valid case is applied:

    .. container:: scrolling-wrapper

        .. math::

            \\sum_{loc}\\boldsymbol{energy_{cap}}(loc::tech)
            \\begin{cases}
                = energy_{cap, equals, systemwide}(loc::tech),&
                    \\text{if } energy_{cap, equals, systemwide}(loc::tech)\\\\
                \\leq energy_{cap, max, systemwide}(loc::tech),&
                    \\text{if } energy_{cap, max, systemwide}(loc::tech)\\\\
                \\text{unconstrained},& \\text{otherwise}
            \\end{cases}
            \\forall tech \\in techs

    """

    max_systemwide = get_param(backend_model, "energy_cap_max_systemwide", tech)
    equals_systemwide = get_param(backend_model, "energy_cap_equals_systemwide", tech)
    energy_cap = po.quicksum(
        backend_model.energy_cap[node, tech]
        for node in backend_model.nodes
        if [node, tech] in backend_model.energy_cap._index
    )
    if not invalid(equals_systemwide):
        return energy_cap == equals_systemwide
    else:
        return energy_cap <= max_systemwide
