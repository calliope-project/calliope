"""
Copyright (C) since 2013 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

capacity.py
~~~~~~~~~~~~~~~~~

Capacity constraints for technologies (output, resource, area, and storage).

"""

import pyomo.core as po  # pylint: disable=import-error
import numpy as np

from calliope.backend.pyomo.util import (
    apply_equals,
    get_param,
    split_comma_list,
    invalid,
)
from calliope import exceptions

ORDER = 10  # order in which to invoke constraints relative to other constraint files


def load_constraints(backend_model):
    sets = backend_model.__calliope_model_data["sets"]

    if backend_model.__calliope_run_config["mode"] == "operate":
        return None
    else:
        if "loc_techs_storage_capacity_constraint" in sets:
            backend_model.storage_capacity_constraint = po.Constraint(
                backend_model.loc_techs_storage_capacity_constraint,
                rule=storage_capacity_constraint_rule,
            )

        if "loc_techs_energy_capacity_storage_min_constraint" in sets:
            backend_model.energy_capacity_storage_min_constraint = po.Constraint(
                backend_model.loc_techs_energy_capacity_storage_min_constraint,
                rule=energy_capacity_storage_min_constraint_rule,
            )

        if "loc_techs_energy_capacity_storage_max_constraint" in sets:
            backend_model.energy_capacity_storage_max_constraint = po.Constraint(
                backend_model.loc_techs_energy_capacity_storage_max_constraint,
                rule=energy_capacity_storage_max_constraint_rule,
            )

        if "loc_techs_energy_capacity_storage_equals_constraint" in sets:
            backend_model.energy_capacity_storage_equals_constraint = po.Constraint(
                backend_model.loc_techs_energy_capacity_storage_equals_constraint,
                rule=energy_capacity_storage_equals_constraint_rule,
            )

        if "loc_techs_energy_capacity_storage_constraint_old" in sets:
            backend_model.energy_capacity_storage_constraint_old = po.Constraint(
                backend_model.loc_techs_energy_capacity_storage_constraint_old,
                rule=energy_capacity_storage_constraint_rule_old,
            )

        if "loc_techs_resource_capacity_constraint" in sets:
            backend_model.resource_capacity_constraint = po.Constraint(
                backend_model.loc_techs_resource_capacity_constraint,
                rule=resource_capacity_constraint_rule,
            )

        if "loc_techs_resource_capacity_equals_energy_capacity_constraint" in sets:
            backend_model.resource_capacity_equals_energy_capacity_constraint = po.Constraint(
                backend_model.loc_techs_resource_capacity_equals_energy_capacity_constraint,
                rule=resource_capacity_equals_energy_capacity_constraint_rule,
            )

        if "loc_techs_resource_area_constraint" in sets:
            backend_model.resource_area_constraint = po.Constraint(
                backend_model.loc_techs_resource_area_constraint,
                rule=resource_area_constraint_rule,
            )

        if "loc_techs_resource_area_per_energy_capacity_constraint" in sets:
            backend_model.resource_area_per_energy_capacity_constraint = po.Constraint(
                backend_model.loc_techs_resource_area_per_energy_capacity_constraint,
                rule=resource_area_per_energy_capacity_constraint_rule,
            )

        if "locs_resource_area_capacity_per_loc_constraint" in sets:
            backend_model.resource_area_capacity_per_loc_constraint = po.Constraint(
                backend_model.locs_resource_area_capacity_per_loc_constraint,
                rule=resource_area_capacity_per_loc_constraint_rule,
            )

        if "loc_techs_energy_capacity_constraint" in sets:
            backend_model.energy_capacity_constraint = po.Constraint(
                backend_model.loc_techs_energy_capacity_constraint,
                rule=energy_capacity_constraint_rule,
            )

        if "techs_energy_capacity_systemwide_constraint" in sets:
            backend_model.energy_capacity_systemwide_constraint = po.Constraint(
                backend_model.techs_energy_capacity_systemwide_constraint,
                rule=energy_capacity_systemwide_constraint_rule,
            )


def get_capacity_constraint(
    backend_model, parameter, loc_tech, _equals=None, _max=None, _min=None, scale=None
):

    decision_variable = getattr(backend_model, parameter)

    if _equals is None:
        _equals = get_param(backend_model, parameter + "_equals", loc_tech)
    if _max is None:
        _max = get_param(backend_model, parameter + "_max", loc_tech)
    if _min is None:
        _min = get_param(backend_model, parameter + "_min", loc_tech)
    if apply_equals(_equals):
        if scale is not None:
            _equals *= scale
        return decision_variable[loc_tech] == _equals
    else:
        if po.value(_min) == 0 and np.isinf(po.value(_max)):
            return po.Constraint.NoConstraint
        else:
            if scale is not None:
                _max *= scale
                _min *= scale
            return (_min, decision_variable[loc_tech], _max)


def storage_capacity_constraint_rule(backend_model, loc_tech):
    """
    Set maximum storage capacity. Supply_plus & storage techs only

    The first valid case is applied:

    .. container:: scrolling-wrapper

        .. math::

            \\boldsymbol{storage_{cap}}(loc::tech)
            \\begin{cases}
                = storage_{cap, equals}(loc::tech),& \\text{if } storage_{cap, equals}(loc::tech)\\\\
                \\leq storage_{cap, max}(loc::tech),& \\text{if } storage_{cap, max}(loc::tech)\\\\
                \\text{unconstrained},& \\text{otherwise}
            \\end{cases}
            \\forall loc::tech \\in loc::techs_{store}

    and (if ``equals`` not enforced):

    .. container:: scrolling-wrapper

        .. math::

            \\boldsymbol{storage_{cap}}(loc::tech) \\geq storage_{cap, min}(loc::tech)
            \\quad \\forall loc::tech \\in loc::techs_{store}

    """
    return get_capacity_constraint(backend_model, "storage_cap", loc_tech)


def energy_capacity_storage_constraint_rule_old(backend_model, loc_tech):
    """
    Set an additional energy capacity constraint on storage technologies,
    based on their use of `charge_rate`.

    This is deprecated and will be removed in Calliope 0.7.0. Instead of
    `charge_rate`, please use `energy_cap_per_storage_cap_max`.

    .. container:: scrolling-wrapper

        .. math::

            \\boldsymbol{energy_{cap}}(loc::tech)
            \\leq \\boldsymbol{storage_{cap}}(loc::tech) \\times charge\\_rate(loc::tech)
            \\quad \\forall loc::tech \\in loc::techs_{store}

    """
    charge_rate = get_param(backend_model, "charge_rate", loc_tech)

    return backend_model.energy_cap[loc_tech] <= (
        backend_model.storage_cap[loc_tech] * charge_rate
    )


def energy_capacity_storage_min_constraint_rule(backend_model, loc_tech):
    """
    Limit energy capacities of storage technologies based on their storage capacities.

    .. container:: scrolling-wrapper

        .. math::

            \\boldsymbol{energy_{cap}}(loc::tech)
                \\geq \\boldsymbol{storage_{cap}}(loc::tech) \\times energy\\_cap\\_per\\_storage\\_cap\\_min(loc::tech)\\\\
            \\forall loc::tech \\in loc::techs_{store}

    """
    return backend_model.energy_cap[loc_tech] >= (
        backend_model.storage_cap[loc_tech]
        * get_param(backend_model, "energy_cap_per_storage_cap_min", loc_tech)
    )


def energy_capacity_storage_max_constraint_rule(backend_model, loc_tech):
    """
    Limit energy capacities of storage technologies based on their storage capacities.

    .. container:: scrolling-wrapper

        .. math::

            \\boldsymbol{energy_{cap}}(loc::tech)
                \\leq \\boldsymbol{storage_{cap}}(loc::tech) \\times energy\\_cap\\_per\\_storage\\_cap\\_max(loc::tech)\\\\
            \\forall loc::tech \\in loc::techs_{store}

    """
    return backend_model.energy_cap[loc_tech] <= (
        backend_model.storage_cap[loc_tech]
        * get_param(backend_model, "energy_cap_per_storage_cap_max", loc_tech)
    )


def energy_capacity_storage_equals_constraint_rule(backend_model, loc_tech):
    """
    Limit energy capacities of storage technologies based on their storage capacities.

    .. container:: scrolling-wrapper

        .. math::

            \\boldsymbol{energy_{cap}}(loc::tech)
                = \\boldsymbol{storage_{cap}}(loc::tech) \\times energy\\_cap\\_per\\_storage\\_cap\\_equals(loc::tech)
            \\forall loc::tech \\in loc::techs_{store}

    """
    energy_cap_per_storage_cap = get_param(
        backend_model, "energy_cap_per_storage_cap_equals", loc_tech
    )
    if apply_equals(energy_cap_per_storage_cap):
        return backend_model.energy_cap[loc_tech] == (
            backend_model.storage_cap[loc_tech] * energy_cap_per_storage_cap
        )
    else:
        return po.Constraint.Skip


def resource_capacity_constraint_rule(backend_model, loc_tech):
    """
    Add upper and lower bounds for resource_cap.

    The first valid case is applied:

    .. container:: scrolling-wrapper

        .. math::

            \\boldsymbol{resource_{cap}}(loc::tech)
            \\begin{cases}
                = resource_{cap, equals}(loc::tech),& \\text{if } resource_{cap, equals}(loc::tech)\\\\
                \\leq resource_{cap, max}(loc::tech),& \\text{if } resource_{cap, max}(loc::tech)\\\\
                \\text{unconstrained},& \\text{otherwise}
            \\end{cases}
            \\forall loc::tech \\in loc::techs_{finite\\_resource\\_supply\\_plus}

    and (if ``equals`` not enforced):

    .. container:: scrolling-wrapper

        .. math::

            \\boldsymbol{resource_{cap}}(loc::tech) \\geq resource_{cap, min}(loc::tech)
            \\quad \\forall loc::tech \\in loc::techs_{finite\\_resource\\_supply\\_plus}
    """

    return get_capacity_constraint(backend_model, "resource_cap", loc_tech)


def resource_capacity_equals_energy_capacity_constraint_rule(backend_model, loc_tech):
    """
    Add equality constraint for resource_cap to equal energy_cap, for any technologies
    which have defined resource_cap_equals_energy_cap.

    .. container:: scrolling-wrapper

        .. math::

            \\boldsymbol{resource_{cap}}(loc::tech) = \\boldsymbol{energy_{cap}}(loc::tech)
            \\quad \\forall loc::tech \\in loc::techs_{finite\\_resource\\_supply\\_plus}
            \\text{ if } resource\\_cap\\_equals\\_energy\\_cap = \\text{True}
    """
    return backend_model.resource_cap[loc_tech] == backend_model.energy_cap[loc_tech]


def resource_area_constraint_rule(backend_model, loc_tech):
    """
    Set upper and lower bounds for resource_area.

    The first valid case is applied:

    .. container:: scrolling-wrapper

        .. math::

            \\boldsymbol{resource_{area}}(loc::tech)
            \\begin{cases}
                = resource_{area, equals}(loc::tech),& \\text{if } resource_{area, equals}(loc::tech)\\\\
                \\leq resource_{area, max}(loc::tech),& \\text{if } resource_{area, max}(loc::tech)\\\\
                \\text{unconstrained},& \\text{otherwise}
            \\end{cases}
            \\forall loc::tech \\in loc::techs_{area}

    and (if ``equals`` not enforced):

    .. container:: scrolling-wrapper

        .. math::

            \\boldsymbol{resource_{area}}(loc::tech) \\geq resource_{area, min}(loc::tech)
            \\quad \\forall loc::tech \\in loc::techs_{area}
    """
    energy_cap_max = get_param(backend_model, "energy_cap_max", loc_tech)
    area_per_energy_cap = get_param(
        backend_model, "resource_area_per_energy_cap", loc_tech
    )

    if po.value(energy_cap_max) == 0 and invalid(area_per_energy_cap):
        # If a technology has no energy_cap here, we force resource_area to zero,
        # so as not to accrue spurious costs
        return backend_model.resource_area[loc_tech] == 0
    else:
        return get_capacity_constraint(backend_model, "resource_area", loc_tech)


def resource_area_per_energy_capacity_constraint_rule(backend_model, loc_tech):
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
        backend_model, "resource_area_per_energy_cap", loc_tech
    )

    return (
        backend_model.resource_area[loc_tech]
        == backend_model.energy_cap[loc_tech] * area_per_energy_cap
    )


def resource_area_capacity_per_loc_constraint_rule(backend_model, loc):
    """
    Set upper bound on use of area for all locations which have `available_area`
    constraint set. Does not consider resource_area applied to demand technologies

    .. container:: scrolling-wrapper

        .. math::

            \\sum_{tech} \\boldsymbol{resource_{area}}(loc::tech) \\leq available\\_area
            \\quad \\forall loc \\in locs \\text{ if } available\\_area(loc)
    """
    model_data_dict = backend_model.__calliope_model_data["data"]
    available_area = model_data_dict["available_area"][loc]

    loc_techs = split_comma_list(model_data_dict["lookup_loc_techs_area"][loc])

    return (
        sum(backend_model.resource_area[loc_tech] for loc_tech in loc_techs)
        <= available_area
    )


def energy_capacity_constraint_rule(backend_model, loc_tech):
    """
    Set upper and lower bounds for energy_cap.

    The first valid case is applied:

    .. container:: scrolling-wrapper

        .. math::

            \\frac{\\boldsymbol{energy_{cap}}(loc::tech)}{energy_{cap, scale}(loc::tech)}
            \\begin{cases}
                = energy_{cap, equals}(loc::tech),& \\text{if } energy_{cap, equals}(loc::tech)\\\\
                \\leq energy_{cap, max}(loc::tech),& \\text{if } energy_{cap, max}(loc::tech)\\\\
                \\text{unconstrained},& \\text{otherwise}
            \\end{cases}
            \\forall loc::tech \\in loc::techs

    and (if ``equals`` not enforced):

    .. container:: scrolling-wrapper

        .. math::

            \\frac{\\boldsymbol{energy_{cap}}(loc::tech)}{energy_{cap, scale}(loc::tech)}
            \\geq energy_{cap, min}(loc::tech)
            \\quad \\forall loc::tech \\in loc::techs
    """
    scale = get_param(backend_model, "energy_cap_scale", loc_tech)
    return get_capacity_constraint(backend_model, "energy_cap", loc_tech, scale=scale)


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

    if tech in getattr(backend_model, "techs_transmission_names", []):
        all_loc_techs = [
            i
            for i in backend_model.loc_techs_transmission
            if i.split("::")[1].split(":")[0] == tech
        ]
        multiplier = 2  # there are always two technologies associated with one link
    else:
        all_loc_techs = [i for i in backend_model.loc_techs if i.split("::")[1] == tech]
        multiplier = 1

    max_systemwide = get_param(backend_model, "energy_cap_max_systemwide", tech)
    equals_systemwide = get_param(backend_model, "energy_cap_equals_systemwide", tech)

    if np.isinf(po.value(max_systemwide)) and not apply_equals(equals_systemwide):
        return po.Constraint.NoConstraint

    sum_expr = sum(backend_model.energy_cap[loc_tech] for loc_tech in all_loc_techs)

    if not invalid(equals_systemwide):
        return sum_expr == equals_systemwide * multiplier
    else:
        return sum_expr <= max_systemwide * multiplier
