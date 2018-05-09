"""
Copyright (C) 2013-2018 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

capacity.py
~~~~~~~~~~~~~~~~~

Capacity constraints for technologies (output, resource, area, and storage).

"""

import pyomo.core as po  # pylint: disable=import-error
import numpy as np

from calliope.backend.pyomo.util import get_param, split_comma_list
from calliope import exceptions


def load_constraints(backend_model):
    sets = backend_model.__calliope_model_data__['sets']

    if 'loc_techs_storage_capacity_constraint' in sets:
        backend_model.storage_capacity_constraint = po.Constraint(
                backend_model.loc_techs_storage_capacity_constraint,
                rule=storage_capacity_constraint_rule
        )

    if 'loc_techs_energy_capacity_storage_constraint' in sets:
        backend_model.energy_capacity_storage_constraint = po.Constraint(
                backend_model.loc_techs_energy_capacity_storage_constraint,
                rule=energy_capacity_storage_constraint_rule
        )

    if 'loc_techs_resource_capacity_constraint' in sets:
        backend_model.resource_capacity_constraint = po.Constraint(
                backend_model.loc_techs_resource_capacity_constraint,
                rule=resource_capacity_constraint_rule
        )

    if 'loc_techs_resource_capacity_equals_energy_capacity_constraint' in sets:
        backend_model.resource_capacity_equals_energy_capacity_constraint = po.Constraint(
                backend_model.loc_techs_resource_capacity_equals_energy_capacity_constraint,
                rule=resource_capacity_equals_energy_capacity_constraint_rule
        )

    if 'loc_techs_resource_area_constraint' in sets:
        backend_model.resource_area_constraint = po.Constraint(
                backend_model.loc_techs_resource_area_constraint,
                rule=resource_area_constraint_rule
        )

    if 'loc_techs_resource_area_per_energy_capacity_constraint' in sets:
        backend_model.resource_area_per_energy_capacity_constraint = po.Constraint(
                backend_model.loc_techs_resource_area_per_energy_capacity_constraint,
                rule=resource_area_per_energy_capacity_constraint_rule
        )

    if 'locs_resource_area_capacity_per_loc_constraint' in sets:
        backend_model.resource_area_capacity_per_loc_constraint = po.Constraint(
            backend_model.locs_resource_area_capacity_per_loc_constraint,
            rule=resource_area_capacity_per_loc_constraint_rule
        )

    if 'loc_techs_energy_capacity_constraint' in sets:
        backend_model.energy_capacity_constraint = po.Constraint(
            backend_model.loc_techs_energy_capacity_constraint,
            rule=energy_capacity_constraint_rule
        )

    if 'techs_energy_capacity_systemwide_constraint' in sets:
        backend_model.energy_capacity_systemwide_constraint = po.Constraint(
            backend_model.techs_energy_capacity_systemwide_constraint,
            rule=energy_capacity_systemwide_constraint_rule
        )


def get_capacity_constraint(backend_model, parameter, loc_tech,
                            _equals=None, _max=None, _min=None, scale=None):

    decision_variable = getattr(backend_model, parameter)

    if not _equals:
        _equals = get_param(backend_model, parameter + '_equals', loc_tech)
    if not _max:
        _max = get_param(backend_model, parameter + '_max', loc_tech)
    if not _min:
        _min = get_param(backend_model, parameter + '_min', loc_tech)
    if po.value(_equals) is not False and po.value(_equals) is not None:
        if np.isinf(po.value(_equals)):
            e = exceptions.ModelError
            raise e('Cannot use inf for {}_equals for loc:tech `{}`'.format(parameter, loc_tech))
        if scale:
            _equals *= scale
        return decision_variable[loc_tech] == _equals
    else:
        if po.value(_min) == 0 and np.isinf(po.value(_max)):
            return po.Constraint.NoConstraint
        else:
            if scale:
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

    return get_capacity_constraint(backend_model, 'storage_cap', loc_tech)


def energy_capacity_storage_constraint_rule(backend_model, loc_tech):
    """
    Set an additional energy capacity constraint on storage technologies,
    based on their use of `charge_rate`.

    .. container:: scrolling-wrapper

        .. math::

            \\boldsymbol{energy_{cap}}(loc::tech)
            \\leq \\boldsymbol{storage_{cap}}(loc::tech) \\times charge\_rate(loc::tech)
            \\quad \\forall loc::tech \\in loc::techs_{store}

    """
    charge_rate = get_param(backend_model, 'charge_rate', loc_tech)

    return backend_model.energy_cap[loc_tech] <= (
        backend_model.storage_cap[loc_tech] * charge_rate
    )


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
            \\forall loc::tech \\in loc::techs_{finite\_resource\_supply\_plus}

    and (if ``equals`` not enforced):

    .. container:: scrolling-wrapper

        .. math::

            \\boldsymbol{resource_{cap}}(loc::tech) \\geq resource_{cap, min}(loc::tech)
            \\quad \\forall loc::tech \\in loc::techs_{finite\_resource\_supply\_plus}
    """

    return get_capacity_constraint(backend_model, 'resource_cap', loc_tech)


def resource_capacity_equals_energy_capacity_constraint_rule(backend_model, loc_tech):
    """
    Add equality constraint for resource_cap to equal energy_cap, for any technologies
    which have defined resource_cap_equals_energy_cap.

    .. container:: scrolling-wrapper

        .. math::

            \\boldsymbol{resource_{cap}}(loc::tech) = \\boldsymbol{energy_{cap}}(loc::tech)
            \\quad \\forall loc::tech \\in loc::techs_{finite\_resource\_supply\_plus}
            \\text{ if } resource\_cap\_equals\_energy\_cap = \\text{True}
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
    energy_cap_max = get_param(backend_model, 'energy_cap_max', loc_tech)
    area_per_energy_cap = get_param(backend_model, 'resource_area_per_energy_cap', loc_tech)

    if po.value(energy_cap_max) == 0 and not po.value(area_per_energy_cap):
        # If a technology has no energy_cap here, we force resource_area to zero,
        # so as not to accrue spurious costs
        return backend_model.resource_area[loc_tech] == 0
    else:
        return get_capacity_constraint(backend_model, 'resource_area', loc_tech)


def resource_area_per_energy_capacity_constraint_rule(backend_model, loc_tech):
    """
    Add equality constraint for resource_area to equal a percentage of energy_cap,
    for any technologies which have defined resource_area_per_energy_cap

    .. container:: scrolling-wrapper

        .. math::

            \\boldsymbol{resource_{area}}(loc::tech) =
            \\boldsymbol{energy_{cap}}(loc::tech) \\times area\_per\_energy\_cap(loc::tech)
            \\quad \\forall loc::tech \\in locs::techs_{area} \\text{ if } area\_per\_energy\_cap(loc::tech)
    """
    area_per_energy_cap = get_param(backend_model, 'resource_area_per_energy_cap', loc_tech)

    return (backend_model.resource_area[loc_tech] ==
                backend_model.energy_cap[loc_tech] * area_per_energy_cap)


def resource_area_capacity_per_loc_constraint_rule(backend_model, loc):
    """
    Set upper bound on use of area for all locations which have `available_area`
    constraint set. Does not consider resource_area applied to demand technologies

    .. container:: scrolling-wrapper

        .. math::

            \\sum_{tech} \\boldsymbol{resource_{area}}(loc::tech) \\leq available\_area
            \\quad \\forall loc \in locs \\text{ if } available\_area(loc)
    """
    model_data_dict = backend_model.__calliope_model_data__['data']
    available_area = model_data_dict['available_area'][loc]

    loc_techs = split_comma_list(model_data_dict['lookup_loc_techs_area'][loc])

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
    scale = get_param(backend_model, 'energy_cap_scale', loc_tech)
    return get_capacity_constraint(backend_model, 'energy_cap', loc_tech, scale=scale)


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
    all_loc_techs = [
        i for i in backend_model.loc_techs
        if i.split('::')[1] == tech
    ]

    max_systemwide = get_param(backend_model, 'energy_cap_max_systemwide', tech)
    equals_systemwide = get_param(backend_model, 'energy_cap_equals_systemwide', tech)

    if np.isinf(po.value(max_systemwide)) and not equals_systemwide:
        return po.Constraint.NoConstraint
    elif equals_systemwide and np.isinf(po.value(equals_systemwide)):
        raise exceptions.ModelError(
            'Cannot use inf for energy_cap_equals_systemwide for tech `{}`'.format(tech)
        )

    sum_expr = sum(backend_model.energy_cap[loc_tech] for loc_tech in all_loc_techs)
    total_expr = equals_systemwide if equals_systemwide else max_systemwide

    if equals_systemwide:
        return sum_expr == total_expr
    else:
        return sum_expr <= total_expr
