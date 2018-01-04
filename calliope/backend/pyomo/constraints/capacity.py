"""
Copyright (C) 2013-2017 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

capacity.py
~~~~~~~~~~~~~~~~~

Capacity constraints for technologies (output, resource, area, and storage).

"""

import pyomo.core as po  # pylint: disable=import-error
import numpy as np

from calliope.backend.pyomo.util import get_param, split_comma_list
from calliope import exceptions



def load_capacity_constraints(backend_model):
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


def get_capacity_constraint(backend_model, parameter, loc_tech,
                            _equals=None, _max=None, _min=None, scale=None):

    decision_variable = getattr(backend_model, parameter)

    if not _equals:
        _equals = get_param(backend_model, parameter + '_equals', loc_tech)
    if not _max:
        _max = get_param(backend_model, parameter + '_max', loc_tech)
    if not _min:
        _min = get_param(backend_model, parameter + '_min', loc_tech)
    if scale:
        _equals = scale * _equals
        _min = scale * _min
        _max = scale * _max
    if po.value(_equals) is not False and po.value(_equals) is not None:
        if np.isinf(po.value(_equals)):
            e = exceptions.ModelError
            raise e('Cannot use inf for {}_equals for loc:tech `{}`'.format(parameter, loc_tech))
        return decision_variable[loc_tech] == _equals
    else:
        if np.isinf(po.value(_max)):
            _max = None  # to disable upper bound
        if po.value(_min) == 0 and po.value(_max) is None:
            return po.Constraint.NoConstraint
        else:
            return (_min, decision_variable[loc_tech], _max)


def storage_capacity_constraint_rule(backend_model, loc_tech):
    """
    Set maximum storage capacity. Supply_plus & storage techs only
    This can be set by either storage_cap (kWh) or by
    energy_cap (charge/discharge capacity) * charge rate.
    If storage_cap.equals and energy_cap.equals are set for the technology, then
    storage_cap * charge rate = energy_cap must hold. Otherwise, take the lowest capacity
    capacity defined by storage_cap.max or energy_cap.max / charge rate.
    """

    storage_cap_equals = get_param(backend_model, 'storage_cap_equals', loc_tech)
    scale = get_param(backend_model, 'energy_cap_scale', loc_tech)
    energy_cap_equals = (scale * get_param(backend_model, 'energy_cap_equals', loc_tech))

    energy_cap_max = get_param(backend_model, 'energy_cap_max', loc_tech)
    storage_cap_max = get_param(backend_model, 'storage_cap_max', loc_tech)

    charge_rate = get_param(backend_model, 'charge_rate', loc_tech)

    # FIXME: working out storage_cap_max or storage_cap_equals from e_cap_max/_equals
    # should be done in preprocessing, not here.

    if po.value(storage_cap_equals):
        return get_capacity_constraint(backend_model, 'storage_cap',
                                        loc_tech, _equals=storage_cap_equals)
    elif po.value(energy_cap_equals) and po.value(charge_rate):
        storage_cap_equals = energy_cap_equals * scale / charge_rate
        return get_capacity_constraint(backend_model, 'storage_cap',
                                        loc_tech, _equals=storage_cap_equals)
    elif po.value(storage_cap_max):
        return get_capacity_constraint(backend_model, 'storage_cap',
                                        loc_tech, _max=storage_cap_max)
    elif po.value(energy_cap_max) and po.value(charge_rate):
        storage_cap_max = energy_cap_max * scale / charge_rate
        return get_capacity_constraint(backend_model, 'storage_cap',
                                        loc_tech, _max=storage_cap_max)
    else:
        po.Constraint.NoConstraint


def energy_capacity_storage_constraint_rule(backend_model, loc_tech):
    """
    Set an additional energy capacity constraint on storage technologies,
    based on their use of `charge_rate`
    """
    energy_cap_scale = get_param(backend_model, 'energy_cap_scale', loc_tech)
    charge_rate = get_param(backend_model, 'charge_rate', loc_tech)

    return backend_model.energy_cap[loc_tech] <= (
        backend_model.storage_cap[loc_tech] * charge_rate * energy_cap_scale
    )


def resource_capacity_constraint_rule(backend_model, loc_tech):
    """
    Add upper and lower bounds for resource_cap
    """
    return get_capacity_constraint(backend_model, 'resource_cap', loc_tech)


def resource_capacity_equals_energy_capacity_constraint_rule(backend_model, loc_tech):
    """
    Add equality constraint for resource_cap to equal energy_cap, for any technologies
    which have defined resource_cap_equals_energy_cap
    """
    return backend_model.resource_cap[loc_tech] == backend_model.energy_cap[loc_tech]


def resource_area_constraint_rule(backend_model, loc_tech):
    """
    Set upper and lower bounds for resource_area.
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
    """
    area_per_energy_cap = get_param(backend_model, 'resource_area_per_energy_cap', loc_tech)

    return (backend_model.resource_area[loc_tech] ==
                backend_model.energy_cap[loc_tech] * area_per_energy_cap)


def resource_area_capacity_per_loc_constraint_rule(backend_model, loc):
    """
    Set upper bound on use of area for all locations which have `available_area`
    constraint set. Does not consider resource_area applied to demand technologies
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
    """
    return get_capacity_constraint(backend_model, 'energy_cap', loc_tech)
