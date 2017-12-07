"""
Copyright (C) 2013-2017 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

energy_balance.py
~~~~~~~~~~~~~~~~~

Energy balance constraints.

"""

import pyomo.core as po  # pylint: disable=import-error
import numpy as np

from calliope.backend.pyomo.util import param_getter, get_previous_timestep
from calliope import exceptions



def load_capacity_constraints(backend_model):
    sets = backend_model.__calliope_model_data__['sets']

    if 'loc_techs_store' in sets:
        backend_model.storage_capacity_constraint = po.Constraint(
                backend_model.loc_techs_store,
                rule=storage_capacity_constraint_rule
        )

    if 'loc_techs_store' in sets:
        backend_model.energy_capacity_storage_constraint = po.Constraint(
                backend_model.loc_techs_store,
                rule=energy_capacity_storage_constraint_rule
        )

    if 'loc_techs_supply_plus' in sets:
        backend_model.resource_capacity_constraint = po.Constraint(
                backend_model.loc_techs_supply_plus,
                rule=resource_capacity_constraint_rule
        )

    if 'loc_techs_supply_plus' in sets:
        backend_model.resource_capacity_equals_energy_capacity_constraint = po.Constraint(
                backend_model.loc_techs_supply_plus,
                rule=resource_capacity_equals_energy_capacity_constraint_rule
        )

    if 'loc_techs_area' in sets:
        backend_model.resource_area_constraint = po.Constraint(
                backend_model.loc_techs_area,
                rule=resource_area_constraint_rule
        )

    if 'loc_techs_area' in sets:
        backend_model.resource_area_per_energy_capacity_constraint = po.Constraint(
                backend_model.loc_techs_area,
                rule=resource_area_per_energy_capacity_constraint_rule
        )

    backend_model.energy_capacity_constraint = po.Constraint(
            backend_model.loc_techs,
            rule=energy_capacity_constraint_rule
    )

    backend_model.energy_capacity_min_constraint = po.Constraint(
            backend_model.loc_techs,
            rule=energy_capacity_min_constraint_rule
    )


def get_capacity_constraint(backend_model, parameter, loc_tech,
                            _equals=None, _max=None, _min=None, scale=None):

    decision_variable = getattr(backend_model, parameter)

    if not _equals:
        _equals = param_getter(backend_model, parameter + '_equals', loc_tech)
    if not _max:
        _max = param_getter(backend_model, parameter + '_max', loc_tech)
    if not _min:
        _min = param_getter(backend_model, parameter + '_min', loc_tech)
    if scale:
        _equals = scale * _equals
        _min = scale * _min
        _max = scale * _max
    if _equals is not False and _equals is not None:
        if np.isinf(_equals):
            e = exceptions.ModelError
            raise e('Cannot use inf for {}_equals for loc:tech `{}`'.format(parameter, loc_tech))
        return decision_variable[loc_tech] == _equals
    #elif model.mode == 'operate':
    #    # Operational mode but 'equals' constraint not set, we use 'max'
    #    # instead
    #    # FIXME this should be logged
    #    if np.isinf(_max):
    #        return po.Constraint.NoConstraint
    #    else:
    #        return model_var == _max
    else:
        if np.isinf(_max):
            _max = None  # to disable upper bound
        if _min == 0 and _max is None:
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

    # FIXME?: energy_cap_equals could be already dealt with in preprocessing, to
    # either be energy_cap_equals or units_equals * energy_cap_per_unit. Similarly for
    # storage_cap_equals
    #if loc_tech in backend_model.loc_tech_milp:
    #    units_equals = param_getter(backend_model, 'units_equals', loc_tech)
    #    storage_cap_equals = (units_equals *
    #        param_getter(backend_model, 'storage_cap_per_unit', loc_tech))
    #    energy_cap_equals = (units_equals *
    #        param_getter(backend_model, 'energy_cap_per_unit', loc_tech))

    storage_cap_equals = param_getter(backend_model, 'storage_cap_equals', loc_tech)
    scale = param_getter(backend_model, 'energy_cap_scale', loc_tech)
    energy_cap_equals = (scale * param_getter(backend_model, 'energy_cap_equals', loc_tech))

    charge_rate = param_getter(backend_model, 'charge_rate', loc_tech)

    # FIXME?: energy_cap_max could be already dealt with in preprocessing, to
    # either be energy_cap_max or units_max * energy_cap_per_unit. Similarly for
    # storage_cap_max
    if not energy_cap_equals:
        #if loc_tech in backend_model.loc_tech_milp:
        #    energy_cap = (param_getter(backend_model, 'units_max', loc_tech)
        #        * param_getter(backend_model, 'energy_cap_per_unit', loc_tech))
        energy_cap = scale * param_getter(backend_model, 'energy_cap_max', loc_tech)
    if not storage_cap_equals:
        storage_cap_max = param_getter(backend_model, 'storage_cap_max', loc_tech)
        return get_capacity_constraint(backend_model, 'storage_cap',
                                        loc_tech, _max=storage_cap_max)
    else:
        return get_capacity_constraint(backend_model, 'storage_cap',
                                        loc_tech, _equals=storage_cap_equals)


def energy_capacity_storage_constraint_rule(backend_model, loc_tech):
    """
    Set an additional energy capacity constraint on storage technologies,
    based on their use of `charge_rate`
    """

    energy_cap_scale = param_getter(backend_model, 'energy_cap_scale', loc_tech)
    charge_rate = param_getter(backend_model, 'charge_rate', loc_tech)

    if charge_rate:
        return backend_model.energy_cap[loc_tech] <= (
            backend_model.storage_cap[loc_tech] * charge_rate * energy_cap_scale
        )
    else:
        return po.Constraint.Skip


def resource_capacity_constraint_rule(backend_model, loc_tech):
    return get_capacity_constraint(backend_model, 'resource_cap', loc_tech)


def resource_capacity_equals_energy_capacity_constraint_rule(backend_model, loc_tech):
    if param_getter(backend_model, 'resource_cap_equals_energy_cap', loc_tech):
        return backend_model.resource_cap[loc_tech] == backend_model.energy_cap[loc_tech]
    else:
        return po.Constraint.Skip


def resource_area_constraint_rule(backend_model, loc_tech):
    """
    Set maximum resource_area.
    """

    energy_cap_max = param_getter(backend_model, 'energy_cap_max', loc_tech)
    area_per_energy_cap = param_getter(backend_model, 'resource_area_per_energy_cap', loc_tech)

    if energy_cap_max == 0 and not area_per_energy_cap:
        # If a technology has no energy_cap here, we force resource_area to zero,
        # so as not to accrue spurious costs
        return backend_model.resource_area[loc_tech] == 0
    else:
        return get_capacity_constraint(backend_model, 'resource_area', loc_tech)


def resource_area_per_energy_capacity_constraint_rule(backend_model, loc_tech):
    area_per_energy_cap = param_getter(backend_model, 'resource_area_per_energy_cap', loc_tech)
    if area_per_energy_cap:
        return (backend_model.resource_area[loc_tech] ==
                    backend_model.energy_cap[loc_tech] * area_per_energy_cap)
    else:
        return po.Constraint.Skip


def energy_capacity_constraint_rule(backend_model, loc_tech):
    """
    Set maximum energy_cap
    """
    sets = backend_model.__calliope_model_data__['sets']

    # Addition of binary variable describing whether a technology has been
    # purchased or not
    if 'loc_tech_purchase' in sets and loc_tech in backend_model.loc_techs_purchase:
        purchased = backend_model.purchased[loc_tech]
    else:
        purchased = 1

    # Addition of integer variable describing how many units of a technology
    # have been purchased
    if 'loc_tech_milp' in sets and loc_tech in backend_model.loc_techs_milp:
        return backend_model.energy_cap[loc_tech] == (
            backend_model.units[loc_tech] *
            param_getter(backend_model, 'energy_cap_per_unit', loc_tech)
        )

    energy_cap_max = param_getter(backend_model, 'energy_cap_max', loc_tech)
    energy_cap_equals = param_getter(backend_model, 'energy_cap_equals', loc_tech)
    energy_cap_scale = param_getter(backend_model, 'energy_cap_scale', loc_tech)

    # energy_cap_equals forces an equality constraint, which cannot be infinite
    # FIXME? move to preprocessing check?
    if energy_cap_equals:
        if energy_cap_equals is None or np.isinf(energy_cap_equals) or np.isnan(energy_cap_equals):
            e = exceptions.ModelError
            raise e('Cannot use inf, NaN, or None for equality constraint: '
                    'energy_cap.equals for loc:tech `{}`'.format(loc_tech))
        else:
            return backend_model.energy_cap[loc_tech] == (
                energy_cap_equals * energy_cap_scale * purchased
            )

    # In operation mode, energy_cap is forced to an equality constraint, even if
    # energy_cap.max is defined.
    # FIXME: commented out as operational mode is currently not allowed
    #if (model.mode == 'operate' and loc_tech not in backend_model.loc_tech_demand
    #    and loc_tech not in backend_model.loc_tech_unmet):
    #    if energy_cap_max is None or np.isinf(e_cap_max) or np.isnan(e_cap_max):
    #        e = exceptions.ModelError
    #        raise e('Cannot use inf, NaN, or None in operational mode, '
    #                'for value of {}.energy_cap.max.{}'.format(y, x))
    #    return backend_model.energy_cap[loc_tech] == energy_cap_max * energy_cap_scale * purchased

    # Infinite or undefined energy_cap_max leads to an ignored constraint
    elif energy_cap_max is None or np.isnan(energy_cap_max) or np.isinf(energy_cap_max):
        return po.Constraint.Skip
    else:
        return backend_model.energy_cap[loc_tech] <= energy_cap_max * energy_cap_scale * purchased


def energy_capacity_min_constraint_rule(backend_model, loc_tech):
    """
    Set minimum energy_cap. All technologies.
    """
    sets = backend_model.__calliope_model_data__['sets']

    # Addition of binary variable describing whether a technology has been
    # purchased or not
    if 'loc_tech_purchase' in sets and loc_tech in backend_model.loc_techs_purchase:
        purchased = backend_model.purchased[loc_tech]
    else:
        purchased = 1

    # Addition of integer variable describing how many units of a technology
    # have been purchased
    if 'loc_tech_milp' in sets and loc_tech in backend_model.loc_techs_milp:
        return po.Constraint.Skip

    energy_cap_min = param_getter(backend_model, 'energy_cap_min', loc_tech)
    energy_cap_equals = param_getter(backend_model, 'energy_cap_equals', loc_tech)

    if energy_cap_equals or not energy_cap_min:
        return po.Constraint.Skip
    else:
        energy_cap_scale = param_getter(backend_model, 'energy_cap_scale', loc_tech)
        return backend_model.energy_cap[loc_tech] >= (
            energy_cap_min * energy_cap_scale * purchased
        )
