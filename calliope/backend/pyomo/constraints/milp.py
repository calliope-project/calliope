"""
Copyright (C) 2013-2018 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

milp.py
~~~~~~~~~~~~~~~~~

Constraints for binary and integer decision variables

"""

import pyomo.core as po  # pylint: disable=import-error

from calliope.backend.pyomo.util import \
    get_param, \
    get_timestep_weight, \
    get_loc_tech, \
    split_comma_list, \
    loc_tech_is_in

from calliope.backend.pyomo.constraints.capacity import get_capacity_constraint


def load_constraints(backend_model):
    sets = backend_model.__calliope_model_data__['sets']

    if 'loc_techs_milp' in sets:
        backend_model.unit_commitment_constraint = po.Constraint(
            backend_model.loc_techs_unit_commitment_constraint, backend_model.timesteps,
            rule=unit_commitment_constraint_rule
        )
        backend_model.unit_capacity_constraint = po.Constraint(
            backend_model.loc_techs_unit_capacity_constraint,
            rule=unit_capacity_constraint_rule
        )

    if 'loc_tech_carriers_carrier_production_max_milp_constraint' in sets:
        backend_model.carrier_production_max_milp_constraint = po.Constraint(
            backend_model.loc_tech_carriers_carrier_production_max_milp_constraint,
            backend_model.timesteps,
            rule=carrier_production_max_milp_constraint_rule
        )

    if 'loc_techs_carrier_production_max_conversion_plus_milp_constraint' in sets:
        backend_model.carrier_production_max_conversion_plus_milp_constraint = po.Constraint(
            backend_model.loc_techs_carrier_production_max_conversion_plus_milp_constraint,
            backend_model.timesteps,
            rule=carrier_production_max_conversion_plus_milp_constraint_rule
        )

    if 'loc_tech_carriers_carrier_consumption_max_milp_constraint' in sets:
        backend_model.carrier_consumption_max_milp_constraint = po.Constraint(
            backend_model.loc_tech_carriers_carrier_consumption_max_milp_constraint,
            backend_model.timesteps,
            rule=carrier_consumption_max_milp_constraint_rule
        )

    if 'loc_tech_carriers_carrier_production_min_milp_constraint' in sets:
        backend_model.carrier_production_min_milp_constraint = po.Constraint(
            backend_model.loc_tech_carriers_carrier_production_min_milp_constraint,
            backend_model.timesteps,
            rule=carrier_production_min_milp_constraint_rule
        )

    if 'loc_techs_carrier_production_min_conversion_plus_milp_constraint' in sets:
        backend_model.carrier_production_min_conversion_plus_milp_constraint = po.Constraint(
            backend_model.loc_techs_carrier_production_min_conversion_plus_milp_constraint,
            backend_model.timesteps,
            rule=carrier_production_min_conversion_plus_milp_constraint_rule
        )

    if 'loc_techs_storage_capacity_units_constraint' in sets:
        backend_model.storage_capacity_units_constraint = po.Constraint(
            backend_model.loc_techs_storage_capacity_units_constraint,
            rule=storage_capacity_units_constraint_rule
        )

    if 'loc_techs_energy_capacity_units_constraint' in sets:
        backend_model.energy_capacity_units_constraint = po.Constraint(
            backend_model.loc_techs_energy_capacity_units_constraint,
            rule=energy_capacity_units_constraint_rule
        )

    if 'loc_techs_update_costs_investment_units_constraint' in sets and backend_model.mode != 'operate':
        for loc_tech, cost in (
            backend_model.loc_techs_update_costs_investment_units_constraint
            * backend_model.costs):

            update_costs_investment_units_constraint(backend_model, cost, loc_tech,)

    if 'loc_techs_update_costs_investment_purchase_constraint' in sets and backend_model.mode != 'operate':
        for loc_tech, cost in (
            backend_model.loc_techs_update_costs_investment_purchase_constraint
            * backend_model.costs):

            update_costs_investment_purchase_constraint(backend_model, cost, loc_tech,)

    if 'loc_techs_energy_capacity_max_purchase_constraint' in sets:
        backend_model.energy_capacity_max_purchase_constraint = po.Constraint(
            backend_model.loc_techs_energy_capacity_max_purchase_constraint,
            rule=energy_capacity_max_purchase_constraint_rule
        )
    if 'loc_techs_energy_capacity_min_purchase_constraint' in sets:
        backend_model.energy_capacity_min_purchase_constraint = po.Constraint(
            backend_model.loc_techs_energy_capacity_min_purchase_constraint,
            rule=energy_capacity_min_purchase_constraint_rule
        )

    if 'loc_techs_storage_capacity_max_purchase_constraint' in sets:
        backend_model.storage_capacity_max_purchase_constraint = po.Constraint(
            backend_model.loc_techs_storage_capacity_max_purchase_constraint,
            rule=storage_capacity_max_purchase_constraint_rule
        )
    if 'loc_techs_storage_capacity_min_purchase_constraint' in sets:
        backend_model.storage_capacity_min_purchase_constraint = po.Constraint(
            backend_model.loc_techs_storage_capacity_min_purchase_constraint,
            rule=storage_capacity_min_purchase_constraint_rule
        )


def unit_commitment_constraint_rule(backend_model, loc_tech, timestep):
    """
    Constraining the number of integer units
    :math:`operating_units(loc_tech, timestep)` of a technology which
    can operate in a given timestep, based on maximum purchased units
    :math:`units(loc_tech)`

    .. container:: scrolling-wrapper

        .. math::

            \\boldsymbol{operating\_units}(loc::tech, timestep) \\leq
            \\boldsymbol{units}(loc::tech)
            \\quad \\forall loc::tech \\in loc::techs_{milp},
            \\forall timestep \\in timesteps

    """

    return (backend_model.operating_units[loc_tech, timestep]
            <= backend_model.units[loc_tech])


def unit_capacity_constraint_rule(backend_model, loc_tech):
    """
    Add upper and lower bounds for purchased units of a technology

    .. container:: scrolling-wrapper

        .. math::

            \\boldsymbol{units}(loc::tech)
            \\begin{cases}
                = units_{equals}(loc::tech),& \\text{if } units_{equals}(loc::tech)\\\\
                \\leq units_{max}(loc::tech),& \\text{if } units_{max}(loc::tech)\\\\
                \\text{unconstrained},& \\text{otherwise}
            \\end{cases}
            \\quad \\forall loc::tech \\in loc::techs_{milp}

    and (if ``equals`` not enforced):

    .. container:: scrolling-wrapper

        .. math::

            \\boldsymbol{units}(loc::tech) \\geq units_{min}(loc::tech)
            \\quad \\forall loc::tech \\in loc::techs_{milp}

    """
    return get_capacity_constraint(backend_model, 'units', loc_tech)


def carrier_production_max_milp_constraint_rule(backend_model, loc_tech_carrier, timestep):
    """
    Set maximum carrier production of MILP techs that aren't conversion plus

    .. container:: scrolling-wrapper

        .. math::

            \\boldsymbol{carrier_{prod}}(loc::tech::carrier, timestep)
            \\leq energy_{cap, per unit}(loc::tech) \\times timestep\_resolution(timestep)
            \\times \\boldsymbol{operating\_units}(loc::tech, timestep)
            \\times \\eta_{parasitic}(loc::tech, timestep)
            \\quad \\forall loc::tech \\in loc::techs_{milp}, \\forall timestep \\in timesteps

    :math:`\\eta_{parasitic}` is only activated for `supply_plus` technologies
    """
    loc_tech = get_loc_tech(loc_tech_carrier)

    carrier_prod = backend_model.carrier_prod[loc_tech_carrier, timestep]
    timestep_resolution = backend_model.timestep_resolution[timestep]
    parasitic_eff = get_param(backend_model, 'parasitic_eff', (loc_tech, timestep))
    energy_cap = get_param(backend_model, 'energy_cap_per_unit', loc_tech)

    return carrier_prod <= (
        backend_model.operating_units[loc_tech, timestep] *
        timestep_resolution * energy_cap * parasitic_eff
    )


def carrier_production_max_conversion_plus_milp_constraint_rule(backend_model, loc_tech, timestep):
    """
    Set maximum carrier production of conversion_plus MILP techs

    .. container:: scrolling-wrapper

        .. math::

            \sum_{loc::tech::carrier \\in loc::tech::carriers_{out}}
            \\boldsymbol{carrier_{prod}}(loc::tech::carrier, timestep)
            \\leq energy_{cap, per unit}(loc::tech) \\times timestep\_resolution(timestep)
            \\times \\boldsymbol{operating\_units}(loc::tech, timestep)
            \\times \\eta_{parasitic}(loc::tech, timestep)
            \\quad \\forall loc::tech \\in loc::techs_{milp, conversion^{+}}, \\forall timestep \\in timesteps

    """
    model_data_dict = backend_model.__calliope_model_data__['data']
    timestep_resolution = backend_model.timestep_resolution[timestep]
    energy_cap = get_param(backend_model, 'energy_cap_per_unit', loc_tech)
    loc_tech_carriers_out = (
        split_comma_list(model_data_dict['lookup_loc_techs_conversion_plus']['out', loc_tech])
    )

    carrier_prod = sum(backend_model.carrier_prod[loc_tech_carrier, timestep]
                       for loc_tech_carrier in loc_tech_carriers_out)

    return carrier_prod <= (
        backend_model.operating_units[loc_tech, timestep] *
        timestep_resolution * energy_cap
    )


def carrier_production_min_milp_constraint_rule(backend_model, loc_tech_carrier, timestep):
    """
    Set minimum carrier production of MILP techs that aren't conversion plus

    .. container:: scrolling-wrapper

        .. math::

            \\boldsymbol{carrier_{prod}}(loc::tech::carrier, timestep)
            \\geq energy_{cap, per unit}(loc::tech) \\times timestep\_resolution(timestep)
            \\times \\boldsymbol{operating\_units}(loc::tech, timestep)
            \\times energy_{cap, min use}(loc::tech)
            \\quad \\forall loc::tech \\in loc::techs_{milp}, \\forall timestep \\in timesteps
    """
    loc_tech = get_loc_tech(loc_tech_carrier)
    carrier_prod = backend_model.carrier_prod[loc_tech_carrier, timestep]
    timestep_resolution = backend_model.timestep_resolution[timestep]
    min_use = get_param(backend_model, 'energy_cap_min_use', (loc_tech, timestep))
    energy_cap = get_param(backend_model, 'energy_cap_per_unit', loc_tech)

    return carrier_prod >= (
        backend_model.operating_units[loc_tech, timestep] *
        timestep_resolution * energy_cap * min_use
    )


def carrier_production_min_conversion_plus_milp_constraint_rule(backend_model, loc_tech, timestep):
    """
    Set minimum carrier production of conversion_plus MILP techs

    .. container:: scrolling-wrapper

        .. math::
            \sum_{loc::tech::carrier \\in loc::tech::carriers_{out}}
            \\boldsymbol{carrier_{prod}}(loc::tech::carrier, timestep)
            \\geq energy_{cap, per unit}(loc::tech) \\times timestep\_resolution(timestep)
            \\times \\boldsymbol{operating\_units}(loc::tech, timestep)
            \\times energy_{cap, min use}(loc::tech)
            \\quad \\forall loc::tech \\in loc::techs_{milp, conversion^{+}},
            \\forall timestep \\in timesteps

    """
    model_data_dict = backend_model.__calliope_model_data__['data']
    timestep_resolution = backend_model.timestep_resolution[timestep]
    energy_cap = get_param(backend_model, 'energy_cap_per_unit', loc_tech)
    min_use = get_param(backend_model, 'energy_cap_min_use', (loc_tech, timestep))
    loc_tech_carriers_out = (
        split_comma_list(model_data_dict['lookup_loc_techs_conversion_plus']['out', loc_tech])
    )

    carrier_prod = sum(backend_model.carrier_prod[loc_tech_carrier, timestep]
                       for loc_tech_carrier in loc_tech_carriers_out)

    return carrier_prod >= (
        backend_model.operating_units[loc_tech, timestep] *
        timestep_resolution * energy_cap * min_use
    )


def carrier_consumption_max_milp_constraint_rule(backend_model, loc_tech_carrier, timestep):
    """
    Set maximum carrier consumption of demand, storage, and transmission MILP techs

    .. container:: scrolling-wrapper

        .. math::

            \\boldsymbol{carrier_{con}}(loc::tech::carrier, timestep)
            \\geq -1 * energy_{cap, per unit}(loc::tech) \\times timestep\_resolution(timestep)
            \\times \\boldsymbol{operating\_units}(loc::tech, timestep)
            \\times \\eta_{parasitic}(loc::tech, timestep)
            \\quad \\forall loc::tech \\in loc::techs_{milp, con}, \\forall timestep \\in timesteps

    """
    loc_tech = get_loc_tech(loc_tech_carrier)
    carrier_con = backend_model.carrier_con[loc_tech_carrier, timestep]
    timestep_resolution = backend_model.timestep_resolution[timestep]
    energy_cap = get_param(backend_model, 'energy_cap_per_unit', loc_tech)

    return carrier_con >= (-1 *
        backend_model.operating_units[loc_tech, timestep] *
        timestep_resolution * energy_cap
    )


def energy_capacity_units_constraint_rule(backend_model, loc_tech):
    """
    Set energy capacity decision variable as a function of purchased units

    .. container:: scrolling-wrapper

        .. math::

            \\boldsymbol{energy_{cap}}(loc::tech) =
            \\boldsymbol{units}(loc::tech) \\times energy_{cap, per unit}(loc::tech)
            \\quad \\forall loc::tech \\in loc::techs_{milp}

    """
    return backend_model.energy_cap[loc_tech] == (
        backend_model.units[loc_tech] *
        get_param(backend_model, 'energy_cap_per_unit', loc_tech)
    )


def storage_capacity_units_constraint_rule(backend_model, loc_tech):
    """
    Set storage capacity decision variable as a function of purchased units

    .. container:: scrolling-wrapper

        .. math::

            \\boldsymbol{storage_{cap}}(loc::tech) =
            \\boldsymbol{units}(loc::tech) \\times storage_{cap, per unit}(loc::tech)
            \\quad \\forall loc::tech \\in loc::techs_{milp, store}

    """
    return backend_model.storage_cap[loc_tech] == (
        backend_model.units[loc_tech] *
        get_param(backend_model, 'storage_cap_per_unit', loc_tech)
    )


def energy_capacity_max_purchase_constraint_rule(backend_model, loc_tech):
    """
    Set maximum energy capacity decision variable upper bound as a function of
    binary purchase variable

    The first valid case is applied:

    .. container:: scrolling-wrapper

        .. math::

            \\frac{\\boldsymbol{energy_{cap}}(loc::tech)}{energy_{cap, scale}(loc::tech)}
            \\begin{cases}
                = energy_{cap, equals}(loc::tech) \\times \\boldsymbol{purchased}(loc::tech),&
                    \\text{if } energy_{cap, equals}(loc::tech)\\\\
                \\leq energy_{cap, max}(loc::tech) \\times \\boldsymbol{purchased}(loc::tech),&
                    \\text{if } energy_{cap, max}(loc::tech)\\\\
                \\text{unconstrained},& \\text{otherwise}
            \\end{cases}
            \\forall loc::tech \\in loc::techs_{purchase}

    """
    energy_cap_max = get_param(backend_model, 'energy_cap_max', loc_tech)
    energy_cap_equals = get_param(backend_model, 'energy_cap_equals', loc_tech)
    energy_cap_scale = get_param(backend_model, 'energy_cap_scale', loc_tech)

    if po.value(energy_cap_equals):
        return backend_model.energy_cap[loc_tech] == (
            energy_cap_equals * energy_cap_scale * backend_model.purchased[loc_tech]
        )

    else:
        return backend_model.energy_cap[loc_tech] <= (
            energy_cap_max * energy_cap_scale * backend_model.purchased[loc_tech]
        )


def energy_capacity_min_purchase_constraint_rule(backend_model, loc_tech):
    """
    Set minimum energy capacity decision variable upper bound as a function of
    binary purchase variable

    and (if ``equals`` not enforced):

    .. container:: scrolling-wrapper

        .. math::

            \\frac{\\boldsymbol{energy_{cap}}(loc::tech)}{energy_{cap, scale}(loc::tech)}
            \\geq energy_{cap, min}(loc::tech) \\times \\boldsymbol{purchased}(loc::tech)
            \\quad \\forall loc::tech \\in loc::techs
    """
    energy_cap_min = get_param(backend_model, 'energy_cap_min', loc_tech)

    energy_cap_scale = get_param(backend_model, 'energy_cap_scale', loc_tech)
    return backend_model.energy_cap[loc_tech] >= (
        energy_cap_min * energy_cap_scale * backend_model.purchased[loc_tech]
    )


def storage_capacity_max_purchase_constraint_rule(backend_model, loc_tech):
    """
    Set maximum storage capacity.

    The first valid case is applied:

    .. container:: scrolling-wrapper

        .. math::

            \\boldsymbol{storage_{cap}}(loc::tech)
            \\begin{cases}
                = storage_{cap, equals}(loc::tech) \\times \\boldsymbol{purchased},&
                    \\text{if } storage_{cap, equals} \\\\
                \\leq storage_{cap, max}(loc::tech) \\times \\boldsymbol{purchased},&
                    \\text{if } storage_{cap, max}(loc::tech)\\\\
                \\text{unconstrained},& \\text{otherwise}
            \\end{cases}
            \\forall loc::tech \\in loc::techs_{purchase, store}

    """
    storage_cap_max = get_param(backend_model, 'storage_cap_max', loc_tech)
    storage_cap_equals = get_param(backend_model, 'storage_cap_equals', loc_tech)

    if po.value(storage_cap_equals):
        return backend_model.storage_cap[loc_tech] == (
            storage_cap_equals * backend_model.purchased[loc_tech]
        )

    elif po.value(storage_cap_max):
        return backend_model.storage_cap[loc_tech] <= (
            storage_cap_max * backend_model.purchased[loc_tech]
        )

    else:
        return po.Constraint.Skip


def storage_capacity_min_purchase_constraint_rule(backend_model, loc_tech):
    """
    Set minimum storage capacity decision variable as a function of
    binary purchase variable

    if ``equals`` not enforced for storage_cap:

    .. container:: scrolling-wrapper

        .. math::

            \\boldsymbol{storage_{cap}}(loc::tech)
            \\geq storage_{cap, min}(loc::tech) \\times \\boldsymbol{purchased}(loc::tech)
            \\quad \\forall loc::tech \\in loc::techs_{purchase, store}
    """
    storage_cap_min = get_param(backend_model, 'storage_cap_min', loc_tech)

    if po.value(storage_cap_min):
        return backend_model.storage_cap[loc_tech] >= (
            storage_cap_min * backend_model.purchased[loc_tech]
        )

    else:
        return po.Constraint.Skip


def update_costs_investment_units_constraint(backend_model, cost, loc_tech):
    """
    Add MILP investment costs (cost * number of units purchased)

    .. container:: scrolling-wrapper

        .. math::

            \\boldsymbol{cost_{investment}}(cost, loc::tech) += \\boldsymbol{units}(loc::tech)
            \\times cost_{purchase}(cost, loc::tech) * timestep_{weight} * depreciation
            \\quad \\forall cost \\in costs, \\forall loc::tech \\in loc::techs_{cost_{investment}, milp}
    """
    model_data_dict = backend_model.__calliope_model_data__
    ts_weight = get_timestep_weight(backend_model)
    depreciation_rate = model_data_dict['data']['cost_depreciation_rate'][(cost, loc_tech)]

    cost_purchase = get_param(backend_model, 'cost_purchase', (cost, loc_tech))
    cost_of_purchase = (
        backend_model.units[loc_tech] * cost_purchase * ts_weight * depreciation_rate
    )

    if loc_tech_is_in(backend_model, loc_tech, 'loc_techs_transmission'):
        cost_of_purchase = cost_of_purchase / 2

    backend_model.cost_investment_rhs[cost, loc_tech].expr += cost_of_purchase

    return None


def update_costs_investment_purchase_constraint(backend_model, cost, loc_tech):
    """
    Add binary investment costs (cost * binary_purchased_unit)

    .. container:: scrolling-wrapper

        .. math::

            \\boldsymbol{cost_{investment}}(cost, loc::tech) += \\boldsymbol{purchased}(loc::tech)
            \\times cost_{purchase}(cost, loc::tech) * timestep_{weight} * depreciation
            \\quad \\forall cost \\in costs, \\forall loc::tech \\in loc::techs_{cost_{investment}, purchase}

    """
    model_data_dict = backend_model.__calliope_model_data__
    ts_weight = get_timestep_weight(backend_model)
    depreciation_rate = model_data_dict['data']['cost_depreciation_rate'][(cost, loc_tech)]

    cost_purchase = get_param(backend_model, 'cost_purchase', (cost, loc_tech))
    cost_of_purchase = (
        backend_model.purchased[loc_tech] * cost_purchase * ts_weight * depreciation_rate
    )

    if loc_tech_is_in(backend_model, loc_tech, 'loc_techs_transmission'):
        cost_of_purchase = cost_of_purchase / 2

    backend_model.cost_investment_rhs[cost, loc_tech].expr += cost_of_purchase

    return None
