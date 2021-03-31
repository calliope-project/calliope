"""
Copyright (C) since 2013 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

milp.py
~~~~~~~~~~~~~~~~~

Constraints for binary and integer decision variables

"""

import pyomo.core as po

from calliope.backend.pyomo.util import get_param


def unit_commitment_milp_constraint_rule(backend_model, node, tech, timestep):
    """
    Constraining the number of integer units
    :math:`operating_units(loc_tech, timestep)` of a technology which
    can operate in a given timestep, based on maximum purchased units
    :math:`units(loc_tech)`

    .. container:: scrolling-wrapper

        .. math::

            \\boldsymbol{operating\\_units}(loc::tech, timestep) \\leq
            \\boldsymbol{units}(loc::tech)
            \\quad \\forall loc::tech \\in loc::techs_{milp},
            \\forall timestep \\in timesteps

    """

    return (
        backend_model.operating_units[node, tech, timestep]
        <= backend_model.units[node, tech]
    )


def carrier_production_max_milp_constraint_rule(
    backend_model, carrier, node, tech, timestep
):
    """
    Set maximum carrier production of MILP techs that aren't conversion plus

    .. container:: scrolling-wrapper

        .. math::

            \\boldsymbol{carrier_{prod}}(loc::tech::carrier, timestep)
            \\leq energy_{cap, per unit}(loc::tech) \\times timestep\\_resolution(timestep)
            \\times \\boldsymbol{operating\\_units}(loc::tech, timestep)
            \\times \\eta_{parasitic}(loc::tech, timestep)
            \\quad \\forall loc::tech \\in loc::techs_{milp}, \\forall timestep \\in timesteps

    :math:`\\eta_{parasitic}` is only activated for `supply_plus` technologies
    """

    carrier_prod = backend_model.carrier_prod[carrier, node, tech, timestep]
    timestep_resolution = backend_model.timestep_resolution[timestep]
    parasitic_eff = get_param(backend_model, "parasitic_eff", (node, tech, timestep))
    energy_cap = get_param(backend_model, "energy_cap_per_unit", (node, tech))

    return carrier_prod <= (
        backend_model.operating_units[node, tech, timestep]
        * timestep_resolution
        * energy_cap
        * parasitic_eff
    )


def carrier_production_max_conversion_plus_milp_constraint_rule(
    backend_model, node, tech, timestep
):
    """
    Set maximum carrier production of conversion_plus MILP techs

    .. container:: scrolling-wrapper

        .. math::

            \\sum_{loc::tech::carrier \\in loc::tech::carriers_{out}}
            \\boldsymbol{carrier_{prod}}(loc::tech::carrier, timestep)
            \\leq energy_{cap, per unit}(loc::tech) \\times timestep\\_resolution(timestep)
            \\times \\boldsymbol{operating\\_units}(loc::tech, timestep)
            \\times \\eta_{parasitic}(loc::tech, timestep)
            \\quad \\forall loc::tech \\in loc::techs_{milp, conversion^{+}}, \\forall timestep \\in timesteps

    """
    timestep_resolution = backend_model.timestep_resolution[timestep]
    energy_cap = get_param(backend_model, "energy_cap_per_unit", (node, tech))
    carriers_out = backend_model.carrier["out", :, tech].index()

    carrier_prod = po.quicksum(
        backend_model.carrier_prod[idx[1], node, tech, timestep] for idx in carriers_out
    )

    return carrier_prod <= (
        backend_model.operating_units[node, tech, timestep]
        * timestep_resolution
        * energy_cap
    )


def carrier_production_min_milp_constraint_rule(
    backend_model, carrier, node, tech, timestep
):
    """
    Set minimum carrier production of MILP techs that aren't conversion plus

    .. container:: scrolling-wrapper

        .. math::

            \\boldsymbol{carrier_{prod}}(loc::tech::carrier, timestep)
            \\geq energy_{cap, per unit}(loc::tech) \\times timestep\\_resolution(timestep)
            \\times \\boldsymbol{operating\\_units}(loc::tech, timestep)
            \\times energy_{cap, min use}(loc::tech)
            \\quad \\forall loc::tech \\in loc::techs_{milp}, \\forall timestep \\in timesteps
    """

    carrier_prod = backend_model.carrier_prod[carrier, node, tech, timestep]
    timestep_resolution = backend_model.timestep_resolution[timestep]
    min_use = get_param(backend_model, "energy_cap_min_use", (node, tech, timestep))
    energy_cap = get_param(backend_model, "energy_cap_per_unit", (node, tech))

    return carrier_prod >= (
        backend_model.operating_units[node, tech, timestep]
        * timestep_resolution
        * energy_cap
        * min_use
    )


def carrier_production_min_conversion_plus_milp_constraint_rule(
    backend_model, node, tech, timestep
):
    """
    Set minimum carrier production of conversion_plus MILP techs

    .. container:: scrolling-wrapper

        .. math::
            \\sum_{loc::tech::carrier \\in loc::tech::carriers_{out}}
            \\boldsymbol{carrier_{prod}}(loc::tech::carrier, timestep)
            \\geq energy_{cap, per unit}(loc::tech) \\times timestep\\_resolution(timestep)
            \\times \\boldsymbol{operating\\_units}(loc::tech, timestep)
            \\times energy_{cap, min use}(loc::tech)
            \\quad \\forall loc::tech \\in loc::techs_{milp, conversion^{+}},
            \\forall timestep \\in timesteps

    """
    timestep_resolution = backend_model.timestep_resolution[timestep]
    energy_cap = get_param(backend_model, "energy_cap_per_unit", (node, tech))
    min_use = get_param(backend_model, "energy_cap_min_use", (node, tech, timestep))
    carriers_out = backend_model.carrier["out", :, tech].index()

    carrier_prod = po.quicksum(
        backend_model.carrier_prod[idx[1], node, tech, timestep] for idx in carriers_out
    )

    return carrier_prod >= (
        backend_model.operating_units[node, tech, timestep]
        * timestep_resolution
        * energy_cap
        * min_use
    )


def carrier_consumption_max_milp_constraint_rule(
    backend_model, carrier, node, tech, timestep
):
    """
    Set maximum carrier consumption of demand, storage, and transmission MILP techs

    .. container:: scrolling-wrapper

        .. math::

            \\boldsymbol{carrier_{con}}(loc::tech::carrier, timestep)
            \\geq -1 * energy_{cap, per unit}(loc::tech) \\times timestep\\_resolution(timestep)
            \\times \\boldsymbol{operating\\_units}(loc::tech, timestep)
            \\times \\eta_{parasitic}(loc::tech, timestep)
            \\quad \\forall loc::tech \\in loc::techs_{milp, con}, \\forall timestep \\in timesteps

    """
    carrier_con = backend_model.carrier_con[carrier, node, tech, timestep]
    timestep_resolution = backend_model.timestep_resolution[timestep]
    energy_cap = get_param(backend_model, "energy_cap_per_unit", (node, tech))

    return carrier_con >= (
        -1
        * backend_model.operating_units[node, tech, timestep]
        * timestep_resolution
        * energy_cap
    )


def energy_capacity_units_milp_constraint_rule(backend_model, node, tech):
    """
    Set energy capacity decision variable as a function of purchased units

    .. container:: scrolling-wrapper

        .. math::

            \\boldsymbol{energy_{cap}}(loc::tech) =
            \\boldsymbol{units}(loc::tech) \\times energy_{cap, per unit}(loc::tech)
            \\quad \\forall loc::tech \\in loc::techs_{milp}

    """
    return backend_model.energy_cap[node, tech] == (
        backend_model.units[node, tech]
        * get_param(backend_model, "energy_cap_per_unit", (node, tech))
    )


def storage_capacity_units_milp_constraint_rule(backend_model, node, tech):
    """
    Set storage capacity decision variable as a function of purchased units

    .. container:: scrolling-wrapper

        .. math::

            \\boldsymbol{storage_{cap}}(loc::tech) =
            \\boldsymbol{units}(loc::tech) \\times storage_{cap, per unit}(loc::tech)
            \\quad \\forall loc::tech \\in loc::techs_{milp, store}

    """
    return backend_model.storage_cap[node, tech] == (
        backend_model.units[node, tech]
        * get_param(backend_model, "storage_cap_per_unit", (node, tech))
    )


def energy_capacity_max_purchase_milp_constraint_rule(backend_model, node, tech):
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
    energy_cap_max = get_param(backend_model, "energy_cap_max", (node, tech))
    energy_cap_equals = get_param(backend_model, "energy_cap_equals", (node, tech))
    energy_cap_scale = get_param(backend_model, "energy_cap_scale", (node, tech))

    if po.value(energy_cap_equals):
        return backend_model.energy_cap[node, tech] == (
            energy_cap_equals * energy_cap_scale * backend_model.purchased[node, tech]
        )

    else:
        return backend_model.energy_cap[node, tech] <= (
            energy_cap_max * energy_cap_scale * backend_model.purchased[node, tech]
        )


def energy_capacity_min_purchase_milp_constraint_rule(backend_model, node, tech):
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
    energy_cap_min = get_param(backend_model, "energy_cap_min", (node, tech))

    energy_cap_scale = get_param(backend_model, "energy_cap_scale", (node, tech))
    return backend_model.energy_cap[node, tech] >= (
        energy_cap_min * energy_cap_scale * backend_model.purchased[node, tech]
    )


def storage_capacity_max_purchase_milp_constraint_rule(backend_model, node, tech):
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
    storage_cap_max = get_param(backend_model, "storage_cap_max", (node, tech))
    storage_cap_equals = get_param(backend_model, "storage_cap_equals", (node, tech))

    if po.value(storage_cap_equals):
        return backend_model.storage_cap[node, tech] == (
            storage_cap_equals * backend_model.purchased[node, tech]
        )

    elif po.value(storage_cap_max):
        return backend_model.storage_cap[node, tech] <= (
            storage_cap_max * backend_model.purchased[node, tech]
        )

    else:
        return po.Constraint.Skip


def storage_capacity_min_purchase_milp_constraint_rule(backend_model, node, tech):
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
    storage_cap_min = get_param(backend_model, "storage_cap_min", (node, tech))

    if po.value(storage_cap_min):
        return backend_model.storage_cap[node, tech] >= (
            storage_cap_min * backend_model.purchased[node, tech]
        )

    else:
        return po.Constraint.Skip


def unit_capacity_systemwide_milp_constraint_rule(backend_model, tech):
    """
    Set constraints to limit the number of purchased units of a single technology
    type across all locations in the model.

    The first valid case is applied:

    .. container:: scrolling-wrapper

        .. math::

            \\sum_{loc}\\boldsymbol{units}(loc::tech) + \\boldsymbol{purchased}(loc::tech)
            \\begin{cases}
                = units_{equals, systemwide}(tech),&
                    \\text{if } units_{equals, systemwide}(tech)\\\\
                \\leq units_{max, systemwide}(tech),&
                    \\text{if } units_{max, systemwide}(tech)\\\\
                \\text{unconstrained},& \\text{otherwise}
            \\end{cases}
            \\forall tech \\in techs

    """

    max_systemwide = get_param(backend_model, "units_max_systemwide", tech)
    equals_systemwide = get_param(backend_model, "units_equals_systemwide", tech)

    def _sum(var_name):
        if hasattr(backend_model, var_name):
            return po.quicksum(
                getattr(backend_model, var_name)[node, tech]
                for node in backend_model.nodes
                if [node, tech] in getattr(backend_model, var_name)._index
            )
        else:
            return 0

    sum_expr_units = _sum("units")
    sum_expr_purchase = _sum("purchased")

    if equals_systemwide:
        return sum_expr_units + sum_expr_purchase == equals_systemwide
    else:
        return sum_expr_units + sum_expr_purchase <= max_systemwide


def asynchronous_con_milp_constraint_rule(backend_model, node, tech, timestep):
    """
    BigM limit set on `carrier_con`, forcing it to either be zero or non-zero,
    depending on whether `con` is zero or one, respectively.

    .. container:: scrolling-wrapper

        .. math::
            - \\boldsymbol{carrier_con}[loc::tech::carrier, timestep] \\leq
            \\text{bigM} \\times (1 - \\boldsymbol{prod_con_switch}[loc::tech, timestep])
            \\forall loc::tech \\in loc::techs_{asynchronous_prod_con},
            \\forall timestep \\in timesteps

    """

    def _sum(var_name):
        return po.quicksum(
            getattr(backend_model, var_name)[carrier, node, tech, timestep]
            for carrier in backend_model.carriers
            if [carrier, node, tech, timestep]
            in getattr(backend_model, var_name)._index
        )

    return (
        -1 * _sum("carrier_con")
        <= (1 - backend_model.prod_con_switch[node, tech, timestep])
        * backend_model.bigM
    )


def asynchronous_prod_milp_constraint_rule(backend_model, node, tech, timestep):
    """
    BigM limit set on `carrier_prod`, forcing it to either be zero or non-zero,
    depending on whether `prod` is zero or one, respectively.

    .. container:: scrolling-wrapper

        .. math::
            \\boldsymbol{carrier_prod}[loc::tech::carrier, timestep] \\leq
            \\text{bigM} \\times \\boldsymbol{prod_con_switch}[loc::tech, timestep]
            \\forall loc::tech \\in loc::techs_{asynchronous_prod_con},
            \\forall timestep \\in timesteps

    """

    def _sum(var_name):
        return po.quicksum(
            getattr(backend_model, var_name)[carrier, node, tech, timestep]
            for carrier in backend_model.carriers
            if [carrier, node, tech, timestep]
            in getattr(backend_model, var_name)._index
        )

    return (
        _sum("carrier_prod")
        <= backend_model.prod_con_switch[node, tech, timestep] * backend_model.bigM
    )
