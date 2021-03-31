"""
Copyright (C) since 2013 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

costs.py
~~~~~~~~

Cost constraints.

"""

import pyomo.core as po

from calliope.backend.pyomo.util import get_param, get_timestep_weight, loc_tech_is_in


def cost_expression_rule(backend_model, cost, node, tech):
    """
    Combine investment and time varying costs into one cost per technology.

    .. container:: scrolling-wrapper

        .. math::

            \\boldsymbol{cost}(cost, loc::tech) = \\boldsymbol{cost_{investment}}(cost, loc::tech)
            + \\sum_{timestep \\in timesteps} \\boldsymbol{cost_{var}}(cost, loc::tech, timestep)

    """

    if loc_tech_is_in(backend_model, (cost, node, tech), "cost_investment_index"):
        cost_investment = backend_model.cost_investment[cost, node, tech]
    else:
        cost_investment = 0

    if hasattr(backend_model, "cost_var"):
        cost_var = po.quicksum(
            backend_model.cost_var[cost, node, tech, timestep]
            for timestep in backend_model.timesteps
            if [cost, node, tech, timestep] in backend_model.cost_var._index
        )
    else:
        cost_var = 0

    return cost_investment + cost_var


def cost_investment_expression_rule(backend_model, cost, node, tech):
    """
    Calculate costs from capacity decision variables.

    Transmission technologies "exist" at two locations, so their cost
    is divided by 2.

    .. container:: scrolling-wrapper

        .. math::

            \\boldsymbol{cost_{investment}}(cost, loc::tech) =
            cost_{fractional\\_om}(cost, loc::tech) +
            cost_{fixed\\_om}(cost, loc::tech) + cost_{cap}(cost, loc::tech)

            cost_{cap}(cost, loc::tech) =
            depreciation\\_rate * ts\\_weight *
            (cost_{energy\\_cap}(cost, loc::tech) \\times \\boldsymbol{energy_{cap}}(loc::tech)
            + cost_{storage\\_cap}(cost, loc::tech) \\times \\boldsymbol{storage_{cap}}(loc::tech)
            + cost_{resource\\_cap}(cost, loc::tech) \\times \\boldsymbol{resource_{cap}}(loc::tech)
            + cost_{resource\\_area}(cost, loc::tech)) \\times \\boldsymbol{resource_{area}}(loc::tech)

            depreciation\\_rate =
            \\begin{cases}
                = 1 / plant\\_life,&
                    \\text{if } interest\\_rate = 0\\\\
                = \\frac{interest\\_rate \\times (1 + interest\\_rate)^{plant\\_life}}{(1 + interest\\_rate)^{plant\\_life} - 1},&
                    \\text{if } interest\\_rate \\gt 0\\\\
            \\end{cases}

            ts\\_weight = \\sum_{timestep \\in timesteps} (time\\_res(timestep) \\times weight(timestep)) \\times \\frac{1}{8760}

    """

    def _get_investment_cost(capacity_decision_variable):
        """
        Conditionally add investment costs, if the relevant set of technologies
        exists. Both inputs are strings.
        """
        if loc_tech_is_in(
            backend_model, (node, tech), capacity_decision_variable + "_index"
        ):
            _cost = getattr(backend_model, capacity_decision_variable)[
                node, tech
            ] * get_param(
                backend_model, "cost_" + capacity_decision_variable, (cost, node, tech)
            )
            return _cost
        else:
            return 0

    cost_energy_cap = backend_model.energy_cap[node, tech] * get_param(
        backend_model, "cost_energy_cap", (cost, node, tech)
    )

    cost_storage_cap = _get_investment_cost("storage_cap")
    cost_resource_cap = _get_investment_cost("resource_cap")
    cost_resource_area = _get_investment_cost("resource_area")

    cost_om_annual_investment_fraction = get_param(
        backend_model, "cost_om_annual_investment_fraction", (cost, node, tech)
    )
    cost_om_annual = get_param(backend_model, "cost_om_annual", (cost, node, tech))

    if loc_tech_is_in(backend_model, (node, tech), "units_index"):
        cost_of_purchase = (
            get_param(backend_model, "cost_purchase", (cost, node, tech))
            * backend_model.units[node, tech]
        )
    elif loc_tech_is_in(backend_model, (node, tech), "purchased_index"):
        cost_of_purchase = (
            get_param(backend_model, "cost_purchase", (cost, node, tech))
            * backend_model.purchased[node, tech]
        )
    else:
        cost_of_purchase = 0

    ts_weight = get_timestep_weight(backend_model)
    depreciation_rate = get_param(
        backend_model, "cost_depreciation_rate", (cost, node, tech)
    )

    cost_cap = (
        depreciation_rate
        * ts_weight
        * (
            cost_energy_cap
            + cost_storage_cap
            + cost_resource_cap
            + cost_resource_area
            + cost_of_purchase
        )
    )

    # Transmission technologies exist at two locations, thus their cost is divided by 2
    if backend_model.inheritance[tech].value.endswith("transmission"):
        cost_cap = cost_cap / 2

    cost_fractional_om = cost_om_annual_investment_fraction * cost_cap
    cost_fixed_om = cost_om_annual * backend_model.energy_cap[node, tech] * ts_weight

    return cost_fractional_om + cost_fixed_om + cost_cap


def cost_var_expression_rule(backend_model, cost, node, tech, timestep):
    """
    Calculate costs from time-varying decision variables

    .. container:: scrolling-wrapper

        .. math::

            \\boldsymbol{cost_{var}}(cost, loc::tech, timestep) = cost_{prod}(cost, loc::tech, timestep) + cost_{con}(cost, loc::tech, timestep)

            cost_{prod}(cost, loc::tech, timestep) = cost_{om\\_prod}(cost, loc::tech, timestep) \\times weight(timestep) \\times \\boldsymbol{carrier_{prod}}(loc::tech::carrier, timestep)

            prod\\_con\\_eff =
            \\begin{cases}
                = \\boldsymbol{resource_{con}}(loc::tech, timestep),&
                    \\text{if } loc::tech \\in loc\\_techs\\_supply\\_plus \\\\
                = \\frac{\\boldsymbol{carrier_{prod}}(loc::tech::carrier, timestep)}{energy_eff(loc::tech, timestep)},&
                    \\text{if } loc::tech \\in loc\\_techs\\_supply \\\\
            \\end{cases}

            cost_{con}(cost, loc::tech, timestep) = cost_{om\\_con}(cost, loc::tech, timestep) \\times weight(timestep) \\times prod\\_con\\_eff

    """

    weight = backend_model.timestep_weights[timestep]

    all_costs = []

    def _sum(var_name, carriers=backend_model.carriers):
        return po.quicksum(
            getattr(backend_model, var_name)[carrier, node, tech, timestep]
            for carrier in carriers
            if [carrier, node, tech, timestep]
            in getattr(backend_model, f"{var_name}_index")
        )

    cost_om_prod = get_param(
        backend_model, "cost_om_prod", (cost, node, tech, timestep)
    )
    if backend_model.inheritance[tech].value.endswith("conversion_plus"):
        carriers = [backend_model.primary_carrier_out[:, tech].index()[0][0]]
        all_costs.append(cost_om_prod * _sum("carrier_prod", carriers=carriers))
    else:
        all_costs.append(cost_om_prod * _sum("carrier_prod"))

    cost_om_con = get_param(backend_model, "cost_om_con", (cost, node, tech, timestep))
    if cost_om_con:
        if loc_tech_is_in(backend_model, (node, tech), "resource_con_index"):
            all_costs.append(
                cost_om_con * backend_model.resource_con[node, tech, timestep]
            )
        elif backend_model.inheritance[tech].value.endswith("supply"):
            energy_eff = get_param(backend_model, "energy_eff", (node, tech, timestep))
            # in case energy_eff is zero, to avoid an infinite value
            if po.value(energy_eff) > 0:
                all_costs.append(cost_om_con * (_sum("carrier_prod") / energy_eff))
        elif backend_model.inheritance[tech].value.endswith("conversion_plus"):
            carriers = [backend_model.primary_carrier_in[:, tech].index()[0][0]]
            all_costs.append(
                cost_om_con * (-1) * _sum("carrier_con", carriers=carriers)
            )
        else:
            all_costs.append(cost_om_con * (-1) * _sum("carrier_con"))
    if hasattr(backend_model, "export_carrier"):
        export_carrier = backend_model.export_carrier[:, node, tech].index()
    else:
        export_carrier = []
    if len(export_carrier) > 0:
        all_costs.append(
            get_param(backend_model, "cost_export", (cost, node, tech, timestep))
            * backend_model.carrier_export[export_carrier[0][0], node, tech, timestep]
        )

    return po.quicksum(all_costs) * weight
