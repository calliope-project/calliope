"""
Copyright (C) since 2013 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

costs.py
~~~~~~~~

Cost constraints.

"""

import pyomo.core as po  # pylint: disable=import-error

from calliope.backend.pyomo.util import (
    get_param,
    get_timestep_weight,
    loc_tech_is_in,
    invalid,
)

ORDER = 10  # order in which to invoke constraints relative to other constraint files


def load_constraints(backend_model):
    sets = backend_model.__calliope_model_data["sets"]
    run_config = backend_model.__calliope_run_config

    if "loc_techs_cost_constraint" in sets:
        backend_model.cost_constraint = po.Constraint(
            backend_model.costs, backend_model.loc_techs_cost, rule=cost_constraint_rule
        )

    # FIXME: remove check for operate from constraint files, avoid investment costs more intelligently?
    if (
        "loc_techs_cost_investment_constraint" in sets
        and run_config["mode"] != "operate"
    ):
        # Right-hand side expression can be later updated by MILP investment costs
        backend_model.cost_investment_rhs = po.Expression(
            backend_model.costs,
            backend_model.loc_techs_cost_investment_constraint,
            initialize=0.0,
        )

        backend_model.cost_investment_constraint = po.Constraint(
            backend_model.costs,
            backend_model.loc_techs_cost_investment_constraint,
            rule=cost_investment_constraint_rule,
        )

    if "loc_techs_om_cost" in sets:
        # Right-hand side expression can be later updated by export costs/revenue
        backend_model.cost_var_rhs = po.Expression(
            backend_model.costs,
            backend_model.loc_techs_om_cost,
            backend_model.timesteps,
            initialize=0.0,
        )
    if "loc_techs_cost_var_constraint" in sets:
        # Constraint is built over a different loc_techs set to expression, as
        # it is updated in conversion.py and conversion_plus.py constraints
        backend_model.cost_var_constraint = po.Constraint(
            backend_model.costs,
            backend_model.loc_techs_cost_var_constraint,
            backend_model.timesteps,
            rule=cost_var_constraint_rule,
        )


def cost_constraint_rule(backend_model, cost, loc_tech):
    """
    Combine investment and time varying costs into one cost per technology.

    .. container:: scrolling-wrapper

        .. math::

            \\boldsymbol{cost}(cost, loc::tech) = \\boldsymbol{cost_{investment}}(cost, loc::tech)
            + \\sum_{timestep \\in timesteps} \\boldsymbol{cost_{var}}(cost, loc::tech, timestep)

    """
    run_config = backend_model.__calliope_run_config

    # FIXME: remove check for operate from constraint files, avoid investment costs more intelligently?
    if (
        loc_tech_is_in(backend_model, loc_tech, "loc_techs_investment_cost")
        and run_config["mode"] != "operate"
    ):
        cost_investment = backend_model.cost_investment[cost, loc_tech]
    else:
        cost_investment = 0

    if loc_tech_is_in(backend_model, loc_tech, "loc_techs_om_cost"):
        cost_var = sum(
            backend_model.cost_var[cost, loc_tech, timestep]
            for timestep in backend_model.timesteps
        )
    else:
        cost_var = 0

    return backend_model.cost[cost, loc_tech] == cost_investment + cost_var


def cost_investment_constraint_rule(backend_model, cost, loc_tech):
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
    model_data_dict = backend_model.__calliope_model_data

    def _get_investment_cost(capacity_decision_variable, calliope_set):
        """
        Conditionally add investment costs, if the relevant set of technologies
        exists. Both inputs are strings.
        """
        if loc_tech_is_in(backend_model, loc_tech, calliope_set):
            _cost = getattr(backend_model, capacity_decision_variable)[
                loc_tech
            ] * get_param(
                backend_model, "cost_" + capacity_decision_variable, (cost, loc_tech)
            )
            return _cost
        else:
            return 0

    cost_energy_cap = backend_model.energy_cap[loc_tech] * get_param(
        backend_model, "cost_energy_cap", (cost, loc_tech)
    )

    cost_storage_cap = _get_investment_cost("storage_cap", "loc_techs_store")
    cost_resource_cap = _get_investment_cost("resource_cap", "loc_techs_supply_plus")
    cost_resource_area = _get_investment_cost("resource_area", "loc_techs_area")

    cost_om_annual_investment_fraction = get_param(
        backend_model, "cost_om_annual_investment_fraction", (cost, loc_tech)
    )
    cost_om_annual = get_param(backend_model, "cost_om_annual", (cost, loc_tech))

    ts_weight = get_timestep_weight(backend_model)
    depreciation_rate = model_data_dict["data"]["cost_depreciation_rate"].get(
        (cost, loc_tech), 0
    )

    cost_cap = (
        depreciation_rate
        * ts_weight
        * (cost_energy_cap + cost_storage_cap + cost_resource_cap + cost_resource_area)
    )

    # Transmission technologies exist at two locations, thus their cost is divided by 2
    if loc_tech_is_in(backend_model, loc_tech, "loc_techs_transmission"):
        cost_cap = cost_cap / 2

    cost_fractional_om = cost_om_annual_investment_fraction * cost_cap
    cost_fixed_om = cost_om_annual * backend_model.energy_cap[loc_tech] * ts_weight

    backend_model.cost_investment_rhs[cost, loc_tech].expr = (
        cost_fractional_om + cost_fixed_om + cost_cap
    )

    return (
        backend_model.cost_investment[cost, loc_tech]
        == backend_model.cost_investment_rhs[cost, loc_tech]
    )


def cost_var_constraint_rule(backend_model, cost, loc_tech, timestep):
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
    model_data_dict = backend_model.__calliope_model_data

    cost_om_prod = get_param(backend_model, "cost_om_prod", (cost, loc_tech, timestep))
    cost_om_con = get_param(backend_model, "cost_om_con", (cost, loc_tech, timestep))
    weight = backend_model.timestep_weights[timestep]

    loc_tech_carrier = model_data_dict["data"]["lookup_loc_techs"][loc_tech]

    if loc_tech_is_in(
        backend_model, loc_tech_carrier, "loc_tech_carriers_prod"
    ) and not invalid(cost_om_prod):
        cost_prod = (
            cost_om_prod
            * weight
            * backend_model.carrier_prod[loc_tech_carrier, timestep]
        )
    else:
        cost_prod = 0

    cost_con = 0
    if not invalid(cost_om_con):
        if loc_tech_is_in(backend_model, loc_tech, "loc_techs_supply_plus"):
            cost_con = (
                cost_om_con * weight * backend_model.resource_con[loc_tech, timestep]
            )
        elif loc_tech_is_in(backend_model, loc_tech, "loc_techs_supply"):
            energy_eff = get_param(backend_model, "energy_eff", (loc_tech, timestep))
            # in case energy_eff is zero, to avoid an infinite value
            if po.value(energy_eff) > 0:
                cost_con = (
                    cost_om_con
                    * weight
                    * (
                        backend_model.carrier_prod[loc_tech_carrier, timestep]
                        / energy_eff
                    )
                )
        elif loc_tech_is_in(backend_model, loc_tech, "loc_techs_demand"):
            cost_con = (
                cost_om_con
                * weight
                * (-1)
                * backend_model.carrier_con[loc_tech_carrier, timestep]
            )

    backend_model.cost_var_rhs[cost, loc_tech, timestep].expr = cost_prod + cost_con
    return (
        backend_model.cost_var[cost, loc_tech, timestep]
        == backend_model.cost_var_rhs[cost, loc_tech, timestep]
    )
