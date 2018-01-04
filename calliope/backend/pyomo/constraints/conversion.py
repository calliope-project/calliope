"""
Copyright (C) 2013-2017 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

conversion.py
~~~~~~~~~~~~~~~~~

Conversion technology constraints.

"""

import pyomo.core as po  # pylint: disable=import-error

from calliope.backend.pyomo.util import \
    get_param


def load_conversion_constraints(backend_model):
    sets = backend_model.__calliope_model_data__['sets']

    backend_model.balance_conversion_constraint = po.Constraint(
        backend_model.loc_techs_balance_conversion_constraint, backend_model.timesteps,
        rule=balance_conversion_constraint_rule
    )

    if 'loc_techs_cost_var_conversion_constraint' in sets:
        backend_model.cost_var_conversion_constraint = po.Constraint(
            backend_model.costs, backend_model.loc_techs_cost_var_conversion_constraint,
            backend_model.timesteps,
            rule=cost_var_conversion_constraint_rule
        )



def balance_conversion_constraint_rule(backend_model, loc_tech, timestep):
    """
    Balance energy carrier consumption and production
    """
    model_data_dict = backend_model.__calliope_model_data__['data']

    loc_tech_carrier_out = model_data_dict['lookup_loc_techs_conversion'][('out', loc_tech)]
    loc_tech_carrier_in = model_data_dict['lookup_loc_techs_conversion'][('in', loc_tech)]

    energy_eff = get_param(backend_model, 'energy_eff', (loc_tech, timestep))

    return (
        backend_model.carrier_prod[loc_tech_carrier_out, timestep] == -1 *
        backend_model.carrier_con[loc_tech_carrier_in, timestep] * energy_eff
    )


def cost_var_conversion_constraint_rule(backend_model, cost, loc_tech, timestep):
    """
    Add time-varying conversion technology costs
    """
    model_data_dict = backend_model.__calliope_model_data__
    weight = model_data_dict['data']['timestep_weights'][timestep]

    loc_tech_carrier_in = (
        model_data_dict['data']['lookup_loc_techs_conversion'][('in', loc_tech)]
    )

    loc_tech_carrier_out = (
        model_data_dict['data']['lookup_loc_techs_conversion'][('out', loc_tech)]
    )

    cost_om_prod = get_param(backend_model, 'cost_om_prod',
                                (cost, loc_tech, timestep))
    cost_om_con = get_param(backend_model, 'cost_om_con',
                            (cost, loc_tech, timestep))
    if po.value(cost_om_prod):
        cost_prod = (cost_om_prod * weight *
            backend_model.carrier_prod[loc_tech_carrier_out, timestep])
    else: cost_prod = 0

    if po.value(cost_om_con):
        cost_con = (cost_om_con * weight *
            backend_model.carrier_con[loc_tech_carrier_in, timestep])
    else: cost_con = 0

    backend_model.cost_var_rhs[cost, loc_tech, timestep] = cost_prod + cost_con

    return (backend_model.cost_var[cost, loc_tech, timestep] ==
            backend_model.cost_var_rhs[cost, loc_tech, timestep])
