"""
Copyright (C) 2013-2018 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

"""

import pyomo.core as po  # pylint: disable=import-error
import numpy as np


def initialize_decision_variables(backend_model):
    """
    Defines variables

    """
    model_data_dict = backend_model.__calliope_model_data__

    ##
    # Variables which are always assigned
    ##
    if backend_model.mode != 'operate':
        backend_model.energy_cap = po.Var(backend_model.loc_techs, within=po.NonNegativeReals)
    backend_model.carrier_prod = po.Var(backend_model.loc_tech_carriers_prod, backend_model.scenarios, backend_model.timesteps, within=po.NonNegativeReals)
    backend_model.carrier_con = po.Var(backend_model.loc_tech_carriers_con, backend_model.scenarios, backend_model.timesteps, within=po.NegativeReals)
    backend_model.cost = po.Var(backend_model.costs, backend_model.loc_techs_cost, backend_model.scenarios, within=po.Reals)

    ##
    # Conditionally assigned variables
    ##

    if 'loc_techs_area' in model_data_dict['sets'] and backend_model.mode != 'operate':
        backend_model.resource_area = po.Var(backend_model.loc_techs_area, within=po.NonNegativeReals)

    if 'loc_techs_store' in model_data_dict['sets']:
        if backend_model.mode != 'operate':
            backend_model.storage_cap = po.Var(backend_model.loc_techs_store, within=po.NonNegativeReals)
        backend_model.storage = po.Var(backend_model.loc_techs_store, backend_model.scenarios, backend_model.timesteps, within=po.NonNegativeReals)

    if 'loc_techs_supply_plus' in model_data_dict['sets']:
        backend_model.resource_con = po.Var(backend_model.loc_techs_supply_plus, backend_model.scenarios, backend_model.timesteps, within=po.Reals)
        if backend_model.mode != 'operate':
            backend_model.resource_cap = po.Var(backend_model.loc_techs_supply_plus, within=po.NonNegativeReals)

    if 'loc_techs_export' in model_data_dict['sets']:
        backend_model.carrier_export = po.Var(backend_model.loc_tech_carriers_export, backend_model.scenarios, backend_model.timesteps, within=po.NonNegativeReals)

    if 'loc_techs_om_cost' in model_data_dict['sets']:
        backend_model.cost_var = po.Var(backend_model.costs, backend_model.loc_techs_om_cost, backend_model.scenarios, backend_model.timesteps, within=po.Reals)

    if 'loc_techs_investment_cost' in model_data_dict['sets'] and backend_model.mode != 'operate':
        backend_model.cost_investment = po.Var(backend_model.costs, backend_model.loc_techs_investment_cost, within=po.Reals)

    if 'loc_techs_purchase' in model_data_dict['sets'] and backend_model.mode != 'operate':
        backend_model.purchased = po.Var(backend_model.loc_techs_purchase, within=po.Binary)

    if 'loc_techs_milp' in model_data_dict['sets']:
        if backend_model.mode != 'operate':
            backend_model.units = po.Var(backend_model.loc_techs_milp, within=po.NonNegativeIntegers)
        backend_model.operating_units = po.Var(backend_model.loc_techs_milp, backend_model.scenarios, backend_model.timesteps, within=po.NonNegativeIntegers)
        # For any milp tech, we need to update energy_cap, as energy_cap_max and energy_cap_equals
        # are replaced by energy_cap_per_unit
        if backend_model.mode == 'operate':
            for k, v in backend_model.units.items():
                backend_model.energy_cap[k] = v * backend_model.energy_cap_per_unit[k]

    if model_data_dict['attrs'].get('run.ensure_feasibility', False):
        backend_model.unmet_demand = po.Var(backend_model.loc_carriers, backend_model.scenarios, backend_model.timesteps, within=po.NonNegativeReals)
        backend_model.bigM = model_data_dict['attrs'].get('run.bigM')


def initialize_robust_decision_variables(backend_model):
    """
    Define decision variables specific to robust optimisation
    (function is triggered if user selects 'robust_plan' as the model mode)
    """
    # Load auxiliary decision variables used to calculate the CVaR
    backend_model.xi = po.Var(within=po.Reals)
    backend_model.eta = po.Var(backend_model.scenarios, within=po.NonNegativeReals)
