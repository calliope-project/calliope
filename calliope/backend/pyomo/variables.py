"""
Copyright (C) 2013-2017 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

"""

import pyomo.core as po  # pylint: disable=import-error


def initialize_decision_variables(backend_model):
    """
    Defines variables

    """
    model_data_dict = backend_model.__calliope_model_data__

    ##
    # Variables which are always assigned
    ##

    backend_model.energy_cap = po.Var(backend_model.loc_techs, within=po.NonNegativeReals)
    backend_model.carrier_prod = po.Var(backend_model.loc_tech_carriers_prod, backend_model.timesteps, within=po.NonNegativeReals)
    backend_model.carrier_con = po.Var(backend_model.loc_tech_carriers_con, backend_model.timesteps, within=po.NegativeReals)
    backend_model.cost = po.Var(backend_model.loc_techs_cost, backend_model.costs, within=po.Reals)

    ##
    # Conditionally assigned variables
    ##

    if 'loc_techs_area' in model_data_dict['sets']:
        backend_model.resource_area = po.Var(backend_model.loc_techs_area, within=po.NonNegativeReals)

    if 'loc_techs_store' in model_data_dict['sets']:
        backend_model.storage_cap = po.Var(backend_model.loc_techs_store, within=po.NonNegativeReals)
        backend_model.storage = po.Var(backend_model.loc_techs_store, backend_model.timesteps, within=po.NonNegativeReals)

    if 'loc_techs_finite_resource' in model_data_dict['sets']:
        backend_model.resource_cap = po.Var(backend_model.loc_techs_finite_resource, within=po.NonNegativeReals)
        backend_model.resource = po.Var(backend_model.loc_techs_finite_resource, backend_model.timesteps, within=po.Reals)

    if 'loc_techs_export' in model_data_dict['sets']:
        backend_model.carrier_export = po.Var(backend_model.loc_techs_export, backend_model.timesteps, within=po.NonNegativeReals)

    if 'loc_techs_om_cost' in model_data_dict['sets']:
        backend_model.cost_var = po.Var(backend_model.loc_techs_om_cost, backend_model.costs, backend_model.timesteps, within=po.Reals)

    if 'loc_techs_investment_cost' in model_data_dict['sets']:
        backend_model.cost_investment = po.Var(backend_model.loc_techs_investment_cost, backend_model.costs, within=po.Reals)

    if 'loc_techs_purchase' in model_data_dict['sets']:
        backend_model.purchased = po.Var(backend_model.loc_techs_purchase, within=po.Binary)

    if 'loc_techs_milp' in model_data_dict['sets']:
        backend_model.units = po.Var(backend_model.loc_techs_milp, within=po.NonNegativeIntegers)
        backend_model.operating_units = po.Var(backend_model.loc_techs_milp, within=po.NonNegativeIntegers)
