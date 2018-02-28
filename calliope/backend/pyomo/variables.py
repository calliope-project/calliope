"""
Copyright (C) 2013-2017 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

"""

import pyomo.core as po  # pylint: disable=import-error
import numpy as np

def initialize_decision_variables(backend_model):
    """
    Defines variables

    """
    model_data_dict = backend_model.__calliope_model_data__

    # In operational mode, capacities are fixed, they're not decision variables
    # so we set these Vars to Params. Pyomo equations in constraints are all still valid
    if backend_model.mode == 'operate':
        Var_or_Param = po.Param
    else:
        Var_or_Param = po.Var

    # We can initialise values in both Var and Param. In Var, it provides a starting
    # value, which it's later free to move from. In Param, it sets the value and sticks with it
    def cap_initializer(cap_var_name, within):
        keyword_args = dict()
        if backend_model.mode != 'operate':
            keyword_args['within'] = within
        else:
            keyword_args['mutable'] = True
            keyword_args['default'] = np.inf
        initialize_data = {}
        if hasattr(backend_model, cap_var_name + '_max'):
            initialize_data.update(model_data_dict['data'][cap_var_name + '_max'])
        # '_equals' overrides '_max' in operational mode
        if hasattr(backend_model, cap_var_name + '_equals'):
            initialize_data.update(model_data_dict['data'][cap_var_name + '_equals'])

        if initialize_data:
            keyword_args['initialize'] = initialize_data

        return keyword_args

    ##
    # Variables which are always assigned
    ##

    backend_model.energy_cap = Var_or_Param(backend_model.loc_techs, **cap_initializer('energy_cap', within=po.NonNegativeReals))
    backend_model.carrier_prod = po.Var(backend_model.loc_tech_carriers_prod, backend_model.timesteps, within=po.NonNegativeReals)
    backend_model.carrier_con = po.Var(backend_model.loc_tech_carriers_con, backend_model.timesteps, within=po.NegativeReals)
    backend_model.cost = po.Var(backend_model.costs, backend_model.loc_techs_cost, within=po.Reals)

    ##
    # Conditionally assigned variables
    ##

    if 'loc_techs_area' in model_data_dict['sets']:
        backend_model.resource_area = Var_or_Param(backend_model.loc_techs_area, **cap_initializer('resource_area', within=po.NonNegativeReals))

    if 'loc_techs_store' in model_data_dict['sets']:
        backend_model.storage_cap = Var_or_Param(backend_model.loc_techs_store, **cap_initializer('storage_cap', within=po.NonNegativeReals))
        backend_model.storage = po.Var(backend_model.loc_techs_store, backend_model.timesteps, within=po.NonNegativeReals)

    if 'loc_techs_finite_resource' in model_data_dict['sets']:
        backend_model.resource_con = po.Var(backend_model.loc_techs_finite_resource, backend_model.timesteps, within=po.Reals)

    if 'loc_techs_finite_resource_supply_plus' in model_data_dict['sets']:
        backend_model.resource_cap = Var_or_Param(backend_model.loc_techs_finite_resource_supply_plus, **cap_initializer('resource_cap', within=po.NonNegativeReals))

    if 'loc_techs_export' in model_data_dict['sets']:
        backend_model.carrier_export = po.Var(backend_model.loc_tech_carriers_export, backend_model.timesteps, within=po.NonNegativeReals)

    if 'loc_techs_om_cost' in model_data_dict['sets']:
        backend_model.cost_var = po.Var(backend_model.costs, backend_model.loc_techs_om_cost, backend_model.timesteps, within=po.Reals)

    if 'loc_techs_investment_cost' in model_data_dict['sets'] and backend_model.mode != 'operate':
        backend_model.cost_investment = po.Var(backend_model.costs, backend_model.loc_techs_investment_cost, within=po.Reals)

    if 'loc_techs_purchase' in model_data_dict['sets']:
        backend_model.purchased = Var_or_Param(backend_model.loc_techs_purchase, within=po.Binary, initialize=1)

    if 'loc_techs_milp' in model_data_dict['sets']:
        backend_model.units = Var_or_Param(backend_model.loc_techs_milp, **cap_initializer('units', within=po.NonNegativeIntegers))
        backend_model.operating_units = po.Var(backend_model.loc_techs_milp, backend_model.timesteps, within=po.NonNegativeIntegers)
        # For any milp tech, we need to update energy_cap, as energy_cap_max and energy_cap_equals
        # are replaced by energy_cap_per_unit
        if backend_model.mode == 'operate':
            for k, v in backend_model.units.items():
                backend_model.energy_cap[k] = v * backend_model.energy_cap_per_unit[k]