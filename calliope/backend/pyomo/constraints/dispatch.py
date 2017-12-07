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
    model_data_dict = backend_model.__calliope_model_data__

    if 'loc_techs_supply_plus' in model_data_dict:
        backend_model.resource_max_constraint = po.Constraint(
            backend_model.loc_techs_supply_plus, backend_model.timestep
            rule=resource_max_constraint_rule
        )

def resource_max_constraint_rule(backend_model, loc_tech, timestep):
    """
    Set maximum resource supply. Supply_plus techs only.
    """
    time_resolution = param_getter(backend_model, 'time_resolution', (timestep))

    return backend_model.resource[loc_tech, timestep] <= (
        time_resolution * backend_model.resource_cap[loc_tech])


def carrier_prod_max_rule(backend_model, loc_tech_carrier, timestep):
    """
    Set maximum carrier production. All technologies.
    """
    model_data_dict = backend_model.__calliope_model_data__['sets']
    carrier_prod = backend_model.carrier_prod[loc_tech_carrier, timestep]
    time_resolution = param_getter(backend_model, 'time_resolution', (timestep))
    parasitic_eff = param_getter(backend_model, 'parasitic_eff', (loc_tech, timestep))
    #if ('loc_tech_conversion_plus' in model_dict
    #     and loc_tech in backend_model.loc_tech_conversion_plus):
    #    carriers_out = model.get_carrier(y, 'out', all_carriers=True)
    #    if isinstance(carriers_out, str):
    #        carriers_out = tuple([carriers_out])
    #    if (c not in carriers_out) or (c in carriers_out and
    #                                   model._locations.at[x, y] == 0):
    #        return c_prod == 0
    #    else:
    #        return po.Constraint.Skip

    if 'loc_tech_milp' in model_data_dict and loc_tech in backend_model.loc_tech_milp:
        energy_cap = param_getter(backend_model, 'energy_cap_per_unit', (loc_tech))
        return carrier_prod <= (
            backend_model.operating_units[loc_tech, timestep] *
            time_resolution * energy_cap * parasitic_eff
        )
    else:
        return carrier_prod <= (
            backend_model.energy_cap[loc_tech] * time_resolution * parasitic_eff
        )