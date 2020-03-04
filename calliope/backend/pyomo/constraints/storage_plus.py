"""
Copyright (C) 2013-2019 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

storage_plus.py
~~~~~~~~~~~~~~~~~

Storage plus technology constraints.

"""
'loc_techs_storage_plus_max_constraint'
'loc_techs_storage_plus_discharge_depth_constraint'
'loc_techs_storage_plus_balance_constraint'
'loc_techs_storage_plus_storage_time_constraint'
'loc_techs_storage_plus_shared_cap_per_time_constraint'
'loc_techs_storage_plus_shared_storage_constraint'

import pyomo.core as po
import numpy as np
import pandas as pd

from calliope.backend.pyomo.util import \
    get_param, \
    split_comma_list, \
    get_previous_timestep
from calliope import exceptions

ORDER = 20  # order in which to invoke constraints relative to other constraint files


def load_constraints(backend_model):
    sets = backend_model.__calliope_model_data['sets']

    if 'loc_techs_storage_plus_max_constraint' in sets:
        backend_model.storage_plus_max_constraint = po.Constraint(
            backend_model.loc_techs_storage_plus_max_constraint,
            backend_model.timesteps,
            rule=storage_plus_max_constraint_rule
        )

    if 'loc_techs_storage_plus_discharge_depth_constraint' in sets:
        backend_model.storage_plus_discharge_depth_constraint = po.Constraint(
            backend_model.loc_techs_storage_plus_discharge_depth_constraint,
            backend_model.timesteps,
            rule=storage_plus_discharge_depth_constraint_rule
        )

    if 'loc_techs_storage_plus_balance_constraint' in sets:
        backend_model.storage_plus_balance_constraint = po.Constraint(
            backend_model.loc_techs_storage_plus_balance_constraint,
            backend_model.timesteps,
            rule=storage_plus_balance_constraint_rule
        )

    if 'loc_techs_storage_plus_storage_time_constraint' in sets:
        backend_model.storage_plus_time_constraint = po.Constraint(
            backend_model.loc_techs_storage_plus_storage_time_constraint,
            backend_model.timesteps,
            rule=storage_plus_time_constraint_rule
        )

    if 'loc_techs_storage_plus_shared_storage_constraint' in sets:
        backend_model.storage_plus_shared_storage_constraint = po.Constraint(
            backend_model.loc_techs_storage_plus_shared_storage_constraint,
            backend_model.timesteps,
            rule=storage_plus_shared_cap_constraint_rule
        )


def storage_plus_max_constraint_rule(backend_model, loc_tech, timestep):
# Storage level cannot exceed the storage cap equals value
    return backend_model.storage[loc_tech,timestep] <= backend_model.storage_cap_equals_per_timestep[loc_tech, timestep]

def storage_plus_discharge_depth_constraint_rule(backend_model, loc_tech, timestep):
# Storage level must be greater than or equal to the discharge depth
    try:# loc_tech in backend_model.loc_techs_storage_plus_cap_per_time: # this set might not exist!
        storage_cap = backend_model.storage_cap_equals_per_timestep[loc_tech, timestep]
    except:
        storage_cap = backend_model.storage_cap[loc_tech]

    try: # loc_tech in backend_model.loc_techs_storage_plus_discharge_depth_per_time:
        sdd = backend_model.storage_discharge_depth_per_timestep[loc_tech, timestep]
    except:
        sdd = get_param(backend_model, 'storage_discharge_depth', loc_tech)

    return backend_model.storage[loc_tech, timestep] >= sdd * storage_cap

def storage_plus_balance_constraint_rule(backend_model, loc_tech, timestep):
# Sum of carriers in and carriers out minus losses plus storage in previous timestep is new storage
    run_config = backend_model.__calliope_run_config
    model_data_dict = backend_model.__calliope_model_data['data']

    # Read in loc_carriers_in and _out from storage plus lookup
    loc_tech_carriers_out = split_comma_list(
        model_data_dict['lookup_loc_techs_storage_plus']['out', loc_tech]
    )
    loc_tech_carriers_in = split_comma_list(
        model_data_dict['lookup_loc_techs_storage_plus']['in', loc_tech]
    )
    energy_eff = get_param(backend_model, 'energy_eff', (loc_tech, timestep))

    # Carrier_con is equal to sum of carrier_con * energy_eff * carrier_ratio for all carriers in
    carrier_con = sum(
        backend_model.carrier_con[loc_tech_carrier, timestep] * energy_eff *
        get_param(backend_model, 'carrier_ratios', ('in', loc_tech_carrier, timestep))
        for loc_tech_carrier in loc_tech_carriers_in
    )

    # Check to see if sum of product of energy_eff and carrier_ratio for carriers out is zero - avoid div 0
    energy_eff_carrier_ratio_check = sum(
        po.value(energy_eff) * get_param(backend_model, 'carrier_ratios', ('out', loc_tech_carrier, timestep))
        for loc_tech_carrier in loc_tech_carriers_out
    )
    if energy_eff_carrier_ratio_check == 0:
        carrier_prod = 0
    else:
        carrier_prod = sum(
            backend_model.carrier_prod[loc_tech_carrier, timestep] / (energy_eff *
            get_param(backend_model, 'carrier_ratios', ('out', loc_tech_carrier, timestep)))
            for loc_tech_carrier in loc_tech_carriers_out
        )
    # Find current timestep and previous timestep
    current_timestep = backend_model.timesteps.order_dict[timestep]
    if current_timestep == 0 and not run_config['cyclic_storage']:
        storage_previous_step = (
            get_param(backend_model, 'storage_initial', loc_tech) *
            backend_model.storage_cap[loc_tech]
        )
    elif (hasattr(backend_model, 'storage_inter_cluster') and
            model_data_dict['lookup_cluster_first_timestep'][timestep]):
        storage_previous_step = 0
    else:
        if (hasattr(backend_model, 'clusters') and
                model_data_dict['lookup_cluster_first_timestep'][timestep]):
            previous_step = model_data_dict['lookup_cluster_last_timestep'][timestep]
        elif current_timestep == 0 and run_config['cyclic_storage']:
            previous_step = backend_model.timesteps[-1]
        else:
            previous_step = get_previous_timestep(backend_model.timesteps, timestep)
        storage_loss = get_param(backend_model, 'storage_loss', loc_tech)
        time_resolution = backend_model.timestep_resolution[previous_step]
        storage_previous_step = (
            ((1 - storage_loss) ** time_resolution) *
            backend_model.storage[loc_tech, previous_step]
        )

    return (backend_model.storage[loc_tech, timestep] == storage_previous_step - carrier_con - carrier_prod)


def storage_plus_time_constraint_rule(backend_model, loc_tech, timestep):

    # if you have set a storage min you have to say what the primary carrier is which this applies to
    # needs to be in terms of loc_tech_carriers

    model_data_dict = backend_model.__calliope_model_data['data']
    loc_tech_carriers_out = split_comma_list(
        model_data_dict['lookup_loc_techs_storage_plus']['out', loc_tech]
    )
    loc_tech_carrier_in = split_comma_list(
        model_data_dict['lookup_loc_techs_storage_plus']['in', loc_tech]
    )[0] # change this so it looks for primary carrier instead - should amount to the same thing
    carrier_prod = backend_model.carrier_prod[loc_tech_carrier_in, timestep]
    try: # loc_tech in backend_model.loc_techs_storage_time_per_timestep:
        try:
            contributing_times = split_comma_list(
                model_data_dict['lookup_storage_time'][(loc_tech, timestep)]
            )
        except(KeyError):
            contributing_times = []
    except:
        contributing_times = list(timestep + pd.to_timedelta(get_param(backend_model, 'storage_time', loc_tech), unit = 'h'))

    try:
        assert loc_tech in backend_model.loc_techs_storage_plus_discharge_depth_per_time #this needs fixing
        sdd_string = 'storage_discharge_depth_per_timestep'
    except:
        sdd_string = 'storage_discharge_depth'

    def storage_cap_finder(loc_tech, contributing_time):
        try: # loc_tech in backend_model.loc_techs_storage_plus_cap_per_time:
            storage_cap = backend_model.storage_cap_equals_per_timestep[loc_tech, pd.Timestamp(contributing_time)]
        except:
            storage_cap = backend_model.storage_cap[loc_tech]
        return storage_cap
    surplus_energy_previous = sum(
        backend_model.storage[loc_tech, pd.Timestamp(contributing_time)] - # previous timesteps storage level
        get_param(backend_model, sdd_string, (loc_tech, pd.Timestamp(contributing_time))) *
        storage_cap_finder(loc_tech, contributing_time)
        for contributing_time in contributing_times
    )

    return carrier_prod == surplus_energy_previous


def storage_plus_shared_cap_constraint_rule(backend_model, loc_tech, timestep):
    share_destination = get_param(backend_model, 'shared_storage_tech', loc_tech).value
    storage_cap_A = backend_model.storage_cap[loc_tech]
    try: # loc_tech in backend_model.loc_techs_storage_plus_cap_per_time:
        storage_cap_A = backend_model.storage_cap_equals_per_timestep[loc_tech, timestep]
    except:
        storage_cap_A = backend_model.storage_cap[loc_tech]

    try: # share_destination in backend_model.loc_techs_storage_plus_cap_per_time:
        storage_cap_B = backend_model.storage_cap_equals_per_timestep[share_destination, timestep]
    except:
        storage_cap_B = backend_model.storage_cap[share_destination]
    return storage_cap_A + backend_model.storage[share_destination, timestep] <= storage_cap_B
