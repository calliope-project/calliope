"""
Copyright (C) 2013-2018 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

postprocess.py
~~~~~~~~~~~~~~

Functionality to post-process model results.

"""

import xarray as xr
import numpy as np

from calliope.core.util.dataset import split_loc_techs
from calliope.core.util.tools import log_time


def postprocess_model_results(results, model_data, timings):
    """
    Adds additional post-processed result variables to
    the given model results in-place. Model must have solved successfully.

    Returns None.

    """
    log_time(
        timings, 'post_process_start',
        comment='Postprocessing: started'
    )

    results['capacity_factor'] = capacity_factor(results, model_data)
    results['systemwide_capacity_factor'] = systemwide_capacity_factor(results, model_data)
    results['systemwide_levelised_cost'] = systemwide_levelised_cost(results)
    results['total_levelised_cost'] = systemwide_levelised_cost(results, total=True)
    results = clean_results(results, model_data.attrs.get('run.zero_thresshold', 0), timings)

    log_time(
        timings, 'post_process_end', time_since_start=True,
        comment='Postprocessing: ended'
    )

    return results


def capacity_factor(results, model_data):
    """
    Returns a DataArray with capacity factor for the given results,
    indexed by loc_tech_carriers_prod and timesteps.

    """
    # In operate mode, energy_cap is an input parameter
    if 'energy_cap' not in results.keys():
        energy_cap = model_data.energy_cap
    else:
        energy_cap = results.energy_cap

    capacities = xr.DataArray([
        energy_cap.loc[dict(loc_techs=i.rsplit('::', 1)[0])].values
        for i in results['loc_tech_carriers_prod'].values
    ], dims=['loc_tech_carriers_prod'], coords={'loc_tech_carriers_prod': results['loc_tech_carriers_prod']})

    capacity_factors = (results['carrier_prod'] / capacities).fillna(0)

    return capacity_factors


def systemwide_capacity_factor(results, model_data):
    """
    Returns a DataArray with systemwide capacity factors for the given
    results, indexed by techs and carriers.

    """
    # In operate mode, energy_cap is an input parameter
    if 'energy_cap' not in results.keys():
        energy_cap = model_data.energy_cap
    else:
        energy_cap = results.energy_cap

    capacity_factors = (
        split_loc_techs(results['carrier_prod']).sum(dim='timesteps').sum(dim='locs') /
        (
            split_loc_techs(energy_cap).sum(dim='locs') *
            model_data.timestep_resolution.sum()
        )
    )

    return capacity_factors


def systemwide_levelised_cost(results, total=False):
    """
    Returns a DataArray with systemwide levelised costs for the given
    results, indexed by techs, carriers and costs if total is False,
    or by carriers and costs if total is True.

    """
    cost = split_loc_techs(results['cost']).sum(dim='locs')
    carrier_prod = split_loc_techs(results['carrier_prod'].sum(dim='timesteps')).sum(dim='locs')

    if total:
        cost = cost.sum(dim='techs')
        carrier_prod = carrier_prod.sum(dim='techs')

    levelised_cost = []

    for carrier in carrier_prod['carriers'].values:
        levelised_cost.append(cost / carrier_prod.loc[dict(carriers=carrier)])

    return xr.concat(levelised_cost, dim='carriers')


def clean_results(results, zero_threshhold, timings):
    """
    Remove unreasonably small values (solver output can lead to floating point
    errors) and remove unmet_demand if it was never used (i.e. sum = zero)

    zero_thresshold is a value set in model configuration. If not set, defaults
    to zero (i.e. doesn't do anything). Reasonable value = 1e-12
    """
    thresshold_applied = []
    for k, v in results.data_vars.items():
        # If there are any values in the data variable which fall below the
        # thresshold, note the data variable name and set those values to zero
        if v.where(abs(v) < zero_threshhold, drop=True).sum():
            thresshold_applied.append(k)
            with np.errstate(invalid='ignore'):
                v.values[abs(v.values) < zero_threshhold] = 0
            v.loc[{}] = v.values

    if thresshold_applied:
        comment = 'All values < {} set to 0 in {}'.format(zero_threshhold, ', '.join(thresshold_applied))
    else:
        comment = 'zero thresshold of {} not required'.format(zero_threshhold)

    log_time(
        timings, 'thresshold_applied',
        comment='Postprocessing: ' + comment
    )

    if 'unmet_demand' in results.data_vars.keys() and not results.unmet_demand.sum():

        log_time(
            timings, 'delete_unmet_demand',
            comment='Postprocessing: Model was feasible, deleting unmet_demand variable'
        )
        results = results.drop('unmet_demand')

    return results
