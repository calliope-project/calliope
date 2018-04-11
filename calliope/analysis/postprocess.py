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
from calliope.core.util.logging import log_time


def postprocess_model_results(results, model_data, timings):
    """
    Adds additional post-processed result variables to
    the given model results in-place. Model must have solved successfully.

    Parameters
    ----------
    results : xarray Dataset
        Output from the solver backend
    model_data : xarray Dataset
        Calliope model data, stored as calliope.Model()._model_data
    timings : dict
        Calliope timing dictionary, stored as calliope.Model()._timings

    Returns
    -------
    results : xarray Dataset
        Input results Dataset, with additional DataArray variables and removed
        all instances of unreasonably low numbers (set by zero_threshold)

    """
    log_time(
        timings, 'post_process_start',
        comment='Postprocessing: started'
    )

    results['capacity_factor'] = capacity_factor(results, model_data)
    results['systemwide_capacity_factor'] = systemwide_capacity_factor(results, model_data)
    results['systemwide_levelised_cost'] = systemwide_levelised_cost(results, model_data)
    results['total_levelised_cost'] = systemwide_levelised_cost(results, model_data, total=True)
    results = clean_results(results, model_data.attrs.get('run.zero_threshold', 0), timings)

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

    capacities = xr.DataArray(
        [
            energy_cap.loc[dict(loc_techs=i.rsplit('::', 1)[0])].values
            for i in results['loc_tech_carriers_prod'].values
        ],
        dims=['loc_tech_carriers_prod'],
        coords={'loc_tech_carriers_prod': results['loc_tech_carriers_prod']}
    )

    capacity_factors = (results['carrier_prod'] / capacities).fillna(0)

    return capacity_factors


def systemwide_capacity_factor(results, model_data):
    """
    Returns a DataArray with systemwide capacity factors over the entire
    model duration, for the given results, indexed by techs and carriers.

    The weight of timesteps is considered when computing capacity factors,
    such that higher-weighted timesteps have a stronger influence
    on the resulting system-wide time-averaged capacity factor.

    """
    # In operate mode, energy_cap is an input parameter
    if 'energy_cap' not in results.keys():
        energy_cap = model_data.energy_cap
    else:
        energy_cap = results.energy_cap

    prod_sum = (
        # Aggregated/clustered days are represented `timestep_weights` times
        split_loc_techs(results['carrier_prod']) * model_data.timestep_weights
    ).sum(dim='timesteps').sum(dim='locs')
    cap_sum = split_loc_techs(energy_cap).sum(dim='locs')
    time_sum = (model_data.timestep_resolution * model_data.timestep_weights).sum()

    capacity_factors = prod_sum / (cap_sum * time_sum)

    return capacity_factors


def systemwide_levelised_cost(results, model_data, total=False):
    """
    Returns a DataArray with systemwide levelised costs for the given
    results, indexed by techs, carriers and costs if total is False,
    or by carriers and costs if total is True.

    The weight of timesteps is considered when computing levelised costs:

    * costs are already multiplied by weight in the constraints, and not
      further adjusted here.

    * production is not multiplied by weight in the contraints, so scaled
      by weight here to be consistent with costs. CAUTION: this scaling
      is temporary duriing levelised cost computation - the actual
      costs in the results remain untouched.

    Parameters
    ----------
    results : xarray.Dataset
        Model results
    model_data : xarray.Dataset
        Model input data
    total : bool, optional
        If False (default) returns per-technology levelised cost, if True,
        returns overall system-wide levelised cost.

    """
    cost = split_loc_techs(results['cost']).sum(dim='locs')
    carrier_prod = (
        # Here we scale production by timestep weight
        split_loc_techs(results['carrier_prod']) * model_data.timestep_weights
    ).sum(dim='timesteps').sum(dim='locs')

    if total:
        cost = cost.sum(dim='techs')
        carrier_prod = carrier_prod.sum(dim='techs')

    levelised_cost = []

    for carrier in carrier_prod['carriers'].values:
        levelised_cost.append(cost / carrier_prod.loc[dict(carriers=carrier)])

    return xr.concat(levelised_cost, dim='carriers')


def clean_results(results, zero_threshold, timings):
    """
    Remove unreasonably small values (solver output can lead to floating point
    errors) and remove unmet_demand if it was never used (i.e. sum = zero)

    zero_threshold is a value set in model configuration. If not set, defaults
    to zero (i.e. doesn't do anything). Reasonable value = 1e-12
    """
    threshold_applied = []
    for k, v in results.data_vars.items():
        # If there are any values in the data variable which fall below the
        # threshold, note the data variable name and set those values to zero
        if v.where(abs(v) < zero_threshold, drop=True).sum():
            threshold_applied.append(k)
            with np.errstate(invalid='ignore'):
                v.values[abs(v.values) < zero_threshold] = 0
            v.loc[{}] = v.values

    if threshold_applied:
        comment = 'All values < {} set to 0 in {}'.format(zero_threshold, ', '.join(threshold_applied))
    else:
        comment = 'zero threshold of {} not required'.format(zero_threshold)

    log_time(
        timings, 'threshold_applied',
        comment='Postprocessing: ' + comment
    )

    if 'unmet_demand' in results.data_vars.keys() and not results.unmet_demand.sum():

        log_time(
            timings, 'delete_unmet_demand',
            comment='Postprocessing: Model was feasible, deleting unmet_demand variable'
        )
        results = results.drop('unmet_demand')

    return results
