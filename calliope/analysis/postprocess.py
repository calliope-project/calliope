"""
Copyright (C) 2013-2018 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

postprocess.py
~~~~~~~~~~~~~~

Functionality to post-process model results.

"""

import xarray as xr

from calliope.core.util.dataset import split_loc_techs


def postprocess_model_results(results, model_data):
    """
    Adds additional post-processed result variables to
    the given model results in-place. Model must have solved successfully.

    Returns None.

    """

    results['capacity_factor'] = capacity_factor(results)
    results['systemwide_capacity_factor'] = systemwide_capacity_factor(results, model_data)
    results['systemwide_levelised_cost'] = systemwide_levelised_cost(results)
    results['total_levelised_cost'] = systemwide_levelised_cost(results, total=True)

    return None


def capacity_factor(results):
    """
    Returns a DataArray with capacity factor for the given results,
    indexed by loc_tech_carriers_prod and timesteps.

    """
    capacities = xr.DataArray([
        results['energy_cap'].loc[dict(loc_techs=i.rsplit('::', 1)[0])].values
        for i in results['loc_tech_carriers_prod'].values
    ], dims=['loc_tech_carriers_prod'], coords={'loc_tech_carriers_prod': results['loc_tech_carriers_prod']})

    capacity_factors = (results['carrier_prod'] / capacities).fillna(0)

    return capacity_factors


def systemwide_capacity_factor(results, model_data):
    """
    Returns a DataArray with systemwide capacity factors for the given
    results, indexed by techs and carriers.

    """
    capacity_factors = (
        split_loc_techs(results['carrier_prod']).sum(dim='timesteps').sum(dim='locs') /
        (
            split_loc_techs(results['energy_cap']).sum(dim='locs') *
            sum(model_data.timestep_resolution)
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
