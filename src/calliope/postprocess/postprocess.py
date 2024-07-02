# Copyright (C) since 2013 Calliope contributors listed in AUTHORS.
# Licensed under the Apache 2.0 License (see LICENSE file).
"""Functionality to post-process model results."""

import logging

import numpy as np
import xarray as xr

LOGGER = logging.getLogger(__name__)


def postprocess_model_results(
    results: xr.Dataset, model_data: xr.Dataset
) -> xr.Dataset:
    """Post-processing of model results.

    Adds additional post-processed result variables to the given model results
    in-place. Model must have solved successfully.
    All instances of unreasonably low numbers (set by zero_threshold) will be removed.

    Args:
        results (xarray.Dataset): Output from the solver backend.
        model_data (xarray.Dataset): Calliope model data.

    Returns:
        xarray.Dataset: input-results dataset.
    """
    zero_threshold = model_data.config.solve.zero_threshold
    results["capacity_factor"] = capacity_factor(results, model_data)
    results["systemwide_capacity_factor"] = capacity_factor(
        results, model_data, systemwide=True
    )
    results["systemwide_levelised_cost"] = systemwide_levelised_cost(
        results, model_data
    )
    results["total_levelised_cost"] = systemwide_levelised_cost(
        results, model_data, total=True
    )

    results = clean_results(results, zero_threshold)

    for var_data in results.data_vars.values():
        if "is_result" not in var_data.attrs.keys():
            var_data.attrs["is_result"] = 1

    return results


def capacity_factor(results, model_data, systemwide=False):
    """Calculation of capacity factors.

    Processes whether `flow_cap` is a parameter or a result, then calculates the
    capacity factor.

    The weight of timesteps is considered when computing systemwide capacity factors,
    such that higher-weighted timesteps have a stronger influence on the resulting
    system-wide time-averaged capacity factor.
    """
    # In operate mode, flow_cap is an input parameter
    if "flow_cap" not in results.keys():
        flow_cap = model_data.flow_cap
    else:
        flow_cap = results.flow_cap

    if systemwide:
        prod_sum = (results["flow_out"] * model_data.timestep_weights).sum(
            dim=["timesteps", "nodes"], min_count=1
        )

        cap_sum = flow_cap.where(lambda x: x > 0).sum(dim="nodes", min_count=1)
        time_sum = (model_data.timestep_resolution * model_data.timestep_weights).sum()

        capacity_factors = (prod_sum / (cap_sum * time_sum)).fillna(0)
    else:
        capacity_factors = (
            results["flow_out"] / flow_cap.where(lambda x: x > 0)
        ).fillna(0)

    return capacity_factors


def systemwide_levelised_cost(
    results: xr.Dataset, model_data: xr.Dataset, total: bool = False
) -> xr.DataArray:
    """Calculates systemwide levelised costs.

    Returns a DataArray with systemwide levelised costs for the given
    results, indexed by techs, carriers and costs if total is False,
    or by carriers and costs if total is True.

    The weight of timesteps is considered when computing levelised costs:

    * costs are already multiplied by weight in the constraints, and not
      further adjusted here.

    * production is not multiplied by weight in the constraints, so scaled
      by weight here to be consistent with costs. CAUTION: this scaling
      is temporary during levelised cost computation - the actual
      costs in the results remain untouched.

    Args:
        results (xarray.Dataset): Model results.
        model_data (xarray.Dataset): Model input data.
        total (bool, optional):
            If False (default) returns per-technology levelised cost, if True,
            returns overall system-wide levelised cost.

    Returns:
        xr.DataArray: Array of levelised costs.
    """
    # Here we scale production by timestep weight
    flow_out = results["flow_out"] * model_data.timestep_weights
    cost = results["cost"].sum(dim="nodes", min_count=1)
    flow_out = (results["flow_out"] * model_data.timestep_weights).sum(
        dim=["timesteps", "nodes"], min_count=1
    )

    if total:
        # cost is the total cost of the system
        # flow_out is only the flow_out of supply and conversion technologies
        allowed_techs = ("supply", "supply_plus", "conversion", "conversion_plus")
        valid_techs = model_data.base_tech.isin(allowed_techs)
        cost = cost.sum(dim="techs", min_count=1)
        flow_out = flow_out.sel(techs=valid_techs).sum(dim="techs", min_count=1)

    levelised_cost = []

    for carrier in flow_out["carriers"].values:
        levelised_cost.append(
            cost / flow_out.loc[{"carriers": carrier}].where(lambda x: x > 0)
        )

    return xr.concat(levelised_cost, dim="carriers")


def clean_results(results, zero_threshold):
    """Remove unreasonably small values and unmet_demand if it was never used.

    Used to avoid floating point errors caused by solver output.
    zero_threshold is a value set in model configuration. If not set, defaults
    to zero (i.e. doesn't do anything). Reasonable value = 1e-12.
    """
    threshold_applied = []
    for k, v in results.data_vars.items():
        # If there are any values in the data variable which fall below the
        # threshold, note the data variable name and set those values to zero
        if v.where(abs(v) < zero_threshold, drop=True).sum():
            threshold_applied.append(k)
            with np.errstate(invalid="ignore"):
                v.values[abs(v.values) < zero_threshold] = 0
            v.loc[{}] = v.values

    if threshold_applied:
        comment = "Postprocessing: All values < {} set to 0 in {}".format(
            zero_threshold, ", ".join(threshold_applied)
        )
        LOGGER.warn(comment)
    else:
        comment = f"Postprocessing: zero threshold of {zero_threshold} not required"
        LOGGER.info(comment)

    # Combine unused_supply and unmet_demand into one variable
    if (
        "unmet_demand" in results.data_vars.keys()
        or "unused_supply" in results.data_vars.keys()
    ):
        results["unmet_demand"] = results.get("unmet_demand", 0) + results.get(
            "unused_supply", 0
        )

        results = results.drop_vars("unused_supply")

    return results
