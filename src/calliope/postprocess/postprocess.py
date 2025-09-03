# Copyright (C) since 2013 Calliope contributors listed in AUTHORS.
# Licensed under the Apache 2.0 License (see LICENSE file).
"""Functionality to post-process model results."""

import logging

import numpy as np
import xarray as xr

from calliope.schemas import ModelStructure

LOGGER = logging.getLogger(__name__)


def postprocess_model_results(results: xr.Dataset, model: ModelStructure) -> xr.Dataset:
    """Post-processing of model results.

    Adds additional post-processed result variables to the given model results
    in-place. Model must have solved successfully.
    All instances of unreasonably low numbers (set by zero_threshold) will be removed.

    Args:
        results (Dataset): Output from the solver backend.
        model (ModelStructure): A calliope model instance.

    Returns:
        Dataset: input-results dataset.
    """
    if model.config.solve.postprocessing_active:
        results["capacity_factor"] = capacity_factor(results, model.inputs)
        results["systemwide_capacity_factor"] = capacity_factor(
            results, model.inputs, systemwide=True
        )
        results["systemwide_levelised_cost"] = systemwide_levelised_cost(
            results, model.inputs
        )
        results["total_levelised_cost"] = systemwide_levelised_cost(
            results, model.inputs, total=True
        )
        results["unmet_sum"] = unmet_sum(results)

    results = apply_zero_threshold(results, model.config.solve.zero_threshold)

    return results


def capacity_factor(
    results: xr.Dataset, model_data: xr.Dataset, systemwide=False
) -> xr.DataArray:
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

        cap_sum = flow_cap.where(lambda x: x > 0)
        if "nodes" in cap_sum.dims:
            cap_sum = cap_sum.sum(dim="nodes", min_count=1)
        time_sum = (model_data.timestep_resolution * model_data.timestep_weights).sum()

        capacity_factors = (prod_sum / (cap_sum * time_sum)).fillna(0)
    else:
        capacity_factors = (
            results["flow_out"]
            / (flow_cap.where(lambda x: x > 0) * model_data.timestep_resolution)
        ).where(results["flow_out"].notnull())

    return capacity_factors


def systemwide_levelised_cost(
    results: xr.Dataset, model_data: xr.Dataset, total: bool = False
) -> xr.DataArray:
    """Calculates systemwide levelised costs.

    Returns a DataArray with systemwide levelised costs for the given
    results, indexed by techs, carriers and costs if total is False,
    or by carriers and costs if total is True.

    The weight of timesteps is considered when computing levelised costs:

    * costs are already multiplied by weight in the constraints, and not further adjusted here.

    * production (`flow_out` + `flow_export`) is not multiplied by weight in the constraints,
      so scaled by weight here to be consistent with costs.
      CAUTION: this scaling is temporary during levelised cost computation -
      the actual costs in the results remain untouched.

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
    cost = results["cost"].sum(dim="nodes", min_count=1)
    generation = (
        (results["flow_out"] + results.get("flow_export", xr.DataArray(0)).fillna(0))
        * model_data.timestep_weights
    ).sum(dim=["timesteps", "nodes"], min_count=1)

    if total:
        # `cost` is the total cost of the system
        # `generation`` is only the generation of supply and conversion technologies
        allowed_techs = ("supply", "conversion")
        valid_techs = model_data.base_tech.isin(allowed_techs)
        cost = cost.sum(dim="techs", min_count=1)
        generation = generation.sel(techs=valid_techs).sum(dim="techs", min_count=1)

    levelised_cost = cost / generation.where(lambda x: x > 0)

    return levelised_cost


def unmet_sum(results: xr.Dataset) -> xr.DataArray:
    """Calculate the sum of unmet demand/supply."""
    if {"unmet_demand", "unused_supply"} & set(results.data_vars.keys()):
        unmet_sum = results.get("unmet_demand", xr.DataArray(0))
        unmet_sum += results.get("unused_supply", xr.DataArray(0))
    else:
        unmet_sum = xr.DataArray(np.nan)
    return unmet_sum


def apply_zero_threshold(results: xr.Dataset, zero_threshold: float) -> xr.Dataset:
    """Remove unreasonably small values in-place.

    Used to avoid floating point errors caused by solver output.
    Reasonable value = 1e-12.
    """
    if zero_threshold != 0:
        for name in list(results.data_vars):
            # If there are any values in the data variable which fall below the
            # threshold, note the data variable name and set those values to zero
            results[name] = xr.where(
                np.abs(results[name]) < zero_threshold, 0, results[name]
            )

        LOGGER.info(
            "Postprocessing: applied zero threshold %s to model results.",
            zero_threshold,
        )
    else:
        LOGGER.info(
            "Postprocessing: skipping zero threshold application (threshold equals 0)."
        )
    return results
