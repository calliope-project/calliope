# Copyright (C) since 2013 Calliope contributors listed in AUTHORS.
# Licensed under the Apache 2.0 License (see LICENSE file).

"""
postprocess.py
~~~~~~~~~~~~~~

Functionality to post-process model results.

"""

import logging

import numpy as np
import xarray as xr

from calliope.util.logging import log_time

logger = logging.getLogger(__name__)


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
    log_time(logger, timings, "post_process_start", comment="Postprocessing: started")

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
    results = clean_results(results, zero_threshold, timings)

    for var_data in results.data_vars.values():
        if "is_result" not in var_data.attrs.keys():
            var_data.attrs["is_result"] = 1

    log_time(
        logger,
        timings,
        "post_process_end",
        time_since_solve_start=True,
        comment="Postprocessing: ended",
    )

    if "run_solution_returned" in timings.keys():
        results.attrs["solution_time"] = (
            timings["run_solution_returned"] - timings["run_start"]
        ).total_seconds()
        results.attrs["time_finished"] = timings["run_solution_returned"].strftime(
            "%Y-%m-%d %H:%M:%S"
        )

    return results


def capacity_factor(results, model_data, systemwide=False):
    """
    # In operate mode, flow_cap is an input parameter
    if "flow_cap" not in results.keys():
        flow_cap = model_data.flow_cap
    else:
        flow_cap = results.flow_cap

    capacity_factors = (results["flow_out"] / flow_cap).fillna(0)

    The weight of timesteps is considered when computing systemwide capacity factors,
    such that higher-weighted timesteps have a stronger influence
    on the resulting system-wide time-averaged capacity factor.

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
        valid_techs = model_data.parent.isin(allowed_techs)
        cost = cost.sum(dim="techs", min_count=1)
        flow_out = flow_out.sel(techs=valid_techs).sum(dim="techs", min_count=1)

    levelised_cost = []

    for carrier in flow_out["carriers"].values:
        levelised_cost.append(
            cost / flow_out.loc[{"carriers": carrier}].where(lambda x: x > 0)
        )

    return xr.concat(levelised_cost, dim="carriers")


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
            with np.errstate(invalid="ignore"):
                v.values[abs(v.values) < zero_threshold] = 0
            v.loc[{}] = v.values

    if threshold_applied:
        comment = "All values < {} set to 0 in {}".format(
            zero_threshold, ", ".join(threshold_applied)
        )
    else:
        comment = "zero threshold of {} not required".format(zero_threshold)

    log_time(logger, timings, "threshold_applied", comment="Postprocessing: " + comment)

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
