"""
Copyright (C) since 2013 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

postprocess.py
~~~~~~~~~~~~~~

Functionality to post-process model results.

"""

import logging

import xarray as xr
import numpy as np

from calliope.core.util.dataset import split_loc_techs
from calliope.core.util.logging import log_time
from calliope.core.attrdict import AttrDict
from calliope.preprocess.util import concat_iterable

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

    run_config = AttrDict.from_yaml_string(model_data.attrs["run_config"])
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
    results = clean_results(results, run_config.get("zero_threshold", 0), timings)

    for var_data in results.data_vars.values():
        if "is_result" not in var_data.attrs.keys():
            var_data.attrs["is_result"] = 1

    log_time(
        logger,
        timings,
        "post_process_end",
        time_since_run_start=True,
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
    Returns a DataArray with capacity factor for the given results.
    The results are either indexed by loc_tech_carriers_prod and timesteps,
    or by techs and carriers if systemwide results are being calculated.

    The weight of timesteps is considered when computing systemwide capacity factors,
    such that higher-weighted timesteps have a stronger influence
    on the resulting system-wide time-averaged capacity factor.

    """
    # In operate mode, energy_cap is an input parameter
    if "energy_cap" not in results.keys():
        energy_cap = model_data.energy_cap
    else:
        energy_cap = results.energy_cap

    _prod = split_loc_techs(results["carrier_prod"])
    _cap = split_loc_techs(energy_cap)
    if systemwide:
        # Aggregated/clustered days are represented `timestep_weights` times
        prod_sum = (_prod * model_data.timestep_weights).sum(["timesteps", "locs"])
        cap_sum = _cap.sum(dim="locs")
        time_sum = (model_data.timestep_resolution * model_data.timestep_weights).sum()
        capacity_factors = prod_sum / (cap_sum * time_sum)

    else:
        extra_dims = {
            i: model_data[i].to_index() for i in _prod.dims if i not in _cap.dims
        }
        capacity_factors = (
            (_prod / _cap.expand_dims(extra_dims))
            .fillna(0)
            .stack({"loc_tech_carriers_prod": ["locs", "techs", "carriers"]})
        )
        new_idx = concat_iterable(
            capacity_factors.loc_tech_carriers_prod.values, ["::", "::"]
        )
        capacity_factors = capacity_factors.assign_coords(
            {"loc_tech_carriers_prod": new_idx}
        ).reindex({"loc_tech_carriers_prod": results.loc_tech_carriers_prod})

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
    cost = results["cost"]
    # Here we scale production by timestep weight
    carrier_prod = results["carrier_prod"] * model_data.timestep_weights

    if total:
        cost = split_loc_techs(cost).sum(dim=["locs", "techs"])
        supply_only_carrier_prod = carrier_prod.sel(
            loc_tech_carriers_prod=list(
                model_data.loc_tech_carriers_supply_conversion_all.values
            )
        )
        carrier_prod = split_loc_techs(supply_only_carrier_prod).sum(
            dim=["timesteps", "locs", "techs"]
        )
    else:
        cost = split_loc_techs(cost).sum(dim=["locs"])
        carrier_prod = split_loc_techs(carrier_prod).sum(["timesteps", "locs"])

    levelised_cost = []

    for carrier in carrier_prod["carriers"].values:
        levelised_cost.append(cost / carrier_prod.loc[dict(carriers=carrier)])

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
