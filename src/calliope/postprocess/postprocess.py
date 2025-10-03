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
    results = apply_zero_threshold(results, model.config.solve.zero_threshold)

    return results


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
