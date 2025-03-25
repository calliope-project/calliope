"""Calliope's optimisation backend module."""

from typing import TYPE_CHECKING

import xarray as xr

from calliope.backend import gurobi_backend_model, pyomo_backend_model
from calliope.backend.latex_backend_model import (
    ALLOWED_MATH_FILE_FORMATS,
    LatexBackendModel,
)
from calliope.exceptions import BackendError
from calliope.preprocess import CalliopeMath

if TYPE_CHECKING:
    from calliope.backend.backend_model import BackendModel
    from calliope.schemas import config_schema


def get_model_backend(
    build_config: "config_schema.Build", data: xr.Dataset, math: CalliopeMath
) -> "BackendModel":
    """Assign a backend using the given configuration.

    Args:
        build_config: Build configuration options.
        data (Dataset): model data for the backend.
        math (CalliopeMath): Calliope math.

    Raises:
        exceptions.BackendError: If invalid backend was requested.

    Returns:
        BackendModel: Initialized backend object.
    """
    match build_config.backend:
        case "pyomo":
            return pyomo_backend_model.PyomoBackendModel(data, math, build_config)
        case "gurobi":
            return gurobi_backend_model.GurobiBackendModel(data, math, build_config)
        case _:
            raise BackendError(f"Incorrect backend '{build_config.backend}' requested.")
