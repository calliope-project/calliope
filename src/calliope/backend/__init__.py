"""Calliope's optimisation backend module."""

from typing import TYPE_CHECKING

import xarray as xr

from calliope.attrdict import AttrDict
from calliope.backend.backend_model import BackendModel
from calliope.backend.gurobi_backend_model import GurobiBackendModel
from calliope.backend.latex_backend_model import (
    ALLOWED_MATH_FILE_FORMATS,
    LatexBackendModel,
)
from calliope.backend.pyomo_backend_model import PyomoBackendModel
from calliope.exceptions import BackendError
from calliope.schemas import config_schema, math_schema


def get_model_backend(
    build_config: "config_schema.Build", data: xr.Dataset, math: "math_schema.Math"
) -> "BackendModel":
    """Assign a backend using the given configuration.

    Args:
        build_config: Build configuration options.
        data (Dataset): model data for the backend.
        math (AttrDict): Calliope math.

    Raises:
        exceptions.BackendError: If invalid backend was requested.

    Returns:
        BackendModel: Initialized backend object.
    """
    match build_config.backend:
        case "pyomo":
            return PyomoBackendModel(data, math, build_config)
        case "gurobi":
            return GurobiBackendModel(data, math, build_config)
        case _:
            raise BackendError(f"Incorrect backend '{build_config.backend}' requested.")
