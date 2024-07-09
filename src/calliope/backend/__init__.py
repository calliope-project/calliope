"""Calliope's optimisation backend module."""

from typing import TYPE_CHECKING

import xarray as xr

from calliope.backend.gurobi_backend_model import GurobiBackendModel
from calliope.backend.latex_backend_model import MathDocumentation
from calliope.backend.parsing import ParsedBackendComponent
from calliope.backend.pyomo_backend_model import PyomoBackendModel
from calliope.exceptions import BackendError

MODEL_BACKENDS = ("pyomo",)

if TYPE_CHECKING:
    from calliope.backend.backend_model import BackendModel


def get_model_backend(name: str, data: xr.Dataset, **kwargs) -> "BackendModel":
    """Assign a backend using the given configuration.

    Args:
        name (str): name of the backend to use.
        data (Dataset): model data for the backend.
        **kwargs: backend keyword arguments corresponding to model.config.build.

    Raises:
        exceptions.BackendError: If invalid backend was requested.

    Returns:
        BackendModel: Initialized backend object.
    """
    match name:
        case "pyomo":
            return PyomoBackendModel(data, **kwargs)
        case "gurobi":
            return GurobiBackendModel(data, **kwargs)
        case _:
            raise BackendError(f"Incorrect backend '{name}' requested.")
