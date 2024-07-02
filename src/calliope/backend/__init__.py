"""Calliope's optimisation backend module."""

import xarray as xr

from calliope.backend.latex_backend_model import MathDocumentation
from calliope.backend.parsing import ParsedBackendComponent
from calliope.backend.pyomo_backend_model import PyomoBackendModel
from calliope.exceptions import BackendError

MODEL_BACKENDS = ("pyomo",)


def get_model_backend(name: str, data: xr.Dataset, **kwargs):
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
            backend = PyomoBackendModel(data, **kwargs)
        case _:
            raise BackendError(f"Incorrect backend '{name}' requested.")

    return backend
