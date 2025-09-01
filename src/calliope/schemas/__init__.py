from dataclasses import dataclass
from typing import TYPE_CHECKING

import xarray as xr

from calliope.schemas.config_schema import CalliopeConfig
from calliope.schemas.general import CalliopeBaseModel as __CalliopeBaseModel
from calliope.schemas.math_schema import CalliopeMath
from calliope.schemas.model_def_schema import CalliopeModelDef
from calliope.schemas.runtime_attrs_schema import CalliopeRuntime

if TYPE_CHECKING:
    from calliope.backend.backend_model import BackendModel


class CalliopeAttrs(__CalliopeBaseModel):
    """All Calliope attributes."""

    definition: CalliopeModelDef = CalliopeModelDef()
    config: CalliopeConfig = CalliopeConfig()
    math: CalliopeMath = CalliopeMath()
    runtime: CalliopeRuntime = CalliopeRuntime()


@dataclass
class ModelStructure:
    """Definition of the structure of a generic Calliope model.

    Helps plug-ins and extensions build in top of default Calliope functionality
    without direct inheritance.
    """

    inputs: xr.Dataset
    backend: "BackendModel"
    config: CalliopeConfig
    definition: CalliopeModelDef
    math: CalliopeMath
    results: xr.Dataset
    runtime: CalliopeRuntime
