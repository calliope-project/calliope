# Copyright (C) since 2013 Calliope contributors listed in AUTHORS.
# Licensed under the Apache 2.0 License (see LICENSE file).
"""Schema for Calliope model attributes."""

from pydantic import Field

from calliope import _version
from calliope.schemas.general import CalliopeBaseModel


class CalliopeRuntime(CalliopeBaseModel):
    """Calliope runtime model attributes."""

    model_config = {"title": "Calliope Model Attributes"}

    applied_overrides: str | None = None
    """Overrides applied when initialising the model."""

    calliope_version_initialised: str = _version.__version__
    """The calliope version this model was initialised with."""

    scenario: str | None = None
    """Scenario applied on initialising the model."""

    timings: dict[str, float] = Field(default_factory=dict)
    """Dictionary of timings that is updated as the model is initialised/built/solved."""

    termination_condition: str | None = None
    """Indicates whether the optimisation problem solved to optimality (`optimal`) or not (e.g. `unbounded`, `infeasible`)."""
