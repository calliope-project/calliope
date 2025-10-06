# Copyright (C) since 2013 Calliope contributors listed in AUTHORS.
# Licensed under the Apache 2.0 License (see LICENSE file).
"""Schema for Calliope model attributes."""

from pydantic import Field

from calliope import _version
from calliope.schemas.config_schema import Resamples, Subsets
from calliope.schemas.general import AttrStr, CalliopeBaseModel


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

    instantiated: bool = False
    """Indicates whether the model has been instantiated."""

    subset: Subsets = Subsets()
    """List all applied subsets."""

    resample: Resamples = Resamples()
    """List all applied resamples."""

    time_cluster: AttrStr | None = None
    """Indicates the time clustering applied to the model."""

    math_priority: list[AttrStr] = Field(default_factory=list)
    """The order of math entries applied to the model, with each subsequent one overwriting the last."""
