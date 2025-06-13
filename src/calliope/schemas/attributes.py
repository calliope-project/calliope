# Copyright (C) since 2013 Calliope contributors listed in AUTHORS.
# Licensed under the Apache 2.0 License (see LICENSE file).
"""Schema for Calliope model attributes."""

from typing import Self

from pydantic import Field, model_validator

from calliope import _version, exceptions
from calliope.schemas.general import CalliopeBaseModel
from calliope.schemas.math_schema import MathSchema
from calliope.schemas.model_def_schema import CalliopeModelDef


class CalliopeModelAttrs(CalliopeBaseModel):
    """Calliope model definition."""

    model_config = {"title": "Calliope Model Attributes"}

    model_def: CalliopeModelDef = CalliopeModelDef()
    """Full model definition (except any data table contents)"""

    applied_math: MathSchema = MathSchema()
    """Math applied when building or updating the optimisation problem."""

    applied_overrides: str | None = None
    """Overrides applied when initialising the model."""

    allow_operate_mode: bool = True
    """If False, building in `operate` mode will not be allowed."""

    calliope_version_initialised: str = _version.__version__
    """The calliope version this model was initialised with."""

    defaults: dict = Field(default_factory=dict)
    """Dictionary of parameter defaults from the calliope model definition schema."""

    scenario: str | None = None
    """Scenario applied on initialising the model."""

    timings: dict[str, float] = Field(default_factory=dict)
    """Dictionary of timings that is updated as the model is initialised/built/solved."""

    termination_condition: str | None = None
    """Indicates whether the optimisation problem solved to optimality (`optimal`) or not (e.g. `unbounded`, `infeasible`)."""

    @property
    def calliope_version_defined(self) -> str | None:
        """Calliope version defined for this model."""
        return self.model_def.config.init.calliope_version

    @model_validator(mode="after")
    def check_versions(self) -> Self:
        """Check the initialised and defined calliope version.

        Returns:
            Self: unchanged pydantic model.
        """
        version_def = self.calliope_version_defined
        version_init = self.calliope_version_initialised

        if not _version.__version__.startswith(version_init):
            exceptions.warn(
                f"Model was initialised with calliope version {version_init}, "
                f"but you are running {_version.__version__}. Proceed with caution!"
            )

        if version_def is not None and not version_init.startswith(version_def):
            exceptions.warn(
                f"Model configuration specifies calliope version {version_def}, "
                f"but you are running {version_init}. Proceed with caution!"
            )
        return self
