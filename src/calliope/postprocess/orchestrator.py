# Copyright (C) since 2013 Calliope contributors listed in AUTHORS.
# Licensed under the Apache 2.0 License (see LICENSE file).
"""Helper registry definition for extensible model postprocessing."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from typing import Final, TypeVar

import numpy as np
import xarray as xr
from pydantic import BaseModel, ConfigDict

from calliope.schemas import ModelStructure
from calliope.schemas.general import AttrStr, NonEmptyUniqueList
from calliope.util.tools import listify

LOGGER = logging.getLogger(__name__)


PostprocessFunction = Callable[[ModelStructure], xr.DataArray | None]
_F = TypeVar("_F", bound=PostprocessFunction)


class PostprocessSettings(BaseModel):
    """Schema for postprocessing function settings."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    active: bool
    """Whether or not the postprocess is active."""
    math: NonEmptyUniqueList
    """List of math names that trigger this postprocess."""
    name: AttrStr
    """Name of the postprocess (defaults to function name)."""
    order: int
    """Defines execution priority (higher == earlier)."""


class PostprocessRegistry:
    """Postprocessing registry class."""

    def __init__(self) -> None:
        """Initialise with an empty registry."""
        self._entries: dict[str, tuple[PostprocessSettings, PostprocessFunction]] = {}
        self._lock = threading.Lock()

    def _register_function(
        self,
        function: PostprocessFunction,
        settings: PostprocessSettings,
        *,
        overwrite: bool,
    ) -> None:
        """Add an entry to the postprocessing registry.

        Args:
            function (PostprocessFunction): decorated function to register.
            settings (PostprocessSettings): decorator settings.
            overwrite (bool): whether or not to allow overwriting a postprocess.

        Raises:
            ValueError: attempted to overwrite a postprocess when `overwrite` is `False`.
        """
        with self._lock:
            if settings.name in self._entries and not overwrite:
                raise ValueError(
                    f"Postprocessor '{settings.name}' is already registered."
                )
            self._entries[settings.name] = (settings, function)

    def _get_applicable(
        self, math_priority: list[str]
    ) -> list[tuple[PostprocessSettings, PostprocessFunction]]:
        """Find postprocessing function entries compatible with this base math file."""
        with self._lock:
            entries = list(self._entries.values())

        model_math = set(math_priority)
        items = [
            (settings, func)
            for settings, func in entries
            if settings.active and set(settings.math) <= model_math
        ]
        items.sort(key=lambda t: (t[0].order, str(t[0].name)))
        return items

    def set_active(self, name: str, active: bool) -> None:
        """Enable or disable a specific function."""
        with self._lock:
            spec, func = self._entries[name]
            self._entries[name] = (spec.model_copy(update={"active": active}), func)


REGISTRY: Final = PostprocessRegistry()


def postprocessor(
    *,
    math: list[str],
    name: str | None = None,
    order: int = 0,
    active: bool = False,
    overwrite: bool = False,
) -> Callable[[_F], _F]:
    """Decorator used to add new functions to the postprocessing registry.

    Args:
        math (list[str]): compatible math files for this function.
        name (str, optional): name to use for this post processing function. Defaults to the function name.
        order (int): priority order.
        active (bool): whether or not this function should be activated.
        overwrite (bool, optional): whether or not to allow overwriting functions. Defaults to False.
    """

    def decorator(func: _F) -> _F:
        settings = PostprocessSettings(
            math=listify(math), name=name or func.__name__, order=order, active=active
        )
        REGISTRY._register_function(func, settings, overwrite=overwrite)
        return func

    return decorator


def _apply_zero_threshold(results: xr.Dataset, zero_threshold: float) -> None:
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


# def postprocess_results(model: ModelStructure) -> xr.Dataset:
#     """Run compatible postprocessors in deterministic order."""
#     if model.config.solve.postprocessing_active:
#         warn(
#             "Model postprocessing will be set to `False` in a future release."
#             "It can be reactivated using `config.solve.postprocessing_active: True`.",
#             FutureWarning,
#         )
#         applicable_postprocesses = REGISTRY._get_applicable(model.math_priority)

#         for settings, func in applicable_postprocesses:
#             name = settings.name
#             try:
#                 # Store postprocesses in the model results.
#                 LOGGER.info("Postprocessing: applying %s postprocess function.", name)
#                 postprocessed_data = func(model)
#                 if postprocessed_data is not None:
#                     model.results[name] = postprocessed_data
#             except Exception as ex:
#                 raise ModelWarning(f"Postprocess '{name}' failed. Skipping.") from ex

#     _apply_zero_threshold(results, config.solve.zero_threshold)

#     return results
