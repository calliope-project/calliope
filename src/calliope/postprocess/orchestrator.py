# Copyright (C) since 2013 Calliope contributors listed in AUTHORS.
# Licensed under the Apache 2.0 License (see LICENSE file).
"""Helper registry definition for extensible model postprocessing."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Final, TypeVar

import numpy as np
import xarray as xr
from pydantic import BaseModel, ConfigDict

from calliope.exceptions import ModelWarning
from calliope.schemas.config_schema import CalliopeConfig
from calliope.schemas.general import AttrStr, NonEmptyUniqueList
from calliope.util.tools import listify

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PostprocessContext:
    """Small runtime context handler for postprocessing functions."""

    inputs: xr.Dataset
    results: xr.Dataset
    config: CalliopeConfig


PostprocessFunction = Callable[[PostprocessContext], xr.DataArray | None]
F = TypeVar("F", bound=PostprocessFunction)


class PostprocessSettings(BaseModel):
    """Schema for postprocessing function settings."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: AttrStr
    order: int
    base_math: NonEmptyUniqueList
    active: bool


class PostprocessRegistry:
    """Postprocessing registry class."""

    def __init__(self) -> None:
        """Initialise with an empty registry."""
        self._entries: dict[str, tuple[PostprocessSettings, PostprocessFunction]] = {}
        self._lock = threading.Lock()

    def register_function(
        self,
        function: PostprocessFunction,
        *,
        name: str,
        base_math: Iterable[str],
        order: int,
        active: bool,
        overwrite: bool,
    ) -> None:
        """Add an entry to the postprocessing registry."""
        with self._lock:
            spec = PostprocessSettings(
                name=name, order=order, base_math=listify(base_math), active=active
            )
            if spec.name in self._entries and not overwrite:
                raise ValueError(f"Postprocessor '{spec.name}' is already registered.")
            self._entries[spec.name] = (spec, function)

    def get_applicable(
        self, base_math: str
    ) -> list[tuple[PostprocessSettings, PostprocessFunction]]:
        """Find postprocessing function entries compatible with this base math file."""
        with self._lock:
            items = [
                (spec, fn)
                for spec, fn in self._entries.values()
                if (base_math in spec.base_math) and (spec.active)
            ]
            items.sort(key=lambda t: (t[0].order, str(t[0].name)))
        return items

    def set_active(self, name: str, active: bool) -> None:
        """Enable or disable a specific function."""
        with self._lock:
            spec, fn = self._entries[name]
            self._entries[name] = (spec.model_copy(update={"active": active}), fn)


REGISTRY: Final = PostprocessRegistry()


def postprocessor(
    *,
    name: str | None = None,
    base_math: Iterable[str],
    order: int,
    active: bool,
    overwrite: bool = False,
) -> Callable[[F], F]:
    """Decorator used to add new functions to the postprocessing registry.

    Args:
        base_math (Iterable[str]): compatible base math files for this function.
        name (str, optional): name to use for this post processing function. Defaults to the function name.
        order (int): priority order.
        active (bool): whether or not this function should be activated.
        overwrite (bool, optional): whether or not to allow overwriting functions. Defaults to False.
    """

    def decorator(func: F) -> F:
        REGISTRY.register_function(
            func,
            base_math=base_math,
            name=name or func.__name__,
            order=order,
            active=active,
            overwrite=overwrite,
        )
        return func

    return decorator


def _apply_zero_threshold(results: xr.Dataset, zero_threshold: float) -> None:
    """Remove unreasonably small values in-place.

    Used to avoid floating point errors caused by solver output.
    Reasonable value = 1e-12.
    """
    for name in list(results.data_vars):
        # If there are any values in the data variable which fall below the
        # threshold, note the data variable name and set those values to zero
        results[name] = xr.where(
            np.abs(results[name]) < zero_threshold, 0, results[name]
        )

    LOGGER.info(
        "Postprocessing: applied zero threshold %s to model results.", zero_threshold
    )


def postprocess_results(
    results: xr.Dataset, inputs: xr.Dataset, config: CalliopeConfig
) -> xr.Dataset:
    """Run compatible postprocessors in deterministic order."""
    if config.solve.enable_postprocessing:
        applicable_postprocesses = REGISTRY.get_applicable(config.init.base_math)
        ctx = PostprocessContext(config=config, inputs=inputs, results=results)

        for settings, function in applicable_postprocesses:
            name = settings.name
            if (name in results) or name in (results.dims):
                raise ValueError(
                    f"Postprocess '{name}' attempted to overwrite an existing variable/coord/dim."
                )
            try:
                # Store postprocesses in the model results.
                LOGGER.info("Postprocessing: applying %s postprocess function.", name)
                postprocessed_data = function(ctx)
                if postprocessed_data is not None:
                    results[name] = postprocessed_data

            except Exception as ex:
                raise ModelWarning(f"Postprocess '{name}' failed. Skipping.") from ex

    _apply_zero_threshold(results, config.solve.zero_threshold)

    return results
