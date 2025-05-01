# Copyright (C) since 2013 Calliope contributors listed in AUTHORS.
# Licensed under the Apache 2.0 License (see LICENSE file).
"""Calliope math handling with interfaces for pre-defined and user-defined files."""

import importlib.resources
import logging
import typing
from pathlib import Path

from calliope.attrdict import AttrDict
from calliope.exceptions import ModelError
from calliope.io import read_rich_yaml
from calliope.schemas.config_schema import InitMath
from calliope.schemas.math_schema import ModeMath
from calliope.util.tools import relative_path

LOGGER = logging.getLogger(__name__)
ORDERED_COMPONENTS_T = typing.Literal[
    "variables",
    "global_expressions",
    "constraints",
    "piecewise_constraints",
    "objectives",
]


def _load_internal_math(math_to_add: str) -> AttrDict:
    """Load standard Calliope math modes."""
    file = importlib.resources.files("calliope") / "math" / f"{math_to_add}.yaml"
    return read_rich_yaml(str(file), allow_override=True)  # TODO-Ivan: remove


def _load_user_math(math_to_add: str, model_def_path: str | Path | None) -> AttrDict:
    """Load user defined math modes."""
    file = relative_path(model_def_path, math_to_add)
    return read_rich_yaml(str(file))


def initialise_math(
    math_config: InitMath, model_def_path: str | Path | None = None
) -> AttrDict:
    """Loads and combines internal and user math files into a unified dataset.

    Args:
        math_config (InitMath): math initialisation configuration.
        model_def_path (str | Path | None): Path to the model definition directory.

    Returns:
        AttrDict: dataset with individual math options.
    """
    LOGGER.info(f"Math init | loading pre-defined {math_config.pre_defined}.")
    math_dataset = AttrDict(
        {name: _load_internal_math(name) for name in math_config.pre_defined}
    )
    if math_config.extra:
        LOGGER.info(f"Math init | loading extras {list(math_config.extra.keys())}.")
        for name, path in math_config.extra.items():
            math_dataset.union({name: _load_user_math(path, model_def_path)})

    return math_dataset


def build_applied_math(
    init_math: dict, names: list[str], add_math: dict | None = None
) -> AttrDict:
    """Construct a validated math dictionary, applying the requested math in order.

    Args:
        init_math (dict): initialised math dataset.
        names (list[str]): names of the math to apply in order.
        add_math (dict | None, optional): additional math to apply at the end. Defaults to None.

    Raises:
        ModelError: a given name was not found in the math dictionary.

    Returns:
        AttrDict: constructed and validated math dataset.
    """
    LOGGER.info(f"Math build | building applied math with {names}.")
    math = AttrDict()
    for name in names:
        try:
            math.union(init_math[name], allow_override=True)
        except KeyError:
            raise ModelError(f"Requested math '{name}' was not initialised.")
    if add_math:
        LOGGER.info("Math build | appending additional math.")
        math.union(add_math, allow_override=True)

    ModeMath(**math)  # TODO-Ivan: respect defaults!
    return math
