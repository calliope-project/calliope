# Copyright (C) since 2013 Calliope contributors listed in AUTHORS.
# Licensed under the Apache 2.0 License (see LICENSE file).
"""Calliope math handling with interfaces for pre-defined and user-defined files."""

import logging
import typing
from importlib import resources
from pathlib import Path

from calliope.attrdict import AttrDict
from calliope.exceptions import ModelError, ModelWarning
from calliope.io import read_rich_yaml
from calliope.schemas.math_schema import MathSchema
from calliope.util.tools import relative_path

LOGGER = logging.getLogger(__name__)
PRE_DEFINED_MATH = ["plan", "operate", "spores", "storage_inter_cluster", "milp"]
ORDERED_COMPONENTS_T = typing.Literal[
    "variables",
    "global_expressions",
    "constraints",
    "piecewise_constraints",
    "objectives",
]


def _load_internal_math(filename: str) -> AttrDict:
    """Load standard Calliope math modes."""
    file = Path(str(resources.files("calliope"))) / "math" / f"{filename}.yaml"
    return read_rich_yaml(file)


def _load_user_math(file_path: str, model_def_path: str | Path | None) -> AttrDict:
    """Load user defined math modes."""
    file = relative_path(model_def_path, file_path)
    return read_rich_yaml(file)


def initialise_math(
    extra_math: dict[str, str] | None = None, model_def_path: str | Path | None = None
) -> AttrDict:
    """Loads and combines internal and user math files into a unified dataset.

    Args:
        base_math (str): name of the file to use as base.
        extra_math (dict[str, str] | None, optional): names and paths to extra math. Defaults to None.
        model_def_path (str | Path | None, optional): Path to the model definition. Defaults to None.

    Raises:
        ModelWarning: pre-defined file has been overwritten.

    Returns:
        AttrDict: dataset with individual math options.
    """
    LOGGER.info("Math init | loading pre-defined math.")

    math_dataset = AttrDict(
        {name: _load_internal_math(name) for name in PRE_DEFINED_MATH}
    )
    if extra_math:
        LOGGER.info(f"Math init | loading extras {list(extra_math.keys())}.")
        for name, path in extra_math.items():
            if name in PRE_DEFINED_MATH:
                raise ModelWarning(f"Overwriting pre-defined '{name}' math.")
            math_dataset.union({name: _load_user_math(path, model_def_path)})

    return math_dataset


def build_applied_math(
    priority: list[str], math_dataset: dict, overwrite: dict | None = None
) -> AttrDict:
    """Construct a validated math dictionary, applying the requested math in order.

    Args:
        priority (list[str]): name of the math to apply in order of priority (lower->upper).
        math_dataset (dict): initialised math dataset.
        overwrite (dict | None, optional): additional math to apply at the end. Defaults to None.

    Raises:
        ModelError: a given name was not found in the math dictionary.

    Returns:
        AttrDict: constructed and validated math dataset.
    """
    LOGGER.info(f"Math build | building applied math with {priority}.")
    math = AttrDict()
    for name in priority:
        try:
            math.union(math_dataset[name], allow_override=True)
        except KeyError:
            raise ModelError(f"Requested math '{name}' was not initialised.")
    if overwrite:
        LOGGER.info("Math build | appending additional math.")
        math.union(overwrite, allow_override=True)

    return AttrDict(MathSchema(**math).model_dump())
