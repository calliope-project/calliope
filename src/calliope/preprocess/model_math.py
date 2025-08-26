# Copyright (C) since 2013 Calliope contributors listed in AUTHORS.
# Licensed under the Apache 2.0 License (see LICENSE file).
"""Calliope math handling with interfaces for pre-defined and user-defined files."""

import logging
import typing
from importlib import resources
from pathlib import Path

from calliope.attrdict import AttrDict
from calliope.backend import parsing
from calliope.exceptions import ModelError, print_warnings_and_raise_errors
from calliope.io import read_rich_yaml
from calliope.schemas.math_schema import CalliopeBuildMath
from calliope.util.tools import relative_path

LOGGER = logging.getLogger(__name__)
MATH_FILE_DIR = resources.files("calliope.math")
BASE_DEFAULT = "plan"
ORDERED_COMPONENTS_T = typing.Literal[
    "variables",
    "global_expressions",
    "constraints",
    "piecewise_constraints",
    "objectives",
]


def initialise_math_paths(
    extra_math: dict[str, str] | None = None, model_def_path: str | Path | None = None
) -> AttrDict:
    """Combines internal and user math file paths into a unified dictionary.

    Args:
        extra_math (dict[str, str] | None, optional): names and paths to extra math. Defaults to None.
        model_def_path (str | Path | None, optional): Path to the model definition. Defaults to None.

    Raises:
        ModelWarning: pre-defined file has been overwritten.

    Returns:
        AttrDict: dataset with individual math paths.
    """
    LOGGER.info("Math init | loading pre-defined math.")

    math_dataset = AttrDict(
        {Path(name).stem: str(name) for name in MATH_FILE_DIR.iterdir()}
    )
    math_dataset["base"] = math_dataset[BASE_DEFAULT]

    if extra_math is not None:
        for name, path in extra_math.items():
            if name in math_dataset:
                LOGGER.warning(
                    f"Math init | Overwriting pre-defined '{name}' math with {path}."
                )
            math_dataset[name] = relative_path(model_def_path, path)

    return math_dataset


def load_math(math_paths: dict[str, str | Path]) -> AttrDict:
    """Load all math files.

    Args:
        math_paths (dict[str, str | Path]): names and paths to math YAML definitions.

    Returns:
        AttrDict: dataset with individual math options.
    """
    LOGGER.info(f"Math init | loading math files {set(math_paths)}.")
    return AttrDict({name: read_rich_yaml(path) for name, path in math_paths.items()})


def build_applied_math(
    priority: list[str],
    math_dataset: dict,
    overwrite: dict | None = None,
    validate: bool = True,
) -> CalliopeBuildMath:
    """Construct a validated math dictionary, applying the requested math in order.

    Args:
        priority (list[str]): name of the math to apply in order of priority (lower->upper).
        math_dataset (dict): initialised math dataset.
        overwrite (dict | None, optional): additional math to apply at the end. Defaults to None.
        validate (bool, optional): whether to validate the math strings. Defaults to True.

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
    math_model = CalliopeBuildMath(**math)
    if validate:
        _validate_math_string_parsing(math_model)
    return math_model


def _validate_math_string_parsing(math_model: CalliopeBuildMath) -> None:
    """Validate that `expression` and `where` strings of the math dictionary can be successfully parsed.

    NOTE: strings are not checked for evaluation validity.
    Evaluation issues will be raised only on adding a component to the backend.
    """
    validation_errors: dict = dict()
    valid_component_names = set(
        math_model.variables.root
        | math_model.parameters.root
        | math_model.global_expressions.root
    )
    for component_group in typing.get_args(ORDERED_COMPONENTS_T):
        for name, dict_ in math_model[component_group].root.items():
            parsed = parsing.ParsedBackendComponent(component_group, name, dict_)
            parsed.parse_top_level_where(errors="ignore")
            parsed.parse_equations(valid_component_names, errors="ignore")
            if not parsed._is_valid:
                validation_errors[f"{component_group}:{name}"] = parsed._errors

    if validation_errors:
        print_warnings_and_raise_errors(
            during="math string parsing (marker indicates where parsing stopped, but may not point to the root cause of the issue)",
            errors=validation_errors,
        )

    LOGGER.info("Math build | Validated math strings.")
