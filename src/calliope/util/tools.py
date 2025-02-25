# Copyright (C) since 2013 Calliope contributors listed in AUTHORS.
# Licensed under the Apache 2.0 License (see LICENSE file).
"""Assorted helper tools."""

from pathlib import Path
from typing import Any, TypeVar

from typing_extensions import ParamSpec

P = ParamSpec("P")
T = TypeVar("T")


def relative_path(base_path_file: str | Path | None, path: str | Path) -> Path:
    """Path standardization.

    If ``path`` is not absolute, it is interpreted as relative to the
    path of the given ``base_path_file``.
    """
    # Check if base_path_file is a string because it might be an AttrDict
    path = Path(path)
    if path.is_absolute():
        return path
    else:
        base_path = Path(base_path_file) if base_path_file is not None else Path.cwd()
        if base_path.is_file():
            base_path = base_path.parent
        return base_path.absolute() / path


def listify(var: Any) -> list:
    """Transform the input into a list.

    If it is a non-string iterable, it will be coerced to a list.
    If it is a string or non-iterable (numeric, boolean, ...), then it will be a single item list.

    Args:
        var (Any): Value to transform.

    Returns:
        list: List containing `var` or elements of `var` (if input was a non-string iterable).
    """
    if var is None:
        var = []
    elif not isinstance(var, str) and hasattr(var, "__iter__"):
        var = list(var)
    else:
        var = [var]
    return var


def get_dot_attr(var: Any, attr: str) -> Any:
    """Get nested attributes in dot notation.

    Works for nested objects (e.g., dictionaries, pydantic models).

    Args:
        var (Any): Object to extract nested attributes from.
        attr (str): Name of the attribute (e.g., "foo.bar").

    Returns:
        Any: Value at the given location.
    """
    levels = attr.split(".", 1)

    if isinstance(var, dict):
        value = var[levels[0]]
    else:
        value = getattr(var, levels[0])

    if len(levels) > 1:
        value = get_dot_attr(value, levels[1])
    return value
