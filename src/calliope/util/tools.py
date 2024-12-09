# Copyright (C) since 2013 Calliope contributors listed in AUTHORS.
# Licensed under the Apache 2.0 License (see LICENSE file).
"""Assorted helper tools."""

from pathlib import Path
from typing import Any, TypeVar

from typing_extensions import ParamSpec

P = ParamSpec("P")
T = TypeVar("T")


def relative_path(base_path_file: str | Path, path: str | Path) -> Path:
    """Path standardization.

    If ``path`` is not absolute, it is interpreted as relative to the
    path of the given ``base_path_file``.
    """
    # Check if base_path_file is a string because it might be an AttrDict
    path = Path(path)
    if path.is_absolute():
        return path
    else:
        base_path_file = Path(base_path_file)
        if base_path_file.is_file():
            base_path_file = base_path_file.parent
        return base_path_file.absolute() / path


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
