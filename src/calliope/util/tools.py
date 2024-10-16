# Copyright (C) since 2013 Calliope contributors listed in AUTHORS.
# Licensed under the Apache 2.0 License (see LICENSE file).
"""Assorted helper tools."""

from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

from typing_extensions import ParamSpec

if TYPE_CHECKING:
    from calliope import AttrDict

P = ParamSpec("P")
T = TypeVar("T")


def relative_path(base_path_file, path) -> Path:
    """Path standardization.

    If ``path`` is not absolute, it is interpreted as relative to the
    path of the given ``base_path_file``.
    """
    # Check if base_path_file is a string because it might be an AttrDict
    path = Path(path)
    if path.is_absolute() or base_path_file is None:
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


def climb_template_tree(
    input_dict: "AttrDict",
    templates: "AttrDict",
    item_name: str | None = None,
    inheritance: list | None = None,
) -> tuple["AttrDict", list | None]:
    """Follow the `template` references from model definition elements to `templates`.

    Model definition elements can inherit template entries (those in `templates`).
    Template entries can also inherit each other, to create an inheritance chain.

    This function will be called recursively until a definition dictionary without `template` is reached.

    Args:
        input_dict (AttrDict): Dictionary (possibly) containing `template`.
        templates (AttrDict): Dictionary of available templates.
        item_name (str | None, optional):
            The current position in the inheritance tree.
            If given, used only for a more expressive KeyError.
            Defaults to None.
        inheritance (list | None, optional):
            A list of items that have been inherited (starting with the oldest).
            If the first `input_dict` does not contain `template`, this will remain as None.
            Defaults to None.

    Raises:
        KeyError: Must inherit from a named template item in `templates`.

    Returns:
        tuple[AttrDict, list | None]: Definition dictionary with inherited data and a list of the inheritance tree climbed to get there.
    """
    to_inherit = input_dict.get("template", None)
    if to_inherit is None:
        updated_input_dict = input_dict
    elif to_inherit not in templates:
        message = f"Cannot find `{to_inherit}` in template inheritance tree."
        if item_name is not None:
            message = f"{item_name} | {message}"
        raise KeyError(message)
    else:
        base_def_dict, inheritance = climb_template_tree(
            templates[to_inherit], templates, to_inherit, inheritance
        )
        updated_input_dict = deepcopy(base_def_dict)
        updated_input_dict.union(input_dict, allow_override=True)
        if inheritance is not None:
            inheritance.append(to_inherit)
        else:
            inheritance = [to_inherit]
    return updated_input_dict, inheritance
