# Copyright (C) since 2013 Calliope contributors listed in AUTHORS.
# Licensed under the Apache 2.0 License (see LICENSE file).

"""
exceptions.py
~~~~~~~~~~~~~

Exceptions and Warnings.

"""

import textwrap
import warnings
from typing import Optional, Union

# Enable simple format when printing ModelWarnings
formatwarning_orig = warnings.formatwarning


def _formatwarning(message, category, filename, lineno, line=None):
    """Formats ModelWarnings as "Warning: message" without extra crud"""
    if category == ModelWarning:
        return "Warning: " + str(message) + "\n"
    else:
        return formatwarning_orig(message, category, filename, lineno, line)


warnings.formatwarning = _formatwarning


class ModelError(Exception):
    """
    ModelErrors should stop execution of the model, e.g. due to a problem
    with the model formulation or input data.

    """

    pass


class BackendError(Exception):
    pass


class ModelWarning(Warning):
    """
    ModelWarnings should be raised for possible model errors, but
    where execution can still continue.

    """

    pass


class BackendWarning(Warning):
    pass


def warn(message, _class=ModelWarning):
    warnings.warn(message, _class)


def print_warnings_and_raise_errors(
    warnings: Optional[Union[list[str], dict[str, list[str]]]] = None,
    errors: Optional[Union[list[str], dict[str, list[str]]]] = None,
    during: str = "model processing",
    bullet: str = " * ",
) -> None:
    """
    Concatenate collections of warnings/errors and print (warnings) / raise ModelError (errors) with a bullet point list of the concatenated collections.

    Lists will return simple bullet lists:
    E.g. warnings=["foo", "bar"] becomes::

        Possible issues found during model processing:
        * foo
        * bar

    Dicts of lists will return nested bullet lists:
    E.g. errors={"foo": ["foobar", "foobaz"]} becomes::

        Errors during model processing:
        * foo
            * foobar
            * foobaz

    Args:
        warnings (Optional[Union[list[str], dict[str, list[str]]]], optional):
            List of warning strings or dictionary of warning strings.
            If None or an empty list, no warnings will be printed.
            Defaults to None.
        errors (Optional[Union[list[str], dict[str, list[str]]]], optional):
            List of error strings or dictionary of error strings.
            If None or an empty list, no errors will be raised.
            Defaults to None.
        during (str, optional):
            Substring that will be placed at the top of the concated list of warnings/errors to point to during which phase of data processing they occured.
            Defaults to "model processing".
        bullet (str, optional): Type of bullet points to use. Defaults to " * ".

    Raises:
        ModelError: If errors is not None or is a non-empty list/dict

    """
    spacer = " " * len(bullet)

    def _sort_strings(stringlist: list[str]) -> list[str]:
        return sorted(list(set(stringlist)))

    def _predicate(string_: str) -> bool:
        return not string_.startswith((bullet, spacer))

    def _indenter(strings: Union[list[str], dict[str, list[str]]]) -> str:
        if isinstance(strings, dict):
            sorted_strings = []
            for k, v in strings.items():
                sorted_strings.append(str(k) + ":")
                sorted_strings.extend(_sort_strings([spacer + bullet + i for i in v]))
        else:
            sorted_strings = _sort_strings(strings)
        return textwrap.indent("\n".join(sorted_strings), bullet, predicate=_predicate)

    if warnings:
        warn(f"Possible issues found during {during}:\n" + _indenter(warnings))

    if errors:
        raise ModelError(f"Errors during {during}:\n" + _indenter(errors))

    return None
