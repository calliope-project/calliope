# Copyright (C) since 2013 Calliope contributors listed in AUTHORS.
# Licensed under the Apache 2.0 License (see LICENSE file).
"""Exceptions and Warning handling."""

import textwrap
import warnings


class ModelError(Exception):
    """Should be raised due to problems with the model formulation or input data."""

    pass


class BackendError(Exception):
    """Should be raised due to issues during backend processing."""

    pass


class ModelWarning(Warning):
    """Should be raised for possible model errors, but where execution can continue."""

    pass


class BackendWarning(Warning):
    """Should be raised for possible backend processing issues where execution may continue."""

    pass


def warn(message: str, _class: type[Warning] = ModelWarning):
    """Raises the specified type of warning."""
    warnings.formatwarning = (
        lambda message, category, *args, **kwargs: f"{category.__name__}: {message}\n"
    )
    warnings.warn(message, _class)
    warnings.formatwarning = warnings._formatwarning_orig


def print_warnings_and_raise_errors(
    warnings: list[str] | dict[str, list[str]] | None = None,
    errors: list[str] | dict[str, list[str]] | None = None,
    during: str = "model processing",
    bullet: str = " * ",
) -> None:
    """Process collections of warnings/errors.

    Prints warnings / raises errors with a bullet point list of the concatenated
    collections.

    Lists will return simple bullet lists:
    E.g. warnings=["foo", "bar"] becomes:

        Possible issues found during model processing:
        * foo
        * bar

    Dicts of lists will return nested bullet lists:
    E.g. errors={"foo": ["foobar", "foobaz"]} becomes:

        Errors during model processing:
        * foo
            * foobar
            * foobaz

    Args:
        warnings (list[str] | dict[str, list[str]] | None, optional):
            List of warning strings or dictionary of warning strings.
            If None or an empty list, no warnings will be printed.
            Defaults to None.
        errors (list[str] | dict[str, list[str]] | None, optional):
            List of error strings or dictionary of error strings.
            If None or an empty list, no errors will be raised.
            Defaults to None.
        during (str, optional):
            Substring that will be placed at the top of the concatenated list of warnings/errors to point to during which phase of data processing they occurred.
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

    def _indenter(strings: list[str] | dict[str, list[str]]) -> str:
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
