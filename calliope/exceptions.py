"""
Copyright (C) since 2013 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

exceptions.py
~~~~~~~~~~~~~

Exceptions and Warnings.

"""

import textwrap
import warnings


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


def print_warnings_and_raise_errors(warnings=None, errors=None):
    """
    Print warnings and raise ModelError from errors.

    Parameters
    ----------
    warnings : list, optional
    errors : list, optional

    """
    if warnings:
        warn(
            "Possible issues found during model processing:\n"
            + textwrap.indent("\n".join(sorted(list(set(warnings)))), " * ")
        )

    if errors:
        raise ModelError(
            "Errors during model processing:\n"
            + textwrap.indent("\n".join(sorted(list(set(errors)))), " * ")
        )

    return None
