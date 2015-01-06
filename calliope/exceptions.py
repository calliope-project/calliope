"""
Copyright (C) 2013-2015 Stefan Pfenninger.
Licensed under the Apache 2.0 License (see LICENSE file).

exceptions.py
~~~~~~~~~~~~~

Exceptions and Warnings.

"""

import warnings


class ModelError(Exception):
    """
    ModelErrors should stop execution of the model, e.g. due to a problem
    with the model formulation or input data.

    """
    pass


class OptionNotSetError(ModelError):
    pass


class ModelWarning(Warning):
    """
    ModelWarnings should be raised for possible model errors, but
    where execution can still continue.

    """
    pass


def _formatwarning(message, category, filename, lineno, line=None):
    """Formats ModelWarnings as "Warning: message" without extra crud"""
    if category == ModelWarning:
        return 'Warning: ' + str(message) + '\n'
    else:
        return warnings.formatwarning(message, category, filename, lineno, line)
