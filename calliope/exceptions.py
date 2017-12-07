"""
Copyright (C) 2013-2017 Calliope contributors listed in AUTHORS.
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


class BackendError(Exception):
    pass


class ModelWarning(Warning):
    """
    ModelWarnings should be raised for possible model errors, but
    where execution can still continue.

    """
    pass


def warn(message, _class=ModelWarning):
    warnings.warn(message, _class)
