"""
Copyright (C) 2013-2017 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

preprocess_checks.py
~~~~~~~~~~~~~~~~~~~~

Checks for model consistency and possible errors during preprocessing.

"""

import os

from .. import utils


def check_initial(config_model, config_run):
    pass


def check_final(model_run):
    comments = utils.AttrDict()
    return comments
