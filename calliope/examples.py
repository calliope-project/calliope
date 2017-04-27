"""
Copyright (C) 2013-2017 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

examples.py
~~~~~~~~~~~

Example models.

"""

import os

from . import core


PATHS = {
    'NationalScale': os.path.join(os.path.dirname(__file__), 'example_models', 'national_scale'),
    'UrbanScale': os.path.join(os.path.dirname(__file__), 'example_models', 'urban_scale'),
}


class NationalScale(core.Model):
    """
    National-scale example model.

    """

    def __init__(self, override=None):
        config_run = os.path.join(PATHS['NationalScale'], 'run.yaml')
        super().__init__(config_run=config_run, override=override)


class UrbanScale(core.Model):
    """
    Urban-scale example model.

    """

    def __init__(self, override=None):
        config_run = os.path.join(PATHS['UrbanScale'], 'run.yaml')
        super().__init__(config_run=config_run, override=override)
