"""
Copyright (C) 2013-2017 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

examples.py
~~~~~~~~~~~

Example models.

"""

import os

from . import core


_PATHS = {
    'national_scale': os.path.join(os.path.dirname(__file__), 'example_models', 'national_scale'),
    'urban_scale': os.path.join(os.path.dirname(__file__), 'example_models', 'urban_scale')
}


def national_scale():
    """Returns the built-in national-scale example model."""
    return core.Model.from_yaml_file(os.path.join(_PATHS['national_scale'], 'run.yaml'))


def urban_scale():
    """Returns the built-in urban-scale example model."""
    return core.Model.from_yaml_file(os.path.join(_PATHS['urban_scale'], 'run.yaml'))


def milp():
    """Returns the built-in urban-scale example model with MILP constraints enabled."""
    return core.Model.from_yaml_file(os.path.join(_PATHS['urban_scale'], 'run_milp.yaml'))
