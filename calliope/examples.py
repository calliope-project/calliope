"""
Copyright (C) 2013-2017 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

examples.py
~~~~~~~~~~~

Example models.

"""

import os

from calliope.core.model import Model


_PATHS = {
    'national_scale': os.path.join(os.path.dirname(__file__), 'example_models', 'national_scale'),
    'urban_scale': os.path.join(os.path.dirname(__file__), 'example_models', 'urban_scale')
}


def national_scale():
    """Returns the built-in national-scale example model."""
    return Model(os.path.join(_PATHS['national_scale'], 'model.yaml'))


def time_clustering():
    """Returns the built-in national-scale example model with time clustering."""
    return Model(
        os.path.join(_PATHS['national_scale'], 'model.yaml'),
        override_file=os.path.join(_PATHS['national_scale'], 'overrides.yaml:time_clustering')
    )


def time_resampling():
    """Returns the built-in national-scale example model with time resampling."""
    return Model(
        os.path.join(_PATHS['national_scale'], 'model.yaml'),
        override_file=os.path.join(_PATHS['national_scale'], 'overrides.yaml:time_resampling')
    )


def urban_scale():
    """Returns the built-in urban-scale example model."""
    return Model(os.path.join(_PATHS['urban_scale'], 'model.yaml'))


def milp():
    """Returns the built-in urban-scale example model with MILP constraints enabled."""
    return Model(
        os.path.join(_PATHS['urban_scale'], 'model.yaml'),
        override_file=os.path.join(_PATHS['urban_scale'], 'overrides.yaml:milp')
    )
