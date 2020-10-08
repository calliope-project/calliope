"""
Copyright (C) since 2013 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

examples.py
~~~~~~~~~~~

Example models.

"""

import os

from calliope.core.model import Model


_PATHS = {
    "national_scale": os.path.join(
        os.path.dirname(__file__), "example_models", "national_scale"
    ),
    "urban_scale": os.path.join(
        os.path.dirname(__file__), "example_models", "urban_scale"
    ),
}


def national_scale(*args, **kwargs):
    """Returns the built-in national-scale example model."""
    return Model(os.path.join(_PATHS["national_scale"], "model.yaml"), *args, **kwargs)


def time_clustering(*args, **kwargs):
    """Returns the built-in national-scale example model with time clustering."""
    return Model(
        os.path.join(_PATHS["national_scale"], "model.yaml"),
        scenario="time_clustering",
        *args,
        **kwargs,
    )


def time_resampling(*args, **kwargs):
    """Returns the built-in national-scale example model with time resampling."""
    return Model(
        os.path.join(_PATHS["national_scale"], "model.yaml"),
        scenario="time_resampling",
        *args,
        **kwargs,
    )


def urban_scale(*args, **kwargs):
    """Returns the built-in urban-scale example model."""
    return Model(os.path.join(_PATHS["urban_scale"], "model.yaml"), *args, **kwargs)


def milp(*args, **kwargs):
    """Returns the built-in urban-scale example model with MILP constraints enabled."""
    return Model(
        os.path.join(_PATHS["urban_scale"], "model.yaml"),
        scenario="milp",
        *args,
        **kwargs,
    )


def operate(*args, **kwargs):
    """Returns the built-in urban-scale example model in operate mode."""
    return Model(
        os.path.join(_PATHS["urban_scale"], "model.yaml"),
        scenario="operate",
        *args,
        **kwargs,
    )


def time_masking(*args, **kwargs):
    """Returns the built-in urban-scale example model with time masking."""
    return Model(
        os.path.join(_PATHS["urban_scale"], "model.yaml"),
        scenario="time_masking",
        *args,
        **kwargs,
    )
