# Copyright (C) since 2013 Calliope contributors listed in AUTHORS.
# Licensed under the Apache 2.0 License (see LICENSE file).

"""
examples.py
~~~~~~~~~~~

Example models.

"""

import importlib

from calliope.core.model import Model

EXAMPLE_MODEL_DIR = importlib.resources.files("calliope") / "example_models"


def national_scale(*args, **kwargs):
    """Returns the built-in national-scale example model."""
    return Model(
        model_definition=EXAMPLE_MODEL_DIR / "national_scale" / "model.yaml",
        *args,
        **kwargs,
    )


def time_clustering(*args, **kwargs):
    """Returns the built-in national-scale example model with time clustering."""
    return national_scale(scenario="time_clustering", *args, **kwargs)


def time_resampling(*args, **kwargs):
    """Returns the built-in national-scale example model with time resampling."""
    return national_scale(scenario="time_resampling", *args, **kwargs)


def urban_scale(*args, **kwargs):
    """Returns the built-in urban-scale example model."""
    return Model(
        model_definition=EXAMPLE_MODEL_DIR / "urban_scale" / "model.yaml",
        *args,
        **kwargs,
    )


def milp(*args, **kwargs):
    """Returns the built-in urban-scale example model with MILP constraints enabled."""
    return urban_scale(scenario="milp", *args, **kwargs)


def operate(*args, **kwargs):
    """Returns the built-in urban-scale example model in operate mode."""
    return urban_scale(scenario="operate", *args, **kwargs)
