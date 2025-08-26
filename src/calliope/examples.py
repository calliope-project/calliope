# Copyright (C) since 2013 Calliope contributors listed in AUTHORS.
# Licensed under the Apache 2.0 License (see LICENSE file).
"""Example models that can be loaded directly into a session."""

import importlib
from pathlib import Path

from calliope.model import read_yaml

_EXAMPLE_MODEL_DIR = Path(importlib.resources.files("calliope")) / "example_models"


def national_scale(*args, **kwargs):
    """Returns the built-in national-scale example model."""
    return read_yaml(
        _EXAMPLE_MODEL_DIR / "national_scale" / "model.yaml", *args, **kwargs
    )


def time_clustering(*args, **kwargs):
    """Returns the built-in national-scale example model with time clustering."""
    return national_scale(scenario="time_clustering", *args, **kwargs)


def time_resampling(*args, **kwargs):
    """Returns the built-in national-scale example model with time resampling."""
    return national_scale(scenario="time_resampling", *args, **kwargs)


def urban_scale(*args, **kwargs):
    """Returns the built-in urban-scale example model."""
    return read_yaml(_EXAMPLE_MODEL_DIR / "urban_scale" / "model.yaml", *args, **kwargs)


def milp(*args, **kwargs):
    """Returns the built-in urban-scale example model with MILP constraints enabled."""
    return urban_scale(scenario="milp", *args, **kwargs)


def operate(*args, **kwargs):
    """Returns the built-in urban-scale example model in operate mode."""
    return urban_scale(scenario="operate", *args, **kwargs)


def operate_milp(*args, **kwargs):
    """Returns the built-in urban-scale example model in operate mode with MILP constraints enabled."""
    return urban_scale(scenario="operate,milp", *args, **kwargs)
