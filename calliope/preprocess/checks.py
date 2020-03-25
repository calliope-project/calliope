"""
Copyright (C) 2013-2019 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

preprocess_checks.py
~~~~~~~~~~~~~~~~~~~~

Checks for model consistency and possible errors during preprocessing.

"""

import os
import logging

import numpy as np
import pandas as pd

from inspect import signature

import calliope
from calliope._version import __version__
from calliope.core.attrdict import AttrDict
from calliope.preprocess.util import get_all_carriers
from calliope.core.util.tools import load_function

logger = logging.getLogger(__name__)

DEFAULTS = AttrDict.from_yaml(
    os.path.join(os.path.dirname(calliope.__file__), "config", "defaults.yaml")
)
POSSIBLE_COSTS = [
    i for i in DEFAULTS.techs.default_tech.costs.default_cost.keys()
]


def check_overrides(config_model, override):
    """
    Perform checks on the override dict and override file inputs to ensure they
    are not doing something silly.
    """
    model_warnings = []
    info = []
    for key in override.as_dict_flat().keys():
        if key in config_model.as_dict_flat().keys():
            info.append(
                "Override applied to {}: {} -> {}".format(
                    key, config_model.get_key(key), override.get_key(key)
                )
            )
        else:
            info.append(
                "`{}`:{} applied from override as new configuration".format(
                    key, override.get_key(key)
                )
            )

    # Check if overriding coordinates are in the same coordinate system. If not,
    # delete all incumbent coordinates, ready for the new coordinates to come in
    if any(
        ["coordinates" in k for k in config_model.as_dict_flat().keys()]
    ) and any(["coordinates" in k for k in override.as_dict_flat().keys()]):

        # get keys that might be deleted and incumbent coordinate system
        config_keys = [
            k
            for k in config_model.as_dict_flat().keys()
            if "coordinates." in k
        ]
        config_coordinates = set(
            [k.split("coordinates.")[-1] for k in config_keys]
        )

        # get overriding coordinate system
        override_coordinates = set(
            k.split("coordinates.")[-1]
            for k in override.as_dict_flat().keys()
            if "coordinates." in k
        )

        # compare overriding and incumbent, deleting incumbent if overriding is different
        if config_coordinates != override_coordinates:
            for key in config_keys:
                config_model.del_key(key)
            model_warnings.append(
                "Updated from coordinate system {} to {}, using overrides".format(
                    config_coordinates, override_coordinates
                )
            )

    if info:
        logger.info("\n".join(info))

    return model_warnings
