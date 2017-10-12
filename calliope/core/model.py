"""
Copyright (C) 2013-2017 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

model.py
~~~~~~~~

Implements the core Model class.

"""

import numpy as np

from . import preprocess_model
from . import debug


class Model(object):
    """
    A Calliope Model.

    """
    def __init__(self, model_run, debug_data):
        self._model_run = model_run
        self._debug_data = debug_data

        # FIXME: name of _modelrun
        # FIXME: add random_seed to example run.yaml files
        random_seed = self._model_run.get_key('run.random_seed', None)
        if random_seed:
            np.random.seed(seed=random_seed)

    @classmethod
    def from_yaml_file(cls, run_config_path):
        """
        Returns a new Model from a YAML file specifiying the run configuration.

        Parameters
        ----------
        run_config_path : str
            Path to YAML file with run configuration.

        """
        return cls(*preprocess_model.model_run_from_yaml(run_config_path))

    @classmethod
    def from_dicts(cls, run_config, model_config):
        """
        Returns a new Model from run_config and model_config dicts.

        Parameters
        ----------
        run_config : dict or AttrDict
        model_config : dict or AttrDict

        """
        return cls(*preprocess_model.model_run_from_dicts(run_config, model_config))

    def save_debug_data(self, path):
        """
        Save fully built and commented model_run to a YAML file at the
        given path, for debug purposes.

        """
        debug.save_debug_data(self._model_run, self._debug_data, path)
