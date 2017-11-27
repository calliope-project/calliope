"""
Copyright (C) 2013-2017 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

model.py
~~~~~~~~

Implements the core Model class.

"""

import numpy as np

from . import preprocess_model
from . import preprocess_data
from . import debug


class Model(object):
    """
    A Calliope Model.

    """
    def __init__(self, config):
        """
        Returns a new Model from either the path to a run configuration file
        or a dict containing 'run' and 'model' keys fully specifying the
        model.

        Parameters
        ----------
        config : str or dict or AttrDict
            If str, must be the path to a run configuration file.
            If dict or AttrDict, must contain two top-level keys, model and run,
            specifying the model and run configuration respectively.

        """
        if isinstance(config, str):
            model_run, debug_data = preprocess_model.model_run_from_yaml(config)
        elif isinstance(config, dict):
            model_run, debug_data = preprocess_model.model_run_from_dict(config)
        else:
            # expected input is a string pointing to a YAML file of the run
            # configuration or a dict/AttrDict in which the run and model
            # configurations are defined
            raise ValueError('input configuration must either be a string or a '
                             'dictionary')

        self._model_run = model_run
        self._debug_data = debug_data

        self._model_data_original = preprocess_data.build_model_data(model_run)

        # FIXME: name of _modelrun
        # FIXME: add random_seed to example run.yaml files
        random_seed = self._model_run.get_key('run.random_seed', None)
        if random_seed:
            np.random.seed(seed=random_seed)

        # After applying the random seed, time clustering can take place
        self.model_data = preprocess_data.apply_time_clustering(
            self._model_data_original, model_run)

    def save_debug_data(self, path):
        """
        Save fully built and commented model_run to a YAML file at the
        given path, for debug purposes.

        """
        debug.save_debug_data(self._model_run, self._debug_data, path)
