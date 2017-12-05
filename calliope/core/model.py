"""
Copyright (C) 2013-2017 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

model.py
~~~~~~~~

Implements the core Model class.

"""

import numpy as np

from calliope.core import debug
from calliope.core.preprocess import generate_model_run, apply_overrides, build_model_data, apply_time_clustering
from calliope.core.attrdict import AttrDict


def model_run_from_yaml(model_file, override_file=None):
    """
    Generate processed ModelRun configuration from a YAML run configuration file.

    Parameters
    ----------
    model_file : str
        Path to YAML file with model configuration.
    override_file : str, optional
        Path to YAML file with model configuration overrides and the override
        group to use, separated by ':', e.g. 'overrides.yaml:group1'.

    """
    config = AttrDict.from_yaml(model_file)
    config.config_path = model_file

    config_with_overrides, debug_comments = apply_overrides(
        config, override_file=override_file
    )

    return generate_model_run(config_with_overrides, debug_comments)


def model_run_from_dict(config_dict, override_dict=None):
    """
    Generate processed ModelRun configuration from
    run and model config dictionaries.

    Parameters
    ----------
    config_dict : dict or AttrDict
    override_dict : dict or AttrDict, optional

    """
    config = config_dict
    config.config_path = None

    config_with_overrides, debug_comments = apply_overrides(
        config, override_dict=override_dict
    )

    return generate_model_run(config_with_overrides, debug_comments)


class Model(object):
    """
    A Calliope Model.

    """
    def __init__(self, config, *args, **kwargs):
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
            model_run, debug_data = model_run_from_yaml(config, *args, **kwargs)
        elif isinstance(config, dict):
            model_run, debug_data = model_run_from_dict(config, *args, **kwargs)
        else:
            # expected input is a string pointing to a YAML file of the run
            # configuration or a dict/AttrDict in which the run and model
            # configurations are defined
            raise ValueError('input configuration must either be a string or a '
                             'dictionary')

        self._model_run = model_run
        self._debug_data = debug_data

        self._model_data_original = build_model_data(model_run)

        # FIXME: name of _modelrun
        # FIXME: add random_seed to example run.yaml files
        random_seed = self._model_run.get_key('run.random_seed', None)
        if random_seed:
            np.random.seed(seed=random_seed)

        # After applying the random seed, time clustering can take place
        self.model_data = apply_time_clustering(
            self._model_data_original, model_run)

    def save_debug_data(self, path):
        """
        Save fully built and commented model_run to a YAML file at the
        given path, for debug purposes.

        """
        debug.save_debug_data(self._model_run, self._debug_data, path)
