"""
Copyright (C) 2013-2017 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

preprocess_model.py
~~~~~~~~~~~~~~~~~~~

Preprocessing of model and run configuration into a unified model_run
AttrDict, and building of associated debug information.

"""

import datetime
import os
import itertools

import seaborn as sns

from .. import exceptions
from .. import utils
from . import preprocess_locations as locations
from . import preprocess_sets as sets
from . import preprocess_checks as checks


_DEFAULT_PALETTE = sns.color_palette('cubehelix', 10).as_hex()


def model_run_from_yaml(run_config_path, run_config_override=None):
    """
    Generate processed ModelRun configuration from a YAML run configuration file.

    Parameters
    ----------
    run_config_path : str
        Path to YAML file with run configuration.
    override : AttrDict, optional
        Provide any additional options or override options from
        ``config_run`` by passing an AttrDict of the form
        ``{'model_settings': 'foo.yaml'}``. Any option possible in
        ``run.yaml`` can be specified in the dict, inluding ``override.``
        options.

    """
    config_run = utils.AttrDict.from_yaml(run_config_path)
    config_run.config_run_path = run_config_path

    # If we have no run name we use the run config file name without extension
    if 'name' not in config_run:
        config_run.name = os.path.splitext(os.path.basename(run_config_path))[0]

    # If passed in, config_run is overridden with any additional overrides...
    if run_config_override:
        assert isinstance(run_config_override, utils.AttrDict)
        config_run.union(
            run_config_override, allow_override=True, allow_replacement=True
        )

    config_model = process_config(config_run)

    return generate_model_run(config_run, config_model)


def model_run_from_dicts(run_config_dict, model_config_dict):
    """
    Generate processed ModelRun configuration from
    run and model config dictionaries.

    Parameters
    ----------
    run_config_dict : dict or AttrDict
    model_config_dict : dict or AttrDict

    """
    config_run = utils.AttrDict(run_config_dict)
    config_run.config_run_path = None

    # If we have no run name we just use current date/time
    if 'name' not in config_run:
        config_run.name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")

    config_model = process_config(model_config_dict, is_model_config=True)

    return generate_model_run(config_run, config_model)


def process_config(config, is_model_config=False):
    """
    Generate processed Model configuration from
    a run config dictionary or model config dictionary.

    Parameters
    ----------
    config : AttrDict
        a run or model configuration AttrDict
    is_model_config : bool, optional
        if True, ``config`` is a model_config, else a run_config

    """

    base_model_config_file = os.path.join(
        os.path.dirname(__file__),
        '..', 'config', 'model.yaml'
    )
    config_model = utils.AttrDict.from_yaml(base_model_config_file)

    default_tech_groups = list(config_model.tech_groups.keys())

    # README CHANGED: `model` is not a list any longer -
    # it is now always a single file

    # README CHANGED: order of arguments to relative_path reversed

    # README CHANGED: data_path option removed -- need to make sure
    # that for parallel runs, data_path relative to the currently
    # open model config file always works

    if not is_model_config:
        # Interpret relative config paths as relative to run.yaml
        config.model = utils.relative_path(
            config.config_run_path, config.model
        )

        config_model_dict = utils.AttrDict.from_yaml(config.model)

        # Interpret data_path as relative to `path`  (i.e the currently
        # open model config file)
        config_model_dict.model.data_path = utils.relative_path(
            config.model, config_model_dict.model.data_path
        )

    # Check whether the model config attempts to override any of the
    # base technology groups
    if 'tech_groups' in config_model_dict:
        overridden_groups = (
            set(default_tech_groups) &
            set(config_model_dict.tech_groups.keys())
        )
        if overridden_groups:
            raise exceptions.ModelError(
                'Trying to re-define base '
                'technology groups: {}'.format(overridden_groups)
            )

    # The input files are allowed to override other model defaults
    config_model.union(config_model_dict, allow_override=True)

    return config_model


def get_parents(tech_id, model_config):
        """
        Returns the full inheritance tree from which ``tech`` descends,
        ending with its base technology group.

        To get the base technology group,
        use ``get_parents(...)[-1]``.

        Parameters
        ----------
        tech : str
        model_config : AttrDict

        """

        tech = model_config.techs[tech_id].essentials.parent
        parents = [tech]

        while True:
            tech = model_config.tech_groups[tech].essentials.parent
            if tech is None:
                break  # We have reached the top of the chain
            parents.append(tech)
        return parents


def process_techs(config_model):

    default_palette_cycler = itertools.cycle(range(len(_DEFAULT_PALETTE)))

    result = utils.AttrDict()
    debug_comments = utils.AttrDict()

    for tech_id, tech_config in config_model.techs.items():
        tech_result = utils.AttrDict()

        # Add inheritance chain
        tech_result.inheritance = get_parents(tech_id, config_model)

        # Process inheritance
        tech_result.essentials = utils.AttrDict()
        for parent in reversed(tech_result.inheritance):
            # Does the parent group have model-wide settings?
            parent_essentials = config_model.tech_groups[parent].essentials
            for k in parent_essentials.as_dict_flat():
                debug_comments.set_key(
                    '{}.essentials.{}'.format(tech_id, k),
                    'From parent tech_group `{}`'.format(parent)
                )
            tech_result.essentials.union(parent_essentials, allow_override=True)

        # Add this tech's essentials, overwriting any essentials from parents
        tech_result.essentials.union(tech_config.essentials, allow_override=True)

        # Add allowed_constraints and required_constraints from base tech
        keys_to_add = ['required_constraints', 'allowed_constraints', 'allowed_costs']
        for k in keys_to_add:
            tech_result[k] = config_model.tech_groups[tech_result.inheritance[-1]].get(k, [])

        # If necessary, populate carrier_in and carrier_out in essentials
        if tech_result.inheritance[-1] in ['supply', 'supply_plus']:
            if 'carrier_in' not in tech_result.essentials:
                tech_result.essentials.carrier_in = 'resource'
            if 'carrier_out' not in tech_result.essentials:
                tech_result.essentials.carrier_out = tech_result.essentials.carrier

        if tech_result.inheritance[-1] == 'demand':
            if 'carrier_in' not in tech_result.essentials:
                tech_result.essentials.carrier_in = tech_result.essentials.carrier
            if 'carrier_out' not in tech_result.essentials:
                tech_result.essentials.carrier_out = 'resource'

        if tech_result.inheritance[-1] == 'transmission':
            if 'carrier_in' not in tech_result.essentials:
                tech_result.essentials.carrier_in = tech_result.essentials.carrier
            if 'carrier_out' not in tech_result.essentials:
                tech_result.essentials.carrier_out = tech_result.essentials.carrier

        # If necessary, pick a color for the tech, cycling through
        # the hardcoded default palette
        if not tech_result.essentials.get_key('color', None):
            color = _DEFAULT_PALETTE[next(default_palette_cycler)]
            tech_result.essentials.color = color

        result[tech_id] = tech_result

    return result, debug_comments


def generate_model_run(config_run, config_model):
    """
    Returns a processed model_run configuration AttrDict and a debug
    YAML object with comments attached, ready to write to disk.

    Parameters
    ----------
    config_run : AttrDict
    config_model : AttrDict

    """
    model_run = utils.AttrDict()
    debug_comments = utils.AttrDict()

    # README CHANGED: if run_config overrides data_path, it is no longer
    # interpreted as relative to the run_config file's path

    # 1) Apply any initiall overrides to config_model
    # 1.a) Via 'model_override', which is the path to a YAML file
    if 'model_override' in config_run:
        override_path = utils.relative_path(
            config_run.config_run_path, config_run.model_override
        )
        override_dict = utils.AttrDict.from_yaml(override_path)
        config_model.union(
            override_dict, allow_override=True, allow_replacement=True
        )
        for k, v in override_dict.as_dict_flat():
            debug_comments.set_key(
                'config_model.{}'.format(k),
                'Overridden via `model_override: {}`'.format(override_path))

    # 1.b) Via 'override', which is an AttrDict
    if ('override' in config_run and isinstance(config_run.override, utils.AttrDict)):
        config_model.union(
            config_run.override, allow_override=True, allow_replacement=True
        )
        for k, v in override_dict.as_dict_flat():
            debug_comments.set_key(
                'config_model.{}'.format(k),
                'Overridden via `override` in run configuration.')

    # 2) Initial sense-checking
    checks.check_initial(config_model, config_run)

    # 3) Fully populate techs
    model_run['techs'], debug_techs = process_techs(config_model)
    debug_comments.set_key('model_run.techs', debug_techs)

    # 4) Fully populate locations
    model_run['locations'], debug_locs = locations.process_locations(
        config_model, model_run['techs']
    )
    debug_comments.set_key('model_run.locations', debug_locs)

    # 5) Initialize sets
    all_sets = sets.generate_simple_sets(model_run)
    all_sets.union(sets.generate_loc_tech_sets(model_run, all_sets))
    model_run['sets'] = all_sets

    # 6) Grab additional relevant bits from run and model config
    model_run['run'] = config_run
    model_run['model'] = config_model['model']

    # 7) Final sense-checking
    final_check_comments = checks.check_final(model_run)
    debug_comments.union(final_check_comments)

    # 8) Build a debug data dict with comments and the original configs
    debug_data = utils.AttrDict({
        'comments': debug_comments,
        'config_model': config_model,
        'config_run': config_run
    })

    return model_run, debug_data
