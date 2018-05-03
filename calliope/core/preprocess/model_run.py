"""
Copyright (C) 2013-2018 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

preprocess_model.py
~~~~~~~~~~~~~~~~~~~

Preprocessing of model and run configuration into a unified model_run
AttrDict, and building of associated debug information.

"""

import datetime
import os
import itertools

import pandas as pd
import numpy as np

import calliope
from calliope import exceptions
from calliope.core.attrdict import AttrDict
from calliope.core.util.tools import relative_path
from calliope.core.preprocess import locations, sets, checks, constraint_sets, util


# Output of: sns.color_palette('cubehelix', 10).as_hex()
_DEFAULT_PALETTE = [
    '#19122b', '#17344c', '#185b48', '#3c7632', '#7e7a36',
    '#bc7967', '#d486af', '#caa9e7', '#c2d2f3', '#d6f0ef'
]


def model_run_from_yaml(model_file, override_file=None, override_dict=None):
    """
    Generate processed ModelRun configuration from a
    YAML model configuration file.

    Parameters
    ----------
    model_file : str
        Path to YAML file with model configuration.
    override_file : str, optional
        Path to YAML file with model configuration overrides and the override
        group to use, separated by ':', e.g. 'overrides.yaml:group1'.
    override_dict : dict or AttrDict, optional

    """
    config = AttrDict.from_yaml(model_file)
    config.config_path = model_file

    config_with_overrides, debug_comments = apply_overrides(
        config, override_file=override_file, override_dict=override_dict
    )

    return generate_model_run(config_with_overrides, debug_comments)


def model_run_from_dict(config_dict, override_dict=None):
    """
    Generate processed ModelRun configuration from a
    model configuration dictionary.

    Parameters
    ----------
    config_dict : dict or AttrDict
    override_dict : dict or AttrDict, optional

    """
    if not isinstance(config_dict, AttrDict):
        config = AttrDict(config_dict)
    else:
        config = config_dict
    config.config_path = None

    config_with_overrides, debug_comments = apply_overrides(
        config, override_dict=override_dict
    )

    return generate_model_run(config_with_overrides, debug_comments)


def combine_overrides(override_file_path, override_groups):
    if ',' in override_groups:
        overrides = override_groups.split(',')
    else:
        overrides = [override_groups]

    override = AttrDict()
    for group in overrides:
        try:
            override_group_from_file = AttrDict.from_yaml(override_file_path)[group]
        except KeyError:
            raise exceptions.ModelError(
                'Override group `{}` does not exist in file `{}`.'.format(
                    group, override_file_path
                )
            )
        try:
            override.union(override_group_from_file, allow_override=False)
        except KeyError as e:
            raise exceptions.ModelError(
                str(e)[1:-1] + '. Already specified but defined again in '
                'override group `{}`.'.format(group)
            )

    return override


def apply_overrides(config, override_file=None, override_dict=None):
    """
    Generate processed Model configuration, applying any overrides.

    Parameters
    ----------
    config : AttrDict
        a model configuration AttrDict
    override_file : str, optional
    override_dict : dict or AttrDict, optional

    """
    debug_comments = AttrDict()

    base_model_config_file = os.path.join(
        os.path.dirname(calliope.__file__),
        'config', 'model.yaml'
    )
    config_model = AttrDict.from_yaml(base_model_config_file)

    default_tech_groups = list(config_model.tech_groups.keys())

    # README CHANGED: `model` is not a list any longer -
    # it is now always a single file

    # README CHANGED: order of arguments to relative_path reversed

    # README CHANGED: data_path option removed -- need to make sure
    # that for parallel runs, data_path relative to the currently
    # open model config file always works

    # Interpret timeseries_data_path as relative
    config.model.timeseries_data_path = relative_path(
        config.config_path, config.model.timeseries_data_path
    )

    # The input files are allowed to override other model defaults
    config_model.union(config, allow_override=True)

    # Apply overrides via 'override_file', which contains the path to a YAML file
    if override_file:
        # Due to the possible occurrance of `C:\path_to_file\file.yaml:override` we have to split
        # override_file into `path_to_file`, `file.yaml` and `override` before
        # merging `path_to_file` and `file.yaml` back together

        path_to_file, override_file_with_group = os.path.split(override_file)
        override_file, override_groups = override_file_with_group.split(':')
        override_file_path = os.path.join(path_to_file, override_file)

        override_from_file = combine_overrides(override_file_path, override_groups)

        warnings = checks.check_overrides(config_model, override_from_file)
        exceptions.print_warnings_and_raise_errors(warnings=warnings)

        config_model.union(
            override_from_file, allow_override=True, allow_replacement=True
        )
        for k, v in override_from_file.as_dict_flat().items():
            debug_comments.set_key(
                '{}'.format(k),
                'Overridden via override: {}'.format(override_file))

    # Apply overrides via 'override', which is an AttrDict
    if override_dict:
        if not isinstance(override_dict, AttrDict):
            override_dict = AttrDict(override_dict)

        warnings = checks.check_overrides(config_model, override_dict)
        exceptions.print_warnings_and_raise_errors(warnings=warnings)

        config_model.union(
            override_dict, allow_override=True, allow_replacement=True
        )
        for k, v in override_dict.as_dict_flat().items():
            debug_comments.set_key(
                '{}'.format(k),
                'Overridden via override dictionary.')

    return config_model, debug_comments


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

    result = AttrDict()
    errors = []
    debug_comments = AttrDict()

    for tech_id, tech_config in config_model.techs.items():

        # If a tech specifies ``exists: false``, we skip it entirely
        if not tech_config.get('exists', True):
            continue

        tech_result = AttrDict()

        # Add inheritance chain
        tech_result.inheritance = get_parents(tech_id, config_model)

        # CHECK: A tech's parent must lead to one of the built-in tech_groups
        builtin_tech_groups = checks.defaults_model.tech_groups.keys()
        if tech_result.inheritance[-1] not in builtin_tech_groups:
            errors.append(
                'tech {} must inherit from a built-in tech group'.format(tech_id)
            )

        # Process inheritance
        tech_result.essentials = AttrDict()
        tech_result.constraints = AttrDict()
        for parent in reversed(tech_result.inheritance):
            # Does the parent group have model-wide settings?
            parent_essentials = config_model.tech_groups[parent].essentials
            parent_systemwide_constraints = util.get_systemwide_constraints(
                config_model.tech_groups[parent]
            )
            for k in parent_essentials.as_dict_flat():
                debug_comments.set_key(
                    '{}.essentials.{}'.format(tech_id, k),
                    'From parent tech_group `{}`'.format(parent)
                )
            tech_result.essentials.union(parent_essentials, allow_override=True)
            tech_result.constraints.union(parent_systemwide_constraints, allow_override=True)

        # Add this tech's essentials and constraints, overwriting any essentials from parents
        tech_result.essentials.union(tech_config.essentials, allow_override=True)
        tech_result.constraints.union(
            util.get_systemwide_constraints(tech_config), allow_override=True
        )

        # Add allowed_constraints and required_constraints from base tech
        keys_to_add = ['required_constraints', 'allowed_constraints', 'allowed_costs']
        for k in keys_to_add:
            tech_result[k] = config_model.tech_groups[tech_result.inheritance[-1]].get(k, [])

        # CHECK: If necessary, populate carrier_in and carrier_out in essentials, but
        # also break on missing carrier data
        if 'carrier_in' not in tech_result.essentials:
            if tech_result.inheritance[-1] in ['supply', 'supply_plus']:
                tech_result.essentials.carrier_in = 'resource'
            elif tech_result.inheritance[-1] in ['demand', 'transmission',
                                                 'storage']:
                try:
                    tech_result.essentials.carrier_in = \
                        tech_result.essentials.carrier
                    debug_comments.set_key(
                        '{}.essentials.carrier_in'.format(tech_id),
                        'Set from essentials.carrier'
                    )
                except KeyError:
                    errors.append(
                        '`carrier` or `carrier_in` must be '
                        'defined for {}'.format(tech_id)
                    )
            else:
                errors.append(
                    '`carrier_in` must be defined for {}'.format(tech_id)
                )

        if 'carrier_out' not in tech_result.essentials:
            if tech_result.inheritance[-1] == 'demand':
                tech_result.essentials.carrier_out = 'resource'
            elif tech_result.inheritance[-1] in ['supply', 'supply_plus',
                                                 'transmission', 'storage']:
                try:
                    tech_result.essentials.carrier_out = \
                        tech_result.essentials.carrier
                except KeyError:
                    errors.append(
                        '`carrier` or `carrier_out` must be '
                        'defined for {}'.format(tech_id)
                    )
            else:
                errors.append(
                    '`carrier_out` must be defined for {}'.format(tech_id)
                )

        # If necessary, pick a color for the tech, cycling through
        # the hardcoded default palette
        if not tech_result.essentials.get_key('color', None):
            color = _DEFAULT_PALETTE[next(default_palette_cycler)]
            tech_result.essentials.color = color
            debug_comments.set_key(
                '{}.essentials.color'.format(tech_id),
                'From Calliope default palette')
        result[tech_id] = tech_result

    return result, debug_comments, errors


def process_tech_groups(config_model, techs):
    tech_groups = AttrDict()
    for group in config_model.tech_groups.keys():
        members = set(
            k for k, v in techs.items()
            if group in v.inheritance
        )
        tech_groups[group] = sorted(list(members))
    return tech_groups


def process_timeseries_data(config_model, model_run):

    if config_model.model.timeseries_data is None:
        timeseries_data = AttrDict()
    else:
        timeseries_data = config_model.model.timeseries_data

    def _parser(x, format):
        return pd.to_datetime(x, format=format, exact=False)

    if 'timeseries_data_path' in config_model.model:
        dtformat = config_model.model['timeseries_dateformat']

        # Generate the set of all files we want to read from file
        location_config = model_run.locations.as_dict_flat()
        model_config = config_model.model.as_dict_flat()

        get_filenames = lambda config: set([
            v.split('=')[1].rsplit(':', 1)[0]
            for v in config.values() if 'file=' in str(v)
        ])
        constraint_filenames = get_filenames(location_config)
        cluster_filenames = get_filenames(model_config)

        datetime_min = []
        datetime_max = []

        for file in constraint_filenames | cluster_filenames:
            file_path = os.path.join(config_model.model.timeseries_data_path, file)
            # load the data, without parsing the dates, to catch errors in the data
            df = pd.read_csv(file_path, index_col=0)
            try:
                df.apply(pd.to_numeric)
            except ValueError as e:
                raise exceptions.ModelError(
                    'Error in loading data from {}. Ensure all entries are '
                    'numeric. Full error: {}'.format(file, e)
                )
            # Now parse the dates, checking for errors specific to this
            try:
                df.index = _parser(df.index, dtformat)
            except ValueError as e:
                raise exceptions.ModelError(
                    'Error in parsing dates in timeseries data from {}, '
                    'using datetime format `{}`: {}'.format(file, dtformat, e)
                )
            timeseries_data[file] = df

            datetime_min.append(df.index[0].date())
            datetime_max.append(df.index[-1].date())

    # Apply time subsetting, if supplied in model_run
    subset_time_config = config_model.model.subset_time
    if subset_time_config is not None:
        # Test parsing dates first, to make sure they fit our required subset format

        try:
            subset_time = _parser(subset_time_config, '%Y-%m-%d %H:%M:%S')
        except ValueError as e:
            raise exceptions.ModelError(
                'Timeseries subset must be in ISO format (anything up to the  '
                'detail of `%Y-%m-%d %H:%M:%S`.\n User time subset: {}\n '
                'Error caused: {}'.format(subset_time_config, e)
            )
        if isinstance(subset_time_config, list) and len(subset_time_config) == 2:
            time_slice = slice(subset_time_config[0], subset_time_config[1])

            # Don't allow slicing outside the range of input data
            if (subset_time[0].date() < max(datetime_min) or
                    subset_time[1].date() > min(datetime_max)):

                raise exceptions.ModelError(
                    'subset time range {} is outside the input data time range '
                    '[{}, {}]'.format(subset_time_config,
                                      max(datetime_min).strftime('%Y-%m-%d'),
                                      min(datetime_max).strftime('%Y-%m-%d'))
                )
        elif isinstance(subset_time_config, list):
            raise exceptions.ModelError(
                'Invalid subset_time value: {}'.format(subset_time_config)
            )
        else:
            time_slice = str(subset_time_config)

        for k in timeseries_data.keys():
            timeseries_data[k] = timeseries_data[k].loc[time_slice, :]
            if timeseries_data[k].empty:
                raise exceptions.ModelError(
                    'The time slice {} creates an empty timeseries array for {}'
                    .format(time_slice, k)
                )

    # Ensure all timeseries have the same index
    indices = [
        (file, df.index) for file, df in timeseries_data.items()
        if file not in cluster_filenames
    ]
    first_file, first_index = indices[0]
    for file, idx in indices[1:]:
        if not first_index.equals(idx):
            raise exceptions.ModelError(
                'Time series indices do not match '
                'between {} and {}'.format(first_file, file)
            )

    return timeseries_data, first_index


def generate_model_run(config, debug_comments):
    """
    Returns a processed model_run configuration AttrDict and a debug
    YAML object with comments attached, ready to write to disk.

    Parameters
    ----------
    config : AttrDict
    debug_comments : AttrDict

    """
    model_run = AttrDict()

    # 1) Initial checks on model configuration
    warnings, errors = checks.check_initial(config)
    exceptions.print_warnings_and_raise_errors(warnings=warnings, errors=errors)

    # 2) Fully populate techs
    # Raises ModelError if necessary
    model_run['techs'], debug_techs, errors = process_techs(config)
    debug_comments.set_key('model_run.techs', debug_techs)
    exceptions.print_warnings_and_raise_errors(errors=errors)

    # 3) Fully populate tech_groups
    model_run['tech_groups'] = process_tech_groups(config, model_run['techs'])

    # 4) Fully populate locations
    model_run['locations'], debug_locs, warnings, errors = locations.process_locations(
        config, model_run['techs']
    )
    debug_comments.set_key('model_run.locations', debug_locs)
    exceptions.print_warnings_and_raise_errors(warnings=warnings, errors=errors)

    # 5) Fully populate timeseries data
    # Raises ModelErrors if there are problems with timeseries data at this stage
    model_run['timeseries_data'], model_run['timesteps'] = (
        process_timeseries_data(config, model_run)
    )

    # 6) Grab additional relevant bits from run and model config
    model_run['run'] = config['run']
    model_run['model'] = config['model']

    # 7) Initialize sets
    all_sets = sets.generate_simple_sets(model_run)
    all_sets.union(sets.generate_loc_tech_sets(model_run, all_sets))
    all_sets = AttrDict({k: list(v) for k, v in all_sets.items()})
    model_run['sets'] = all_sets
    model_run['constraint_sets'] = constraint_sets.generate_constraint_sets(model_run)

    # 8) Final sense-checking
    final_check_comments, warnings, errors = checks.check_final(model_run)
    debug_comments.union(final_check_comments)
    exceptions.print_warnings_and_raise_errors(warnings=warnings, errors=errors)

    # 9) Build a debug data dict with comments and the original configs
    debug_data = AttrDict({
        'comments': debug_comments,
        'config_initial': config,
    })

    return model_run, debug_data
