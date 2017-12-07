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

import pandas as pd
import seaborn as sns

import calliope
from calliope import exceptions
from calliope.core.attrdict import AttrDict
from calliope.core.util.tools import relative_path
from calliope.core.preprocess import locations, sets, checks


_DEFAULT_PALETTE = sns.color_palette('cubehelix', 10).as_hex()


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

    # Check whether the model config attempts to override any of the
    # base technology groups
    if 'tech_groups' in config:
        overridden_groups = (
            set(default_tech_groups) &
            set(config.tech_groups.keys())
        )
        if overridden_groups:
            raise exceptions.ModelError(
                'Trying to re-define base '
                'technology groups: {}'.format(overridden_groups)
            )

    # The input files are allowed to override other model defaults
    config_model.union(config, allow_override=True)

    # FIXME: if applying an override that doesn't exist in model, should warn
    # the user about possible mis-spelling

    # Apply overrides via 'override_file', which contains the path to a YAML file
    if override_file:
        # Due to the possible occurrance of `C:\path_to_file\file.yaml:override` we have to split
        # override_file into `path_to_file`, `file.yaml` and `override` before
        # merging `path_to_file` and `file.yaml` back together

        path_to_file, override_file_with_group = os.path.split(override_file)
        override_file, override_groups = override_file_with_group.split(':')
        override_file_path = os.path.join(path_to_file, override_file)

        override_from_file = combine_overrides(override_file_path, override_groups)

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

        # CHECK: If necessary, populate carrier_in and carrier_out in essentials, but
        # also break on missing carrier data
        if 'carrier_in' not in tech_result.essentials:
            if tech_result.inheritance[-1] in ['supply', 'supply_plus', 'unmet_demand']:
                tech_result.essentials.carrier_in = 'resource'
            elif tech_result.inheritance[-1] in ['demand', 'transmission',
                                                 'storage']:
                try:
                    tech_result.essentials.carrier_in = \
                        tech_result.essentials.carrier
                    debug_comments.set_key(
                        '{}.essentials.carrier_in', 'From run configuration filename'
                    )
                except KeyError:
                    errors.append('`carrier` or `carrier_in` must be '
                        'defined for {}'.format(tech_id))
            else:
                errors.append(
                    '`carrier_in` must be defined for {}'.format(tech_id)
                )

        if 'carrier_out' not in tech_result.essentials:
            if tech_result.inheritance[-1] == 'demand':
                tech_result.essentials.carrier_out = 'resource'
            elif tech_result.inheritance[-1] in ['supply', 'supply_plus', 'unmet_demand',
                                                 'transmission', 'storage']:
                try:
                    tech_result.essentials.carrier_out = \
                        tech_result.essentials.carrier
                except KeyError:
                    errors.append('`carrier` or `carrier_out` must be '
                        'defined for {}'.format(tech_id))
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

    if 'timeseries_data_path' in config_model.model:
        dtformat = config_model.model['timeseries_dateformat']

        # Generate the set of all files we want to read from file
        flattened_config = model_run.locations.as_dict_flat()
        csv_files = set([
            v.split('=')[1].rsplit(':', 1)[0]
            for v in flattened_config.values() if 'file=' in str(v)
        ])

        for file in csv_files:
            file_path = os.path.join(config_model.model.timeseries_data_path, file)
            parser = lambda x: datetime.datetime.strptime(x, dtformat)
            try:
                df = pd.read_csv(
                    file_path, index_col=0, parse_dates=True, date_parser=parser
                )
            except ValueError as e:
                raise exceptions.ModelError(
                    "Incorrect datetime format used in {}, expecting "
                    "`{}`, got `{}` instead"
                    "".format(file, dtformat, e.args[0].split("'")[1]))
            timeseries_data[file] = df

    # Apply time subsetting, if supplied in model_run
    subset_time_config = config_model.model.subset_time
    if subset_time_config is not None:
        if isinstance(subset_time_config, list):
            if len(subset_time_config) == 2:
                time_slice = slice(subset_time_config[0], subset_time_config[1])
            else:
                raise exceptions.ModelError(
                    'Invalid subset_time value: {}'.format(subset_time_config)
                )
        else:
            time_slice = str(subset_time_config)
        for k in timeseries_data.keys():
            timeseries_data[k] = timeseries_data[k].loc[time_slice, :]

    # Ensure all timeseries have the same index
    indices = [(file, df.index) for file, df in timeseries_data.items()]
    first_file, first_index = indices[0]
    for file, idx in indices[1:]:
        if not first_index.equals(idx):
            raise exceptions.ModelError('Time series indices do not match '
                'between {} and {}'.format(first_file, file))

    return timeseries_data


def generate_model_run(config, debug_comments):
    """
    Returns a processed model_run configuration AttrDict and a debug
    YAML object with comments attached, ready to write to disk.

    Parameters
    ----------
    config_run : AttrDict
    config_model : AttrDict

    """
    model_run = AttrDict()

    # 1) Initial checks on model configuration
    warnings, errors = checks.check_initial(config)
    checks.print_warnings_and_raise_errors(warnings=warnings, errors=errors)

    # 2) Fully populate techs
    # Raises ModelError if necessary
    model_run['techs'], debug_techs, errors = process_techs(config)
    debug_comments.set_key('model_run.techs', debug_techs)
    checks.print_warnings_and_raise_errors(errors=errors)

    # 3) Fully populate tech_groups
    model_run['tech_groups'] = process_tech_groups(config, model_run['techs'])

    # 4) Fully populate locations
    model_run['locations'], debug_locs, warnings, errors = locations.process_locations(
        config, model_run['techs']
    )
    debug_comments.set_key('model_run.locations', debug_locs)
    checks.print_warnings_and_raise_errors(warnings=warnings, errors=errors)

    # 5) Fully populate timeseries data
    # Raises ModelErrors if there are problems with timeseries data at this stage
    model_run['timeseries_data'] = process_timeseries_data(config, model_run)

    # 6) Initialize sets
    all_sets = sets.generate_simple_sets(model_run)
    all_sets.union(sets.generate_loc_tech_sets(model_run, all_sets))
    all_sets = AttrDict({k: list(v) for k, v in all_sets.items()})
    model_run['sets'] = all_sets

    # 7) Grab additional relevant bits from run and model config
    model_run['run'] = config['run']
    model_run['model'] = config['model']

    # 8) Final sense-checking
    final_check_comments, warnings, errors = checks.check_final(model_run)
    debug_comments.union(final_check_comments)
    checks.print_warnings_and_raise_errors(warnings=warnings, errors=errors)

    # 9) Build a debug data dict with comments and the original configs
    debug_data = AttrDict({
        'comments': debug_comments,
        'config_initial': config,
    })

    return model_run, debug_data

## Storage_time_max should be removed in preprocessing, with the result being a
## change in storage_cap_min and storage_cap_max to account for it. Half completed
## function is dumped here
#def get_storage_cap(backend_model, loc_tech, energy_cap, charge_rate):
#    """
#    Get storage_cap.max from storage_time.max, where applicable. If storage_time.max is used,
#    the maximum storage possible is the minimum value of storage possible for
#    any time length which meets the storage_time.max value.
#    """
#    # TODO:
#    # incorporate timeseries resolution. Currently assumes that each
#    # timestep is worth one unit of time.
#    storage_time_max = param_getter(backend_model, 'storage_time_max', (loc_tech))
#    if loc_tech in model_data_dict['sets']['loc_tech_milp']:
#        units = param_getter(backend_model, 'units_max', (loc_tech))
#        storage_cap_max = (units *
#            param_getter(backend_model, 'storage_cap_per_unit', (loc_tech)))
#    else:
#        storage_cap_max = param_getter(backend_model, 'storage_cap_max', (loc_tech))
#    if not storage_cap_max:
#        storage_cap_max = np.inf
#    if not storage_time_max:
#        return storage_cap_max
#    if not energy_cap and not charge_rate:
#        return 0
#    elif not energy_cap:
#        energy_cap = storage_cap_max * charge_rate
#    elif energy_cap and charge_rate:
#        energy_cap = min(energy_cap, storage_cap_max * charge_rate)
#
#    storage_loss = param_getter(backend_model, 'storage_loss', (loc_tech)))
#    energy_eff = model.data.loc[dict(y=y, x=x)].get('energy_eff',
#                            model.get_option(y+ '.constraints.energy_eff', x=x))
#    try:
#        leakage = 1 / (1 - storage_loss)  # if storage_loss is timeseries dependant, will be a DataArray
#        discharge = energy_cap / energy_eff  # if energy_eff is timeseries dependant, will be a DataArray
#    except ZeroDivisionError:  # if storage_loss = 1 or energy_eff = 0
#        return np.inf  # i.e. no upper limit on storage
#    exponents = [i for i in range(storage_time_max)]
#    if isinstance(leakage, xr.DataArray) or isinstance(discharge, xr.DataArray):
#        roll = model.data.t.rolling(t=storage_time_max)  # create arrays of all rolling horizons
#        S = {storage_cap_max}
#        for label, arr_window in roll:
#            if len(arr_window) == storage_time_max:  # only consider arrays of the maximum time length
#                S.add(sum(discharge * np.power(leakage, exponents)))
#        return min(S)  # smallest value of storage is the maximum allowed
#    else:  # no need to loop through rolling horizons if all values are static in time
#        return min(storage_cap_max, sum(discharge * np.power(leakage, exponents)))