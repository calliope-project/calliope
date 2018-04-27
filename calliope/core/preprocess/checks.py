"""
Copyright (C) 2013-2018 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

preprocess_checks.py
~~~~~~~~~~~~~~~~~~~~

Checks for model consistency and possible errors during preprocessing.

"""

import os
import logging

import numpy as np
import xarray as xr

from inspect import signature

import calliope
from calliope._version import __version__
from calliope.core.attrdict import AttrDict
from calliope.core.util.tools import flatten_list
from calliope.core.preprocess.util import get_all_carriers
from calliope.core.util.logging import logger
from calliope.core.util.tools import load_function

_defaults_files = {
    k: os.path.join(os.path.dirname(calliope.__file__), 'config', k + '.yaml')
    for k in ['model', 'defaults']
}
defaults = AttrDict.from_yaml(_defaults_files['defaults'])
defaults_model = AttrDict.from_yaml(_defaults_files['model'])


def check_overrides(config_model, override):
    """
    Perform checks on the override dict and override file inputs to ensure they
    are not doing something silly.
    """
    warnings = []
    info = []
    for key in override.as_dict_flat().keys():
        if key in config_model.as_dict_flat().keys():
            info.append(
                'Override applied to {}: {} -> {}'
                .format(key, config_model.get_key(key), override.get_key(key))
            )
        else:
            info.append(
                '`{}`:{} applied from override as new configuration'
                .format(key, override.get_key(key))
            )

    # Check if overriding coordinates are in the same coordinate system. If not,
    # delete all incumbent coordinates, ready for the new coordinates to come in
    if (any(['coordinates' in k for k in config_model.as_dict_flat().keys()]) and
            any(['coordinates' in k for k in override.as_dict_flat().keys()])):

        # get keys that might be deleted and incumbent coordinate system
        config_keys = [k for k in config_model.as_dict_flat().keys() if 'coordinates.' in k]
        config_coordinates = set([k.split('coordinates.')[-1] for k in config_keys])

        # get overriding coordinate system
        override_coordinates = set(
            k.split('coordinates.')[-1] for k in override.as_dict_flat().keys()
            if 'coordinates.' in k
        )

        # compare overriding and incumbent, deleting incumbent if overriding is different
        if config_coordinates != override_coordinates:
            for key in config_keys:
                config_model.del_key(key)
            warnings.append(
                'Updated from coordinate system {} to {}, using overrides'
                .format(config_coordinates, override_coordinates)
            )

    if info:
        logger.info('\n'.join(info))

    return warnings


def check_initial(config_model):
    """
    Perform initial checks of model and run config dicts.

    Returns
    -------
    warnings : list
        possible problems that do not prevent the model run
        from continuing
    errors : list
        serious issues that should raise a ModelError

    """
    errors = []
    warnings = []

    # Check for version mismatch
    model_version = config_model.model.get('calliope_version', False)
    if model_version:
        if not str(model_version) in __version__:
            warnings.append(
                'Model configuration specifies calliope_version={}, '
                'but you are running {}. Proceed with caution!'.format(
                    model_version, __version__)
            )

    # Check top-level keys
    for k in config_model.keys():
        if k not in ['model', 'run', 'locations', 'tech_groups', 'techs', 'links', 'config_path']:
            warnings.append(
                'Unrecognised top-level configuration item: {}'.format(k)
            )

    # Check run configuration
    # Exclude solver_options from checks, as we don't know all possible
    # options for all solvers
    for k in config_model['run'].keys_nested():
        if (k not in defaults_model['run'].keys_nested() and
                'solver_options' not in k):
            warnings.append(
                'Unrecognised setting in run configuration: {}'.format(k)
            )

    # Check model configuration, but top-level keys only
    for k in config_model['model'].keys():
        if k not in defaults_model['model'].keys():
            warnings.append(
                'Unrecognised setting in model configuration: {}'.format(k)
            )

    # Only ['in', 'out', 'in_2', 'out_2', 'in_3', 'out_3']
    # are allowed as carrier tiers
    for key in config_model.as_dict_flat().keys():
        if ('.carrier_' in key and key.split('.carrier_')[-1].split('.')[0] not
                in ['in', 'out', 'in_2', 'out_2', 'in_3', 'out_3', 'ratios'] and
                'group_share' not in key):
            errors.append(
                "Invalid carrier tier found at {}. Only "
                "'carrier_' + ['in', 'out', 'in_2', 'out_2', 'in_3', 'out_3'] "
                "is valid.".format(key)
            )

    # No techs may have the same identifier as a tech_group
    name_overlap = (
        set(config_model.tech_groups.keys()) &
        set(config_model.techs.keys())
    )
    if name_overlap:
        errors.append(
            'tech_groups and techs with '
            'the same name exist: {}'.format(name_overlap)
        )

    # Checks for techs and tech_groups:
    # * All user-defined tech and tech_groups must specify a parent
    # * techs cannot be parents, only tech groups can
    # * No carrier may be called 'resource'
    default_tech_groups = list(defaults_model.tech_groups.keys())
    for tg_name, tg_config in config_model.tech_groups.items():
        if tg_name in default_tech_groups:
            continue
        if not tg_config.get_key('essentials.parent'):
            errors.append(
                'tech_group {} does not define '
                '`essentials.parent`'.format(tg_name)
            )
        elif tg_config.get_key('essentials.parent') in config_model.techs.keys():
            errors.append(
                'tech_group `{}` has a tech as a parent, only another tech_group '
                'is allowed'.format(tg_name)
            )
        if 'resource' in get_all_carriers(tg_config.essentials):
            errors.append(
                'No carrier called `resource` may '
                'be defined (tech_group: {})'.format(tg_name)
            )

    for t_name, t_config in config_model.techs.items():
        if not t_config.get_key('essentials.parent'):
            errors.append(
                'tech {} does not define '
                '`essentials.parent`'.format(t_name)
            )
        elif t_config.get_key('essentials.parent') in config_model.techs.keys():
            errors.append(
                'tech `{}` has another tech as a parent, only a tech_group '
                'is allowed'.format(tg_name)
            )
        if 'resource' in get_all_carriers(t_config.essentials):
            errors.append(
                'No carrier called `resource` may '
                'be defined (tech: {})'.format(t_name)
            )

    # Error if a constraint is loaded from file that must not be
    allowed_from_file = defaults['file_allowed']
    for k, v in config_model.as_dict_flat().items():
        if 'file=' in str(v):
            constraint_name = k.split('.')[-1]
            if constraint_name not in allowed_from_file:
                errors.append(
                    'Cannot load `{}` from file for configuration {}'
                    .format(constraint_name, k)
                )

    # Check the objective function being used has all the appropriate
    # options set in objective_options, and that no options are unused
    objective_function = 'calliope.backend.pyomo.objective.' + config_model.run.objective
    objective_args_expected = list(signature(load_function(objective_function)).parameters.keys())
    objective_args_expected = [arg for arg in objective_args_expected
                               if arg not in ['backend_model', 'kwargs']]
    for arg in objective_args_expected:
        if arg not in config_model.run.objective_options:
            errors.append(
                'Objective function argument `{}` not found in run.objective_options'
                .format(arg)
            )
    for arg in config_model.run.objective_options:
        if arg not in objective_args_expected:
            warnings.append(
                'Objective function argument `{}` given but not used by objective function `{}`'
                .format(arg, config_model.run.objective)
            )

    return warnings, errors


def _check_tech(model_run, tech_id, tech_config, loc_id, warnings, errors, comments):
    """
    Checks individual tech/tech groups at specific locations.
    NOTE: Updates `warnings` and `errors` lists in-place.
    """
    if tech_id not in model_run.techs:
        warnings.append(
            'Tech {} was removed by setting ``exists: False`` - not checking '
            'the consistency of its constraints at location {}.'.format(tech_id, loc_id)
        )
        return warnings, errors

    required = model_run.techs[tech_id].required_constraints
    allowed = model_run.techs[tech_id].allowed_constraints
    allowed_costs = model_run.techs[tech_id].allowed_costs
    all_defaults = list(defaults.default_tech.constraints.keys())

    # Error if required constraints are not defined
    for r in required:
        # If it's a string, it must be defined
        single_ok = isinstance(r, str) and r in tech_config.constraints
        # If it's a list of strings, one of them must be defined
        multiple_ok = (
            isinstance(r, list) and
            any([i in tech_config.constraints for i in r])
        )
        if not single_ok and not multiple_ok:
            errors.append(
                '`{}` at `{}` fails to define '
                'all required constraints: {}'.format(tech_id, loc_id, required)
            )
            # print('{} -- {}-{}: {}, {}'.format(r, loc_id, tech_id, single_ok, multiple_ok))

    # If the technology is supply_plus, check if it has storage_cap_max. If yes, it needs charge rate
    if model_run.techs[tech_id].essentials.parent == 'supply_plus':
        if (any(['storage_cap_' in k for k in tech_config.constraints.keys()])
            and 'charge_rate' not in tech_config.constraints.keys()):
            errors.append(
                '`{}` at `{}` fails to define '
                'charge_rate, but is using storage'.format(tech_id, loc_id, required)
            )
    # If a technology is defined by units (i.e. integer decision variable), it must define energy_cap_per_unit
    if (any(['units_' in k for k in tech_config.constraints.keys()])
        and 'energy_cap_per_unit' not in tech_config.constraints.keys()):
        errors.append(
            '`{}` at `{}` fails to define energy_cap_per_unit when specifying '
            'technology in units_max/min/equals'.format(tech_id, loc_id, required)
        )

    # If a technology is defined by units & is a storage tech, it must define storage_cap_per_unit
    if (any(['units_' in k for k in tech_config.constraints.keys()])
            and model_run.techs[tech_id].essentials.parent in ['storage', 'supply_plus']
            and any(['storage' in k for k in tech_config.constraints.keys()])
            and 'storage_cap_per_unit' not in tech_config.constraints.keys()):
        errors.append(
            '`{}` at `{}` fails to define storage_cap_per_unit when specifying '
            'technology in units_max/min/equals'.format(tech_id, loc_id, required)
        )

    # If a technology is defines force_resource but is not in loc_techs_finite_resource
    if ('force_resource' in tech_config.constraints.keys() and
            loc_id + '::' + tech_id not in model_run.sets.loc_techs_finite_resource):

        warnings.append(
            '`{}` at `{}` defines force_resource but not a finite resource, so '
            'force_resource will not be applied'.format(tech_id, loc_id)
        )

    # Flatten required list and gather remaining unallowed constraints
    required_f = flatten_list(required)
    remaining = set(tech_config.constraints) - set(required_f) - set(allowed)

    # Error if something is defined that's not allowed, but is in defaults
    # Warn if something is defined that's not allowed, but is not in defaults
    # (it could be a misspelling)
    for k in remaining:
        if k in all_defaults:
            errors.append(
                '`{}` at `{}` defines non-allowed '
                'constraint `{}`'.format(tech_id, loc_id, k)
            )
        else:
            warnings.append(
                '`{}` at `{}` defines unrecognised '
                'constraint `{}` - possibly a misspelling?'.format(tech_id, loc_id, k)
            )

    # Error if an `export` statement does not match the given carrier_outs
    if 'export_carrier' in tech_config.constraints:
        essentials = model_run.techs[tech_id].essentials
        export = tech_config.constraints.export_carrier
        if (export and export not in [essentials.get_key(k, '')
                for k in ['carrier_out', 'carrier_out_2', 'carrier_out_3']]):
            errors.append(
                '`{}` at `{}` is attempting to export a carrier '
                'not given as an output carrier: `{}`'.format(tech_id, loc_id, export)
            )

    # Error if non-allowed costs are defined
    for cost_class in tech_config.get_key('costs', {}):
        for k in tech_config.costs[cost_class]:
            if k not in allowed_costs:
                errors.append(
                    '`{}` at `{}` defines non-allowed '
                    '{} cost: `{}`'.format(tech_id, loc_id, cost_class, k)
                )
    return warnings, errors


def check_final(model_run):
    """
    Perform final checks of the completely built model_run.

    Returns
    -------
    comments : AttrDict
        debug output
    warnings : list
        possible problems that do not prevent the model run
        from continuing
    errors : list
        serious issues that should raise a ModelError

    """
    warnings, errors = [], []
    comments = AttrDict()

    # Go through all loc-tech combinations and check validity
    for loc_id, loc_config in model_run.locations.items():
        if 'techs' in loc_config:
            for tech_id, tech_config in loc_config.techs.items():
                _check_tech(
                    model_run, tech_id, tech_config, loc_id,
                    warnings, errors, comments
                )

        if 'links' in loc_config:
            for link_id, link_config in loc_config.links.items():
                for tech_id, tech_config in link_config.techs.items():
                    _check_tech(
                        model_run, tech_id, tech_config,
                        'link {}:{}'.format(loc_id, link_id),
                        warnings, errors, comments
                    )
    # Either all locations or no locations must have coordinates
    all_locs = list(model_run.locations.keys())
    locs_with_coords = [
        k for k in model_run.locations.keys()
        if 'coordinates' in model_run.locations[k]
    ]
    if len(locs_with_coords) != 0 and len(all_locs) != len(locs_with_coords):
        errors.append(
            'Either all or no locations must have `coordinates` defined. '
            'Locations defined: {} - Locations with coordinates: {}'.format(
                all_locs, locs_with_coords)
        )

    # If locations have coordinates, they must all be either lat/lon or x/y
    elif len(locs_with_coords) != 0:
        first_loc = list(model_run.locations.keys())[0]
        try:
            coord_keys = sorted(list(model_run.locations[first_loc].coordinates.keys()))
            if coord_keys != ['lat', 'lon'] and coord_keys != ['x', 'y']:
                errors.append(
                    'Unidentified coordinate system. All locations must either'
                    'use the format {lat: N, lon: M} or {x: N, y: M}.'
                )
        except AttributeError:
            errors.append(
                'Coordinates must be given in the format {lat: N, lon: M} or '
                '{x: N, y: M}, not ' + str(model_run.locations[first_loc].coordinates)
            )

        for loc_id, loc_config in model_run.locations.items():
            try:
                if sorted(list(loc_config.coordinates.keys())) != coord_keys:
                    errors.append('All locations must use the same coordinate format.')
                    break
            except AttributeError:
                errors.append(
                    'Coordinates must be given in the format {lat: N, lon: M} or '
                    '{x: N, y: M}, not ' + str(model_run.locations[first_loc].coordinates)
                )
                break

    # Ensure that timeseries have no non-unique index values
    for k, df in model_run['timeseries_data'].items():
        if df.index.duplicated().any():
            errors.append('Time series `{}` contains non-unique timestamp values.'.format(k))

    # FIXME:
    # make sure `comments` is at the the base level:
    # i.e. comments.model_run.xxxxx....

    return comments, warnings, errors


def check_model_data(model_data):
    """
    Perform final checks of the completely built xarray Dataset `model_data`.

    Returns
    -------
    comments : AttrDict
        debug output
    warnings : list
        possible problems that do not prevent the model run
        from continuing
    errors : list
        serious issues that should raise a ModelError

    """
    warnings, errors = [], []
    comments = AttrDict()

    # Ensure that no loc-tech specifies infinite resource and force_resource=True
    if "force_resource" in model_data.data_vars:
        relevant_loc_techs = [
            i.loc_techs_finite_resource.item()
            for i in model_data.force_resource if i.item() is True
        ]
        forced_resource = model_data.resource.loc[
            dict(loc_techs_finite_resource=relevant_loc_techs)
        ]
        conflict = forced_resource.where(forced_resource == np.inf).to_pandas().dropna()
        if not conflict.empty:
            errors.append(
                'loc_tech(s) {} cannot have `force_resource` set as infinite '
                'resource values are given'.format(', '.join(conflict.index))
            )

    # Ensure that if a tech has negative costs, there is a max cap defined
    # FIXME: doesn't consider capacity being set by a linked constraint e.g.
    # `resource_cap_per_energy_cap`.
    relevant_caps = [
        i for i in ['energy_cap', 'storage_cap', 'resource_cap', 'resource_area']
        if 'cost_' + i in model_data.data_vars.keys()
    ]
    for cap in relevant_caps:
        relevant_loc_techs = (model_data['cost_' + cap]
                              .where(model_data['cost_' + cap] < 0, drop=True)
                              .to_pandas())
        cap_max = cap + '_max'
        cap_equals = cap + '_equals'
        for loc_tech in relevant_loc_techs.columns:
            try:
                cap_val = model_data[cap_max][loc_tech].item()
            except KeyError:
                try:
                    cap_val = model_data[cap_equals][loc_tech].item()
                except KeyError:
                    cap_val = np.nan
            if np.isinf(cap_val) or np.isnan(cap_val):
                errors.append(
                    'loc_tech {} cannot have a negative cost_{} as the '
                    'corresponding capacity constraint is not set'
                    .format(loc_tech, cap)
                )

    for loc_tech in set(model_data.loc_techs_demand.values).intersection(model_data.loc_techs_finite_resource.values):
        if any(model_data.resource.loc[loc_tech].values > 0):
            errors.append(
                'Positive resource given for demand loc_tech {}. All demands '
                'must have negative resource'.format(loc_tech)
            )

    # Delete all empty dimensions & the variables associated with them
    for dim_name, dim_length in model_data.dims.items():
        if dim_length == 0:
            if dim_name in model_data.coords.keys():
                del model_data[dim_name]
            associated_vars = [
                var for var, data in model_data.data_vars.items() if dim_name in data.dims
            ]
            model_data = model_data.drop(associated_vars)
            warnings.append(
                'dimension {} and associated variables {} were empty, so have '
                'been deleted'.format(dim_name, ', '.join(associated_vars))
            )

    # Check if we're allowed to use operate mode
    if 'allow_operate_mode' not in model_data.attrs.keys():
        daily_timesteps = [
            model_data.timestep_resolution.loc[i].values
            for i in np.unique(model_data.timesteps.to_index().strftime('%Y-%m-%d'))
        ]
        if not np.all(daily_timesteps == daily_timesteps[0]):
            model_data.attrs['allow_operate_mode'] = 0
            warnings.append(
                'Operational mode requires the same timestep resolution profile '
                'to be emulated on each date'
            )
        else:
            model_data.attrs['allow_operate_mode'] = 1

    return model_data, comments, warnings, errors
