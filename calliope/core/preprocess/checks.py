"""
Copyright (C) 2013-2017 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

preprocess_checks.py
~~~~~~~~~~~~~~~~~~~~

Checks for model consistency and possible errors during preprocessing.

"""

import os

import calliope
from calliope.core.attrdict import AttrDict
from calliope.core.util.tools import flatten_list
from calliope.core.util.loc_tech import get_all_carriers
from calliope import exceptions


_defaults_files = {
    k: os.path.join(os.path.dirname(calliope.__file__), 'config', k + '.yaml')
    for k in ['model', 'defaults']
}
defaults = AttrDict.from_yaml(_defaults_files['defaults'])
defaults_model = AttrDict.from_yaml(_defaults_files['model'])


def print_warnings_and_raise_errors(warnings=None, errors=None):
    """
    Print warnings and raise ModelError from errors.

    Params
    ------
    warnings : list, optional
    errors : list, optional

    """
    if warnings:
        exceptions.warn(
            'Possible issues found during pre-processing:\n' +
            '\n'.join(warnings)
        )

    if errors:
        raise exceptions.ModelError(
            'Errors during pre-processing:\n' +
            '\n'.join(errors)
        )

    return None


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

    # Check run configuration
    for k in config_model['run'].keys_nested():
        if k not in defaults_model['run'].keys_nested():
            warnings.append(
                'Unrecognized setting in run configuration: {}'.format(k)
            )

    # Only ['in', 'out', 'in_2', 'out_2', 'in_3', 'out_3']
    # are allowed as carrier tiers
    for key in config_model.as_dict_flat().keys():
        if ('.carrier_' in key and key.split('.carrier_')[-1].split('.')[0] not
                in ['in', 'out', 'in_2', 'out_2', 'in_3', 'out_3', 'ratios']):
            errors.append(
                "Invalid carrier tier found at {}. Only "
                "'carrier_' + ['in', 'out', 'in_2', 'out_2', 'in_3', 'out_3'] "
                "is valid.".format(key)
            )

    # No tech_groups/techs may have the same identifier as the built-in groups
    # tech_groups are checked in preprocess_model.process_config()
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
    # * No carrier may be called 'resource'
    default_tech_groups = list(config_model.tech_groups.keys())
    for tg_name, tg_config in config_model.tech_groups.items():
        if tg_name in default_tech_groups:
            continue
        if not tg_config.get_key('essentials.parent'):
            errors.append(
                'tech_group {} does not define '
                '`essentials.parent`'.format(tg_name)
            )
        if 'resource' in get_all_carriers(tg_config):
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
        if 'resource' in get_all_carriers(t_config):
            errors.append(
                'No carrier called `resource` may '
                'be defined (tech: {})'.format(t_name)
            )

    return errors, warnings


def _check_tech(model_run, tech_id, tech_config, loc_id, warnings, errors, comments):
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
    if 'export' in tech_config.constraints:
        export = tech_config.constraints.export
        if export not in [tech_config.essentials.get_key(k) for k in ['carrier_out', 'carrier_out_2', 'carrier_out_3']]:
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

    # Error if a constraint is loaded from file that must not be
    allowed_from_file = defaults['file_allowed']
    for k, v in tech_config.as_dict_flat().items():
        if 'file=' in str(v):
            constraint_name = k.split('.')[-1]
            if constraint_name not in allowed_from_file:
                errors.append(
                    '`{}` at `{}` is trying to load `{}` from file, '
                    'which is not allowed'.format(tech_id, loc_id, constraint_name)
                )

    return None


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
            'Either all or no locations must have `coordinates` defined'
        )

    # If locations have coordinates, they must all be either lat/lon or x/y
    first_loc = list(model_run.locations.keys())[0]
    coord_keys = sorted(list(model_run.locations[first_loc].coordinates.keys()))
    if coord_keys != ['lat', 'lon'] and coord_keys != ['x', 'y']:
        errors.append(
            'Unidentified coordinate system. All locations must either'
            'use the format {lat: N, lon: M} or {x: N, y: M}.'
        )
    for loc_id, loc_config in model_run.locations.items():
        if sorted(list(loc_config.coordinates.keys())) != coord_keys:
            errors.append('All locations must use the same coordinate format.')
            break

    # FIXME: check that constraints are consistent with desired mode:
    # planning or operational
    # if operational, print a single warning, and
    # turn _max constraints into _equals constraints with added comments
    # make sure `comments` is at the the base level:
    # i.e. comments.model_run.xxxxx....

    # FIXME: check that any storage/supply_plus technologies correctly define
    # energy_cap, storage_cap, and charge_rate so as not to clash with each other
    # given that energy_cap = storage_cap * charge_rate

    return comments, warnings, errors

def check_model_data(model_data):
    # FIXME: verify timestep consistency a la verification in get_timeres of
    # old calliope

    # FIXME: ensure that no loc-tech specifies infinite resource and force_resource=True

    # FIXME: raise error with time clustering if it no longer fits with opmode
    # a la last section of initialize_time in old calliope

    # if data.model.mode == 'operate':
    #     opmode_safe = data.attrs.get('opmode_safe', False)
    #     if opmode_safe:
    #         data.attrs['time_resolution'] = self.get_timeres()
    #     else:
    #         msg = 'Time settings incompatible with operational mode'
    #         raise exceptions.ModelError(msg)

    # FIXME ensure that if a tech has negative costs, there is a max cap defined

    comments = None
    warnings = None
    errors = None

    return comments, warnings, errors
