"""
Copyright (C) 2013-2017 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

preprocess_checks.py
~~~~~~~~~~~~~~~~~~~~

Checks for model consistency and possible errors during preprocessing.

"""

import os

from .. import utils


_defaults_files = {
    k: os.path.join(os.path.dirname(__file__), '..', 'config', k + '.yaml')
    for k in ['run', 'model', 'defaults']
}

defaults = utils.AttrDict.from_yaml(_defaults_files['defaults'])

defaults_model = utils.AttrDict.from_yaml(_defaults_files['model'])

defaults_run = utils.AttrDict.from_yaml(_defaults_files['run'])
# Hardcode additional keys into allowed defaults:
# Auto-generated string to run_config file's path
defaults_run['config_run_path'] = None


def _check_config_run(config_run):
    errors = []
    warnings = []

    for k in config_run.keys_nested():
        if k not in defaults_run.keys_nested():
            warnings.append(
                'Unrecognized setting in run configuration: {}'.format(k)
            )

    return errors, warnings


def _get_all_carriers(config):
    return ([config.get_key('carrier', '')] +
            list(config.get_key('carrier_in', '')) +
            list(config.get_key('carrier_out', ''))
            )


def _check_config_model(config_model):
    errors = []
    warnings = []

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
        if 'resource' in _get_all_carriers(tg_config):
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
        if 'resource' in _get_all_carriers(t_config):
            errors.append(
                'No carrier called `resource` may '
                'be defined (tech: {})'.format(t_name)
            )

    # Either all locations or no locations must have coordinates
    all_locs = list(config_model.locations.keys())
    locs_with_coords = [
        k for k in config_model.locations.keys()
        if 'coordinates' in config_model.locations[k]
    ]
    if len(locs_with_coords) != 0 and len(all_locs) != len(locs_with_coords):
        errors.append(
            'Either all or no locations must have `coordinates` defined'
        )

    # If locations have coordinates, they must all be either lat/lon or x/y
    first_loc = list(config_model.locations.keys())[0]
    coord_keys = sorted(list(config_model.locations[first_loc].coordinates.keys()))
    if coord_keys != ['lat', 'lon'] and coord_keys != ['x', 'y']:
        errors.append(
            'Unidentified coordinate system. All locations must either'
            'use the format {lat: N, lon: M} or {x: N, y: M}.'
        )
    for loc_id, loc_config in config_model.locations.items():
        if sorted(list(loc_config.coordinates.keys())) != coord_keys:
            errors.append('All locations must use the same coordinate format.')
            break

    return errors, warnings


def check_initial(config_model, config_run):
    """
    Perform initial checks of model and run config dicts.

    May stop and raise ModelError on serious issues, or print
    warnings for possible problems that do not prevent the model run
    from continuing.

    """
    errors_run, warnings_run = _check_config_run(config_run)
    errors_model, warnings_model = _check_config_model(config_model)

    errors = errors_run + errors_model
    warnings = warnings_run + warnings_model

    return warnings, errors


def check_final(model_run):
    """
    Perform final checks of the completely built model_run.

    This may:
    * add comments to debug output;
    * stop and raise ModelError on serious issues;
    * print warnings for possible problems that do not prevent the run from
      continuing.

    """
    warnings, errors = [], []
    comments = utils.AttrDict()

    # FIXME: Confirm that all techs specify essentials
    # At this point, `essensials` from model_config have been added directly
    # to the top level of each tech key in model_run.techs

    # FIXME: Check that all techs have a carrier (might have been inherited)

    # FIXME: Confirm that all required constraints are defined for each tech

    # FIXME If something is defined that's not allowed, but is in defaults:
    # error

    # FIXME: if something defined that's not allowed, but is not in defaults:
    # warn (it could be a mis-spelling)

    # All `export` statements must be equal to one of the carrier_outs
    # FIXME (pared down version of old self.check_and_set_export())

    # FIXME: check that constraints are consistent with desired mode:
    # planning or operational
    # if operational, warn but turn _max constraints into _equals constraints

    # FIXME: make sure comments is at the the base level:
    # i.e. comments must be comments.model_run.xxxxx....

    return comments, warnings, errors
