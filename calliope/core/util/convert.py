"""
Copyright (C) 2013-2018 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

convert.py
~~~~~~~~~~

Convert Calliope model configurations from 0.5.x to 0.6.0.

"""

from calliope.core.util.logging import logger
import os
import glob

import pandas as pd

from calliope.core.attrdict import AttrDict, __Missing

_MISSING = __Missing()


_CONVERSIONS = AttrDict.from_yaml(
    os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'conversion_0.6.0.yaml')
)

_TECH_GROUPS = [
    'supply', 'supply_plus',
    'conversion', 'conversion_plus',
    'demand', 'transmission', 'storage'
]


def load_with_import_resolution(in_path):
    files = {}
    top = AttrDict.from_yaml(in_path, resolve_imports=False)
    files[in_path] = top
    base_path = os.path.dirname(in_path)
    for import_path in top.get('import', []):
        result = load_with_import_resolution(os.path.join(base_path, import_path))
        for k, v in result.items():
            files[k] = v
    return files


def convert_run_dict(in_dict, conversion_dict):
    return convert_subdict(in_dict, conversion_dict['run_config'])


def convert_model_dict(in_dict, conversion_dict, state, tech_groups=None):
    out_dict = AttrDict()

    # process techs
    if 'techs' in in_dict:
        for k, v in in_dict.techs.items():

            # Remove now unsupported `unmet_demand` techs
            if (v.get('parent', '') in ['unmet_demand', 'unmet_demand_as_supply_tech'] or
                    'unmet_demand_' in k):
                out_dict.set_key('__disabled.techs.{}'.format(k), v)
                # We will want to enable ``ensure_feasibility`` to replace
                # ``unmet_demand``
                state['ensure_feasibility'] = True
                continue

            new_tech_config = convert_subdict(v, conversion_dict['tech_config'])

            if 'constraints_per_distance' in v:
                # Convert loss to efficiency
                if 'e_loss' in v.constraints_per_distance:
                    v.constraints_per_distance.e_loss = 1 - v.constraints_per_distance.e_loss
                new_tech_config.update(
                    convert_subdict(
                        v.constraints_per_distance,
                        conversion_dict['tech_constraints_per_distance_config']
                    )
                )

            # Costs are a little more involved -- need to get each cost class
            # as a subdict and merge the results back together
            new_cost_dict = AttrDict()
            if 'costs' in v:
                for cost_class in v.costs:
                    new_cost_dict[cost_class] = convert_subdict(v.costs[cost_class], conversion_dict['tech_costs_config'])
            if 'costs_per_distance' in v:
                for cost_class in v.costs_per_distance:
                    # FIXME update not overwrite
                    per_distance_config = convert_subdict(v.costs_per_distance[cost_class], conversion_dict['tech_costs_per_distance_config'])
                    if cost_class in new_cost_dict:
                        new_cost_dict[cost_class].union(per_distance_config)
                    else:
                        new_cost_dict[cost_class] = per_distance_config
            if 'depreciation' in v:
                # 'depreciation.interest.{cost_class}' goes to 'costs.{cost_class}.interest_rate'
                if 'interest' in v.depreciation:
                    for cost_class, interest in v.depreciation.interest.items():
                        new_cost_dict.set_key(
                            '{}.interest_rate'.format(cost_class),
                            interest
                        )
                # 'depreciation.lifetime' goes to 'constraints.lifetime'
                if 'lifetime' in v.depreciation:
                    new_tech_config.set_key(
                        'constraints.lifetime',
                        v.depreciation.lifetime
                    )

            if new_cost_dict:
                new_tech_config['costs'] = new_cost_dict

            # After conversion, remove legacy _per_distance top-level entries
            try:
                del new_tech_config['constraints_per_distance']
                del new_tech_config['costs_per_distance']
            except KeyError:
                pass

            # Assign converted techs to either tech_groups or techs
            if tech_groups and k in tech_groups:
                out_key = 'tech_groups.{}'.format(k)
            else:
                out_key = 'techs.{}'.format(k)

            out_dict.set_key(out_key, new_tech_config)

        del in_dict['techs']

    # process locations
    if 'locations' in in_dict:
        new_locations_dict = AttrDict()
        for k, v in in_dict.locations.items():
            new_locations_dict[k] = convert_subdict(v, conversion_dict['location_config'])

        # convert per-location constraints now in [locname].techs[techname].constraints
        for k, v in new_locations_dict.items():
            if 'techs' in v:
                for tech, tech_dict in v.techs.items():
                    new_locations_dict[k].techs[tech] = convert_subdict(
                        tech_dict, conversion_dict['tech_config']
                    )

            # Add techs that do not specify any overrides as keys
            missing_techs = set(v.get_key('__disabled.techs', [])) - set(v.get('techs', {}).keys())
            for tech in missing_techs:
                new_locations_dict[k].set_key('techs.{}'.format(tech), None)

        # Remove now unsupported `unmet_demand` techs
        for k, v in new_locations_dict.items():
            for tech in list(v.techs.keys()):
                parent = v.get_key('techs.{}.parent'.format(tech), '')
                if (parent in ['unmet_demand', 'unmet_demand_as_supply_tech']
                        or 'unmet_demand_' in tech):
                    new_locations_dict[k].del_key('techs.{}'.format(tech))
                    if '__disabled.techs' in new_locations_dict[k]:
                        new_locations_dict[k].get_key('__disabled.techs').append(tech)
                    else:
                        new_locations_dict[k].set_key('__disabled.techs', [tech])

        out_dict['locations'] = new_locations_dict
        del in_dict['locations']

    # process links
    if 'links' in in_dict:
        new_links_dict = AttrDict()
        for k, v in in_dict.links.items():
            for tech, tech_dict in v.items():
                new_links_dict.set_key(
                    '{}.techs.{}'.format(k, tech),
                    convert_subdict(tech_dict, conversion_dict['tech_config'])
                )

        out_dict['links'] = new_links_dict
        del in_dict['links']

    # process metadata
    if 'metadata' in in_dict:
        # manually transfer location coordinates
        if 'location_coordinates' in in_dict.metadata:
            for k, v in in_dict.metadata.location_coordinates.items():
                if isinstance(v, list):  # Assume it was lat/lon
                    new_coords = AttrDict({'lat': v[0], 'lon': v[1]})
                else:
                    new_coords = v
                in_dict.set_key('locations.{}.coordinates'.format(k), new_coords)
        del in_dict['metadata']

    # Fix up any 'resource' keys that refer to 'file' only
    for k in [i for i in out_dict.keys_nested() if i.endswith('.resource')]:
        if out_dict.get_key(k) == 'file':
            tech = k.split('techs.')[-1].split('.')[0]
            out_dict.set_key(k, 'file={}_r.csv'.format(tech))

    # process remaining top-level entries
    out_dict.union(convert_subdict(in_dict, conversion_dict['model_config']))

    return out_dict


def convert_subdict(in_dict, conversion_dict):
    out_dict = AttrDict()

    for old_k in conversion_dict.keys_nested():
        new_k = conversion_dict.get_key(old_k)
        value = in_dict.get_key(old_k, _MISSING)

        if value != _MISSING:
            if new_k is None:
                out_dict.set_key('__disabled.{}'.format(old_k), value)
            else:
                out_dict.set_key(conversion_dict.get_key(old_k), value)
            in_dict.del_key(old_k)  # Remove from in_dict

    out_dict.union(in_dict)  # Merge remaining (unchanged) keys

    return out_dict


def convert_model(run_config_path, model_config_path, out_path):
    """
    Convert a model specified by a model YAML file

    Parameters
    ----------
    run_config_path: str
        is merged with the model configuration and saved into the
        main model configuration file given by ``model_config``
    model_config_path: str
        model configuration file
    out_path: str
        path into which to save ``model_config`` and all other YAML
        files imported by it -- recreates original directory structure
        at that location, so recommendation is to specify an empty
        subdirectory or a new directory (will be created)

    Returns
    -------
    None

    """
    state = {'ensure_feasibility': False}
    converted_run_config = AttrDict()
    run_config = load_with_import_resolution(run_config_path)
    for k, v in run_config.items():
        # We consider any files imported in run configuration, but
        # disregard file names and simply merge everything together
        # into the new model configuration
        converted_run_config.update(convert_run_dict(v, _CONVERSIONS))

    new_model_config = AttrDict()
    model_config = load_with_import_resolution(model_config_path)

    # Get all techs from old model that need to be tech_groups in the new one
    merged_model_config = AttrDict.from_yaml(model_config_path)
    run_config_overrides = AttrDict.from_yaml(run_config_path).get_key('override', None)
    if run_config_overrides:
        merged_model_config.union(run_config_overrides, allow_override=True)
    tech_groups = set()
    for tech, tech_dict in merged_model_config.techs.items():
        parent = tech_dict.get('parent', None)
        if parent and parent not in _TECH_GROUPS:
            tech_groups.add(parent)

    for k, v in model_config.items():
        new_model_config[k] = convert_model_dict(
            v, _CONVERSIONS, tech_groups=tech_groups, state=state
        )

    # Merge run_config into main model config file
    new_model_config[model_config_path].union(converted_run_config)

    # README: For future use we probably want a configuration to specify
    # a calliope version it's compatible with / built for
    new_model_config[model_config_path].set_key('model.calliope_version', '0.6.0')

    # Set ensure_feasibility if the old model used unmet_demand
    if state['ensure_feasibility']:
        new_model_config[model_config_path].set_key('run.ensure_feasibility', True)
        logger.info(
            'Found no longer supported `unmet_demand` techs, setting `run.ensure_feasibility` \n'
            'to True to replace them. See the docs for more info:\n'
            'https://calliope.readthedocs.io/en/stable/user/building.html#allowing-for-unmet-demand'
        )

    # README: adding top-level interest_rate and lifetime definitions
    # for all techs EXCEPT demand,
    # to mirror the fact that there used to be defaults
    defaults_v05 = AttrDict()
    cost_classes = [  # Get a list of all cost classes in model
        k.split('costs.', 1)[-1].split('.', 1)[0]
        for k in new_model_config.keys_nested()
        if 'costs.' in k
    ]
    for t in [i for i in _TECH_GROUPS if i != 'demand']:
        defaults_v05.set_key('tech_groups.{}.constraints.lifetime'.format(t), 25)
        for cc in cost_classes:
            interest = 0.1 if cc == 'monetary' else 0
            defaults_v05.set_key('tech_groups.{}.costs.{}.interest_rate'.format(t, cc), interest)
    new_model_config[model_config_path].union(defaults_v05)

    # For each file in new_model_config, save it to its same
    # position from the old path in the `out_path`
    for f in new_model_config:
        out_dir, out_filename = os.path.split(
            f.replace(os.path.commonpath([model_config_path, f]), '.')
        )
        if f == model_config_path:
            out_dir_model_config_path = out_dir
            out_filename = os.path.basename(model_config_path)
        out_file = os.path.join(out_path, out_dir, out_filename)
        os.makedirs(os.path.join(out_path, out_dir), exist_ok=True)
        new_model_config[f].to_yaml(out_file)

    # Read each CSV file in the model data dir and apply index
    full_new_config = AttrDict.from_yaml(
        os.path.join(out_path, out_dir_model_config_path, os.path.basename(model_config_path))
    )
    ts_dir = full_new_config.get_key('model.timeseries_data_path')
    ts_path_in = os.path.join(
        os.path.dirname(model_config_path), ts_dir
    )
    ts_path_out = os.path.join(
        os.path.join(out_path, ts_dir)
    )
    os.makedirs(ts_path_out, exist_ok=True)

    index_t = pd.read_csv(os.path.join(ts_path_in, 'set_t.csv'), index_col=0, header=None)[1]

    for f in glob.glob(os.path.join(ts_path_in, '*.csv')):
        if 'set_t.csv' not in f:
            df = pd.read_csv(f, index_col=0)
            df.index = index_t
            df.index.name = None
            df.to_csv(os.path.join(ts_path_out, os.path.basename(f)))

    # FIXME: override_run_configs:
    # for each run config, create an override config
    # based on file name with top-level overrides and additional override
    # groups with any parallel configs named:
    # "parallel_{}".format(k.replace(' ', '_'))
