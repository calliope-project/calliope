"""
Copyright (C) 2013-2017 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

convert.py
~~~~~~~~~~

Convert Calliope model configurations from 0.5.x to 0.6.0.

"""

import os
import glob

import pandas as pd

from calliope.core.attrdict import AttrDict


_CONVERSIONS = AttrDict.from_yaml(
    os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'conversion_0.6.0.yaml')
)


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


def convert_model_dict(in_dict, conversion_dict):
    out_dict = AttrDict()

    # process techs
    if 'techs' in in_dict:
        for k, v in in_dict.techs.items():
            new_tech_config = convert_subdict(v, conversion_dict['tech_config'])

            if 'constraints_per_distance' in v:
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
                    per_distance_config = convert_subdict(v.costs_per_distance[cost_class], conversion_dict['tech_costs_config'])
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

            out_dict.set_key('techs.{}'.format(k), new_tech_config)

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

        out_dict['locations'] = new_locations_dict
        del in_dict['locations']

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

    # process remaining top-level entries
    out_dict.union(convert_subdict(in_dict, conversion_dict['model_config']))

    return out_dict


def convert_subdict(in_dict, conversion_dict):
    out_dict = AttrDict()

    for old_k in conversion_dict.keys_nested():
        new_k = conversion_dict.get_key(old_k)
        value = in_dict.get_key(old_k, None)

        if value:
            if new_k is None:
                out_dict.set_key('__disabled.{}'.format(old_k), value)
            else:
                out_dict.set_key(conversion_dict.get_key(old_k), value)
            in_dict.del_key(old_k)  # Remove from in_dict

    out_dict.union(in_dict)  # Merge remaining (unchanged) keys

    return out_dict


def convert_model(run_config_path, model_config_path, out_path, override_run_config_paths=None):
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
    override_run_config_paths: list of strs, optional
        any additional run configuration files given are converted
        into override groups in a single overrides.yaml in the out_path

    Returns
    -------
    None

    """
    converted_run_config = AttrDict()
    run_config = load_with_import_resolution(run_config_path)
    for k, v in run_config.items():
        # We consider any files imported in run configuration, but
        # disregard file names and simply merge everything together
        # into the new model configuration
        converted_run_config.update(convert_run_dict(v, _CONVERSIONS))

    new_model_config = AttrDict()
    model_config = load_with_import_resolution(model_config_path)

    for k, v in model_config.items():
        new_model_config[k] = convert_model_dict(v, _CONVERSIONS)

    # Merge run_config into main model config file
    new_model_config[model_config_path].union(converted_run_config)

    # README: For future use we probably want a configuration to specify
    # a calliope version it's compatible with / built for
    new_model_config[model_config_path]['calliope_version'] = '0.6.0'

    # For each file in new_model_config, save it to its same
    # position from the old path in the `out_path`
    for f in new_model_config:
        out_dir, out_filename = os.path.split(
            f.replace(os.path.commonpath([model_config_path, f]), '.')
        )
        if f == model_config_path:
            out_filename = os.path.basename(model_config_path)
        out_file = os.path.join(out_path, out_dir, out_filename)
        os.makedirs(os.path.join(out_path, out_dir), exist_ok=True)
        new_model_config[f].to_yaml(out_file)

    # Read each CSV file in the model data dir and apply index
    ts_dir = new_model_config[model_config_path].get_key('model.timeseries_data_path')
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
