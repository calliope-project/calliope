"""
Copyright (C) 2013-2018 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

debug.py
~~~~~~~~

Debugging tools.

"""

from functools import reduce
import operator

import ruamel.yaml as ruamel_yaml


def get_from_dict(data_dict, map_list):
    return reduce(operator.getitem, map_list, data_dict)


def apply_to_dict(data_dict, map_list, func, args):
    getattr(get_from_dict(data_dict, map_list[:-1])[map_list[-1]], func)(*args)


def save_debug_data(model_run, debug_data, out_file):
    # README: currently based on ruamel.yaml 0.15 which is a mix of old
    # and new API - possibly needs a bit of rewriting once ruamel.yaml
    # has progressed a bit further
    yaml = ruamel_yaml.YAML()

    model_run_debug = model_run.copy()
    del model_run_debug['timeseries_data']  # Can't be serialised!

    # Turn sets in model_run into lists for YAML serialization
    for k, v in model_run_debug.sets.items():
        model_run_debug.sets[k] = list(v)

    debug_comments = debug_data['comments']
    debug_yaml = yaml.load(yaml.dump(model_run_debug.as_dict()))
    for k in debug_comments.model_run.keys_nested():
        v = debug_comments.model_run.get_key(k)
        keys = k.split('.')
        apply_to_dict(debug_yaml, keys[:-1], 'yaml_add_eol_comment', (v, keys[-1]))

    dumper = ruamel_yaml.dumper.RoundTripDumper
    dumper.ignore_aliases = lambda self, data: True

    with open(out_file, 'w') as f:
        ruamel_yaml.dump(
            debug_yaml, f,
            Dumper=dumper, default_flow_style=False
        )
