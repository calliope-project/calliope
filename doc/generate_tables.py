"""
Copyright (C) 2013-2018 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

generate_tables.py
~~~~~~~~~~~~~~~~~~

Parses, extracts and formats default configuration data
for the documentation.

"""

import csv
import ruamel.yaml as yaml


def get_section(commented_map):
    """ Returns list of (setting, default, comment) tuples processed
    from a YAML section."""
    result = []
    for k, v in commented_map.items():
        comment = commented_map.ca.items[k][2].value.strip('#').strip()
        if '¦' in comment:
            comment = comment.split('¦')
            comment[0] = comment[0].replace('name:', '').strip()
            comment[1] = comment[1].replace('unit:', '').strip()
            comment[2] = comment[2].strip()
        else:
            comment = [comment]
        # Special case: empty dict gets turned into CommentedMap,
        # turn it back
        if isinstance(v, yaml.comments.CommentedMap):
            v = {}
        result.append((k, v, *comment))
    return result


def write_csv(filename, iterable):
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(iterable)


def process():
    with open('../calliope/config/defaults.yaml', 'r') as f:
        defaults = yaml.round_trip_load(f)

    write_csv(
        './user/includes/default_essentials.csv',
        get_section(defaults['default_tech']['essentials'])
    )
    write_csv(
        './user/includes/default_constraints.csv',
        get_section(defaults['default_tech']['constraints'])
    )
    write_csv(
        './user/includes/default_costs.csv',
        get_section(defaults['default_tech']['costs']['default'])
    )

    with open('../calliope/config/model.yaml', 'r') as f:
        model = yaml.round_trip_load(f)

    write_csv(
        './user/includes/model_settings.csv',
        get_section(model['model'])
    )
    write_csv(
        './user/includes/run_settings.csv',
        get_section(model['run'])
    )

    for tech_group in model['tech_groups']:
        defaults = {
            'essentials': model['tech_groups'][tech_group].get('essentials', {}),
            'constraints': model['tech_groups'][tech_group].get('constraints', {}),
            'costs': model['tech_groups'][tech_group].get('costs', {})
        }
        with open('./user/includes/basetech_{}.yaml'.format(tech_group), 'w') as f:
            f.write(yaml.dump(defaults, Dumper=yaml.RoundTripDumper))

        required_allowed = {
            'required_constraints': model['tech_groups'][tech_group].get('required_constraints', {}),
            'allowed_constraints': model['tech_groups'][tech_group].get('allowed_constraints', {}),
            'allowed_costs': model['tech_groups'][tech_group].get('allowed_costs', {})
        }
        with open('./user/includes/required_allowed_{}.yaml'.format(tech_group), 'w') as f:
            f.write(yaml.dump(required_allowed, Dumper=yaml.RoundTripDumper))


# Run the process function when exec'd -- this is bad style, yes
process()
