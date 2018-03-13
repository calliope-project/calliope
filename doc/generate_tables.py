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

    essentials = get_section(defaults['default_tech']['essentials'])
    constraints = get_section(defaults['default_tech']['constraints'])
    costs = get_section(defaults['default_tech']['costs']['default'])

    # Write files
    write_csv('./user/includes/default_essentials.csv', essentials)
    write_csv('./user/includes/default_constraints.csv', constraints)
    write_csv('./user/includes/default_costs.csv', costs)

    # # Process the abstract base technologies
    # # FIXME: here actually read model.yaml
    # for tech in ['supply', 'supply_plus', 'demand'
    #              'storage',
    #              'transmission', 'conversion', 'conversion_plus']:
    #     block = get_block(filename, tech + ':')
    #     with open('./user/includes/basetech_{}.yaml'.format(tech), 'w') as f:
    #         f.write(block)


# Run the process function when exec'd -- this is bad style, yes
process()
