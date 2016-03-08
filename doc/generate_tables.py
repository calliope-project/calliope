"""
Copyright (C) 2013-2016 Stefan Pfenninger.
Licensed under the Apache 2.0 License (see LICENSE file).

generate_tables.py
~~~~~~~~~~~~~~~~~~

Parses, extracts and formats default configuration data
for the documentation.

"""

import csv


def get_line(f, strip_comments=False):
    line = f.readline()
    if strip_comments:
        line = line.split('#', 1)[0].rstrip()
    return line


def seek(f, line_content, strip_comments=False):
    line = get_line(f, strip_comments)
    while line.strip() != line_content:
        line = get_line(f, strip_comments)
    return line


def leading_spaces(line):
    return len(line) - len(line.lstrip(' '))


def get_block(filename, block):
    with open(filename, 'r') as f:
        line = seek(f, block, strip_comments=True)
        spaces = leading_spaces(line)
        lines = []
        while True:
            line = f.readline()
            # Check if we reached beginning of next tech block, if so, break
            if not line or leading_spaces(line) == spaces:
                break
            lines.append(line[spaces + 4:])  # Remove leading indentation
    return ''.join(lines)


def get_section(f, outside_indentation):
    collector = []

    while True:
        line = f.readline()
        # If leading spaces are outside_indentation, we've reached the end
        # of the section
        if leading_spaces(line) == outside_indentation:
            break
        setting, line = line.split(':', maxsplit=1)
        try:
            default, comment = line.split('#', maxsplit=1)
        except ValueError:  # need more than 1 value to unpack
            default = line
            comment = ''
        line_tuple = (setting.strip(), default.strip(), comment.strip())
        # Skip things marked as 'UNCODUMENTED'
        if not line_tuple[2].startswith('UNDOCUMENTED'):
            collector.append(line_tuple)

    return collector, line


def write_csv(filename, iterable):
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(iterable)


def process():
    filename = '../calliope/config/defaults.yaml'

    with open(filename, 'r') as f:
        # Seek until 'techs:'
        line = seek(f, 'techs:')
        # Make sure we're looking at 'defaults:'
        line = f.readline()
        assert line.strip() == 'defaults:'

        # Read 'constraints:'
        line = seek(f, 'constraints:')  # Seek until 'constraints:'
        constraints, line = get_section(f, 8)

        # Read 'costs.default:', which directly follows 'constraints:'
        assert line.startswith('        costs:')
        line = get_line(f)  # Skip one line
        costs, line = get_section(f, 8)

    # Write files
    write_csv('./user/includes/default_constraints.csv', constraints)
    write_csv('./user/includes/default_costs.csv', costs)

    # Read entire depreciation block
    depreciation = get_block(filename, 'depreciation:')
    with open('./user/includes/default_depreciation.yaml', 'w') as f:
        f.write(depreciation)

    # Process the abstract base technologies
    for tech in ['supply', 'demand', 'unmet_demand',
                 'unmet_demand_as_supply_tech', 'storage',
                 'transmission', 'conversion']:
        block = get_block(filename, tech + ':')
        with open('./user/includes/basetech_{}.yaml'.format(tech), 'w') as f:
            f.write(block)

# Run the process function when exec'd -- this is bad style, yes
process()
