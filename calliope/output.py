"""
Copyright (C) 2013-2015 Stefan Pfenninger.
Licensed under the Apache 2.0 License (see LICENSE file).

output.py
~~~~~~~~~

Functionality for generating model outputs.

"""

import numpy as np

from . import utils


def generate_constraints(solution, output_path=None, techs=None,
                         constraints=None,
                         include_transmission=True, transmission_techs=None,
                         transmission_constraints=['e_cap'], fillna=None):
    """
    Generate constraints from a given solution.

    If ``output_path`` is specified, write the resulting YAML to disk,
    else return the YAML string.

    Can use ``techs``, ``locations``, ``constraints``,
    ``include_transmission``, ``transmission_techs``,
    ``transmission_constraints`` to specify only a subset of the
    solution to be turned into constraints.

    If ``fillna`` set to something other than None, NA values will be
    replaced with the given value.

    """
    def _setkey(d, key, value, fillna):
        if fillna is not None and np.isnan(value):
            value = fillna
        d.set_key(key, value)

    d = utils.AttrDict()

    # Get a list of default constraints, so that we know which constraints
    # exist in a '_max' form
    o = solution.config_model
    default_constraints = list(o.techs.defaults.constraints.keys())

    # Set up the list of locations, techs, constraints
    locations = solution.parameters.major_axis
    techs_in_solution = [i for i in solution.parameters.minor_axis
                         if ':' not in i]
    if not techs:
        techs = techs_in_solution
    if not constraints:
        constraints = solution.parameters.items

    # Non-transmission techs
    # FIXME only include techs that are allowed by the model_config at
    # a given location
    key_string = 'locations.{0}.override.{1}.constraints.{2}'
    for x in locations:
        for y in techs:
            for var in constraints:
                key = key_string.format(x, y, var)
                if var + '_max' in default_constraints:
                    key += '_max'
                _setkey(d, key, solution.parameters.at[var, x, y], fillna)

    # Transmission techs
    if include_transmission:
        transmission_techs_in_sol = [i for i in solution.parameters.minor_axis
                                     if ':' in i]
        if not transmission_techs:
            transmission_techs = set([i.split(':')[0]
                                      for i in transmission_techs_in_sol])
            print(transmission_techs)
        if not transmission_constraints:
            transmission_constraints = solution.parameters.items

        d.links = utils.AttrDict()
        t_key_string = 'links.{0}.{1}.constraints.{2}'
        for x in locations:
            for y in transmission_techs_in_sol:
                for var in transmission_constraints:
                    value = solution.parameters.at[var, x, y]
                    y_bare, x_rem = y.split(':')
                    if x_rem == x:
                        continue
                    g = lambda x: o.links.get_key(x, default=False)
                    if (g('{},{}.{}'.format(x, x_rem, y_bare)) is not False or
                            g('{},{}.{}'.format(x_rem, x, y_bare)) is not False):
                        exists = True
                    else:
                        exists = False
                    if exists and y_bare in transmission_techs:
                        key = t_key_string.format(x + ',' + x_rem, y_bare, var)
                        if var + '_max' in default_constraints:
                            key += '_max'
                        _setkey(d, key, value, fillna)

    if output_path is not None:
        d.to_yaml(output_path)
    else:
        return d
