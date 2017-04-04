"""
Copyright (C) 2013-2017 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

output.py
~~~~~~~~~

Functionality for generating model outputs.

"""

import numpy as np

from . import utils


def generate_constraints(solution, output_path=None, specifier='max',
                         techs=None, constraints=None,
                         include_transmission=True, transmission_techs=None,
                         transmission_constraints=['e_cap'], fillna=None,
                         map_numeric=None, map_any=None):
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

    Use ``map_numeric`` and ``map_any`` to give functions that will be
    applied to numeric or any value to modify it.

    """
    # TODO: add unit tests
    excluded_vars = ['e_cap_net']

    def _setkey(d, key, value):  # fillna, round, multiply passed implicitly
        if fillna is not None and np.isnan(value):
            value = fillna
        if map_numeric:
            try:
                # TypeError if not a number, we don't want to multiply strings
                value = map_numeric(value)
            except TypeError:
                pass  # Ignore if not a number
        if map_any:
            value = map_any(value)
        d.set_key(key, value)

    d = utils.AttrDict()

    # Get a list of default constraints, so that we know which constraints
    # exist in a form that includes sub-constraints (like '.max')
    o = solution.config_model
    possible_constraints = list(o.techs.defaults.constraints.keys())
    default_constraints = list(o.techs.defaults.constraints.keys_nested())
    max_min_equals_constraints = set([c.split('.')[0]
                                      for c in default_constraints
                                      if '.max' in c])

    # Set up the list of locations, techs, constraints
    locations = solution.coords['x'].values
    techs_in_solution = [i for i in solution.coords['y'].values if ':' not in i]
    if not techs:
        techs = techs_in_solution
    if not constraints:
        constraints = [i for i in possible_constraints if i in solution.data_vars]

    # Non-transmission techs
    # FIXME only include techs that are allowed by the model_config at
    # a given location
    key_string = 'locations.{0}.override.{1}.constraints.{2}'
    for x in locations:
        for y in techs:
            for var in [v for v in constraints
                        if v not in excluded_vars]:
                key = key_string.format(x, y, var)
                if var in max_min_equals_constraints:
                    key += '.{}'.format(specifier)
                value = solution[var].loc[dict(x=x, y=y)].item()
                if not np.isnan(value):
                    _setkey(d, key, value)

    # Transmission techs
    if include_transmission:
        transmission_techs_in_sol = [i for i in solution.coords['y'].values
                                     if ':' in i]
        if not transmission_techs:
            transmission_techs = set([i.split(':')[0]
                                      for i in transmission_techs_in_sol])
        if not transmission_constraints:
            transmission_constraints = [i for i in possible_constraints if i in solution.data_vars]

        d.links = utils.AttrDict()
        t_key_string = 'links.{0}.{1}.constraints.{2}'
        for x in locations:
            for y in transmission_techs_in_sol:
                for var in [v for v in transmission_constraints
                            if v not in excluded_vars]:
                    value = solution[var].loc[dict(x=x, y=y)].item()
                    y_bare, x_rem = y.split(':')
                    if x_rem == x:
                        continue
                    g = lambda x: o.links.get_key(x, default=False)
                    if (g('{},{}.{}'.format(x, x_rem, y_bare)) is not False or
                            g('{},{}.{}'.format(x_rem, x, y_bare)) is not False):
                        exists = True
                    else:
                        exists = False
                    if exists and y_bare in transmission_techs and not np.isnan(value):
                        key = t_key_string.format(x + ',' + x_rem, y_bare, var)
                        if var in max_min_equals_constraints:
                            key += '.{}'.format(specifier)
                        _setkey(d, key, value)

    if output_path is not None:
        d.to_yaml(output_path)
    else:
        return d
