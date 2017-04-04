"""
Copyright (C) 2013-2017 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

locations.py
~~~~~~~~~~~~

Functions to deal with locations and their configuration.

"""

import pandas as pd

from . import utils


def _generate_location(location, items, techs):
    """
    Returns a dict for a given location. Dict keys for permitted technologies
    are the only ones that don't start with '_'.

    Args:
        location : (str) name of the location
        items : (AttrDict) location settings
        techs : (list) list of available technologies
    """
    # Mandatory basics
    within = items.get('within', None)
    level = items.get('level', None)
    d = {'_location': location, '_level': level,
         '_within': str(within)}
    # Override
    if 'override' in items:
        for k in items.override.keys_nested():
            target_key = '_override.' + k
            try:
                d[target_key] = items.override.get_key(k)
            except KeyError:
                # Allow locations to not define 'within'
                if k == 'within':
                    d[target_key] = None
    # Permitted echnologies
    for y in techs:
        if y in items.techs:
            d[y] = 1
        else:
            d[y] = 0
    return d


def explode_locations(k):
    """Expands the given key ``k``. ``k``s of the form ``'1--3'`` or
    ``'1,2,3'`` are both expanded into the list ``['1', '2', '3']``.

    Can deal with any combination, e.g. ``'1--3,6,9--11,a'`` results in::

        ['1', '2', '3', '6', '9', '10', '11', 'a']

    Always returns a list, even if ``k`` is just a simple key,
    i.e. ``explode_locations('1')`` returns ``['1']``.

    """
    # Ensure sure we don't pass in other things
    assert isinstance(k, str)
    finalkeys = []
    subkeys = k.split(',')
    for sk in subkeys:
        if '--' in sk:
            begin, end = sk.split('--')
            finalkeys += [str(i).strip()
                          for i in range(int(begin), int(end) + 1)]
        else:
            finalkeys += [sk.strip()]
    if finalkeys == [] or finalkeys == ['']:
        raise KeyError('Empty key')
    return finalkeys


def set_location_levels(locations):
    locset = set(locations.keys())

    curr_lvl = 0

    for l in tuple(locset):
        if 'within' not in locations[l] or locations[l].within is None:
            locations[l].level = curr_lvl
            locset.remove(l)

    while len(locset) > 0:
        curr_lvl += 1
        to_subtract = set()
        for l in tuple(locset):
            if locations[l].within not in locset:
                locations[l].level = curr_lvl
                to_subtract.add(l)
        locset = locset - to_subtract

    return locations


def process_locations(d):
    """
    Process locations by taking an AttrDict that may include compact keys
    such as ``1,2,3``, and returning an AttrDict with exactly one key per
    location with all of its settings.

    """
    def _set_loc_key(d, k, value):
        """Set key ``k`` in ``d`` to ``value```."""
        if k in d:
            try:
                d[k].union(value)
            except KeyError as e:
                raise KeyError('Problem at location {}: {}'.format(k, str(e)))
        else:
            d[k] = value
    loc_dict = utils.AttrDict()
    for key in d:
        if ('--' in key) or (',' in key):
            key_locs = explode_locations(key)
            for subkey in key_locs:
                _set_loc_key(loc_dict, subkey, d[key].copy())
        else:
            _set_loc_key(loc_dict, key, d[key].copy())
    locations = set_location_levels(loc_dict)
    return locations


def generate_location_matrix(d, techs):
    """Generate a pandas DataFrame indexed by locations, containing a column
    for each technology in `techs` and 1 if that location is allowed to
    use the technology, else 0.

    The DataFrame also contains _level and _within columns for grouping
    locations into layers and zones.

    """
    rows = []
    for k, v in sorted(d.items()):
        rows.append(_generate_location(k, v, techs))
    df = pd.DataFrame.from_records(rows)
    df.index = df._location
    df = df.drop(['_location'], axis=1)
    return df
