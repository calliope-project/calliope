"""
Copyright (C) 2013-2017 Stefan Pfenninger.
Licensed under the Apache 2.0 License (see LICENSE file).

sets.py
~~~~~~~

"""

import numpy as np

from . import exceptions
from . import transmission


def init_set_y(model, _x):
    _y = set()
    try:
        for k, v in model.config_model.locations.items():
            for y in v.techs:
                if y in model.config_model.techs:
                    _y.add(y)
                else:
                    e = exceptions.ModelError
                    raise e('Location `{}` '
                            'uses undefined tech `{}`.'.format(k, y))
    except KeyError:
        e = exceptions.ModelError
        raise e('The region `' + k + '` does not allow '
                'any technologies via `techs`. Must give '
                'at least one technology per region.')
    _y = list(_y)

    # Potentially subset _y
    if model.config_run.get_key('subset_y', default=False):
        _y = [y for y in _y if y in model.config_run.subset_y]

    # Subset of transmission technologies, if any defined
    # Used to initialized transmission techs further below
    # (not yet added to main `y` set here)
    links = model.config_model.get_key('links', None)
    if links:
        _y_trans = transmission.get_transmission_techs(links)
        transmission_techs = list(set([list(v.keys())[0]
                                  for k, v in links.items()]))
    else:
        _y_trans = []
        transmission_techs = []

    # Subset of conversion technologies
    _y_conv = [y for y in _y if model.ischild(y, of='conversion')]

    # Subset of supply, demand, storage technologies
    _y_pc = [y for y in _y
             if not model.ischild(y, of='conversion')
             or model.ischild(y, of='transmission')]

    # Subset of technologies that define es_prod/es_con
    _y_prod = ([y for y in _y if not model.ischild(y, of='demand')]
               + _y_trans)
    _y_con = ([y for y in _y if not model.ischild(y, of='supply')]
              + _y_trans)

    # Subset of technologies that allow rb
    _y_rb = []
    for y in _y:
        for x in _x:
            if model.get_option(y + '.constraints.allow_rb', x=x) is True:
                _y_rb.append(y)
                break  # No need to look at other x

    # Subset of technologies with parasitics (carrier efficiency != 1.0)
    _y_p = []
    for y in _y:
        for x in _x:
            if model.get_option(y + '.constraints.c_eff', x=x) != 1.0:
                _y_p.append(y)
                break  # No need to look at other x

    sets = {
        'y': _y,
        'y_conv': _y_conv,
        'y_pc': _y_pc,
        'y_prod': _y_prod,
        'y_con': _y_con,
        'y_rb': _y_rb,
        'y_p': _y_p,
        'y_trans': _y_trans,
        'techs_transmission': transmission_techs

    }

    return sets


def init_y_trans(model):
    # Add transmission technologies to y, if any defined
    _y_trans = model._sets['y_trans']

    if _y_trans:
        model._sets['y'].extend(_y_trans)

        # Add transmission tech columns to locations matrix
        for y in _y_trans:
            model._locations[y] = 0

        # Create representation of location-tech links
        tree = transmission.explode_transmission_tree(
            model.config_model.links, model._sets['x']
        )

        # Populate locations matrix with allowed techs and overrides
        if tree:
            for x in tree:
                for y in tree[x]:
                    # Allow the tech
                    model._locations.at[x, y] = 1
                    # Add constraints if needed
                    for c in tree[x][y].keys_nested():
                        colname = '_override.' + y + '.' + c
                        if colname not in model._locations.columns:
                            model._locations[colname] = np.nan
                        model._locations.at[x, colname] = tree[x][y].get_key(c)
