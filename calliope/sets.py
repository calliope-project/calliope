"""
Copyright (C) 2013-2017 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

sets.py
~~~~~~~

Sets & sub-sets defined to reduce size of decision variables & constraints.

The markers preceded with `###` are used to auto-include the set description
in the documentation

Sets are:

###PART TO INCLUDE IN DOCUMENTATION STARTS HERE###

Main sets & sub-sets
====================

Technologies:

* ``m.y_demand``: all demand sources
    * ``m.y_sd_r_area``: if any r_area constraints are defined (shared)
    * ``m.y_sd_finite_r``: if finite resource limit is defined (shared)
* ``m.y_supply``: all basic supply technologies
    * ``m.y_sd_r_area``: if any r_area constraints are defined (shared)
    * ``m.y_sd_finite_r``: if finite resource limit is defined (shared)
* ``m.y_storage``: specifically storage technologies
* ``m.y_supply_plus``: all supply+ technologies
    * ``m.y_sp_r_area``: If any r_area constraints are defined
    * ``m.y_sp_finite_r``: if finite resource limit is defined
    * ``m.y_sp_r2``: if secondary resource is allowed
* ``m.y_conversion``: all basic conversion technologies
* ``m.y_conversion_plus``: all conversion+ technologies
    * ``m.y_cp_2out``: secondary carrier(s) out
    * ``m.y_cp_3out``: tertiary carrier(s) out
    * ``m.y_cp_2in``: secondary carrier(s) in
    * ``m.y_cp_3in``: tertiary carrier(s) in
* ``m.y_transmission``: all transmission technologies
* ``m.y_unmet``: dummy supply technologies to log

Locations:

* ``m.x_trans``: all transmission locations
* ``m.x_r``: all locations which act as system sources/sinks
* ``m.x_store``: all locations in which storage is allowed

Shared subsets
==============

* ``m.y_finite_r``: shared between y_demand, y_supply, and y_supply_plus. Contains:
    * ``m.y_sd_finite_r``
    * ``m.y_sp_finite_r``
* ``m.y_r_area``: shared between y_demand, y_supply, and y_supply_plus. Contains:
    * ``m.y_sd_r_area``
    * ``m.y_sp_r_area``

Meta-sets
=========

Technologies:

* ``m.y``: all technologies, includes:
    * ``m.y_demand``
    * ``m.y_supply``
    * ``m.y_storage``
    * ``m.y_supply_plus``
    * ``m.y_conversion``
    * ``m.y_conversion_plus``
    * ``m.y_transmission``
* ``m.y_sd``: all basic supply & demand technologies, includes:
    * ``m.y_demand``
    * ``m.y_supply``
* ``m.y_store``: all technologies that have storage capabilities, includes:
    * ``m.y_storage``
    * ``m.y_supply_plus``

Locations:

* ``m.x``: all locations, includes:
    * ``m.x_trans``
    * ``m.x_r``
    * ``m.x_store``

###PART TO INCLUDE IN DOCUMENTATION ENDS HERE###

"""

import numpy as np

from . import exceptions
from . import transmission


def init_set_y(model, _x):
    """
    Initialise sets and subsets of y
    """

    # Subset of transmission technologies, if any defined
    # Used to initialize transmission techs further below
    # (not yet added to main `y` set here)
    links = model.config_model.get_key('links', None)
    if links:
        _y_trans = transmission.get_transmission_techs(links)
        transmission_techs = list(set([list(v.keys())[0]
                                  for k, v in links.items()]))
        _x_trans = set()
        for link in links.keys():
            locs = link.split(',')
            _x_trans.add(locs[0])
            _x_trans.add(locs[1])
        _x_trans = list(_x_trans)
    else:
        _y_trans = []
        _x_trans = []
        transmission_techs = []

    _y = set()
    _x = _x.copy()
    for k, v in model.config_model.locations.items():
        if 'techs' in v.keys():
            if set(v.techs).intersection(transmission_techs): # transmission nodes have only tranmission techs in techs list
                _x.remove(k)
            else:
                for y in v.techs:
                    if y in model.config_model.techs:
                        _y.add(y)
                    else:
                        e = exceptions.ModelError
                        raise e('Location `{}` '
                                'uses undefined tech `{}`.'.format(k, y))
        else:
            e = exceptions.ModelError
            raise e('The region `' + k + '` does not allow any '
                    'technologies via `techs` nor does it have '
                    'links to other locations. Must give at '
                    'least one technology per region or link '
                    'to other regions.')
    _y = list(_y)

    # Potentially subset _y
    if model.config_run.get_key('subset_y', default=False):
        _y = [y for y in _y if y in model.config_run.subset_y]

    # Subset of technologies that are demand sources
    _y_demand = [y for y in _y if model.ischild(y, of='demand')]

    # Subset of supply technologies
    _y_supply = [y for y in _y if model.ischild(y, of='supply')]

    # Subset of supply_plus technologies
    _y_supply_plus = [y for y in _y if model.ischild(y, of='supply_plus')]

    # Subset of storage technologies
    _y_storage = [y for y in _y if model.ischild(y, of='storage')]

    # Subset of conversion technologies

    _y_conversion = [y for y in _y if model.ischild(y, of='conversion')]

    # Subset of conversion+ technologies
    _y_conversion_plus = [y for y in _y if model.ischild(y, of='conversion_plus')]

    # Subset of unmet technologies
    _y_unmet = [y for y in _y if model.ischild(y, of='unmet_demand')]

    # Subset of basic supply & demand technologies
    _y_sd = _y_supply + _y_demand + _y_unmet

    # Subset of storage technologies
    _y_store = [y for y in np.concatenate((_y_storage,_y_supply_plus)) if
                any([model.get_option(y + '.constraints.s_cap.max', x=x) +
                     model.get_option(y + '.constraints.s_time.max', x=x) +
                     model.get_option(y + '.constraints.s_cap.equals', x=x) +
                     model.get_option(y + '.constraints.c_rate', x=x)
                     for x in _x])]

    _y_sd_r_area = [y for y in _y_sd if
                    any([model.get_option(y + '.constraints.r_area.max', x=x) +
                         model.get_option(y + '.constraints.r_area.equals', x=x) +
                         model.get_option(y + '.constraints.r_area_per_e_cap', x=x)
                         for x in _x])]
    _y_sp_r_area = [y for y in _y_supply_plus if
                    any([model.get_option(y + '.constraints.r_area.max', x=x) +
                         model.get_option(y + '.constraints.r_area.equals', x=x) +
                         model.get_option(y + '.constraints.r_area_per_e_cap', x=x)
                         for x in _x])]

    _y_r_area = _y_sd_r_area + _y_sp_r_area

    _y_sp_r2 = [y for y in _y_supply_plus if
               any([model.get_option(y + '.constraints.allow_r2', x=x)
                    for x in _x])]

    _y_cp_2out = [y for y in _y_conversion_plus
                  if model.get_option(y + '.carrier_out_2')]
    _y_cp_3out = [y for y in _y_conversion_plus
                  if model.get_option(y + '.carrier_out_3')]
    _y_cp_2in = [y for y in _y_conversion_plus
                 if model.get_option(y + '.carrier_in_2')]
    _y_cp_3in = [y for y in _y_conversion_plus
                 if model.get_option(y + '.carrier_in_3')]

    # subset of technologies allowing export
    _y_export = [y for y in _y
                 if any([model.get_option(y + '.export', x=x) for x in _x])]
    sets = {
        'y': _y,
        'y_demand': _y_demand,
        'y_supply': _y_supply,
        'y_supply_plus': _y_supply_plus,
        'y_storage': _y_storage,
        'y_conversion': _y_conversion,
        'y_conversion_plus': _y_conversion_plus,
        'y_unmet': _y_unmet,
        'y_sd': _y_sd,
        'y_store': _y_store,
        'y_sd_r_area': _y_sd_r_area,
        'y_sp_r_area': _y_sp_r_area,
        'y_r_area': _y_r_area,
        'y_sp_r2': _y_sp_r2,
        'y_cp_2out': _y_cp_2out,
        'y_cp_3out': _y_cp_3out,
        'y_cp_2in': _y_cp_2in,
        'y_cp_3in': _y_cp_3in,
        'y_transmission': _y_trans,
        'techs_transmission': transmission_techs,
        'x_transmission': _x_trans,
        'y_export': _y_export,
        'x_transmission_plus': _x
    }


    return sets

def init_set_x(model):
    """
    Add subsets of x. x_transmission already set in init_set_y
    This function has to be run *after* init_set_y to ensure technology sets are
    already included in model._sets.
    """
    locations = model.config_model.locations
    # All locations which interact with the spatial boundary
    _x_r = [x for x in model._sets['x'] if
            set(model._sets['y_supply_plus'] +
            model._sets['y_sd']).intersection(locations[x].techs)]
    # All locations in which energy is stored
    _x_store = [x for x in model._sets['x'] if
                set(model._sets['y_store']).intersection(locations[x].techs)]
    # All locations which demand energy
    _x_demand = [x for x in model._sets['x'] if
                 set(model._sets['y_demand']).intersection(locations[x].techs)]
    # All locations in which energy is converted between carriers
    _x_conversion = [x for x in model._sets['x'] if
                     set(model._sets['y_conversion_plus'] +
                     model._sets['y_conversion']).intersection(locations[x].techs)]
    # All locations in which energy is exported
    _x_export = [x for x in model._sets['x'] if
                     set(model._sets['y_export']).intersection(locations[x].techs)]

    return {
        'x_r': _x_r,
        'x_store': _x_store,
        'x_demand': _x_demand,
        'x_conversion': _x_conversion,
        'x_export': _x_export
    }

def init_set_c(model):
    """
    Add all energy carriers to the set `c`.
    """
    def _check_if_dict(option):
        if isinstance(option, dict):
            return option.keys()
        else:
            return [option]
    _c = set()
    sets = model._sets
    for y in sets['y']:  # Only add carriers for allowed technologies
        if y in sets['y_supply'] or y in sets['y_supply_plus']:
            _c.update([model.get_option(y + '.carrier',
                                default=y + '.carrier_out')])
        if y in sets['y_demand']:
            _c.update([model.get_option(y + '.carrier',
                                default=y + '.carrier_in')])
        if y in sets['y_transmission'] or y in sets['y_storage']:
            _c.update([model.get_option(y + '.carrier')])
        if y in sets['y_conversion']:
            _c.update([model.get_option(y + '.carrier_in')])
            _c.update([model.get_option(y + '.carrier_out')])
        if y in sets['y_conversion_plus']:
            _c.update(_check_if_dict(model.get_option(y + '.carrier_in')))
            _c.update(_check_if_dict(model.get_option(y + '.carrier_out')))
        if y in sets['y_cp_2out']:
            _c.update(_check_if_dict(model.get_option(y + '.carrier_out_2')))
        if y in sets['y_cp_3out']:
            _c.update(_check_if_dict(model.get_option(y + '.carrier_out_3')))
        if y in sets['y_cp_2in']:
            _c.update(_check_if_dict(model.get_option(y + '.carrier_in_2')))
        if y in sets['y_cp_3in']:
            _c.update(_check_if_dict(model.get_option(y + '.carrier_in_3')))
    _c = _c.difference([True, False])
    return list(_c)



def init_y_trans(model):
    # Add transmission technologies to y, if any defined
    _y_trans = model._sets['y_transmission']

    if _y_trans:
        model._sets['y'].extend(_y_trans)

        # Add transmission tech columns to locations matrix
        for y in _y_trans:
            model._locations[y] = 0

        # Create representation of location-tech links
        tree = transmission.explode_transmission_tree(
            model.config_model.links, model._sets['x_transmission']
        )

        # Populate locations matrix with allowed techs and overrides
        if tree:
            for x in tree:
                for y in tree[x]:
                    # Allow the tech
                    model._locations.at[x, y] = 1
                    # Add constraints if needed. Speficying one way transmission
                    # leads to e_prod in the direction opposite to the defined
                    # link being set to False. E.g. if 'a,b' is one way for tech
                    # 'y' then y:a.constraints.e_prod @ x=b -> False
                    for c in tree[x][y].keys_nested():
                        if c == 'constraints.one_way':
                            colname = '_override.' + y + '.constraints.e_prod'
                        else:
                            colname = '_override.' + y + '.' + c
                        if colname not in model._locations.columns:
                            model._locations[colname] = np.nan
                        if (c == 'constraints.one_way'
                                and '{},{}'.format(y.split(':')[1], x) in
                                model.config_model.links.keys()):
                            # set the option in model._locations and model.config_model.locations
                            model.set_option(y + '.constraints.e_prod', False, x=x)
                        else:
                            model._locations.at[x, colname] = tree[x][y].get_key(c)
