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
* ``m.y_export``: all technologies allowing export of their carrier outside the system
* ``m.y_purchase``: technology defining a 'purchase' cost, thus triggering an associated binary decision variable
* ``m.y_milp``: technology defining a 'units' maximum, minimum, or equality, thus triggering an associated integer decision variable

Locations:

* ``m.x_transmission``: all transmission locations
* ``m.x_r``: all locations which act as system sources/sinks
* ``m.x_conversion``: all locations in which there are conversion/conversion_plus technologies
* ``m.x_store``: all locations in which storage is allowed
* ``m.x_export``: locations allowing 'y_export' technologies to export outside the system
* ``m.x_purchase``: locations associated with 'y_purchased' technologies
* ``m.x_milp``: locations associated with 'y_milp' technologies

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
    * ``m.y_unmet``
    * ``m.y_export``
    * ``m.y_purchase``
    * ``m.y_milp``
* ``m.y_sd``: all basic supply & demand technologies, includes:
    * ``m.y_demand``
    * ``m.y_supply``
    * ``m.y_unmet``
* ``m.y_store``: all technologies that have storage capabilities, includes:
    * ``m.y_storage``
    * ``m.y_supply_plus``

Locations:

* ``m.x``: all locations, includes:
    * ``m.x_transmission``
    * ``m.x_r``
    * ``m.x_conversion``
    * ``m.x_store``
    * ``m.x_export``
    * ``m.x_purchase``
    * ``m.x_milp``

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
        links = {k:v for k, v in model.config_model.links.items() if
            len(set(k.split(",")).difference(_x)) == 0}
        _y_trans = transmission.get_transmission_techs(links)
        transmission_techs = list(set([list(v.keys())[0]
                                  for k, v in links.items()]))
        _x_trans = set()
        for link in links.keys():
            locs = link.split(',')
            _x_trans.add(locs[0])
            _x_trans.add(locs[1])
        # Avoid adding in locations that may have been removed by use of "subset_x"
        _x_trans = list(_x_trans.intersection(_x))
    else:
        _y_trans = []
        _x_trans = []
        transmission_techs = []

    _y = set()
    _x = _x.copy()

    for k, v in model.config_model.locations.items():
        if k not in _x:
            continue
        if 'techs' in v.keys():
            # transmission nodes have only transmission techs in techs list
            if set(v.techs).intersection(transmission_techs) and k in _x:
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
    _y_store = _y_storage + [y for y in _y_supply_plus if
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

    # subset of technologies associated with the model MILP functionality
    _y_milp = [y for y in _y if
               any([model.get_option(y + '.constraints.units.max', x=x) +
                    model.get_option(y + '.constraints.units.equals', x=x) +
                    model.get_option(y + '.constraints.units.min', x=x)
                    for x in _x])
                  ]
    # subset of technologies associated with a binary purchase variable
    _y_purchase = [y for y in _y if y not in _y_milp and
                   any([model.get_cost('purchase', y, k, x=x)
                        for x in _x for k in model._sets['k']])
                  ]
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
        'y_purchase': _y_purchase,
        'y_milp': _y_milp,
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
    # All locations with technologies associated with a MILP variable
    _x_milp = [x for x in model._sets['x'] if
                     set(model._sets['y_milp']).intersection(locations[x].techs)]
    # All locations with technologies associated with a binary purchase variable
    _x_purchase = [x for x in model._sets['x'] if
                     set(model._sets['y_purchase']).intersection(locations[x].techs)]


    return {
        'x_r': _x_r,
        'x_store': _x_store,
        'x_demand': _x_demand,
        'x_conversion': _x_conversion,
        'x_export': _x_export,
        'x_purchase': _x_purchase,
        'x_milp': _x_milp
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
        links = {k:v for k, v in model.config_model.links.items() if
            len(set(k.split(",")).difference(model._sets['x_transmission'])) == 0}
        tree = transmission.explode_transmission_tree(
            links, model._sets['x_transmission']
        )

        # Populate locations matrix with allowed techs and overrides
        if tree:
            for x in tree:
                for y in tree[x]:
                    # Allow the tech
                    model._locations.at[x, y] = 1
                    # Add constraints if needed. Specifying one way transmission
                    # leads to e_con in the direction of the link being set to
                    # False. E.g. if 'a,b' is one way for transmission tech 'y'
                    # then y:a.constraints.e_con @ x=b -> False. Here, this stops
                    # transmission from b to a (no carrier can be consumed by the
                    # tranmission technology at b).
                    for c in tree[x][y].keys_nested():
                        if c == 'constraints.one_way':
                            colname = '_override.' + y + '.constraints.e_con'
                        else:
                            colname = '_override.' + y + '.' + c
                        if colname not in model._locations.columns:
                            model._locations[colname] = np.nan
                        if (c == 'constraints.one_way'
                                and '{},{}'.format(y.split(':')[1], x) in
                                model.config_model.links.keys()):
                            # set the option in model._locations and model.config_model.locations
                            model.set_option(y + '.constraints.e_con', False, x=x)
                        else:
                            model._locations.at[x, colname] = tree[x][y].get_key(c)

def init_set_loc_tech(model):
    def create_loc_tech(tech):
        return [":".join([x, y]) for x in model._sets["x"]
        for y in model._sets[tech] if model._locations.at[x, y]]
    return_dict = dict()
    return_dict["loc_tech_area"] = create_loc_tech("y_r_area")
    return_dict["loc_tech_store"] = create_loc_tech("y_store")
    return_dict["loc_tech_storage"] = create_loc_tech("y_storage")
    return_dict["loc_tech_supply_plus_finite_r"] = create_loc_tech("y_sp_finite_r")
    return_dict["loc_tech"] = create_loc_tech("y")
    return_dict["loc_tech_r2"] = create_loc_tech("y_sp_r2")
    return_dict["loc_tech_export"] = create_loc_tech("y_export")
    return_dict["loc_tech_purchase"] = create_loc_tech("y_purchase")
    return_dict["loc_tech_milp"] = create_loc_tech("y_milp")
    return_dict["loc_tech_transmission"] = create_loc_tech("y_transmission")
    return_dict["loc_tech_finite_r"] = create_loc_tech("y_finite_r")
    return_dict["loc_tech_demand"] = create_loc_tech('y_demand')
    return_dict["loc_tech_supply"] = create_loc_tech('y_supply')
    return_dict["loc_tech_supply_plus"] = create_loc_tech('y_supply_plus')
    return_dict["loc_tech_conversion"] = create_loc_tech('y_conversion')
    return_dict["loc_tech_conversion_plus"] = create_loc_tech('y_conversion_plus')
    return_dict["loc_tech_unmet"] = create_loc_tech('y_unmet')
    return_dict["loc_tech_2out"] = create_loc_tech('y_cp_2out')
    return_dict["loc_tech_3out"] = create_loc_tech('y_cp_3out')
    return_dict["loc_tech_2in"] = create_loc_tech('y_cp_2in')
    return_dict["loc_tech_3in"] = create_loc_tech('y_cp_3in')

    return return_dict
