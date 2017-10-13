"""
Copyright (C) 2013-2017 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

preprocess_sets.py
~~~~~~~~~~~~~~~~~~

Sets & sub-sets defined to reduce size of decision variables & constraints.

The markers preceded with `###` are used to auto-include the set description
in the documentation

###PART TO INCLUDE IN DOCUMENTATION STARTS HERE###

Main sets
=========

* timesteps
* techs
* techs_non_transmission
* techs_transmission
* locs
* costs
* carriers

Location-technology subsets
===========================

* loc_techs

Technology groups

* loc_techs_storage
* loc_techs_transmission
* loc_techs_demand
* loc_techs_supply
* loc_techs_supply_plus
* loc_techs_conversion
* loc_techs_conversion_plus
* loc_techs_unmet_demand

Subsets based on active constraints

* loc_techs_area
* loc_techs_store
* loc_techs_finite_resource
* loc_techs_supply_plus_finite_resource
* loc_techs_export
* loc_techs_purchase
* loc_techs_milp
* loc_techs_out_2
* loc_techs_out_3
* loc_techs_in_2
* loc_techs_in_3

###PART TO INCLUDE IN DOCUMENTATION ENDS HERE###

"""

from itertools import product

import numpy as np

from .. import utils


def generate_simple_sets(model_run):
    """
    Generate basic sets for a given pre-processed ``model_run``.

    Parameters
    ----------
    model_run : AttrDict

    """
    sets = utils.AttrDict()

    flat_techs = model_run.techs.as_dict(flat=True)
    flat_locations = model_run.locations.as_dict(flat=True)

    sets.resources = set(
        v for k, v in flat_techs.items()
        if '.carrier' in k)

    sets.carriers = sets.resources - set(['resource'])

    sets.carrier_tiers = set(key.split('.carrier_')[1]
        for key in flat_techs.keys()
        if '.carrier_' in key)

    sets.costs = set(
        k.split('costs.')[-1].split('.')[0]
        for k in flat_locations.keys()
        if '.costs.' in k)

    sets.locs = set(model_run.locations.keys())

    sets.techs_non_transmission = set(
        k for k, v in model_run.techs.items()
        if v.inheritance[-1] != 'transmission')

    sets.techs_transmission_names = set(
        k for k, v in model_run.techs.items()
        if v.inheritance[-1] == 'transmission')

    # This builds the "tech:loc" expansion of transmission technologies
    techs_transmission = set()
    for loc_name, loc_config in model_run.locations.items():
        for link_name, link_config in loc_config.get('links', {}).items():
            for tech_name in link_config.techs:
                techs_transmission.add('{}:{}'.format(tech_name, link_name))
    sets.techs_transmission = techs_transmission

    sets.techs = sets.techs_non_transmission | sets.techs_transmission

    # `timesteps` set is only built later, when processing time series
    # input data
    sets.timesteps = set()

    return sets


def concat_with_colon(iterable):
    """
    Take an interable containing iterables of strings,
    return a list of strings concatenating inner iterables with ':'.

    E.g.:
    ``
    result = concat_with_colon([('x', 'y', 'z'), ('1', '2')])
    result == ['x:y:z', '1:2']
    ``

    """
    return [':'.join(i) for i in iterable]


def _check_finite_resource(loc_techs_config, k):
    """
    Check in loc_techs_config (AttrDict) whether `k` (str) has
    specified a finite `resource`.

    """
    return ('resource' in loc_techs_config[k].constraints and
            (isinstance(loc_techs_config[k].constraints.resource, str) or
                not np.isinf(loc_techs_config[k].constraints.resource))
            )


def generate_loc_tech_sets(model_run, simple_sets):
    """
    Generate loc-tech sets for a given pre-processed ``model_run``

    Parameters
    ----------
    model_run : AttrDict
    simple_sets : AttrDict
        Simple sets returned by ``generate_simple_sets(model_run)``.

    """
    sets = utils.AttrDict()

    ##
    # First deal with transmission techs, which can show up only in
    # loc_techs_transmission, loc_techs_milp, and loc_techs_purchase
    ##

    # All `tech:loc` expanded transmission technologies
    sets.loc_techs_transmission = set(concat_with_colon([
        (i, u, j) for i, j, u in product(  # (loc, loc, tech) product
            simple_sets.locs,
            simple_sets.locs,
            simple_sets.techs_transmission_names)
        if model_run.get_key(
            'locations.{}.links.{}.techs.{}'.format(i, j, u), None
        )
    ]))

    # A dict of transmission tech config objects
    # to make parsing for set membership easier
    loc_techs_transmission_config = {
        k: model_run.get_key(
            'locations.{0}.links.{2}.techs.{1}'.format(*k.split(':'))
        )
        for k in sets.loc_techs_transmission
    }

    ##
    # Now deal with the rest of the techs and other sets
    ##

    # Only loc-tech combinations that actually exist
    sets.loc_techs_non_transmission = set(concat_with_colon([
        (l, t) for l, t in product(
            simple_sets.locs,
            simple_sets.techs_non_transmission)
        if model_run.get_key('locations.{}.techs.{}'.format(l, t), None)
    ]))

    sets.loc_techs = sets.loc_techs_non_transmission | sets.loc_techs_transmission

    # A dict of non-transmission tech config objects
    # to make parsing for set membership easier
    loc_techs_config = {
        k: model_run.get_key(
            'locations.{}.techs.{}'.format(*k.split(':'))
        )
        for k in sets.loc_techs_non_transmission
    }

    ##
    # Sets based on membership in abstract base technology groups
    ##

    for group in [
            'storage', 'demand', 'supply', 'supply_plus',
            'unmet_demand', 'conversion', 'conversion_plus']:
        tech_set = set(
            k for k in sets.loc_techs_non_transmission
            if model_run.techs[k.split(':')[1]].essentials.parent == group
        )
        sets['loc_techs_{}'.format(group)] = tech_set

    ##
    # Sets based on specific constraints being active
    ##

    # Technologies that specify resource_area constraints
    sets.loc_techs_area = set(
        k for k in sets.loc_techs_non_transmission
        if any('.resource_area' in i
               for i in loc_techs_config[k].constraints.keys_nested())
    )

    # Technologies that define storage, which can include `supply_plus`
    # and `storage` groups
    sets.loc_techs_store = set(
        k for k in sets.loc_techs_supply_plus
        if any('.storage_' in i
               for i in loc_techs_config[k].constraints.keys_nested())
    ) | sets.loc_techs_storage

    # `supply` or `demand` technologies that specify a finite resource
    sets.loc_techs_finite_resource = set(
        k for k in (sets.loc_techs_supply | sets.loc_techs_demand)
        if _check_finite_resource(loc_techs_config, k)
    )

    # `supply_plus` technologies that specify a finite resource
    sets.loc_techs_supply_plus_finite_resource = set(
        k for k in sets.loc_techs_supply_plus
        if _check_finite_resource(loc_techs_config, k)
    )

    # Technologies that allow export
    sets.loc_techs_export = set(
        k for k in sets.loc_techs_non_transmission
        if 'export_carrier' in loc_techs_config[k].constraints
    )

    # Technologies that allow purchasing discrete units
    # NB: includes transmission techs!
    loc_techs_purchase = set(
        k for k in sets.loc_techs_non_transmission
        if any('.purchase' in i
               for i in loc_techs_config[k].constraints.keys_nested())
    )

    transmission_purchase = set(
        k for k in sets.loc_techs_transmission
        if any('.purchase' in i
               for i in loc_techs_transmission_config[k].constraints.keys_nested())
    )

    sets.loc_techs_purchase = loc_techs_purchase | transmission_purchase

    # Technologies with MILP constraints
    loc_techs_milp = set(
        k for k in sets.loc_techs_non_transmission
        if any('.units_' in i
               for i in loc_techs_config[k].constraints.keys_nested())
    )

    transmission_milp = set(
        k for k in sets.loc_techs_transmission
        if any('.units_' in i
               for i in loc_techs_transmission_config[k].constraints.keys_nested())
    )

    sets.loc_techs_milp = loc_techs_milp | transmission_milp

    ##
    # Subsets of `conversion_plus` technologies
    ##

    # `conversion_plus` technologies with secondary carrier(s) out
    sets.loc_techs_out_2 = set(
        k for k in sets.loc_techs_conversion_plus
        if 'carrier_out_2' in loc_techs_config[k].constraints.get_key('carrier_ratios', {})
    )

    # `conversion_plus` technologies with  tertiary carrier(s) out
    sets.loc_techs_out_3 = set(
        k for k in sets.loc_techs_conversion_plus
        if 'carrier_out_3' in loc_techs_config[k].constraints.get_key('carrier_ratios', {})
    )

    # `conversion_plus` technologies with  secondary carrier(s) in
    sets.loc_techs_in_2 = set(
        k for k in sets.loc_techs_conversion_plus
        if 'carrier_in_2' in loc_techs_config[k].constraints.get_key('carrier_ratios', {})
    )

    # `conversion_plus` technologies with  tertiary carrier(s) in
    sets.loc_techs_in_3 = set(
        k for k in sets.loc_techs_conversion_plus
        if 'carrier_in_3' in loc_techs_config[k].constraints.get_key('carrier_ratios', {})
    )

    return sets
