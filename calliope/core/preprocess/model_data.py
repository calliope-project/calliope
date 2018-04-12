"""
Copyright (C) 2013-2018 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

model_data.py
~~~~~~~~~~~~~~~~~~

Functionality to build the model-internal data array and process
time-varying parameters.

"""

import collections

import ruamel.yaml
import xarray as xr
import numpy as np

from calliope.core.attrdict import AttrDict
from calliope._version import __version__
from calliope.core.preprocess import checks
from calliope.core.preprocess.util import split_loc_techs_transmission, concat_iterable
from calliope.core.preprocess.time import add_time_dimension
from calliope.core.preprocess.lookup import add_lookup_arrays


def build_model_data(model_run, debug=False):
    """
    Take a Calliope model_run and convert it into an xarray Dataset, ready for
    constraint generation. Timeseries data is also extracted from file at this
    point, and the time dimension added to the data

    Parameters
    ----------
    model_run : AttrDict
        preprocessed model_run dictionary, as produced by
        Calliope.core.preprocess_model
    debug : bool, default = False
        Used to debug steps within build_model_data, particularly before/after
        time dimension addition. If True, more information is returned

    Returns
    -------
    data : xarray Dataset
        Dataset with optimisation parameters as variables, optimisation sets as
        coordinates, and other information in attributes.
    data_dict : dict, only returned if debug = True
        dictionary of parameters, prior to time dimension addition. Used here to
        populate the Dataset (using `from_dict()`)
    data_pre_time : xarray Dataset, only returned if debug = True
        Dataset, prior to time dimension addition, with optimisation parameters
        as variables, optimisation sets as coordinates, and other information
        in attributes.
    """
    # We build up a dictionary of the data, then convert it to an xarray Dataset
    # before applying time dimensions
    data = xr.Dataset(
        coords=add_sets(model_run),
        attrs=add_attributes(model_run)
    )

    data_dict = dict()
    data_dict.update(constraints_to_dataset(model_run))
    data_dict.update(costs_to_dataset(model_run))
    data_dict.update(location_specific_to_dataset(model_run))
    data_dict.update(tech_specific_to_dataset(model_run))
    data_dict.update(carrier_specific_to_dataset(model_run))

    data.merge(xr.Dataset.from_dict(data_dict), inplace=True)

    add_lookup_arrays(data, model_run)

    if debug:
        data_pre_time = data.copy(deep=True)

    add_time_dimension(data, model_run)

    # Carrier information uses DataArray indexing in the function, so we merge
    # these directly into the main xarray Dataset

    if debug:
        return data, data_dict, data_pre_time
    else:
        return data


def add_sets(model_run):
    coords = dict()
    for key, value in model_run.sets.items():
        if value:
            coords[key] = value
    for key, value in model_run.constraint_sets.items():
        if value:
            coords[key] = value
    return coords


def constraints_to_dataset(model_run):
    """
    Extract all constraints from the processed dictionary (model.model_run) and
    return an xarray Dataset with all the constraints as DataArray variables and
    model sets as Dataset dimensions.

    Parameters
    ----------
    model_run : AttrDict
        processed Calliope model_run dict

    Returns
    -------
    data_dict : dict conforming to xarray conventions

    """
    data_dict = dict()

    # FIXME: hardcoding == bad
    def _get_set(constraint):
        """
        return the set of loc_techs over which the given constraint should be
        built
        """
        if '_area' in constraint:
            return 'loc_techs_area'
        elif any(i in constraint for i in ['resource_cap',  'parasitic', 'resource_min_use']):
            return 'loc_techs_supply_plus'
        elif 'resource' in constraint: # i.e. everything with 'resource' in the name that isn't resource_cap
            return 'loc_techs_finite_resource'
        elif 'storage' in constraint or 'charge_rate' in constraint:
            return 'loc_techs_store'
        elif 'purchase' in constraint:
            return 'loc_techs_purchase'
        elif 'units_' in constraint:
            return 'loc_techs_milp'
        elif 'export' in constraint:
            return 'loc_techs_export'
        else:
            return 'loc_techs'

    # find all constraints which are actually defined in the yaml file
    relevant_constraints = set(i.split('.constraints.')[1]
                               for i in model_run.locations.as_dict_flat().keys()
                               if '.constraints.' in i and
                               '.carrier_ratios.'not in i)
    for constraint in relevant_constraints:
        data_dict[constraint] = dict(dims=_get_set(constraint), data=[])
        for loc_tech in model_run.sets[_get_set(constraint)]:
            loc, tech = loc_tech.split('::', 1)
            # for transmission technologies, we also need to go into link nesting
            if ':' in tech:  # i.e. transmission technologies
                tech, link = tech.split(':')
                loc_tech_dict = model_run.locations[loc].links[link].techs[tech]
            else:  # all other technologies
                loc_tech_dict = model_run.locations[loc].techs[tech]
            constraint_value = loc_tech_dict.constraints.get(constraint, np.nan)
            # inf is assumed to be string on import, so we need to np.inf it
            if constraint_value == 'inf':
                constraint_value = np.inf
            # add the value for the particular location & technology combination to the list
            data_dict[constraint]['data'].append(constraint_value)
        # once we've looped through all technology & location combinations, add the array to the dataset

    # Additional system-wide constraints from model_run.model
    # FIXME: hardcoding == bad
    data_dict['reserve_margin'] = {
        'data': [model_run.model.get('reserve_margin', {}).get(c, np.nan)
                 for c in model_run.sets['carriers']],
        'dims': 'carriers'
    }

    group_share_data = {}
    group_constraints = ['energy_cap_min', 'energy_cap_max', 'energy_cap_equals']
    group_constraints_carrier = ['carrier_prod_min', 'carrier_prod_max', 'carrier_prod_equals']

    for constraint in [  # Only process constraints that are defined
            c for c in group_constraints
            if c in ''.join(model_run.model.get_key('group_share', AttrDict()).keys_nested())]:
        group_share_data[constraint] = [
            model_run.model.get_key('group_share.{}.{}'.format(techlist, constraint), np.nan)
            for techlist in model_run.sets['techlists']
        ]

    for constraint in [  # Only process constraints that are defined
            c for c in group_constraints_carrier
            if c in ''.join(model_run.model.get_key('group_share', AttrDict()).keys_nested())]:
        group_share_data[constraint] = [
            [
                model_run.model.get_key('group_share.{}.{}.{}'.format(techlist, constraint, carrier), np.nan)
                for techlist in model_run.sets['techlists']
            ]
            for carrier in model_run.sets['carriers']
        ]

    # Add to data_dict and set dims correctly
    for k in group_share_data:
        data_dict['group_share_' + k] = {
            'data': group_share_data[k],
            'dims': 'techlists' if k in group_constraints else ('carriers', 'techlists')
        }

    return data_dict


def costs_to_dataset(model_run):
    """
    Extract all costs from the processed dictionary (model.model_run) and
    return an xarray Dataset with all the costs as DataArray variables. Variable
    names will be prepended with `cost_` to differentiate from other constraints

    Parameters
    ----------
    model_run : AttrDict
        processed Calliope model_run dict

    Returns
    -------
    data_dict : dict conforming to xarray conventions

    """
    data_dict = dict()

    # FIXME: hardcoding == bad
    def _get_set(cost):
        """
        return the set of loc_techs over which the given cost should be built
        """
        if any(i in cost for i in ['_cap', 'depreciation_rate', 'purchase', 'area']):
            return 'loc_techs_investment_cost'
        elif any(i in cost for i in ['om_', 'export']):
            return 'loc_techs_om_cost'
        else:
            return 'loc_techs'

    # find all cost classes and associated costs which are actually defined in the model_run
    costs = set(i.split('.costs.')[1].split('.')[1]
                for i in model_run.locations.as_dict_flat().keys()
                if '.costs.' in i)
    cost_classes = model_run.sets['costs']

    # loop over unique costs, cost classes and technology & location combinations
    for cost in costs:
        data_dict['cost_' + cost] = dict(dims=["costs", _get_set(cost)], data=[])
        for cost_class in cost_classes:
            cost_class_array = []
            for loc_tech in model_run.sets[_get_set(cost)]:
                loc, tech = loc_tech.split('::', 1)
                # for transmission technologies, we also need to go into link nesting
                if ':' in tech:  # i.e. transmission technologies
                    tech, link = tech.split(':')
                    loc_tech_dict = model_run.locations[loc].links[link].techs[tech]
                else:  # all other technologies
                    loc_tech_dict = model_run.locations[loc].techs[tech]
                cost_dict = loc_tech_dict.get_key('costs.' + cost_class, None)

                # inf is assumed to be string on import, so need to np.inf it
                cost_value = np.nan if not cost_dict else cost_dict.get(cost, np.nan)
                # add the value for the particular location & technology combination to the correct cost class list
                cost_class_array.append(cost_value)
            data_dict['cost_' + cost]['data'].append(cost_class_array)

    return data_dict


def carrier_specific_to_dataset(model_run):
    """
    Extract carrier information from the processed dictionary (model.model_run)
    and return an xarray Dataset with DataArray variables describing carrier_in,
    carrier_out, and carrier_ratio (for conversion plus technologies) information.

    Parameters
    ----------
    model_run : AttrDict
        processed Calliope model_run dict

    Returns
    -------
    data_dict : dict conforming to xarray conventions

    """
    carrier_tiers = model_run.sets['carrier_tiers']
    loc_tech_dict = {k: [] for k in model_run.sets['loc_techs_conversion_plus']}
    data_dict = dict()
    # Set information per carrier tier ('out', 'out_2', 'in', etc.)
    # for conversion-plus technologies
    if model_run.sets['loc_techs_conversion_plus']:
        # carrier ratios are the floating point numbers used to compare one
        # carrier_in/_out value with another carrier_in/_out value
        data_dict['carrier_ratios'] = dict(
            dims=['carrier_tiers', 'loc_tech_carriers_conversion_plus'], data=[]
        )
        data_dict['carrier_ratios_min'] = dict(
            dims=['carrier_tiers', 'loc_techs_conversion_plus'], data=[]
        )
        for carrier_tier in carrier_tiers:
            data = []
            for loc_tech_carrier in model_run.sets['loc_tech_carriers_conversion_plus']:
                loc, tech, carrier = loc_tech_carrier.split('::')
                carrier_ratio = (
                    model_run.locations[loc].techs[tech].constraints.get_key(
                        'carrier_ratios.carrier_' + carrier_tier + '.' + carrier, 1
                    )
                )
                data.append(carrier_ratio)
                loc_tech_dict[loc + '::' + tech].append(carrier_ratio)
            data_dict['carrier_ratios']['data'].append(data)
            data_dict['carrier_ratios_min']['data'].append(
                [min(i) for i in loc_tech_dict.values()]
            )

    # Additional system-wide constraints from model_run.model
    if 'reserve_margin' in model_run.model.keys():
        data_dict['reserve_margin'] = {
            'data': [model_run.model.reserve_margin.get(c, np.nan)
                     for c in model_run.sets['carriers']],
            'dims': 'carriers'
        }

    return data_dict


def location_specific_to_dataset(model_run):
    """
    Extract location specific information from the processed dictionary
    (model.model_run) and return an xarray Dataset with DataArray variables
    describing distance, coordinate and available area information.

    Parameters
    ----------
    model_run : AttrDict
        processed Calliope model_run dict

    Returns
    -------
    data_dict : dict conforming to xarray conventions

    """
    # for every transmission technology, we extract distance information, if it
    # is available
    data_dict = dict()

    data_dict['distance'] = dict(dims='loc_techs_transmission', data=[
        model_run.get_key(
            'locations.{loc_from}.links.{loc_to}.techs.{tech}.distance'
            .format(**split_loc_techs_transmission(loc_tech)), np.nan)
        for loc_tech in model_run.sets['loc_techs_transmission']
    ])
    # If there is no distance information stored, distance array is deleted
    if data_dict['distance']['data'].count(np.nan) == len(data_dict['distance']['data']):
        del data_dict['distance']

    data_dict['lookup_remotes'] = dict(
        dims='loc_techs_transmission',
        data=concat_iterable([
            (k['loc_to'], k['tech'], k['loc_from'])
            for k in [
                split_loc_techs_transmission(loc_tech)
                for loc_tech in model_run.sets['loc_techs_transmission']
            ]
        ], ['::', ':'])
    )
    # If there are no remote locations stored, lookup_remotes array is deleted
    if data_dict['lookup_remotes']['data'].count(np.nan) == len(data_dict['lookup_remotes']['data']):
        del data_dict['lookup_remotes']

    data_dict['available_area'] = dict(dims='locs', data=[
        model_run.locations[loc].get('available_area', np.nan)
        for loc in model_run.sets['locs']
    ])

    # remove this dictionary element if nothing is defined in it
    if set(data_dict['available_area']['data']) == {np.nan}:
        del data_dict['available_area']

    # Coordinates are defined per location, but may not be defined at all for
    # the model
    if 'coordinates' in model_run.sets:
        data_dict['loc_coordinates'] = dict(dims=['locs', 'coordinates'], data=[])
        for loc in model_run.sets['locs']:
            data_dict['loc_coordinates']['data'].append([
                model_run.locations[loc].coordinates[coordinate]
                for coordinate in model_run.sets.coordinates])

    return data_dict


def tech_specific_to_dataset(model_run):
    """
    Extract technology (location inspecific) information from the processed
    dictionary (model.model_run) and return an xarray Dataset with DataArray
    variables describing color and inheritance chain information.

    Parameters
    ----------
    model_run : AttrDict
        processed Calliope model_run dict

    Returns
    -------
    data_dict : dict conforming to xarray conventions

    """
    data_dict = collections.defaultdict(
        lambda: {'dims': ['techs'], 'data': []}
    )

    systemwide_constraints = [
        k.split('.')[-1] for k in model_run.techs.keys_nested()
        if '.constraints.' in k and
        k.endswith('_systemwide')
    ]

    for tech in model_run.sets['techs']:
        if tech in model_run.sets['techs_transmission']:
            tech = tech.split(':')[0]
        data_dict['colors']['data'].append(model_run.techs[tech].get_key(
            'essentials.color'))
        data_dict['inheritance']['data'].append('.'.join(
            model_run.techs[tech].get_key('inheritance')))
        data_dict['names']['data'].append(
            # Default to tech ID if no name is set
            model_run.techs[tech].get_key('essentials.name', tech))
        for k in systemwide_constraints:
            data_dict[k]['data'].append(
                model_run.techs[tech].constraints.get_key(k, np.nan)
            )

    return data_dict


def add_attributes(model_run):
    attr_dict = AttrDict()
    attr_dict['model'] = model_run.model.copy()
    attr_dict['run'] = model_run.run.copy()

    # Some keys are killed right away
    for k in ['model.time', 'model.data_path', 'model.timeseries_data_path',
              'run.config_run_path', 'run.model']:
        try:
            attr_dict.del_key(k)
        except KeyError:
            pass

    # Now we flatten the AttrDict into a dict
    attr_dict = attr_dict.as_dict(flat=True)

    # Anything empty or None in the flattened dict is also killed
    for k in list(attr_dict.keys()):
        if not attr_dict[k]:
            del attr_dict[k]

    attr_dict['calliope_version'] = __version__

    default_tech_dict = checks.defaults.default_tech.as_dict()
    default_location_dict = checks.defaults.default_location.as_dict()
    attr_dict['defaults'] = ruamel.yaml.dump({
        **default_tech_dict['constraints'],
        **{'cost_{}'.format(k): v for k, v in default_tech_dict['costs']['default'].items()},
        **default_location_dict
    })

    return attr_dict
