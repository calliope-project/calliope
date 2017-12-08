"""
Copyright (C) 2013-2017 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

preprocess_data.py
~~~~~~~~~~~~~~~~~~

Functionality to build the model-internal data array and process
time-varying parameters.

"""

import json

import xarray as xr
import numpy as np
import pandas as pd

from calliope import exceptions
from calliope.core.attrdict import AttrDict
from calliope.core.util.tools import plugin_load
from calliope._version import __version__
from calliope.core.preprocess import checks
from calliope.core.util.dataset import reorganise_dataset_dimensions
from calliope.core.preprocess.util import split_loc_techs_transmission, concat_iterable


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

    data.merge(xr.Dataset.from_dict(data_dict), inplace=True)

    data.merge(carriers_to_dataset(model_run), inplace=True)

    if debug:
        data_pre_time = data.copy(deep=True)

    add_time_dimension(data, model_run)

    # Carrier information uses DataArray indexing in the function, so we merge
    # these directly into the main xarray Dataset

    if debug:
        return data, data_dict, data_pre_time
    else:
        return data


def apply_time_clustering(model_data_original, model_run):
    """
    Take a Calliope model_data post time dimension addition, prior to any time
    clustering, and apply relevant time clustering/masking techniques.
    See doi: 10.1016/j.apenergy.2017.03.051 for applications.

    Techniques include:
    - Clustering timeseries into a selected number of 'representative' days.
        Days with similar profiles and daily magnitude are grouped together and
        represented by one 'representative' day with a greater weight per time
        step.
    - Masking timeseries, leading to variable timestep length
        Only certain parts of the input are shown at full resolution, with other
        periods being clustered together into a single timestep.
        E.g. Keep high resolution in the week with greatest wind power variability,
        smooth all other timesteps to 12H
    - Timestep resampling
        Used to reduce problem size by reducing resolution of all timeseries data.
        E.g. resample from 1H to 6H timesteps


    Parameters
    ----------
    model_data_original : xarray Dataset
        Preprocessed Calliope model_data, as produced using
        `calliope.core.preprocess_data.build_model_data`
        and found in model._model_data_original
    model_run : bool
        preprocessed model_run dictionary, as produced by
        Calliope.core.preprocess_model

    Returns
    -------
    data : xarray Dataset
        Dataset with optimisation parameters as variables, optimisation sets as
        coordinates, and other information in attributes. Time dimension has
        been updated as per user-defined clustering techniques (from model_run)
    """

    # Carry y_ subset sets over to data for easier data analysis
    time_config = model_run.model.get('time', None)
    if not time_config:
        return reorganise_dataset_dimensions(model_data_original)  # Nothing more to do here
    else:
        data = model_data_original.copy(deep=True)

    # Add temporary 'timesteps per day' attribute
    timestep_resolution = data['timestep_resolution'].values[0]
    if not np.all(data['timestep_resolution'] == timestep_resolution):
        raise exceptions.ModelError('For clustering, timestep resolution must be uniform. ')
    timesteps_per_day = 24 / timestep_resolution
    if isinstance(timesteps_per_day, float):
        assert timesteps_per_day.is_integer(), 'Timesteps/day must be integer.'
        timesteps_per_day = int(timesteps_per_day)
    data.attrs['_timesteps_per_day'] = timesteps_per_day

    ##
    # Process masking and get list of timesteps to keep at high res
    ##
    if 'masks' in time_config:
        masks = {}
        # time.masks is a list of {'function': .., 'options': ..} dicts
        for entry in time_config.masks:
            entry = AttrDict(entry)
            mask_func = plugin_load(entry.function, builtin_module='calliope.core.time.masks')
            mask_kwargs = entry.get_key('options', default={})
            masks[entry.to_yaml()] = mask_func(data, **mask_kwargs)
        data.attrs['masks'] = masks
        # Concatenate the DatetimeIndexes by using dummy Series
        chosen_timesteps = pd.concat([pd.Series(0, index=m)
                                     for m in masks.values()]).index
        # timesteps: a list of timesteps NOT picked by masks
        timesteps = pd.Index(data.time.values).difference(chosen_timesteps)
    else:
        timesteps = None

    ##
    # Process function, apply resolution adjustments
    ##
    if 'function' in time_config:
        func = plugin_load(
            time_config.function, builtin_module='calliope.core.time.funcs')
        func_kwargs = time_config.get('function_options', {})
        data = func(data=data, timesteps=timesteps, **func_kwargs)

    # Final checking of the data
    final_check_comments, warnings, errors = checks.check_model_data(data)
    checks.print_warnings_and_raise_errors(warnings=warnings, errors=errors)

    # Temporary timesteps per day attribute is no longer needed
    try:
        del data.attrs['_timesteps_per_day']
    except KeyError:
        pass

    return reorganise_dataset_dimensions(data)


def add_sets(model_run):
    coords = dict()
    for key, value in model_run.sets.items():
        if value:
            coords[key] = value  # turn set into list
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
    data : xarray Dataset

    """
    data_dict = dict()

    # FIXME? should set finding be hardcoded like this?
    def _get_set(constraint):
        """
        return the set of loc_techs over which the given constraint should be
        built
        """
        if constraint == 'resource':
            return 'loc_techs_finite_resource'
        elif '_area' in constraint:
            return 'loc_techs_area'
        elif 'resource_' in constraint or 'parasitic' in constraint:
            return 'loc_techs_supply_plus'
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
        data_dict[constraint]=dict(dims=_get_set(constraint), data=[])
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
    data : xarray Dataset

    """
    data_dict = dict()

    # FIXME? should set finding be hardcoded like this?
    def _get_set(cost):
        """
        return the set of loc_techs over which the given cost should be built
        """
        if '_cap' in cost or 'depreciation_rate' in cost or 'purchase' in cost:
            return 'loc_techs_investment_cost'
        elif 'om_' in cost or 'export' in cost:
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


def carriers_to_dataset(model_run):
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
    data : xarray Dataset

    """
    carrier_tiers = model_run.sets['carrier_tiers']

    # get the technologies associated with a certain loc_carrier
    lookup_loc_carriers_dict = dict(dims=['loc_carriers'])
    data = []
    for loc_carrier in model_run.sets['loc_carriers']:
        loc_tech_carrier = list(set(
            i for i in
            model_run.sets['loc_tech_carriers_prod'] +
            model_run.sets['loc_tech_carriers_con']
            if loc_carrier == i.split("::", 1)[0] + "::" + i.rsplit("::", 1)[1]
        ))
        data.append(",".join(loc_tech_carrier))
    lookup_loc_carriers_dict['data'] = data
    lookup_loc_carriers = xr.DataArray.from_dict(lookup_loc_carriers_dict)
    dataset = lookup_loc_carriers.to_dataset(name='lookup_loc_carriers')

    # Get the string name for a loc_tech which includes the carrier associated
    # with that technology (for non_conversion technologies)
    lookup_loc_techs_dict = dict(dims=['loc_techs_non_conversion'])

    data = []
    for loc_tech in model_run.sets['loc_techs_non_conversion']:
        # For any non-conversion technology, there is only one carrier (either
        # produced or consumed)
        loc_tech_carrier = list(set(
            i for i in
            model_run.sets['loc_tech_carriers_prod'] +
            model_run.sets['loc_tech_carriers_con']
            if loc_tech == i.rsplit("::", 1)[0]
        ))
        if len(loc_tech_carrier) > 1:
            raise exceptions.ModelError("More than one carrier associated with"
            " non-conversion location:technology `{}`".format(loc_tech))
        else:
            data.append(loc_tech_carrier[0])
    lookup_loc_techs_dict['data'] = data
    lookup_loc_techs = xr.DataArray.from_dict(lookup_loc_techs_dict)
    dataset.merge(lookup_loc_techs.to_dataset(name='lookup_loc_techs'), inplace=True)

    # Following only added if conversion technologies are defined:
    if model_run.sets['loc_techs_conversion']:
        # Get the string name for a loc_tech which includes the carriers in and out
        # associated with that technology (for conversion technologies)
        lookup_loc_techs_conversion = (
            xr.DataArray(np.empty((len(model_run.sets['loc_techs_conversion']),
                                   len(carrier_tiers)), dtype=np.object),
                        dims=['loc_techs_conversion', 'carrier_tiers'],
                        coords=[('loc_techs_conversion',
                                list(model_run.sets['loc_techs_conversion'])),
                                ('carrier_tiers', list(carrier_tiers))]
            )
        )
        for loc_tech in model_run.sets['loc_techs_conversion']:
            # For any non-conversion technology, there are only two carriers
            # (one produced and one consumed)
            loc_tech_carrier_in = [
                i for i in
                model_run.sets['loc_tech_carriers_con']
                if loc_tech == i.rsplit("::", 1)[0]
            ]

            loc_tech_carrier_out = [
                i for i in
                model_run.sets['loc_tech_carriers_prod']
                if loc_tech == i.rsplit("::", 1)[0]
            ]
            if len(loc_tech_carrier_in) > 1 or len(loc_tech_carrier_out) > 1:
                raise exceptions.ModelError("More than one carrier in or out "
                "associated with conversion location:technology `{}`".format(loc_tech))
            else:
                lookup_loc_techs_conversion.loc[
                    dict(loc_techs_conversion=loc_tech, carrier_tiers=["in", "out"])
                ] = [loc_tech_carrier_in[0],  loc_tech_carrier_out[0]]
        dataset.merge(lookup_loc_techs_conversion
            .to_dataset(name="lookup_loc_techs_conversion"), inplace=True)

    # Following only added if conversion_plus technologies are defined:
    if model_run.sets['loc_techs_conversion_plus']:
        # Get the string name for a loc_tech which includes all the carriers in
        # and out associated with that technology (for conversion_plus technologies)
        lookup_loc_techs_conversion_plus = (
            xr.DataArray(np.empty((len(model_run.sets['loc_techs_conversion_plus']),
                                   len(model_run.sets['resources']),
                                   len(carrier_tiers)), dtype=np.object),
                         dims=['loc_techs_conversion_plus', 'resources', 'carrier_tiers'],
                         coords=[('loc_techs_conversion_plus',
                                  list(model_run.sets['loc_techs_conversion_plus'])),
                                 ('resources', list(model_run.sets['resources'])),
                                 ('carrier_tiers', list(carrier_tiers))]
            )
        )
        for loc_tech in model_run.sets['loc_techs_conversion_plus']:
            _tech = loc_tech.split('::', 1)[1]
            for carrier_tier in carrier_tiers:
                # create a list of carriers for the given technology that fits
                # the current carrier_tier.
                relevant_carriers = model_run.techs[_tech].essentials.get(
                    'carrier_' + carrier_tier, [])
                if relevant_carriers and isinstance(relevant_carriers, list):
                    loc_tech_carriers = [loc_tech + "::" + i for i in relevant_carriers]
                elif relevant_carriers:
                    loc_tech_carriers = loc_tech + "::" + relevant_carriers
                lookup_loc_techs_conversion_plus.loc[
                    dict(loc_techs_conversion_plus=loc_tech,
                         resources=relevant_carriers, carrier_tiers=carrier_tier)
                ] = loc_tech_carriers

        # carrier ratios are the floating point numbers used to compare one
        # carrier_in/_out value with another carrier_in/_out value
        carrier_ratios = xr.DataArray(
            np.zeros((len(model_run.sets['loc_tech_carriers_conversion_plus']),
                      len(carrier_tiers))),
            dims=['loc_tech_carriers_conversion_plus', 'carrier_tiers'],
            coords=[('loc_tech_carriers_conversion_plus',
                     list(model_run.sets['loc_tech_carriers_conversion_plus'])),
                    ('carrier_tiers', list(carrier_tiers))]
        )
        for loc_tech_carrier in model_run.sets['loc_tech_carriers_conversion_plus']:
            loc, tech, carrier = loc_tech_carrier.split('::')
            for carrier_tier in carrier_tiers:
                carrier_ratio = (model_run.locations[loc].techs[tech].constraints
                    .get_key('carrier_ratios.carrier_' + carrier_tier + '.' + carrier, 1))
                # find the location of the information in the xr DataArray and
                # replace with the carrier_ratio
                carrier_ratios.loc[dict(loc_tech_carriers_conversion_plus=loc_tech_carrier,
                                        carrier_tiers=carrier_tier)] = carrier_ratio
        dataset.merge(lookup_loc_techs_conversion_plus.to_dataset(
            name='lookup_loc_techs_conversion_plus'), inplace=True)
        dataset.merge(
            carrier_ratios.to_dataset(name='carrier_ratios'), inplace=True)
    return dataset


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
    data : xarray Dataset

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
    k = split_loc_techs_transmission
    data_dict['lookup_remotes'] = dict(dims='loc_techs_transmission',
        data=concat_iterable([(k['loc_to'], k['tech'], k['loc_from'])
            for k in [split_loc_techs_transmission(loc_tech)
                for loc_tech in model_run.sets['loc_techs_transmission']
            ]
        ], ['::', ':'])
    )

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
    variables describing color, inheritance chain and stack_weight information.

    Parameters
    ----------
    model_run : AttrDict
        processed Calliope model_run dict

    Returns
    -------
    data : xarray Dataset

    """
    # for every technology, we extract location inspecific information
    information = ['essentials.color', 'essentials.stack_weight']
    data_dict = {'color':{'dims':['techs'], 'data':[]},
                 'stack_weight':{'dims':['techs'], 'data':[]},
                 'inheritance':{'dims':['techs'], 'data':[]}}

    for tech in model_run.sets['techs']:
        if tech in model_run.sets['techs_transmission']:
            tech = tech.split(':')[0]
        data_dict['color']['data'].append(model_run.techs[tech].get_key(
            'essentials.color'))
        data_dict['stack_weight']['data'].append(model_run.techs[tech].get_key(
            'essentials.stack_weight'))
        data_dict['inheritance']['data'].append('.'.join(
            model_run.techs[tech].get_key('inheritance')))

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

    default_dict = checks.defaults.default_tech.as_dict()
    attr_dict['defaults'] = json.dumps({
        **default_dict['constraints'],
        **{'cost_{}'.format(k): v for k, v in default_dict['costs']['default'].items()}
    })

    return attr_dict


def add_time_dimension(data, model_run):
    """
    Once all constraints and costs have been loaded into the model dataset, any
    timeseries data is loaded from file and substituted into the model dataset

    Parameters:
    -----------
    data : xarray Dataset
        A data structure which has already gone through `constraints_to_dataset`,
        `costs_to_dataset`, and `add_attributes`
    model_run : AttrDict
        Calliope model_run dictionary

    Returns:
    --------
    data : xarray Dataset
        A data structure with an additional time dimension to the input dataset,
        with all relevant `file=` entries replaced with data from file.

    """
    data['timesteps'] = pd.to_datetime(data.timesteps)

    # Search through every constraint/cost for use of '='
    for variable in data.data_vars:
        # 1) If '=' in variable, it will give the variable a string data type
        if data[variable].dtype.kind != 'U':
            continue

        # 2) convert to a Pandas Series to do 'string contains' search
        data_series = data[variable].to_series()

        # 3) get a Series of all the uses of 'file=' for this variable
        filenames = data_series[data_series.str.contains('=')]

        # 4) If no use of 'file=' then we can be on our way
        if filenames.empty:
            continue

        # 5) remove all before '=' and split filename and location column
        filenames = filenames.str.split('=').str[1].str.rsplit(':', 1)

        # 6) Get all timeseries data from dataframes stored in model_run
        timeseries_data = [model_run.timeseries_data[file].loc[:, column].values
                           for (file, column) in filenames.values]

        timeseries_data_series = pd.DataFrame(index=filenames.index,
                                              columns=data.timesteps.values,
                                              data=timeseries_data).stack()
        timeseries_data_series.index.rename('timesteps', -1, inplace=True)

        # 7) Add time dimension to the relevent DataArray and update the '='
        # dimensions with the time varying data (static data is just duplicated
        # at each timestep)
        timeseries_data_array = xr.broadcast(data[variable], data.timesteps)[0].copy()

        timeseries_data_array.loc[
            xr.DataArray.from_series(timeseries_data_series).coords
        ] = xr.DataArray.from_series(timeseries_data_series).values

        # 8) assign correct dtype (might be string/object accidentally)
        if all(np.where(
            ((timeseries_data_array=='True') | (timeseries_data_array=='False'))
            )):
            # Turn to bool
            timeseries_data_array.loc[dict()] = timeseries_data_array == 'True'
            timeseries_data_array = timeseries_data_array.astype(np.bool, copy=False)
        else:
            try:
                timeseries_data_array = timeseries_data_array.astype(np.float, copy=False)
            except:
                None
        data[variable] = timeseries_data_array

    # Add time_resolution and timestep_weight variables
    # parsed_timesteps = pd.to_datetime(data.timesteps)
    seconds = abs(
        pd.to_datetime(data.timesteps.values[0]) -
        pd.to_datetime(data.timesteps.values[1])
    ).total_seconds()
    hours = seconds / 3600

    data['timestep_resolution'] = xr.DataArray(
            np.ones(len(data.timesteps)) * hours,
            dims=['timesteps']
    )

    data['timestep_weights'] = xr.DataArray(
            np.ones(len(data.timesteps)),
            dims=['timesteps']
    )

    return None
