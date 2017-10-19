"""
Copyright (C) 2013-2017 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

preprocess_data.py
~~~~~~~~~~~~~~~~~~

Functionality to build the model-internal data array and process
time-varying parameters.

"""

import os

import xarray as xr
import numpy as np
import pandas as pd

from .. import utils
from .. _version import __version__


def build_model_data(model_run):
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

    add_time_dimension(data, model_run)

    # Carrier information uses DataArray indexing in the function, so we merge
    # these directly into the main xarray Dataset
    data.merge(carriers_to_dataset(model_run), inplace=True)

    return data


def apply_time_clustering(model_data_original):
    # FIXME: Not yet implemented
    return model_data_original


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
    model_run : utils.AttrDict
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
            loc, tech = loc_tech.split(':', 1)
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
    model_run : utils.AttrDict
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
        if '_cap' in cost or 'interest_rate' in cost or 'purchase' in cost:
            return 'loc_techs_investment_costs'
        elif 'om_' in cost or 'export' in cost:
            return 'loc_techs_om_costs'
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
                loc, tech = loc_tech.split(':', 1)
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
    model_run : utils.AttrDict
        processed Calliope model_run dict

    Returns
    -------
    data : xarray Dataset

    """
    carrier_tiers = model_run.sets['carrier_tiers']
    loc_tech_carriers = xr.DataArray(
        np.zeros((len(model_run.sets['techs']),
                  len(model_run.sets['resources']),
                  len(carrier_tiers))),
                            dims=['techs', 'resources', 'carrier_tiers'],
                            coords=[('techs', list(model_run.sets['techs'])),
                                    ('resources',
                                        list(model_run.sets['resources'])),
                                    ('carrier_tiers', list(carrier_tiers))])
    # for every technology, 1 is given if that carrier is in/out
    for tech in model_run.sets['techs']:
        # we need the location inspecific name for tranmission technologies
        _tech = tech.split(':')[0] if ':' in tech else tech
        for i in carrier_tiers:
            # create a list of carriers for the given technology that fits the
            # current carrier_tier. This will be one value for all but
            # conversion_plus technologies
            relevant_carriers = model_run.techs[_tech].essentials.get(
                'carrier_' + i, [])
            # find the location of the information in the xr DataArray and
            # replace with 1 (i.e. True, that carrier is active at that
            # carrier_tier for that technology)
            loc_tech_carriers.loc[dict(techs=tech, resources=relevant_carriers,
                                       carrier_tiers=i)] = 1
    data = loc_tech_carriers.to_dataset(name='loc_tech_carriers')

    # Following only added if conversion_plus technologies are defined:
    if model_run.sets['loc_techs_conversion_plus']:
        # conversion ratios are the floating point numbers used to compare one
        # carrier_in/_out value with another carrier_in/_out value
        carrier_ratios = xr.DataArray(
            np.zeros((len(model_run.sets['loc_techs_conversion_plus']),
                      len(model_run.sets['resources']),
                      len(carrier_tiers)
            )),
            dims=['loc_techs_conversion_plus', 'resources', 'carrier_tiers'],
            coords=[('loc_techs_conversion_plus',
                        list(model_run.sets['loc_techs_conversion_plus'])),
                    ('resources', list(model_run.sets['resources'])),
                    ('carrier_tiers', list(carrier_tiers))]
        )
        for loc_tech in model_run.sets['loc_techs_conversion_plus']:
            loc, tech = loc_tech.split(':', 1)
            for i in carrier_tiers:
                relevant_carriers = model_run.techs[tech].essentials.get(
                    'carrier_' + i, [])
                # listify 'relevant_carriers' if not already a list
                if isinstance(relevant_carriers, str):
                    relevant_carriers = [relevant_carriers]
                # for the relevant carriers at this carrier tier, get the ratio
                # of input/output against the primary input/output carrier
                carrier_ratio = [model_run.locations[loc].techs[tech].constraints
                               .get_key('carrier_ratios.carrier_' + i + '.' + j, 1)
                      for j in relevant_carriers]
                # find the location of the information in the xr DataArray and
                # replace with the carrier_ratio
                carrier_ratios.loc[dict(loc_techs_conversion_plus=loc_tech,
                                        resources=relevant_carriers,
                                        carrier_tiers=i)] = carrier_ratio
        data.merge(carrier_ratios.to_dataset(name='carrier_ratios'), inplace=True)
    return data


def location_specific_to_dataset(model_run):
    """
    Extract location specific information from the processed dictionary
    (model.model_run) and return an xarray Dataset with DataArray variables
    describing distance, coordinate and available area information.

    Parameters
    ----------
    model_run : utils.AttrDict
        processed Calliope model_run dict

    Returns
    -------
    data : xarray Dataset

    """
    # for every transmission technology, we extract distance information, if it
    # is available
    data_dict = dict()
    data_dict['distance'] = dict(dims='loc_techs_transmission', data=[])
    for loc_tech in model_run.sets['loc_techs_transmission']:
        loc, tech, link = loc_tech.split(':')
        data_dict['distance']['data'].append(
            model_run.locations[loc].links[link].techs[tech].get(
                'distance', np.nan
                )
            )
    data_dict['available_area'] = dict(dims='locs', data=[])
    data_dict['available_area']['data'] = [
        model_run.locations[loc].get('available_area', np.nan)
        for loc in model_run.sets['locs']
    ]
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
    Extract technology (location inspecific) information from the processed dictionary
    (model.model_run) and return an xarray Dataset with DataArray variables
    describing color, inheritance chain and stack_weight information.

    Parameters
    ----------
    model_run : utils.AttrDict
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
    attr_dict = utils.AttrDict()
    attr_dict['model'] = model_run.model.copy()
    attr_dict['run'] = model_run.run.copy()

    # Some keys are killed right away
    for k in ['model.time', 'model.data_path', 'run.config_run_path', 'run.model']:
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

    Returns:
    --------
    data : xarray Dataset
        A data structure with an additional time dimension to the input dataset,
        with all relevant `file=` entries replaced with data from file.

    """
    data_path = model_run.model.data_path
    set_t = pd.read_csv(
        os.path.join(data_path, 'time_set.csv'),
        header=None, index_col=1, parse_dates=[1], squeeze=True
    )
    subset_time_config = model_run.model.subset_time
    time_slice = None
    if subset_time_config:
        time_slice = (subset_time_config[0] if len(subset_time_config) == 1
                      else slice(subset_time_config[0], subset_time_config[1]))
    subset_time = set_t[time_slice] if time_slice else set_t

    # search through every constraint/cost for use of 'file'
    for variable in data.data_vars:
        if (data[variable].dtype.kind == 'U' and
                any(data[variable].to_dataframe().stack().str.contains('file='))):
            # convert to a Pandas Series to do file search
            data_df = data[variable].to_dataframe().stack()
        else:
            continue

        filenames = data_df[data_df.str.contains('file=')]
        filenames_df = pd.DataFrame()
        filenames_df['filename'], filenames_df['column'] = \
            filenames.str.split(':', 1).str
        # store all the information about variables which will need to be given
        # over all timesteps, including those which are duplicates
        all_data = data_df[data_df != 'nan']
        # create an empty pandas DataFrame
        timeseries_df = pd.DataFrame(index=all_data.index,
                                     columns=[i for i in subset_time.index])
        # fill in values that are just duplicates, not actually from file
        if any(all_data.drop(filenames.index)):
            timeseries_df.loc[all_data.drop(filenames.index).index] = \
                np.vstack(all_data.drop(filenames.index).values)
        timeseries_df.columns.name = 'time'
        # create xarray DataArray from DataFrame with the correct dimensions
        timeseries_dataarray = (timeseries_df.stack(dropna=False).unstack(level=-2)
                                          .to_xarray().to_array())
        # Each DataArray in model_data could exist over a different subset of
        # loc_techs (e.g. loc_techs_finite_resource), so we extract which one it
        # is for this particular DataArray, for later indexing
        loc_techs = [d for d in timeseries_dataarray.dims if 'loc_tech' in d][0]

        for file in set(filenames_df.filename):
            d_path = os.path.join(data_path, file.split('file=')[1])
            cols = set(filenames_df[filenames_df.filename == file].column)
            data_from_csv = pd.read_csv(d_path, usecols=cols)
            # Apply time subset
            data_from_csv = data_from_csv.loc[subset_time.values, :]
            for col in cols:
                col_df = filenames_df[(filenames_df.filename == file) &
                                   (filenames_df.column == col)]
                loc_dict = {loc_techs:col_df.index.get_level_values(loc_techs)}
                if 'costs' in timeseries_dataarray.dims:  # cost
                    loc_dict.update({'costs':col_df.index.get_level_values('costs')})
                timeseries_dataarray.loc[loc_dict] = data_from_csv[col].values
        for variable in timeseries_dataarray['variable']:
            data[variable.item()] = timeseries_dataarray.loc[dict(variable=variable)].drop('variable')
    return None


## Not currently implemented as time_funcs and time_clustering will fail
def initialize_time(data_original, model_run):
    # Carry y_ subset sets over to data for easier data analysis
    time_config = model_run.run.get('time', None)
    if not time_config:
        return None  # Nothing more to do here
    ##
    # Process masking and get list of timesteps to keep at high res
    ##
    if 'masks' in time_config:
        masks = {}
        # time.masks is a list of {'function': .., 'options': ..} dicts
        for entry in time_config.masks:
            entry = utils.AttrDict(entry)
            mask_func = utils.plugin_load(entry.function,
                                          builtin_module='time_masks')
            mask_kwargs = entry.get_key('options', default={})
            masks[entry.to_yaml()] = mask_func(data, **mask_kwargs)
        data.attrs['masks'] = masks
        # Concatenate the DatetimeIndexes by using dummy Series
        chosen_timesteps = pd.concat([pd.Series(0, index=m)
                                     for m in masks.values()]).index
        # timesteps: a list of timesteps NOT picked by masks
        timesteps = pd.Index(data.timesteps.values).difference(chosen_timesteps)
    else:
        timesteps = None
    ##
    # Process function, apply resolution adjustments
    ##
    if 'function' in time_config:
        func = utils.plugin_load(
            time_config.function, builtin_module='time_funcs')
        func_kwargs = time_config.get('function_options', {})
        data = func(data=data_original, timesteps=timesteps, **func_kwargs)

        ## Removed while operational mode is inexistant
        # Raise error if we've made adjustments incompatible
        # with operational mode
        # if data.model.mode == 'operate':
        #     opmode_safe = data.attrs.get('opmode_safe', False)
        #     if opmode_safe:
        #         data.attrs['time_resolution'] = self.get_timeres()
        #     else:
        #         msg = 'Time settings incompatible with operational mode'
        #         raise exceptions.ModelError(msg)
    return None
