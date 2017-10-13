import os

import xarray as xr
import numpy as np
import pandas as pd

from .. import utils

def build_model_data(model_run):
    data = xr.Dataset()
    data.coords.update(add_sets(model_run))
    data.merge(constraints_to_dataset(model_run), inplace=True)
    data.merge(costs_to_dataset(model_run), inplace=True)

    data = add_time_dimension(data)

    data.merge(carriers_to_dataset(model_run), inplace=True)
    data.merge(distance_to_dataset(model_run), inplace=True)
    data.merge(essentials_to_dataset(model_run), inplace=True)

    data.attrs = add_attributes(model_run)

    return data

def apply_time_clustering(model_data_original):
    ## Not yet implemented
    return model_data_original

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
    data = xr.Dataset()
    # find all constraints which are actually defined in the yaml file
    relevant_constraints = set(i.split('.constraints.')[1]
                               for i in model_run.locations.as_dict_flat().keys()
                               if '.constraints.' in i
                               and '.carrier_ratios.'not in i)
    for constraint in relevant_constraints:
        constraint_array = []
        for loc_tech in model_run.sets['loc_techs']:
            loc, tech = loc_tech.split(':', 1)
            # for transmission technologies, we also need to go into link nesting
            if ':' in tech: # i.e. transmission technologies
                tech, link = tech.split(':')
                loc_tech_dict = model_run.locations[loc].links[link].techs[tech]
            else: # all other technologies
                loc_tech_dict = model_run.locations[loc].techs[tech]
            constraint_value = loc_tech_dict.constraints.get(constraint, np.nan)
            # inf is assumed to be string on import, so we need to np.inf it
            if constraint_value == 'inf':
                constraint_value = np.inf
            # add the value for the particular location & technology combination to the list
            constraint_array.append(constraint_value)
        # once we've looped through all technology & location combinations, add the array to the dataset
        data.merge(xr.DataArray(constraint_array,
                                dims=['loc_techs'],
                                coords=[('loc_techs',
                                         model_run.sets['loc_techs'])])
                                .to_dataset(name=constraint), inplace=True)
    return data

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
    data = xr.Dataset()
    # find all cost classes and associated costs which are actually defined in the processed.yaml file
    costs = set(i.split('.costs.')[1].split('.')[1]
                for i in model_run.locations.as_dict_flat().keys()
                if '.costs.' in i)
    cost_classes = model_run.sets['costs']
    # loop over unique costs, cost classes and technology & location combinations
    for cost in set(costs):
        cost_class_dict = {i:[] for i in cost_classes}
        for cost_class in cost_classes:
            for loc_tech in model_run.sets['loc_techs']:
                loc, tech = loc_tech.split(':', 1)
                 # for transmission technologies, we also need to go into link nesting
                if ':' in tech: # i.e. transmission technologies
                    tech, link = tech.split(':')
                    loc_tech_dict = model_run.locations[loc].links[link].techs[tech]
                else: # all other technologies
                    loc_tech_dict = model_run.locations[loc].techs[tech]
                cost_dict = loc_tech_dict.get_key('costs.' + cost_class, None)
                # inf is assumed to be string on import, so need to np.inf it
                cost_value = np.nan if not cost_dict else cost_dict.get(cost, np.nan)
                # add the value for the particular location & technology combination to the correct cost class list
                cost_class_dict[cost_class].append(cost_value)
        # once you've looped through all technology, location, & cost class
        # combinations, add the array to the dataset. All cost based DataArrays
        # are prepended with 'cost_'
        data.merge(xr.DataArray([value for value in cost_class_dict.values()],
                                dims=['costs', 'loc_techs',],
                                coords=[('costs', model_run.sets['costs']),
                                        ('loc_techs', model_run.sets['loc_techs'])])
                                .to_dataset(name='cost_' + cost), inplace=True)
    return data

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
    # relevant carriers are all those defined in the model
    relevant_carriers = ['resource'] + model_run.sets['carriers']
    # carrier tiers are all levels of carrier defined in the model
    # as conversion_plus technologies can have secondary and tertiary carriers in/out
    carrier_tiers = set(key.split('.carrier_')[1]
                        for key in model_run.techs.as_dict_flat().keys()
                        if '.carrier_' in key)
    carriers = xr.DataArray(np.zeros((len(model_run.sets['techs']),
                                      len(relevant_carriers),
                                      len(carrier_tiers))),
                                dims=['techs', 'resources', 'carrier_tiers'],
                                coords=[('techs', model_run.sets['techs']),
                                        ('resources', list(relevant_carriers)),
                                        ('carrier_tiers', list(carrier_tiers))])
    # for every technology, 1 is given if that carrier is in/out
    for tech in model_run.sets['techs']:
        # we need the location inspecific name for tranmission technologies
        _tech = tech.split(':')[0] if ':' in tech else tech
        for i in carrier_tiers:
            _carriers = model_run.techs[_tech].essentials.get('carrier_' + i, [])
            carriers.loc[dict(techs=tech, resources=_carriers, carrier_tier=i)] = 1
    data = carriers.to_dataset(name='carriers')
    # Following only added if conversion_plus technologies are defined:
    if model_run.sets['loc_techs_conversion_plus']:
        # conversion ratios are the floating point numbers used to compare one
        # carrier_in/_out value with another carrier_in/_out value
        carrier_ratios = xr.DataArray(
            np.zeros((len(model_run.sets['loc_techs_conversion_plus']),
                      len(relevant_carriers), len(carrier_tiers))),
            dims=['loc_techs_conversion_plus', 'carrier_resource', 'carrier_tier'],
            coords=[('loc_techs_conversion_plus',
                        model_run.sets['loc_techs_conversion_plus']),
                    ('carrier_resource', list(relevant_carriers)),
                    ('carrier_tier', list(carrier_tiers))]
            )
        for loc_tech in model_run.sets['loc_techs_conversion_plus']:
            loc, tech = loc_tech.split(':', 1)
            for i in carrier_tiers:
                _carriers = model_run.techs[tech].essentials.get('carrier_' + i, [])
                _carriers = [_carriers] if isinstance(_carriers, str) else _carriers
                _c = [model_run.locations[loc].techs[tech].constraints
                               .get_key('carrier_ratios.carrier_' + i + '.' + j, 1)
                      for j in _carriers]
                carrier_ratios.loc[dict(loc_techs_conversion_plus=loc_tech,
                                        resources=_carriers,
                                        carrier_tiers=i)] = _c
        data.merge(carrier_ratios.to_dataset(name='carrier_ratios'), inplace=True)
    return data

def distance_to_dataset(model_run):
    """
    Extract distance and coordinate information from the processed dictionary
    (model.model_run) and return an xarray Dataset with DataArray variables
    describing distance and coordinate information.

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
    distance = []
    for loc_tech in model_run.sets['loc_techs_transmission']:
        loc, tech, link = loc_tech.split(':')
        distance.append(model_run.locations[loc].links[link].techs[tech].get(
            'distance', np.nan
        ))
    data = (xr.DataArray(distance, dims=['loc_techs_transmission'])
              .to_dataset(name='distance'))

    # Coordinates are defined per location:
    coordinates = []
    coordinate_dims = []
    for loc in model_run.sets['locs']:
        coordinate = model_run.locations[loc].get('coordinates', None)
        if not coordinate:
            data_coord = None
            # either all coordinates exist, or none of them,
            # so no point looking for coordinates after this
            break
        else:
            data_coord = []
            if not coordinate_dims:
                coordinate_dims = list(coordinate.keys())
            coordinates.append([coordinate[coordinate_dims[0]],
                                coordinate[coordinate_dims[1]]])
    if data_coord:
        data.merge(xr.DataArray(coordinates,
                              dims=['locs', 'coordinates'],
                              coords=[('locs', model_run.sets.loc),
                                      ('coordinate', coordinate_dims)])
                     .to_dataset(name='coordinates')
                   )
    return data

def essentials_to_dataset(model_run):
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
    information = ['essentials.color', 'inheritance', 'essentials.stack_weight']
    data = xr.Dataset()
    for info in information:
        name = info.split('.')[1] if '.' in info else info
        data_info = []
        for tech in model_run.sets['techs']:
            if tech in model_run.sets['tech_transmission']:
                tech = tech.split(':')[0]
            data_info.append(model_run.techs[tech].get_key(info))
        data.append(xr.DataArray(data, dims=['techs']).to_dataset(name=name))

    return data

def add_sets(model_run):
    coords = dict()
    for key, value in model_run.sets.items():
        if value:
            coords[key] = value
    return coords

# TODO? Warn on overriding a value from model_run.model with something from
#  model_run.run
def add_attributes(model_run):
    attr_dict = model_run.model
    attr_dict.update(model_run.run)
    return attr_dict

def add_time_dimension(data):
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
    data_path = data.data_path
    set_t = pd.read_csv(os.path.join(data_path, 'time_set.csv'),
                        header=None, index_col=1, parse_dates=[1])
    if data.subset_time:
        time_slice = (data.subset_time[0] if len(data.subset_time) == 1
                      else slice(data.subset_time[0], data.subset_time[1]))
    subset_time = set_t[time_slice] if time_slice else set_t

    # Initialise DataFrames and Datasets
    constraint_data = pd.DataFrame()
    cost_data = pd.DataFrame()
    # search through every constraint/cost for use of 'file'
    for variable in data.data_vars:
        if (data[variable].dtype.kind == 'U'
            and any(data[variable].to_dataframe().stack().str.contains('file='))):
            _data = data[variable].to_dataframe()
        else:
            continue
        if 'cost_' in variable:
            cost_data = cost_data.append(_data)
        else:
            constraint_data = constraint_data.append(_data)
    for variable_data in [constraint_data, cost_data]:
        # convert to a Pandas Series to do file search
        variable_data = variable_data.stack()
        filenames = variable_data[variable_data.str.contains('file=')]
        # store all the information about variables which will need to be given
        # over all timesteps, including those which are duplicates
        all_data = variable_data[variable_data != 'nan']
        # create an empty pandas DataFrame
        timeseries_data = pd.DataFrame(index=all_data.index,
                                       columns=[i for i in subset_time.index])
        # fill in values that are just duplicates, not actually from file
        timeseries_data.loc[all_data.drop(filenames.index).index] = \
            np.vstack(all_data.drop(filenames.index).values)
        timeseries_data.columns.name = 'time'
        # create xarray DataArray from DataFrame with the correct dimensions
        timeseries_data = (timeseries_data.stack(dropna=False).unstack(level=-2)
                                          .to_xarray().to_array())
        # create
        df = pd.DataFrame()
        df['filename'], df['column'] = filenames.str.split(':', 1).str
        for file in set(df.filename):
            d_path = os.path.join(data_path, file.split('file=')[1])
            cols = set(df[df.filename==file].column)
            data_from_csv = pd.read_csv(d_path, usecols=cols)
            for col in cols:
                col_df = df[(df.filename == file) & (df.column == col)]
                if len(variable_data.index.levels) == 3: # cost
                    timeseries_data.loc[dict(loc_techs=col_df.index.get_level_values('loc_techs'),
                                             costs=col_df.index.get_level_values('costs'))] = data_from_csv[col].values
                else: # constraints
                    timeseries_data.loc[dict(loc_techs=col_df.index.get_level_values('loc_techs'))] = data_from_csv[col].values
        for variable in timeseries_data['variable']:
            data[variable.item()] = timeseries_data.loc[dict(variable=variable)].drop('variable')
    return data

## Not yet updated from calliope 0.5.3 to next gen.
def initialize_time(data):
    # Carry y_ subset sets over to data for easier data analysis
    time_config = model_run.run.get('time', False)
    if not time_config:
        return None  # Nothing more to do here
    else:
        # For analysis purposes, keep old data around
        data_original = data.copy(deep=True)
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
        model_run.masks = masks
        # Concatenate the DatetimeIndexes by using dummy Series
        chosen_timesteps = pd.concat([pd.Series(0, index=m)
                                     for m in masks.values()]).index
        # timesteps: a list of timesteps NOT picked by masks
        timesteps = pd.Index(data.t.values).difference(chosen_timesteps)
    else:
        timesteps = None
    ##
    # Process function, apply resolution adjustments
    ##
    if 'function' in time_config:
        func = utils.plugin_load(
            time_config.function, builtin_module='time_funcs')
        func_kwargs = time_config.get('function_options', {})
        self.data = func(data=self.data, timesteps=timesteps, **func_kwargs)
        self._sets['t'] = self.data['t'].to_index()
        # Raise error if we've made adjustments incompatible
        # with operational mode
        if self.mode == 'operate':
            opmode_safe = self.data.attrs.get('opmode_safe', False)
            if opmode_safe:
                self.data.attrs['time_res'] = self.get_timeres()
            else:
                msg = 'Time settings incompatible with operational mode'
                raise exceptions.ModelError(msg)
    return None