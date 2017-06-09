"""
Copyright (C) 2013-2017 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

core.py
~~~~~~~

Core model functionality via the Model class.

"""

import datetime
import functools
import inspect
import itertools
import logging
import os
import random
import shutil
import time
import warnings

import pyomo.opt as popt  # pylint: disable=import-error
import pyomo.core as po  # pylint: disable=import-error
# pyomo.environ is needed for pyomo solver plugins
import pyomo.environ  # pylint: disable=unused-import,import-error
import numpy as np
import pandas as pd
import xarray as xr
from pyutilib.services import TempfileManager  # pylint: disable=import-error

from ._version import __version__
from . import exceptions
from . import constraints
from . import locations
from . import output
from . import sets
from . import time_funcs  # pylint: disable=unused-import
from . import time_masks  # pylint: disable=unused-import
from . import utils

# Enable simple format when printing ModelWarnings
formatwarning_orig = warnings.formatwarning
_time_format = '%Y-%m-%d %H:%M:%S'


def _get_time():
    return time.strftime(_time_format)


def _formatwarning(message, category, filename, lineno, line=None):
    """Formats ModelWarnings as "Warning: message" without extra crud"""
    if category == exceptions.ModelWarning:
        return 'Warning: ' + str(message) + '\n'
    else:
        return formatwarning_orig(message, category, filename, lineno, line)

warnings.formatwarning = _formatwarning


@functools.lru_cache(maxsize=1)
def get_default_techs(foo=0):  # pylint: disable=unused-argument
    """
    Get list of techs pre-defined in defaults.yaml.

    The foo=0 parameter makes sure that lru_cache has an argument to cache,
    the function must always be called as get_default_techs() with no
    arguments, ensuring that the values are only read from disk once and
    then cached.

    """
    module_config = os.path.join(os.path.dirname(__file__), 'config')
    o = utils.AttrDict.from_yaml(os.path.join(module_config, 'defaults.yaml'))
    return list(o.techs.keys())


def get_model_config(cr, config_run_path, adjust_data_path=None,
                     insert_defaults=True):
    """
    cr is the run configuration AttrDict,
    config_run_path the path to the run configuration file

    If ``adjust_data_path`` is given, the data_path setting is adjusted
    using the given path, else, it is forced to an absolute path.

    If ``insert_defaults`` is False, the default settings from
    defaults.yaml will not be included, which is necessary when
    generating model settings file for parallel runs.

    """
    # Ensure 'model' key is a list
    if not isinstance(cr.model, list):
        cr.model = [cr.model]

    # Interpret relative config paths as relative to run.yaml
    cr.model = [utils.relative_path(i, config_run_path) for i in cr.model]

    # Load defaults from module path
    module_conf = os.path.join(os.path.dirname(__file__), 'config')
    o = utils.AttrDict.from_yaml(os.path.join(module_conf, 'defaults.yaml'))

    # If defaults should not be inserted, replace the loaded AttrDict
    # with an empty one (a bit of a hack, but we also want the
    # default_techs list so we need to load the AttrDict anyway)
    if not insert_defaults:
        o = utils.AttrDict()
        o.techs = utils.AttrDict()

    # Load all additional files, continuously checking consistency
    for path in cr.model:
        new_o = utils.AttrDict.from_yaml(path)
        if 'techs' in list(new_o.keys()):
            overlap = set(get_default_techs()) & set(new_o.techs.keys())
            if overlap:
                e = exceptions.ModelError
                raise e('Trying to re-define a default technology in '
                        '{}: {}'.format(path, list(overlap)))
        # Interpret data_path as relative to `path`  (i.e the currently
        # open model config file), unless `adjust_data_path` is given
        if 'data_path' in new_o:
            if adjust_data_path:
                new_o.data_path = os.path.join(adjust_data_path,
                                               new_o.data_path)
            else:
                new_o.data_path = utils.relative_path(new_o.data_path, path)
        # The input files are allowed to override defaults
        o.union(new_o, allow_override=True)

    return o


class Model(object):
    """
    Calliope model.

    Parameters
    ----------
    config_run : str or AttrDict, optional
        Path to YAML file with run settings, or AttrDict containing run
        settings. If not given, the included default run and model
        settings are used.
    override : AttrDict, optional
        Provide any additional options or override options from
        ``config_run`` by passing an AttrDict of the form
        ``{'model_settings': 'foo.yaml'}``. Any option possible in
        ``run.yaml`` can be specified in the dict, inluding ``override.``
        options.

    """
    def __init__(self, config_run=None, override=None):
        super().__init__()
        self.verbose = False
        self.debug = utils.AttrDict()

        # Populate self.config_run and self.config_model
        self.initialize_configuration(config_run, override)
        self._get_option = utils.option_getter(self.config_model)
        self.get_cost = utils.cost_getter(self._get_option)

        # Set random seed if specified in run configuration
        random_seed = self.config_run.get('random_seed', None)
        if random_seed:
            np.random.seed(seed=random_seed)

        # Populate config_model with link distances, where metadata is given
        # but no distances given in locations.yaml
        self.get_distances()

        # Initialize sets
        self.initialize_parents()
        self.initialize_sets()

        # Get timeseries constraints/costs
        self.initialize_timeseries()

        # For any exporting technology, set 'export' value to carrier_out
        self.check_and_set_export()

        # Read data and apply time resolution adjustments
        self.read_data()
        self.mode = self.config_run.mode
        self.initialize_time()

    def override_model_config(self, override_dict):
        od = override_dict
        if 'data_path' in od.keys_nested():
            # If run_config overrides data_path, interpret it as
            # relative to the run_config file's path
            od['data_path'] = utils.relative_path(od['data_path'],
                                                  self.config_run_path)
        self.config_model.union(od, allow_override=True,
                                allow_replacement=True)

    def initialize_configuration(self, config_run, override):
        self.flush_option_cache()
        # Load run configuration
        if not config_run:
            raise exceptions.ModelError(
                'Must specify run configuration as either a path or an AttrDict.'
            )
        if isinstance(config_run, str):
            # 1) config_run is a string, assume it's a path
            cr = utils.AttrDict.from_yaml(config_run)
            # self.run_id is used to set an output folder for logs, if
            # debug.keep_temp_files is set to True
            self.run_id = os.path.splitext(os.path.basename(config_run))[0]
            self.config_run_path = config_run
        else:
            # 2) config_run is not a string, assume it's an AttrDict
            cr = config_run
            assert isinstance(cr, utils.AttrDict)
            # we have no filename so we just use current date/time
            self.run_id = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            # Use current working directory as config_run path
            self.config_run_path = os.getcwd()
        self.config_run = cr
        if override:
            assert isinstance(override, utils.AttrDict)
            cr.union(override, allow_override=True, allow_replacement=True)
        # If manually specify a run_id in debug, overwrite the generated one
        if 'debug.run_id' in cr.keys_nested():
            self.run_id = cr.debug.run_id
        self.config_model = get_model_config(cr, self.config_run_path)
        # Override config_model settings if specified in config_run
        # 1) Via 'model_override', which is the path to a YAML file
        if 'model_override' in cr:
            override_path = utils.relative_path(cr.model_override,
                                                self.config_run_path)
            override_dict = utils.AttrDict.from_yaml(override_path)
            self.override_model_config(override_dict)
        # 2) Via 'override', which is an AttrDict
        if ('override' in cr and isinstance(cr.override, utils.AttrDict)):
            self.override_model_config(cr.override)
        # Initialize locations
        locs = self.config_model.locations
        self.config_model.locations = locations.process_locations(locs)
        # As a final step, flush the option cache
        self.flush_option_cache()

    def initialize_timeseries(self):
        """
        Find any constraints/costs values requested as from 'file' in YAMLs
        and store that information.
        """
        timeseries_constraint = ['r']
        allowed_timeseries_constraints = ['r_eff', 'r_scale', 'r2_eff', 's_loss',
                                          'e_prod', 'e_con', 'e_eff', 'p_eff',
                                          'e_cap_min_use', 'e_ramping',
                                          'om_var', 'om_fuel', 'om_r2', 'export']
        # Flatten the dictionary to get e.g. techs.ccgt.constraints.e_eff as keys
        for k, v in self.config_model.as_dict_flat().items():
            if isinstance(v, str):
                if v.startswith("file"):  # Find any referring to a file
                    params = k.split('.')  # Split the elements of the key to get constraint/cost type
                    if params[-1] == 'r':
                        None  # 'r' already in the list automatically
                    elif params[-1] in allowed_timeseries_constraints:  #Look for e.g. e_eff
                        timeseries_constraint.append(params[-1])
                    else:
                        raise Exception("unable to handle loading data from "
                                        "file for '{}'".format(params[-1]))
        # Send list of parameters to config_model AttrDict
        self.config_model['timeseries_constraints'] = list(set(timeseries_constraint))

    def check_and_set_export(self):
        """
        In instances where a technology is allowing export, e.g. techs.ccgt.export: true
        then change 'true' to the carrier of that technology.
        """
        for y in self._sets['y_export']:
            for x in self._sets['x_export']:
                export = self.get_option(y + '.export', x=x)
                if y not in self._sets['y_conversion_plus']:
                    carrier = self.get_option(
                        y + '.carrier', x=x, default=y + '.carrier_out'
                    )
                    other_carriers = []
                else:
                    carrier, other_carriers = self.get_cp_carriers(y, x)
                if export is True:
                    self.set_option(y + '.export', carrier, x=x)
                # any instance where export is not False, but is set to some string value
                elif export != carrier and export not in other_carriers:
                    raise exceptions.ModelError(
                        'Attempting to export carrier {} '.format(export) +
                        'from {}:{} '.format(y, x) +
                        'when this technology does not produce this carrier'
                    )

    def initialize_time(self):
        # Carry y_ subset sets over to data for easier data analysis
        self.data.attrs['_sets'] = {k: v for k, v in self._sets.items() if 'y_' in k}
        self.data['_weights'] = xr.DataArray(
            pd.Series(1, index=self.data['t'].values),
            dims=['t']
        )

        time_config = self.config_run.get('time', False)
        if not time_config:
            return None  # Nothing more to do here
        else:
            # For analysis purposes, keep old data around
            self.data_original = self.data.copy(deep=True)

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
                masks[entry.to_yaml()] = mask_func(self.data, **mask_kwargs)

            self._masks = masks  # FIXME a better place to put masks

            # Concatenate the DatetimeIndexes by using dummy Series
            chosen_timesteps = pd.concat([pd.Series(0, index=m)
                                         for m in masks.values()]).index
            # timesteps: a list of timesteps NOT picked by masks
            timesteps = pd.Index(self.data.t.values).difference(chosen_timesteps)
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

    def get_distances(self):
        """
        Where distances are not given for links, use any metadata to fill
        in the gap.
        Distance calculated using vincenty inverse formula (given in utils module).
        """
        # Check if metadata & links are loaded
        if (not self.config_model.get('metadata', None)
            or not self.config_model.get('links', None)):
            return None

        # We continue if both links and metadata are defined in config_model
        link_tech = set() # fill with 'link.tech' strings
        link_tech_distance = set() # fill with 'link.tech' strings if 'link.tech.distance' exists

        for key in self.config_model.links.as_dict_flat().keys():
            link_tech.update(['.'.join(key.split('.',2)[:2])]) # get only 'link.tech'
            if 'distance' in key:
                link_tech_distance(['.'.join(key.split('.',2)[:2])]) # get only 'link.tech'
            # create set with 'link.tech' strings where 'link.tech.distance' does not exist
            fill_distance = link_tech.difference(link_tech_distance)
        for i in fill_distance:
            # Links are given as 'a,b', so need to split them into individuals
            locs = i.split('.')[0].split(',')
            # Find distance using vincenty func & metadata of lat-long,
            # ignoring curvature if metadata is given as x-y, not lat-long
            loc_coords = self.config_model.metadata.location_coordinates
            loc1 = getattr(loc_coords, locs[0]) # look for first location in a,b
            loc2 = getattr(loc_coords, locs[1]) # look for second location in a,b
            if all(['lat' in key or 'lon' in key for key in # geographic
                   loc_coords.as_dict_flat().keys()]):
                dist = utils.vincenty([loc1.lat, loc1.lon], [loc2.lat, loc2.lon])
            elif all(['x' in key or 'y' in key for key in # cartesian
                     loc_coords.as_dict_flat().keys()]):
                dist = np.sqrt((loc1.x - loc2.x)**2 + (loc1.y - loc2.y)**2)
            else:
                raise KeyError('unidentified coordinate system. Expecting data '
                       'in the format {lat: N, lon: M} or {x: N, y: M} for '
                       'user coordinate values of N, M.')
            # update config_model
            self.config_model.links.set_key(i+'.distance', dist)

    def get_t(self, timestamp, offset=0):
        """
        Get a timestamp before/after (by offset) from the given timestamp
        in the model's set of timestamps. Raises ModelError if out of bounds.

        """
        idx = self.data['t'].to_index()
        if isinstance(offset, pd.tslib.Timedelta):
            loc = idx.get_loc(timestamp + offset)
        else:  # integer
            loc = idx.get_loc(timestamp) + offset
        if loc < 0:
            raise exceptions.ModelError(
                'Attempted to get a timestep before the first one.'
            )
        else:
            try:
                return idx[loc]
            except IndexError:  # Beyond length of index
                raise exceptions.ModelError(
                    'Attempted to get a timestep beoynd the last one.'
                )

    def prev_t(self, timestamp):
        """Return the timestep prior to the given timestep."""
        return self.get_t(timestamp, offset=-1)

    def get_timeres(self, verify=False):
        """Returns resolution of data in hours.

        If ``verify=True``, verifies that the entire file is at the same
        resolution. ``self.get_timeres(verify=True)`` can be called
        after Model initialization to verify this.

        """
        datetime_index = self._sets['t']
        seconds = (datetime_index[0] - datetime_index[1]).total_seconds()
        if verify:
            for i in range(len(datetime_index) - 1):
                assert ((datetime_index[i] - datetime_index[i + 1]).total_seconds()
                        == seconds)
        hours = abs(seconds) / 3600
        return hours

    def get_option(self, option, x=None, default=None,
                   ignore_inheritance=False):
        """
        Retrieves options from model settings for the given tech,
        falling back to the default if the option is not defined for the
        tech.

        If ``x`` is given, will attempt to use location-specific override
        from the location matrix first before falling back to model-wide
        settings.

        If ``default`` is given, it is used as a fallback if no default value
        can be found in the regular inheritance chain. If ``default`` is None
        and the regular inheritance chain defines no default, an error
        is raised.

        If ``ignore_inheritance`` is True, the default is immediately used
        instead of a search through the inheritance chain if the option
        has not been set for the given tech.

        If the first segment of the option contains ':', it will be
        interpreted as implicit tech subsetting: e.g. asking for
        'hvac:r1' implicitly uses 'hvac:r1' with the parent 'hvac', even
        if that has not been defined, to search the option inheritance
        chain.

        Examples:

        * ``model.get_option('ccgt.costs.om_var')``
        * ``model.get_option('csp.weight')``
        * ``model.get_option('csp.r', x='33')``
        * ``model.get_option('ccgt.costs.om_var',\
          default='defaults.costs.om_var')``

        """
        key = (option, x, default, ignore_inheritance)
        try:
            result = self.option_cache[key]
        except KeyError:
            # self._get_option is defined inside __init__
            result = self.option_cache[key] = self._get_option(*key)
        return result

    def set_option(self, option, value, x=None):
        """
        Set ``option`` to ``value``. Returns None on success.

        A default can be set by passing an option like
        ``defaults.constraints.e_eff``.

        """
        if x is None:
            self.config_model.set_key('techs.' + option, value)
        else:  # Setting a specific x
            self._locations.at[x, '_override.' + option] = value
            self.config_model.set_key('locations.' + x + '.override.' + option,
                                      value)
        self.flush_option_cache()

    def flush_option_cache(self):
        self.option_cache = {}

    def get_name(self, y):
        try:
            return self.get_option(y + '.name')
        except exceptions.OptionNotSetError:
            return y

    def get_carrier(self, y, direction, level=None, primary=False, all_carriers=False):
        """
        Get the ``carrier_in`` or ``carrier_out`` of a technology in the model

        Parameters
        ----------
        y: str
            technology
        direction: str, `in` or `out`
            For `carrier_in` and `carrier_out` repectively
        level: int; 2 or 3; optional, default = None
            for conversion_plus technologies, define the carrier level if
            not top level, e.g. level=3 gives `carrier_out_3`
        primary: bool, optional, default = False
            give `primary carrier` for a given technology, which is a carrier in
            `carrier_out` given ass `primary carrier` in the technology definition
        all_carriers: bool, optional, default = False
            give all carriers for tech y and given direction. For conversion_plus
            technologies, this will give an array of carriers, if more than one
            carrier has been defined in the given direction. All levels are combined.
        """
        if y in self._sets['y_conversion_plus']:
            if level:  # Either 2 or 3
                return self.get_option(y + '_'.join('.carrier',
                                                    direction, str(level)))
            primary_carrier, all_carriers = self.get_cp_carriers(y,
                                                           direction=direction)
            if primary:
                return primary_carrier
            if all_carriers:
                return all_carriers

        carrier = self.get_option(
            y + '.carrier', default=y + '.carrier_' + direction
        )
        if not carrier:  # no carrier_in/carrier_out defined
            return 'resource'
        else:
            return carrier

    def get_weight(self, y):
        return self.get_option(y + '.stack_weight')

    def get_color(self, y):
        color = self.get_option(y + '.color')
        if color is False:
            # If no color defined, choose one by seeding random generator
            # with the tech name to get pseudo-random one
            random.seed(y)
            r = lambda: random.randint(0, 255)
            color = '#{:0>2x}{:0>2x}{:0>2x}'.format(r(), r(), r())
        return color

    def get_parent(self, y):
        """Returns the abstract base technology from which ``y`` descends."""
        if y in self._sets['y_transmission']:
            return 'transmission'
        else:
            while True:
                parent = self.get_option(y + '.parent')
                if parent == 'defaults':
                    break
                y = parent
            return y

    def get_group_members(self, group, in_model=True, head_nodes_only=True,
                          expand_transmission=True):
        """
        Return the member technologies of a group. If ``in_model`` is
        True, only technologies (head nodes) in use in the current model
        are returned.

        Returns:
            * A list of group members if there are any.
            * If a group has no members (is only member of other
              groups, i.e. a head node), a list with a single item
              containing only the group/technology itself.
            * An empty list if the group is defined but not allowed
              in the current model.
            * None if the group doesn't exist.

        Other arguments:

            ``head_nodes_only`` : if True, don't return intermediate
                                  groups, i.e. technology definitions
                                  that are inherited from. Setting this
                                  to False only makes sense if in_model
                                  is also False, because in_model=True
                                  implies that only head nodes are
                                  returned.

            ``expand_transmission`` : if True, return in-model
                                      transmission technologies in the
                                      form ``tech:location``.

        """
        def _get(self, group, memberset):
            members = [i for i in self.parents if self.parents[i] == group]
            if members:
                for i, member in enumerate(members):
                    if not head_nodes_only:
                        memberset.add(member)
                    members[i] = _get(self, member, memberset)
                return members
            else:
                memberset.add(group)

        if group not in self.parents:
            return None

        memberset = set()
        _get(self, group, memberset)  # Fills memberset

        if in_model:
            memberset = set([y for y in memberset
                             if (y in self._sets['y']
                                 or y in self._sets['techs_transmission'])])
            # Expand transmission techs
            if expand_transmission:
                for y in list(memberset):
                    if y in self._sets['techs_transmission']:
                        memberset.remove(y)
                        memberset.update([yt
                                          for yt in self._sets['y_transmission']
                                          if yt.startswith(y + ':')])
        return list(memberset)

    @utils.memoize_instancemethod
    def get_eff_ref(self, var, y, x=None):
        """Get reference efficiency, falling back to efficiency if no
        reference efficiency has been set."""
        base = y + '.constraints.' + var
        eff_ref = self.get_option(base + '_eff_ref', x=x, default=False)
        if eff_ref is False:
            eff_ref = self.get_option(base + '_eff', x=x)
        # Check for case wher e_eff is a timeseries file (so the option
        # is a string), and no e_eff_ref has been set to override that
        # string with a numeric option
        if isinstance(eff_ref, str):
            raise exceptions.ModelError(
                'Must set `e_eff_ref` if `e_eff` is a file.'
            )
        return eff_ref

    @utils.memoize_instancemethod
    def get_cp_carriers(self, y, x=None, direction='out'):
        """
        Find all carriers for conversion_plus technology & return the primary
        output carrier as string and all other output carriers as list of strings
        """
        c = self.get_option(y + '.carrier_{}'.format(direction), x=x)
        primary_carrier = self.get_option(y + '.primary_carrier', x=x)
        c_2 = self.get_option(y + '.carrier_{}_2'.format(direction), x=x)
        c_3 = self.get_option(y + '.carrier_{}_3'.format(direction), x=x)
        other_c = set()
        # direction = 'out', 'primary_carrier' has to be defined if 'carrier_in'
        # contains several carriers (i.e. is a dict)
        if direction == 'out' and isinstance(c, dict):
            if not primary_carrier:
                e = exceptions.ModelError
                raise e('primary_carrier must be set for conversion_plus technology '
                        '`{}` as carrier_out contains multiple carriers'.format(y))
            other_c.update(c.keys())
        elif direction == 'out' and isinstance(c, str):
            other_c.update([c])
            primary_carrier = c
        # direction = 'in', there is no "primary_carrier"
        elif direction == 'in':
            other_c.update(c.keys() if isinstance(c, dict) else [c])
            primary_carrier = None
        other_c.update(c_2.keys() if isinstance(c_2, dict) else [c_2])
        other_c.update(c_3.keys() if isinstance(c_3, dict) else [c_3])
        other_c.discard(False) # if there is no secondary or tertiary carrier, they return False
        if len(other_c) == 1: # only one value
            other_c = str(list(other_c)[0])
        else:
            other_c = tuple(other_c)
        return primary_carrier, other_c

    def scale_to_peak(self, df, peak, scale_time_res=True):
        """Returns the given dataframe scaled to the given peak value.

        If ``scale_time_res`` is True, the peak is multiplied by the model's
        time resolution. Set it to False to scale things like efficiencies.

        """
        # Normalize to (0, 1), then multiply by the desired maximum,
        # scaling this peak according to time_res
        if scale_time_res:
            adjustment = self.get_timeres()
        else:
            adjustment = 1
        if peak < 0:
            scale = float(df.min())
        else:
            scale = float(df.max())
        return (df / scale) * peak * adjustment

    def initialize_parents(self):
        techs = self.config_model.techs
        try:
            self.parents = {i: techs[i].parent for i in techs.keys()
                            if i != 'defaults'}
        except KeyError:
            tech = inspect.trace()[-1][0].f_locals['i']
            if 'parent' not in list(techs[tech].keys()):
                e = exceptions.ModelError
                raise e('Technology `' + tech + '` defines no parent!')
        # Verify that no technologies apart from the default technologies
        # inherit from 'defaults'
        for k, v in self.parents.items():
            if k not in get_default_techs() and v == 'defaults':
                e = exceptions.ModelError
                raise e('Tech `' + k + '` inherits from `defaults` but ' +
                        'should inherit from a built-in default technology.')
        # Verify that all parents are themselves actually defined
        for k, v in self.parents.items():
            if v not in list(techs.keys()):
                e = exceptions.ModelError
                raise e('Parent `' + v + '` of technology `' +
                        k + '` is not defined.')

    @utils.memoize_instancemethod
    def ischild(self, y, of):
        """Returns True if ``y`` is a child of ``of``, else False"""
        result = False
        while (result is False) and (y != 'defaults'):
            parent = self.parents[y]
            if parent == of:
                result = True
            y = parent
        return result

    @utils.memoize_instancemethod
    def functionality_switch(self, func_name):
        """
        Check if a given functionality of the model is required, based on whether
        there is any reference to it in model configuration that isn't defaults.

        Args:
         - func_name: str; the funcitonality to check

        Returns: bool; Whether the functionality is switched is on (True) or off (False)
        """
        return any([
            func_name in i for i in self.config_model.as_dict_flat().keys()
            if 'default' not in i
        ])

    def get_time_slice(self):
        if self.config_run.get_key('subset_t', default=False):
            return slice(None)
        else:
            return slice(
                self.config_run.subset_t[0],
                self.config_run.subset_t[1]
            )

    def initialize_sets(self):
        self._sets = utils.AttrDict()
        self.config_model

        # t: time
        _t = pd.read_csv(
            os.path.join(self.config_model.data_path, 'set_t.csv'),
            header=None, index_col=1, parse_dates=[1]
        )
        self._set_t_original = _t.index
        if self.config_run.get_key('subset_t', default=False):
            _t = _t.loc[self.config_run.subset_t[0]:self.config_run.subset_t[1]]
        self._sets['t'] = _t.index

        # x: locations
        _x = list(self.config_model.locations.keys())
        if self.config_run.get_key('subset_x', default=False):
            _x = [i for i in _x if i in self.config_run.subset_x]
        self._sets['x'] = _x

        # y: techs
        sets_y = sets.init_set_y(self, _x)
        self._sets = {**self._sets, **sets_y}

        # x subsets
        sets_x = sets.init_set_x(self)
        self._sets = {**self._sets, **sets_x}
        # c: carriers
        sets_c = sets.init_set_c(self)
        self._sets['c'] = sets_c

        # k: cost classes
        classes_c = [
            list(self.config_model.techs[k].costs.keys())
            for k in self.config_model.techs
            if k != 'defaults'  # Prevent 'default' from entering set
            if 'costs' in self.config_model.techs[k]
        ]
        # Flatten list and make sure 'monetary' is in it
        classes_c = (
            [i for i in itertools.chain.from_iterable(classes_c)]
            + ['monetary']
        )
        # Remove any duplicates by going from list to set and back
        self._sets['k'] = list(set(classes_c))

        # Locations settings matrix and transmission technologies
        self._locations = locations.generate_location_matrix(
            self.config_model.locations, techs=self._sets['y']
        )
        # Locations table: only keep rows that are actually in set `x`
        self._locations = self._locations.loc[_x, :]

        # Initialize transmission technologies
        sets.init_y_trans(self)

        # set 'y' is now complete, ensure that all techs conform to the
        # rule that only "head" techs can be used in the model
        for y in self._sets['y']:
            if self.get_option(y + '.parent') in self._sets['y']:
                e = exceptions.ModelError
                raise e('Only technologies without children can be used '
                        'in the model definition '
                        '({}, {}).'.format(y, self.get_option(y + '.parent')))

    @utils.memoize
    def _get_option_from_csv(self, filename):
        """Read CSV time series"""
        d_path = os.path.join(self.config_model.data_path, filename)
        df = pd.read_csv(d_path, index_col=0)
        df.index = self._set_t_original
        df = df.loc[self._sets['t'], :]  # Subset in case necessary
        # Fill columns that weren't defined with NaN
        # missing_cols = list(set(self.data._x) - set(df.columns))
        # for c in missing_cols:
        #     df[c] = np.nan

        # Ensure that the read file's index matches the data's timesteps
        mismatch = df.index.difference(self._sets['t'])
        if len(mismatch) > 0:
            e = exceptions.ModelError
            entries = mismatch.tolist()
            raise e('File has invalid index. Ensure that it has the same '
                    'date range and resolution as set_t.csv: {}.\n\n'
                    'Invalid entries: {}'.format(filename, entries))
        return df

    def _get_filename(self, param, option, y, x):
        # If we have a string, it must be `file` or `file=..`
        if not option.startswith('file'):
            e = exceptions.ModelError
            raise e('Invalid value for `{}.{}.{}`:'
                    ' `{}`'.format(param, y, x, option))
        # Parse the option and return the filename
        else:
            try:
                # Parse 'file=filename' option
                f = option.split('=')[1]
            except IndexError:
                # If set to just 'file', set filename with y and
                # param, e.g. 'csp_r_eff.csv'
                if isinstance(param, str):
                    f = y + '_' + param + '.csv'
                else:
                    f = y + '_' + '_'.join(param) + '.csv'
        return f

    def _apply_x_map(self, df, x_map, x=None):
        # Format is <name in model config>:<name in data>
        x_map_dict = {i.split(':')[0].strip():
                      i.split(':')[1].strip()
                      for i in x_map.split(',')}
        x_map_str = 'x_map: \'{}\''.format(x_map)
        # Get the mapping for this x from x_map
        # NB not removing old columns in case
        # those are also used somewhere!
        if x is None:
            x = list(x_map_dict.keys())
        elif x in x_map_dict:
            x = [x]
        else:
            x = []
        for this_x in x:
            try:
                x_m = x_map_dict[this_x]
            except KeyError:
                e = exceptions.ModelError
                raise e('x_map defined but does not map '
                        'location defined in model config: '
                        '{}, with {}'.format(this_x, x_map_str))
            if x_m not in df.columns:
                e = exceptions.ModelError
                raise e('Trying to map to a column not '
                        'contained in data: {}, for region '
                        '{}, with {}'
                        .format(x_m, this_x, x_map_str))
            df[this_x] = df[x_m]
        return df

    def _read_param_for_tech(self, param, y, time_res, option, x=None):
        # added check that it is a constraint (string param)
        if option != float('inf') and param == 'r':
            self._sets['y_finite_r'].add(y)
        k = '{}.{}:{}'.format(param, y, x)

        if isinstance(option, str):  # if option is string, read a file
            self._sets['y_' + param + '_timeseries'].add(y)
            f = self._get_filename(param, option, y, x)
            df = self._get_option_from_csv(f)
            self.debug.data_sources.set_key(k, 'file:' + f)

        else:  # option is numeric
            df = pd.DataFrame(
                option,
                index=self._sets['t'], columns=self._sets['x']
            )
            self.debug.data_sources.set_key(k, 'model_config')

        # Apply x_map if necessary
        x_map = self.get_option(y + '.x_map', x=x)
        if x_map is not None:
            df = self._apply_x_map(df, x_map, x)

        if param == 'r' and (x in df.columns or x is None):
            if x is None:
                x_slice = slice(None)
            else:
                x_slice = x
            # Convert power to energy for r, if necessary
            r_unit = self.get_option(y + '.constraints.r_unit', x=x)
            if r_unit == 'power':
                df.loc[:, x_slice] = df.loc[:, x_slice] * time_res

            # Scale r to a given maximum if necessary
            scale = self.get_option(
                y + '.constraints.r_scale_to_peak', x=x
            )
            if scale:
                df.loc[:, x_slice] = self.scale_to_peak(df.loc[:, x_slice], scale)

        if x is not None:
            df = df.loc[:, x]
        return df

    def _validate_param_df(self, param, y, df):
        for x in self._sets['x']:
            if x not in df.columns:
                if self._locations.at[x, y] == 0:
                    df[x] = np.nan
                else:
                    df[x] = 0
                    k = '{}.{}:{}'.format(param, y, x)
                    w = exceptions.ModelWarning
                    message = 'Could not load data for {}'.format(k)
                    warnings.warn(message, w)
                    v = '_NOT_FOUND_'
                    self.debug.data_sources.set_key(k, v)

    def _validate_param_dataset_consistency(self, dataset):
        sources = self.debug.data_sources
        missing_data = [kk for kk in sources.keys_nested()
                        if sources.get_key(kk) == '_NOT_FOUND_']
        if len(missing_data) > 0:
            message = ('The following parameter values could not be read '
                       'from file. They were automatically set to `0`: '
                       + ', '.join(missing_data))
            warnings.warn(message, exceptions.ModelWarning)

        # Finally, check data consistency. For now, demand must be <= 0,
        # and supply >=0, at all times.
        # FIXME update these checks on implementing conditional param updates.
        for y in self._sets['y_finite_r']:
            base_tech = self.get_parent(y)
            possible_x = [x for x in self._sets['x']
                          if self._locations.at[x, y] != 0]
            for x in possible_x:
                series = dataset['r'].loc[{'y': y, 'x': x}].to_pandas()
                err_suffix = 'for tech: {}, at location: {}'.format(y, x)
                if base_tech == 'demand':
                    err = 'Demand resource must be <=0, ' + err_suffix
                    assert (series <= 0).all(), err
                elif base_tech == 'supply':
                    err = 'Supply resource must be >=0, ' + err_suffix
                    assert (series >= 0).all(), err

    def read_data(self):
        """
        Populate parameter data from CSV files or model configuration.

        """
        data = {}
        attrs = {}
        self.debug.data_sources = utils.AttrDict()

        # `time_res` never changes, so always reflects the spacing
        # of time step indices
        attrs['time_res'] = time_res = self.get_timeres()
        time_res_series = pd.Series(time_res, index=self._sets['t'])
        time_res_series.index.name = 't'
        data['_time_res'] = xr.DataArray(time_res_series)

        # Last index t for which model may still use startup exceptions
        startup_time_idx = int(self.config_model.startup_time / time_res)
        attrs['startup_time_bounds'] = self._sets['t'][startup_time_idx]

        # Storage initialization parameter, defined over (x, y)
        s_init = {y: [self.get_option(y + '.constraints.s_init', x=x)
                      for x in self._sets['x']]
                  for y in self._sets['y']}
        s_init = pd.DataFrame(s_init, index=self._sets['x'])
        s_init.columns.name = 'y'
        s_init.index.name = 'x'
        data['s_init'] = xr.DataArray(s_init)

        # Parameters that may be defined over (x, y, t)

        ts_constraint_sets = {
            'y_' + k + '_timeseries': set()
            for k in self.config_model.timeseries_constraints
        }
        self._sets = {**self._sets, **ts_constraint_sets}

        # Add all instances of finite resource (either timeseries dependant or not)
        self._sets['y_finite_r'] = set()

        for param in self.config_model.timeseries_constraints:  # Constraints
            param_data = {}
            if 'om' in param or 'export' in param:  # Cost constraints
                for y in self._sets['y']:
                    cost_ts = {}
                    for k in self._sets['k']:
                        # First, set up each parameter without considering
                        # potential per-location (per-x) overrides
                        option = self.get_cost(param, y, k)

                        cost_ts[k] = self._read_param_for_tech(
                            param, y, time_res, option, x=None)
                        for x in self._sets['x']:
                            # Check for each x whether it defines an override
                            # that is different from the generic option, and if so,
                            # update the dataframe
                            option_x = self.get_cost(param, y, k, x=x)
                            if option != option_x:
                                cost_ts[k].loc[:, x] = self._read_param_for_tech(
                                    param, y, time_res, option_x, x=x)

                        self._validate_param_df(param, y, cost_ts[k])  # Have all `x` been set?
                    # Create
                    cost_ts = {k: xr.DataArray(v, dims=['t', 'x']) for k, v in cost_ts.items()}
                    param_data[y] = xr.Dataset(cost_ts).to_array(dim='k')
            else:
                for y in self._sets['y']:
                    # First, set up each parameter without considering
                    # potential per-location (per-x) overrides
                    j = '.'.join([y, 'constraints', param])
                    option = self.get_option(j)

                    constraint_ts = self._read_param_for_tech(
                        param, y, time_res, option, x=None)
                    for x in self._sets['x']:
                        # Check for each x whether it defines an override
                        # that is different from the generic option, and if so,
                        # update the dataframe
                        option_x = self.get_option(j, x=x)
                        if option != option_x:
                            constraint_ts.loc[:, x] = self._read_param_for_tech(
                                param, y, time_res, option_x, x=x)

                    self._validate_param_df(param, y, constraint_ts)  # Have all `x` been set?
                    param_data[y] = xr.DataArray(constraint_ts, dims=['t', 'x'])

            # Turn param_data into a DataArray
            data[param] = xr.Dataset(param_data).to_array(dim='y')

        dataset = xr.Dataset(data)
        dataset.attrs = attrs

        # Check data consistency
        self._validate_param_dataset_consistency(dataset)

        # Make sure there are no NaNs anywhere in the data
        # to prevent potential solver problems
        dataset = dataset.fillna(0)

        # initialise an additional set now that we know y_finite_r
        self._sets['y_sp_finite_r'] = list(
            self._sets['y_finite_r'].intersection(self._sets['y_supply_plus'])
        )

        self.data = dataset

    def _get_t_max_demand(self):
        """Return timestep index with maximum demand"""
        # FIXME needs unit tests
        t_max_demands = utils.AttrDict()
        for c in self._sets['c']:
            ys = [y for y in self.data['y'].values
                  if self.get_carrier(y, 'in') == c]
            # Get copy of r data array
            r_carrier = self.data['r'].loc[{'y': ys}].copy()
            # Only kep negative (=demand) values
            r_carrier.values[r_carrier.values > 0] = 0
            t_max_demands[c] = (r_carrier.sum(dim='y').sum(dim='x')
                                         .to_dataframe()
                                         .sum(axis=1).idxmin())
        return t_max_demands

    def add_constraint(self, constraint, *args, **kwargs):
        try:
            constraint(self, *args, **kwargs)
        # If there is an error in a constraint, make sure to also get
        # the index where the error happened and pass that along
        except ValueError as e:
            index = inspect.trace()[-1][0].f_locals['index']
            index_string = ', at index: {}'.format(index)
            if not e.args:
                e.args = ('',)
            e.args = (e.args[0] + index_string,) + e.args[1:]
            # Also log it because that is what Pyomo does, and want to ensure
            # that the log entry contains the info we added
            logging.error('Error generating constraint' + index_string)
            raise

    def _param_populator(self, src_data, src_param, levels):

        """
        Returns a `getter` function that returns (x, t)-specific
        values for parameters, used in parameter updating
        """
        getter_data = (src_data[src_param].to_dataframe().reorder_levels(levels)
                                                         .to_dict()[src_param])

        def getter_constraint(m, y, x, t):  # pylint: disable=unused-argument
            return getter_data[(y, x, t)]

        def getter_cost(m, y, x, t, k):  # pylint: disable=unused-argument
            return getter_data[(y, x, t, k)]

        if len(src_data[src_param].dims) == 4:  # Costs
            return getter_cost
        else:  # All other constraints
            return getter_constraint

    def _param_updater(self, src_data, src_param, t_offset=None):
        """
        Returns a `getter` function that returns (x, t)-specific
        values for parameters, used in parameter updating
        """
        if len(src_data[src_param].dims) == 4:  # Costs
            levels = ['y', 'x', 't', 'k']
        else: # all other constraints
            levels = ['y', 'x', 't']
        getter_data = (src_data[src_param].to_dataframe()
                                          .reorder_levels(levels)
                                          .to_dict()[src_param])

        def getter(m, i):  # pylint: disable=unused-argument
            # i is the key of the Pyomo Param, which is (y, x, t)
            # for constraints and (y, x, t, k) for costs
            j = list(i)
            if t_offset:
                j[2] = self.get_t(j[2], t_offset)  # Change time
            return getter_data[tuple(j)]

        return getter

    def update_parameters(self, t_offset):
        d = self.data

        for param in self.config_model.timeseries_constraints:
            y_set = self._sets['y_' + param + '_timeseries']
            initializer = self._param_updater(d, param, t_offset)
            param_object = getattr(self.m, param + '_param')
            for i in param_object.iterkeys():
                param_object[i] = initializer(self.m, i)

        s_init = self.data['s_init'].to_dataframe().to_dict()['s_init']
        s_init_initializer = lambda m, y, x: float(s_init[x, y])
        for y in self.m.y_store:
            for x in self.m.x:
                self.m.s_init[y, x] = s_init_initializer(self.m, y, x)

    def _set_t_end(self):
        # t_end is the timestep previous to t_start + horizon,
        # because the .loc[start:end] slice includes the end
        try:
            offset = int(self.config_model.opmode.horizon /
                         self.data.attrs['time_res']) - 1
            self.t_end = self.get_t(self.t_start, offset=offset)
        except KeyError:
            # If t_end is beyond last timestep, cap it to last one, and
            # log the occurance
            t_bound = self._sets['t'][-1]
            msg = 'Capping t_end to {}'.format(t_bound)
            logging.debug(msg)
            self.t_end = t_bound

    def generate_model(self, t_start=None):
        """
        Generate the model and store it under the property `m`.

        Args:
            t_start : if self.mode == 'operate', this must be specified,
            but that is done automatically via solve_iterative() when
            calling run()

        """
        #
        # Setup
        #
        self.m = m = po.ConcreteModel()
        d = self.data
        self.t_start = t_start
        self.t_max_demand = self._get_t_max_demand()

        #
        # Sets
        #

        # Time steps
        # datetimes = self.data['t'].to_pandas().reset_index(drop=True)
        # datetimes = pd.Series(range(len(self.data['t'])), index=self.data['t'].values)
        if self.mode == 'plan':
            m.t = po.Set(initialize=d['t'].to_index(), ordered=True)
        elif self.mode == 'operate':
            self._set_t_end()
            m.t = po.Set(initialize=d['t'].to_series()[self.t_start:self.t_end].index,
                         ordered=True)

        # Carriers
        m.c = po.Set(initialize=self._sets['c'], ordered=True)
        # Locations
        m.x = po.Set(initialize=self._sets['x'], ordered=True)
        # Locations with only transmission technologies defined
        m.x_transmission = po.Set(
            initialize=self._sets['x_transmission'], within=m.x, ordered=True)
        # Locations which have transmission AND other technologies
        m.x_transmission_plus = po.Set(
            initialize=self._sets['x_transmission_plus'], within=m.x, ordered=True)
        # Source/sink locations (i.e. have `r` defined)
        m.x_r = po.Set(initialize=self._sets['x_r'], within=m.x, ordered=True)
        # Storage locations
        m.x_store = po.Set(
            initialize=self._sets['x_store'], within=m.x, ordered=True)
        # Deman locations
        m.x_demand = po.Set(
            initialize=self._sets['x_demand'], within=m.x, ordered=True)
        # Conversion locations
        m.x_conversion = po.Set(
            initialize=self._sets['x_conversion'], within=m.x, ordered=True)
        # Export locations
        m.x_export = po.Set(
            initialize=self._sets['x_export'], within=m.x, ordered=True)
        # Cost classes
        m.k = po.Set(initialize=self._sets['k'], ordered=True)

        #
        # Technologies and various subsets of technologies
        #

        m.y = po.Set(initialize=self._sets['y'], ordered=True)
        # Supply technologies
        m.y_supply = po.Set(
            initialize=self._sets['y_supply'], within=m.y, ordered=True)
        # Supply+ technologies
        m.y_supply_plus = po.Set(
            initialize=self._sets['y_supply_plus'], within=m.y, ordered=True)
        # Storage only technologies
        m.y_storage = po.Set(
            initialize=self._sets['y_storage'], within=m.y, ordered=True)
        # Transmission technologies
        m.y_transmission = po.Set(
            initialize=self._sets['y_transmission'], within=m.y, ordered=True)
        # Conversion technologies
        m.y_conversion = po.Set(
            initialize=self._sets['y_conversion'], within=m.y, ordered=True)
        # Conversion+ technologies
        m.y_conversion_plus = po.Set(
            initialize=self._sets['y_conversion_plus'], within=m.y, ordered=True)
        # Demand sources
        m.y_demand = po.Set(
            initialize=self._sets['y_demand'], within=m.y, ordered=True)
        # Technologies to deal with unmet demand
        m.y_unmet = po.Set(
            initialize=self._sets['y_unmet'], within=m.y, ordered=True)
        # Supply/Demand sources
        m.y_sd = po.Set(
            initialize=self._sets['y_sd'], within=m.y, ordered=True)
        # Technologies that contain storage
        m.y_store = po.Set(
            initialize=self._sets['y_store'], within=m.y, ordered=True)
        # Technologies with a finite resource defined (timeseries or otherwise)
        m.y_finite_r = po.Set(
            initialize=self._sets['y_finite_r'], within=m.y, ordered=True)
        # Supply+ technologies with a finite resource defined (timeseries or otherwise)
        m.y_sp_finite_r = po.Set(
            initialize=self._sets['y_sp_finite_r'], within=m.y_finite_r,
            ordered=True)
        # Supply/demand technologies with r_area constraints
        m.y_sd_r_area = po.Set(
            initialize=self._sets['y_sd_r_area'], within=m.y, ordered=True)
        # Supply+ technologies with r_area constraints
        m.y_sp_r_area = po.Set(
            initialize=self._sets['y_sp_r_area'], within=m.y, ordered=True)
        # All technologies with r_area constraints
        m.y_r_area = po.Set(
            initialize=self._sets['y_r_area'], within=m.y, ordered=True)
        # Supply+ technologies that allow secondary resource
        m.y_sp_r2 = po.Set(
            initialize=self._sets['y_sp_r2'], within=m.y, ordered=True)
        # Conversion+ technologies that allow secondary carrier_out
        m.y_cp_2out = po.Set(
            initialize=self._sets['y_cp_2out'], within=m.y, ordered=True)
        # Conversion+ technologies that allow tertiary carrier_out
        m.y_cp_3out = po.Set(
            initialize=self._sets['y_cp_3out'], within=m.y, ordered=True)
        # Conversion+ technologies that allow secondary carrier_in
        m.y_cp_2in = po.Set(
            initialize=self._sets['y_cp_2in'], within=m.y, ordered=True)
        # Conversion+ technologies that allow tertiary carrier_in
        m.y_cp_3in = po.Set(
            initialize=self._sets['y_cp_3in'], within=m.y, ordered=True)
        # Technologies that allow export
        m.y_export = po.Set(initialize=self._sets['y_export'], within=m.y, ordered=True)

        # Timeseries
        for param in self.config_model.timeseries_constraints:
            setattr(m, 'y_' + param + '_timeseries',
                    po.Set(initialize=self._sets['y_' + param + '_timeseries'],
                           within=m.y))

        #
        # Parameters
        #

        # Here we set timeseries data as Pyomo parameters as it makes constraints
        # generation significantly quicker. Other data comes from config_model
        # dictionary to avoid over-dependance on Pyomo

        for param in self.config_model.timeseries_constraints:
            y_set = list(getattr(m, 'y_' + param + '_timeseries'))
            if len(d[param].dims) == 4:  # Costs
                initializer = self._param_populator(d, param, ['y', 'x', 't', 'k'])
                setattr(
                    m, param + '_param',
                    po.Param(y_set, m.x, m.t, m.k,
                             initialize=initializer, mutable=True)
                )
            else:  # All other constraints
                initializer = self._param_populator(d, param, ['y', 'x', 't'])
                setattr(
                    m, param + '_param',
                    po.Param(y_set, m.x, m.t,
                             initialize=initializer, mutable=True)
                )

        s_init = self.data['s_init'].to_dataframe().to_dict()['s_init']
        s_init_initializer = lambda m, y, x: float(s_init[x, y])
        m.s_init = po.Param(m.y_store, m.x, initialize=s_init_initializer,
                            mutable=True)

        #
        # Variables and constraints
        #

        # Generate variables, uses same error checking as constraint generation
        self.add_constraint(constraints.base.generate_variables)

        # 1. Required
        constr = [constraints.base.node_resource,
                  constraints.base.node_energy_balance,
                  constraints.base.node_constraints_build,
                  constraints.base.node_constraints_operational,
                  constraints.base.node_constraints_transmission,
                  constraints.base.node_costs,
                  constraints.base.model_constraints]
        if self.mode == 'plan':
            constr += [constraints.planning.system_margin,
                       constraints.planning.node_constraints_build_total]
        for c in constr:
            self.add_constraint(c)

        # 2. Optional
        if self.config_model.get_key('constraints', default=False):
            for c in self.config_model.constraints:
                self.add_constraint(utils._load_function(c))

        # 3. Objective function
        default_obj = 'constraints.objective.objective_cost_minimization'
        objective = self.config_model.get_key('objective', default=default_obj)
        self.add_constraint(utils._load_function(objective))

    def _log_time(self):
        self.run_times["end"] = time.time()
        self.run_times["runtime"] = int(time.time() - self.run_times["start"])
        logging.info('Runtime: ' + str(self.run_times["runtime"]) + ' secs')

    def run(self, iterative_warmstart=True):
        """
        Instantiate and solve the model

        """
        o = self.config_model
        d = self.data
        cr = self.config_run
        self.run_times = {}
        self.run_times["start"] = time.time()
        if self.verbose:
            print('[{}] Model run started.'.format(_get_time()))
        if self.mode == 'plan':
            self.generate_model()  # Generated model goes to self.m
            self.solve()
            self.load_solution()
        elif self.mode == 'operate':
            assert len(self.data['_time_res'].to_series().unique()) == 1, \
                'Operational mode only works with uniform time step lengths.'
            time_res = self.data.attrs['time_res']
            assert (time_res <= self.config_model.opmode.horizon and
                    time_res <= self.config_model.opmode.window), \
                'Timestep length must be smaller than horizon and window.'
            # solve_iterative() generates, solves, and loads the solution
            self.solve_iterative(iterative_warmstart)
        else:
            e = exceptions.ModelError
            raise e('Invalid model mode: `{}`'.format(self.mode))
        self._log_time()
        if self.verbose:
            print('[{}] Solution ready. '
                  'Total run time was {} seconds.'
                  .format(_get_time(), self.run_times["runtime"]))
        if cr.get_key('output.save', default=False) is True:
            output_format = cr.get_key('output.format', default=['netcdf'])
            if not isinstance(output_format, list):
                output_format = [output_format]
            for fmt in output_format:
                self.save_solution(fmt)
            if self.verbose:
                print('[{}] Solution saved to file.'.format(_get_time()))
            save_constr = cr.get_key('output.save_constraints', default=False)
            if save_constr:
                options = cr.get_key('output.save_constraints_options',
                                     default={})
                output.generate_constraints(self.solution,
                                            output_path=save_constr,
                                            **options)
                if self.verbose:
                    print('[{}] Constraints saved to file.'.format(_get_time()))

    def _solve_with_output_capture(self, warmstart, solver_kwargs):
        if self.config_run.get_key('debug.echo_solver_log', default=False):
            return self._solve(warmstart, solver_kwargs)
        else:
            # Silencing output by redirecting stdout and stderr
            with utils.capture_output() as self.pyomo_output:
                return self._solve(warmstart, solver_kwargs)

    def _solve(self, warmstart, solver_kwargs):
        warning = None
        solver = self.config_run.get_key('solver')
        if warmstart:
            try:
                results = self.opt.solve(self.m, warmstart=True,
                                         tee=True, **solver_kwargs)
            except ValueError as e:
                if 'warmstart' in e.args[0]:
                    warning = ('The chosen solver, {}, '
                               'does not support warmstart, '
                               'which may impact performance.').format(solver)
                    results = self.opt.solve(self.m, tee=True, **solver_kwargs)
        else:
            results = self.opt.solve(self.m, tee=True, **solver_kwargs)
        return results, warning

    def solve(self, warmstart=False):
        """
        Args:
            warmstart : (default False) re-solve an updated model
                        instance

        Returns: None

        """
        cr = self.config_run
        solver_kwargs = {}
        if not warmstart:
            solver_io = cr.get_key('solver_io', default=False)
            if solver_io:
                self.opt = popt.SolverFactory(cr.solver, solver_io=solver_io)
            else:
                self.opt = popt.SolverFactory(cr.solver)
            # Set solver options from run_settings file, if it exists
            try:
                for k in cr.solver_options.keys_nested():
                    self.opt.options[k] = cr.solver_options.get_key(k)
            except KeyError:
                pass
            if cr.get_key('debug.symbolic_solver_labels', default=False):
                solver_kwargs['symbolic_solver_labels'] = True
        if cr.get_key('debug.keep_temp_files', default=False):
            solver_kwargs['keepfiles'] = True
            if self.mode == 'plan':
                logdir = os.path.join('Logs', self.run_id)
            elif self.mode == 'operate':
                logdir = os.path.join('Logs', self.run_id
                                      + '_' + str(self.t_start))
            if (cr.get_key('debug.overwrite_temp_files', default=False)
                    and os.path.exists(logdir)):
                shutil.rmtree(logdir)
            os.makedirs(logdir)
            TempfileManager.tempdir = logdir

        self.run_times["preprocessed"] = time.time()
        if self.verbose:
            print('[{}] Model preprocessing took {:.2f} seconds.'
                  .format(_get_time(), self.run_times["preprocessed"]
                          - self.run_times["start"]))

        try:
            self.results, warnmsg = self._solve_with_output_capture(warmstart, solver_kwargs)
        except:
            logging.critical('Solver output:\n{}'.format('\n'.join(self.pyomo_output)))
            raise

        if warnmsg:
            warnings.warn(warnmsg, exceptions.ModelWarning)

        self.load_results()
        self.run_times["solved"] = time.time()
        if self.verbose:
            print('[{}] Solving model took {:.2f} seconds.'
                  .format(_get_time(), self.run_times["solved"]
                          - self.run_times["preprocessed"]))

    def process_solution(self):
        """
        Called from both load_solution() and load_solution_iterative()
        """
        # Add levelized cost
        self.solution = (self.solution
                             .merge(self.get_levelized_cost()
                                        .to_dataset(name='levelized_cost')))
        # Add capacity factor
        self.solution = (self.solution
                             .merge(self.get_capacity_factor()
                                        .to_dataset(name='capacity_factor')))
        # Add metadata
        md = self.get_metadata()
        md.columns.name = 'cols_metadata'
        md.index.name = 'y'
        self.solution = (self.solution.merge(xr.DataArray(md)
                                               .to_dataset(name='metadata')))
        # Add summary
        summary = self.get_summary()
        summary.columns.name = 'cols_summary'
        summary.index.name = 'techs'
        self.solution = (self.solution.merge(xr.DataArray(summary)
                                               .to_dataset(name='summary')))
        # Add groups
        groups = self.get_groups()
        groups.columns.name = 'cols_groups'
        groups.index.name = 'techs'
        self.solution = (self.solution.merge(xr.DataArray(groups)
                                               .to_dataset(name='groups')))
        # Add shares
        shares = self.get_shares(groups)
        shares.columns.name = 'cols_shares'
        shares.index.name = 'techs'
        self.solution = (self.solution.merge(xr.DataArray(shares)
                                               .to_dataset(name='shares')))
        # Add time resolution
        self.solution = (self.solution
                             .merge(self.data['_time_res']
                                        .copy(deep=True)
                                        .to_dataset(name='time_res')))
        # Add model and run configuration
        self.solution.attrs['config_run'] = self.config_run
        self.solution.attrs['config_model'] = self.config_model

    def load_solution(self):
        sol = self.get_node_variables()
        sol = sol.merge(self.get_totals())
        sol = sol.merge(self.get_node_parameters())
        sol = sol.merge(self.get_costs().to_dataset(name='costs'))
        self.solution = sol
        self.process_solution()

    def get_var(self, var, dims=None, standardize_coords=True):
        """
        Return output for variable `var` as a pandas.Series (1d),
        pandas.Dataframe (2d), or xarray.DataArray (3d and higher).

        Parameters
        ----------
        var : variable name as string, e.g. 'es_prod'
        dims : list, optional
            indices as strings, e.g. ('y', 'x', 't');
            if not given, they are auto-detected

        """
        m = self.m
        try:
            var_container = getattr(m, var)
        except AttributeError:
            raise exceptions.ModelError('Variable {} inexistent.'.format(var))
        # Get dims
        if not dims:
            dims = [i.name for i in var_container.index_set().set_tuple]
        # Make sure standard coordinate names are used
        if standardize_coords:
            dims = [i.split('_')[0] for i in dims]
        result = pd.DataFrame.from_dict(var_container.get_values(), orient='index')
        if result.empty:
            raise exceptions.ModelError('Variable {} has no data.'.format(var))
        result.index = pd.MultiIndex.from_tuples(result.index, names=dims)
        result = result[0]  # Get the only column in the dataframe
        # Unstack and sort by time axis
        if len(dims) == 1:
            result = result.sort_index()
        elif len(dims) == 2:
            # if len(dims) is 2, we already have a well-formed DataFrame
            result = result.unstack(level=0)
            result = result.sort_index()
        else:  # len(dims) >= 3
            result = xr.DataArray.from_series(result)
        return result

    def get_c_sum(self):
        c = self.get_var('c_prod') + self.get_var('c_con')
        return c.fillna(0)

    def get_node_variables(self):
        detail = ['s', 'r', 'r2', 'export']
        p = xr.Dataset()
        p['e'] = self.get_c_sum()
        for v in detail:
            try:
                p[v] = self.get_var(v)
            except exceptions.ModelError:
                continue
        return p

    def get_e_cap_net(self):
        # Create a DataFrame of p_eff to combine with the decision variable e_cap
        # to get e_cap_net
        m = self.m
        p_eff = pd.DataFrame.from_dict({
            (y, x): self.get_option(y + '.constraints.p_eff', x=x)
            for y in m.y for x in m.x}, orient='index')
        p_eff.index = pd.MultiIndex.from_tuples(p_eff.index, names=['y', 'x'])
        p_eff = p_eff[0].unstack(level=0).sort_index()

        return self.get_var('e_cap') * p_eff

    def get_node_parameters(self):
        detail = ['e_cap', 's_cap', 'r_cap', 'r_area', 'r2_cap']
        result = xr.Dataset()
        for v in detail:
            try:
                result[v] = self.get_var(v)
            except exceptions.ModelError:
                continue
        result['e_cap_net'] = self.get_e_cap_net()
        return result

    def get_costs(self, t_subset=None):
        if t_subset is None:
            return self.get_var('cost')
        else:
            # len_adjust is the fraction of construction and fixed costs
            # that is accrued to the chosen t_subset. NB: construction and fixed
            # operation costs are calculated for a whole year
            len_adjust = (sum(self.data['_time_res'].to_series().iloc[t_subset])
                          / sum(self.data['_time_res'].to_series()))
            # Adjust for the fact that fixed costs accrue over a smaller length
            # of time as per len_adjust
            cost_fixed = self.get_var('cost_fixed') * len_adjust

            # Adjust for the fact that variable costs are only accrued over
            # the t_subset period
            cost_variable = self.get_var('cost_var')[{'t': t_subset}].sum(dim='t')

            return cost_fixed + cost_variable

    def get_totals(self, t_subset=None, apply_weights=True):
        """Get total produced and consumed per technology and location."""
        if t_subset is None:
            t_subset = slice(None)

        if apply_weights:
            try:
                weights = self.data['_weights'][dict(t=t_subset)]
            except AttributeError:
                weights = 1
        else:
            weights = 1

        p = xr.Dataset({i: (self.get_var(i)[dict(t=t_subset)]
                        * weights).sum(dim='t')
                        for i in ['c_prod', 'c_con']})
        return p

    def get_levelized_cost(self):
        """
        Get levelized costs.

        NB: Only production, not consumption, is used in calculations.

        """
        sol = self.solution
        cost_dict = {}
        for cost in self._sets['k']:
            carrier_dict = {}
            for carrier in self._sets['c']:
                # Levelized cost of electricity (LCOE)
                with np.errstate(divide='ignore', invalid='ignore'):  # don't warn about division by zero
                    lc = (sol['costs'].loc[dict(k=cost)] /
                          sol['c_prod'].loc[dict(c=carrier)])
                lc = lc.to_pandas()

                # Make sure the dataframe has y as columns and x as index
                if lc.index.name == 'y':
                    lc = lc.T

                lc = lc.replace(np.inf, 0)
                carrier_dict[carrier] = lc
            cost_dict[cost] = xr.Dataset(carrier_dict).to_array(dim='c')
        arr = xr.Dataset(cost_dict).to_array(dim='k')
        return arr

    def _get_time_res_sum(self):
        m = self.m
        time_res = self.data['_time_res'].to_series()
        weights = self.data['_weights'].to_series()

        try:  # Try loading time_res_sum from operational mode
            time_res_sum = self.data.attrs['time_res_sum']
        except KeyError:
            time_res_sum = sum(time_res.at[t] * weights.at[t] for t in m.t)
        return time_res_sum

    def get_capacity_factor(self):
        """
        Get capacity factor.

        NB: Only production, not consumption, is used in calculations.

        """
        sol = self.solution
        cfs = {}
        for carrier in sol.coords['c'].values:
            time_res_sum = self._get_time_res_sum()
            with np.errstate(divide='ignore', invalid='ignore'):
                cf = sol['c_prod'].loc[dict(c=carrier)] / (sol['e_cap_net']
                                                           * time_res_sum)
            cf = cf.to_pandas()

            # Make sure the dataframe has y as columns and x as index
            if cf.index.name == 'y':
                cf = cf.T

            cf = cf.fillna(0)
            cfs[carrier] = cf
        arr = xr.Dataset(cfs).to_array(dim='c')
        return arr

    def get_metadata(self):
        df = pd.DataFrame(index=self._sets['y'])
        df.loc[:, 'type'] = df.index.map(lambda y: self.get_parent(y))
        df.loc[:, 'name'] = df.index.map(lambda y: self.get_name(y))
        df.loc[:, 'carrier_in'] = df.index.map(
            lambda y: self.get_carrier(y, direction='in', all_carriers=True))
        df.loc[:, 'carrier_out'] = df.index.map(
            lambda y: self.get_carrier(y, direction='out', all_carriers=True))
        df.loc[:, 'stack_weight'] = df.index.map(lambda y: self.get_weight(y))
        df.loc[:, 'color'] = df.index.map(lambda y: self.get_color(y))
        return df

    def get_summary(self, sort_by='e_cap', carrier=None):
        sol = self.solution

        c_prod = sol['c_prod'].sum(dim='x')
        c_con = sol['c_con'].sum(dim='x')
        df = pd.DataFrame(index=sol.y, columns=['e_con', 'e_prod'])

        if carrier is not None:
            # If a specific carrier is asked for, we only grab that one
            df['e_prod'] = c_prod.loc[dict(c=carrier)]
            df['e_con'] = c_con.loc[dict(c=carrier)]
        else:
            # If no specific carrier asked for, we intelligently pick
            # the correct one for each technology
            for y in sol.y:
                carrier_out = self.get_carrier(
                    y.item(), direction='out', all_carriers=True
                )
                carrier_in = self.get_carrier(
                    y.item(), direction='in', all_carriers=True
                )
                if isinstance(carrier_out, tuple):  # conversion_plus
                    # for carrier_out, we take the primary_carrier
                    carrier_out = self.get_carrier(y.item(), direction='out',
                                                   primary=True)
                if isinstance(carrier_in, tuple):  # conversion_plus
                    # here we don't have a 'primary_carrier_in' value,
                    # so we just take the first in the list
                    carrier_in = list(carrier_in)[0]

                df.loc[y.item(), 'e_prod'] = (
                    c_prod.loc[dict(y=y, c=carrier_out)].item()
                    if carrier_out != 'resource' else np.nan
                )
                df.loc[y.item(), 'e_con'] = (
                    c_con.loc[dict(y=y, c=carrier_in)].item()
                    if carrier_in != 'resource' else np.nan
                )
        df = df.astype(np.float32)  # np.float needed to avoid divide by zero error

        # Total (over locations) capacity factors per carrier
        time_res_sum = self._get_time_res_sum()
        with np.errstate(divide='ignore', invalid='ignore'):  # don't warn about division by zero
            cf = np.divide(df.loc[:, 'e_prod'],
                           (sol['e_cap_net'].sum(dim='x') * time_res_sum))
        df['cf'] = cf

        # Total (over locations) levelized costs per carrier
        for k in sorted(sol['levelized_cost'].coords['k'].values):
            with np.errstate(divide='ignore', invalid='ignore'):  # don't warn about division by zero
                df['levelized_cost_' + k] = np.divide(
                    sol['costs'].loc[dict(k=k)].sum(dim='x'),
                    df.loc[:, 'e_prod'])

        # Add other carrier-independent stuff
        df['e_cap'] = sol['e_cap'].sum(dim='x')
        # Optional characteristics:
        optionals = ['r_area', 's_cap', 'r_cap']
        for optional in optionals:
            try:
                df[optional] = sol[optional].sum(dim='x')
            except:
                continue

        # # Add technology type
        # df['type'] = df.index.map(self.get_parent)

        # Get the basename of each tech (i.e., 'hvac' for 'hvac:location1')
        df['index_name'] = df.index
        basenames = [i[0] for i in df.index_name.str.split(':').tolist()]

        # Add this to the summary df
        df['basename'] = basenames

        # Now go through each transmission tech and sum it up into one row,
        # appending this to the summary df
        transmission_basetechs = set([t for t in df.basename
                                      if self.get_parent(t)
                                      == 'transmission'])

        for basename in transmission_basetechs:
            if df.basename.str.contains(basename).any():
                temp = df.query('basename == "{}"'.format(basename))
                temp_sum = temp.sum()
                cf_cost_cols = (['cf'] +
                                [c for c in df.columns if 'cost_' in c])
                temp_cf_cost = temp.loc[:, cf_cost_cols] \
                                   .mul(temp.loc[:, 'e_prod'], axis=0) \
                                   .sum() / temp.loc[:, 'e_prod'].sum()
                temp_sum.loc[cf_cost_cols] = temp_cf_cost
                temp_sum.index_name = basename
                temp_sum.type = 'transmission'
                df = df.append(temp_sum, ignore_index=True)

        # Finally, drop the transmission techs with ':' in their name,
        # only keeping the summary rows, drop temporary columns, and re-set
        # the index
        df = df[~df.index_name.str.contains(':')]
        df = df.set_index('index_name')
        df.index.name = 'y'
        df = df.drop(['basename'], axis=1)

        return df.sort_values(by=sort_by, ascending=False)

    def get_groups(self):
        ggm = self.get_group_members
        s = pd.Series({k: '|'.join(ggm(k, head_nodes_only=True))
                      for k in self.config_model.techs
                      if ggm(k, head_nodes_only=True) != []
                      and ggm(k, head_nodes_only=True) is not None})

        df = pd.DataFrame(s, columns=['members'])

        # Forcing booleans to strings so that groups table has
        # uniform datatypes
        gg = lambda y: str(self.get_option(
            y + '.group', default='defaults.group',
            ignore_inheritance=True)
        )

        df['group'] = df.index.map(gg)
        df['type'] = df.index.map(self.get_parent)
        return df

    def get_shares(self, groups):
        from . import analysis
        vars_ = ['e_prod', 'e_con', 'e_cap']
        df = pd.DataFrame(index=groups.index, columns=vars_)
        for var in vars_:
            for index, row in groups.iterrows():
                group_members = row['members'].split('|')
                group_type = row['type']
                share = analysis.get_group_share(self.solution, group_members,
                                                 group_type, var=var)
                df.at[index, var] = share.to_pandas()
        return df

    def load_solution_iterative(self, node_vars, total_vars, cost_vars):
        totals = sum(total_vars)
        costs = sum(cost_vars)
        node = xr.concat(node_vars, dim='t')
        # We are simply concatenating the same timesteps over and over again
        # when we concatenate the indivudal runs, so we need to set the
        # correct time axis again
        node['t'] = self._sets['t']

        sol = self.get_node_parameters()
        sol = sol.merge(totals)
        sol = sol.merge(costs)
        sol = sol.merge(node)
        self.solution = sol
        self.process_solution()

    def solve_iterative(self, iterative_warmstart=True):
        """
        Solve iterative by updating model parameters.

        By default, on optimizations subsequent to the first one,
        warmstart is used to speed up the model generation process.

        Returns None on success, storing results under self.solution

        """
        o = self.config_model
        d = self.data
        time_res = d['_time_res'].to_series()
        window_adj = int(self.config_model.opmode.window / d.attrs['time_res'])
        steps = [self._sets['t'][i]
                 for i in range(len(self._sets['t']))
                 if (i % window_adj) == 0]
        # Remove the last step - since we look forward at each step,
        # it would take us beyond actually existing data
        steps = steps[:-1]
        node_vars = []
        total_vars = []
        cost_vars = []
        d.attrs['time_res_sum'] = 0
        # This will fail if the time range given is too short, i.e. there are
        # no future timesteps to consider.
        if len(steps) == 0:
            raise exceptions.ModelError('Unable to solve iteratively with '
                                        'current time subset and step-size')
        self.generate_model(t_start=steps[0])
        for index, step in enumerate(steps):
            if index == 0:
                self.solve(warmstart=False)
            else:
                self.t_start = step
                self._set_t_end()
                # Note: we don't update the timestep set, so it keeps the
                # values it got on first construction. Instead,
                # we use an offset when updating parameter data so that
                # the correct values are read into the "incorrect" timesteps.
                self.update_parameters(t_offset=step - steps[0])
                self.solve(warmstart=iterative_warmstart)
            self.load_results()

            # Gather relevant model results over decision interval, so
            # we only grab [0:window/time_res_static] steps, where
            # window/time_res_static will be an iloc index
            if index == (len(steps) - 1):
                # Final iteration saves data from entire horizon
                stepsize = int(self.config_model.opmode.horizon / d.attrs['time_res'])
            else:
                # Non-final iterations only save data from window
                stepsize = int(self.config_model.opmode.window / d.attrs['time_res'])

            node = self.get_node_variables()
            node_vars.append(node[dict(t=slice(0, stepsize))])
            # Get totals
            totals = self.get_totals(t_subset=slice(0, stepsize))
            total_vars.append(totals)
            costs = (self.get_costs(t_subset=slice(0, stepsize))
                         .to_dataset(name='costs'))
            cost_vars.append(costs)
            timesteps = [time_res.at[t] for t in self.m.t][0:stepsize]
            d.attrs['time_res_sum'] += sum(timesteps)

            # Save state of storage for carry over to next iteration
            if self._sets['y_store']:
                s = self.get_var('s')
                # Convert from timestep length to absolute index
                storage_state_index = stepsize - 1
                assert (isinstance(storage_state_index, int) or
                        storage_state_index.is_integer())
                storage_state_index = int(storage_state_index)
                d['s_init'] = s[dict(t=storage_state_index)].to_pandas().T

        self.load_solution_iterative(node_vars, total_vars, cost_vars)

    def load_results(self):
        """Load results into model instance for access via model variables."""
        not_optimal = (self.results['Solver'][0]['Termination condition'].key
                       != 'optimal')
        r = self.m.solutions.load_from(self.results)
        if r is False or not_optimal:
            logging.critical('Solver output:\n{}'.format('\n'.join(self.pyomo_output)))
            logging.critical(self.results.Problem)
            logging.critical(self.results.Solver)
            if not_optimal:
                message = 'Model solution was non-optimal.'
            else:
                message = 'Could not load results into model instance.'
            raise exceptions.ModelError(message)

    def save_solution(self, how):
        """Save model solution. ``how`` can be 'netcdf' or 'csv'"""

        if 'path' not in self.config_run.output:
            self.config_run.output['path'] = 'Output'
            logging.warning('`config_run.output.path` not set, using default: `Output`')

        # Create output dir, but ignore if it already exists
        try:
            os.makedirs(self.config_run.output.path)
        except OSError:  # Hoping this isn't raised for more serious stuff
            pass
        # except KeyError:  # likely because `path` or `output` not defined
        #     raise exceptions.ModelError('`config_run.output.path` not configured.')

        # Add input time series (r and e_eff) alongside the solution
        for param in self.config_model.timeseries_constraints:
            subset_name = 'y_' + param + '_timeseries'
            # Only if the set has some members
            if len(self._sets[subset_name]) > 0:
                self.solution[param] = self.data[param]

        if how == 'netcdf':
            self._save_netcdf4()
        elif how == 'csv':
            self._save_csv()
        else:
            raise ValueError('Unsupported value for `how`: {}'.format(how))

        # Remove time series from solution again after writing it to disk
        for param in self.config_model.timeseries_constraints:
            if param in self.solution:
                del self.solution[param]
        return None

    def _save_netcdf4(self):
        """
        Save solution as NetCDF4 to the file ``solution.nc`` in
        ``self.config_run.output.path``

        """
        sol = self.solution
        store_file = os.path.join(self.config_run.output.path, 'solution.nc')
        # Raise error if file exists already, to make sure we don't destroy
        # existing data
        if os.path.exists(store_file):
            i = 0
            alt_file = os.path.join(self.config_run.output.path,
                                    'solution_{}.nc')
            while os.path.exists(alt_file.format(i)):
                i += 1
            alt_file = alt_file.format(i)  # Now "pick" the first free filename
            message = ('File `{}` exists, '
                       'using `{}` instead.'.format(store_file, alt_file))
            logging.warning(message)
            store_file = alt_file

        # Metadata
        for k in ['config_model', 'config_run']:
            # Serialize config dicts to YAML strings
            sol.attrs[k] = sol.attrs[k].to_yaml()
        sol.attrs['run_time'] = self.run_times["runtime"]
        sol.attrs['calliope_version'] = __version__

        encoding = {k: {'zlib': True, 'complevel': 4} for k in self.solution.data_vars}
        self.solution.to_netcdf(store_file, format='netCDF4', encoding=encoding)
        self.solution.close()  # Force-close NetCDF file after writing

        return store_file  # Return the path to the NetCDF file we used

    def _save_csv(self):
        """Save solution as CSV files to ``self.config_run.output.path``"""
        for k in self.solution.data_vars:
            out_path = os.path.join(self.config_run.output.path, '{}.csv'.format(k))
            self.solution[k].to_dataframe().to_csv(out_path)

        # Metadata
        md = utils.AttrDict()
        md['config_run'] = self.config_run
        md['config_model'] = self.config_model
        md['run_time'] = self.run_times["runtime"]
        md['calliope_version'] = __version__
        md.to_yaml(os.path.join(self.config_run.output.path, 'metadata.yaml'))

        return self.config_run.output.path
