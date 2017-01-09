"""
Copyright (C) 2013-2017 Stefan Pfenninger.
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

# Parameters that may be defined as time series
_TIMESERIES_PARAMS = ['r', 'e_eff']

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


class BaseModel(object):
    """
    Base model class.

    """
    def __init__(self):
        super().__init__()


class Model(BaseModel):
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
            config_run = os.path.join(os.path.dirname(__file__),
                                      'example_model', 'run.yaml')
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
        if ('override' in cr
                and isinstance(cr.override, utils.AttrDict)):
            self.override_model_config(cr.override)
        # Initialize locations
        locs = self.config_model.locations
        self.config_model.locations = locations.process_locations(locs)
        # As a final step, flush the option cache
        self.flush_option_cache()

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
            func = utils.plugin_load(time_config.function, builtin_module='time_funcs')
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
        if 'metadata' not in self.config_model or 'links' not in self.config_model:
            return
        elif self.config_model.links:
            for link, v in self.config_model.links.items():
                for trans, v2 in self.config_model.links[link].items():
                    # for a given link and transmission type (e.g. 'hvac'), check if distance is set.
                    if 'distance' not in self.config_model.links[link][trans]:
                        # Links are given as 'a,b', so need to split them into individuals
                        links = link.split(',')
                        # Find distance using geopy package & metadata of lat-long
                        dist = utils.vincenty(getattr(self.config_model.metadata.location_coordinates, links[0]),
                                        getattr(self.config_model.metadata.location_coordinates, links[1]))
                        # update config_model
                        self.config_model.links[link][trans]['distance'] = dist

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
                assert ((datetime_index[i] - datetime_index[i+1]).total_seconds()
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
        self.flush_option_cache()

    def flush_option_cache(self):
        self.option_cache = {}

    def get_name(self, y):
        try:
            return self.get_option(y + '.name')
        except exceptions.OptionNotSetError:
            return y

    def get_carrier(self, y):
        return self.get_option(y + '.carrier')

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

    def get_source_carrier(self, y):
        source_carrier = self.get_option(y + '.source_carrier')
        if source_carrier:
            return source_carrier
        else:
            return None

    def get_parent(self, y):
        """Returns the abstract base technology from which ``y`` descends."""
        if y in self._sets['y_trans']:
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
        if not group in self.parents:
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
                                          for yt in self._sets['y_trans']
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

        # c: carriers
        _c = set()
        for y in self._sets['y']:  # Only add carriers for allowed technologies
            _c.update([self.get_option(y + '.carrier')])
            if self.get_option(y + '.source_carrier'):
                _c.update([self.get_option(y + '.source_carrier')])
        _c = list(_c)
        self._sets['c'] = _c

        # k: cost classes
        classes = [list(self.config_model.techs[k].costs.keys())
                   for k in self.config_model.techs
                   if k != 'defaults'  # Prevent 'default' from entering set
                   if 'costs' in self.config_model.techs[k]]
        # Flatten list and make sure 'monetary' is in it
        classes = ([i for i in itertools.chain.from_iterable(classes)]
                   + ['monetary'])
        # Remove any duplicates by going from list to set and back
        self._sets['k'] = list(set(classes))

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

    def _get_filename(self, param, y, x):
        option = self.get_option(y + '.constraints.' + param, x=x)
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
                f = y + '_' + param + '.csv'
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

    def _read_param_for_tech(self, param, y, time_res, x=None):
        option = self.get_option(y + '.constraints.' + param, x=x)
        if option != float('inf'):
            self._sets['y_def_' + param].add(y)
        k = '{}.{}:{}'.format(param, y, x)

        if isinstance(option, str):  # if option is string, read a file
            f = self._get_filename(param, y, x)
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
        for y in self._sets['y_def_r']:
            base_tech = self.get_parent(y)
            possible_x = [x for x in dataset['x'].values
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
        ts_sets = {'y_def_' + k: set() for k in _TIMESERIES_PARAMS}
        self._sets = {**self._sets, **ts_sets}

        for param in _TIMESERIES_PARAMS:
            param_data = {}
            for y in self._sets['y']:
                # First, set up each parameter without considering
                # potential per-location (per-x) overrides
                df = self._read_param_for_tech(param, y, time_res, x=None)
                k = y + '.constraints.' + param

                option = self.get_option(k)
                for x in self._sets['x']:
                    # Check for each x whether it defines an override
                    # that is different from the generic option, and if so,
                    # update the dataframe
                    option_x = self.get_option(k, x=x)
                    if option != option_x:
                        df.loc[:, x] = self._read_param_for_tech(param, y, time_res, x=x)

                self._validate_param_df(param, y, df)  # Have all `x` been set?

                param_data[y] = xr.DataArray(df, dims=['t', 'x'])

            # Turn param_data into a DataArray
            data[param] = xr.Dataset(param_data).to_array(dim='y')

        dataset = xr.Dataset(data)
        dataset.attrs = attrs

        # Check data consistency
        self._validate_param_dataset_consistency(dataset)

        # Make sure there are no NaNs anywhere in the data
        # to prevent potential solver problems
        dataset = dataset.fillna(0)

        self.data = dataset

    def _get_t_max_demand(self):
        """Return timestep index with maximum demand"""
        # FIXME needs unit tests
        t_max_demands = utils.AttrDict()
        for c in self._sets['c']:
            ys = [y for y in self.data['y'].values
                  if self.get_option(y + '.carrier') == c]
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

    def _param_populator(self, src_data, src_param, t_offset=None):
        """
        Returns a `getter` function that returns (x, t)-specific
        values for parameters

        """
        getter_data = (src_data[src_param].to_dataframe()
                                          .reorder_levels(['y', 'x', 't'])
                                          .to_dict()[src_param])

        def getter(m, y, x, t):  # pylint: disable=unused-argument
            if t_offset:
                t = self.get_t(t, t_offset)
            return getter_data[(y, x, t)]
            # return src.loc[{'y': y, 'x': x, 't': t}]
        return getter

    def update_parameters(self, t_offset):
        d = self.data

        for param in _TIMESERIES_PARAMS:
            initializer = self._param_populator(d, param, t_offset)
            y_set = self._sets['y_def_' + param]
            param_object = getattr(self.m, param)
            for y in y_set:
                for x in self.m.x:
                    for t in self.m.t:
                        param_object[y, x, t] = initializer(self.m, y, x, t)

        s_init = self.data['s_init'].to_dataframe().to_dict()['s_init']
        s_init_initializer = lambda m, y, x: float(s_init[x, y])
        for y in self.m.y_pc:
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
        # Cost classes
        m.k = po.Set(initialize=self._sets['k'], ordered=True)
        #
        # Technologies and various subsets of technologies
        #
        m.y = po.Set(initialize=self._sets['y'], ordered=True)
        # Production technologies
        m.y_prod = po.Set(initialize=self._sets['y_prod'], within=m.y, ordered=True)
        # Production technologies
        m.y_con = po.Set(initialize=self._sets['y_con'], within=m.y, ordered=True)
        # Production/consumption technologies
        m.y_pc = po.Set(initialize=self._sets['y_pc'], within=m.y, ordered=True)
        # Transmission technologies
        m.y_trans = po.Set(initialize=self._sets['y_trans'], within=m.y, ordered=True)
        # Conversion technologies
        m.y_conv = po.Set(initialize=self._sets['y_conv'], within=m.y, ordered=True)
        # Technologies with specified `r`
        m.y_def_r = po.Set(initialize=self._sets['y_def_r'], within=m.y)
        # Technologies with specified `e_eff`
        m.y_def_e_eff = po.Set(initialize=self._sets['y_def_e_eff'], within=m.y)
        # Technologies that allow `rb`
        m.y_rb = po.Set(initialize=self._sets['y_rb'], within=m.y)
        # Technologies with parasitics
        m.y_p = po.Set(initialize=self._sets['y_p'], within=m.y)
        # Technologies without parasitics
        set_no_parasitics = set(self._sets['y']) - set(self._sets['y_p'])
        m.y_np = po.Set(initialize=set_no_parasitics, within=m.y)

        #
        # Parameters
        #

        for param in _TIMESERIES_PARAMS:
            y = getattr(m, 'y_def_' + param)
            # param_data = self.data[param].to_dataframe().reorder_levels(['y', 'x', 't']).to_dict()[param]
            initializer = self._param_populator(self.data, param)
            setattr(m, param, po.Param(y, m.x, m.t, initialize=initializer, mutable=True))

        s_init = self.data['s_init'].to_dataframe().to_dict()['s_init']
        s_init_initializer = lambda m, y, x: float(s_init[x, y])
        m.s_init = po.Param(m.y_pc, m.x, initialize=s_init_initializer,
                            mutable=True)

        #
        # Variables and constraints
        #

        # 1. Required
        constr = [constraints.base.node_resource,
                  constraints.base.node_energy_balance,
                  constraints.base.node_constraints_build,
                  constraints.base.node_constraints_operational,
                  constraints.base.node_constraints_transmission,
                  constraints.base.node_parasitics,
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
                  .format(_get_time(), self.run_times["preprocessed"] - self.run_times["start"]))

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
                  .format(_get_time(), self.run_times["solved"] - self.run_times["preprocessed"]))

    def process_solution(self):
        """
        Called from both load_solution() and load_solution_iterative()
        """
        # Add levelized cost
        self.solution = self.solution.merge(self.get_levelized_cost().to_dataset(name='levelized_cost'))
        # Add capacity factor
        self.solution = self.solution.merge(self.get_capacity_factor().to_dataset(name='capacity_factor'))
        # Add metadata
        md = self.get_metadata()
        md.columns.name = 'cols_metadata'
        md.index.name = 'y'
        self.solution = self.solution.merge(xr.DataArray(md).to_dataset(name='metadata'))
        # Add summary
        summary = self.get_summary()
        summary.columns.name = 'cols_summary'
        summary.index.name = 'techs'
        self.solution = self.solution.merge(xr.DataArray(summary).to_dataset(name='summary'))
        # Add groups
        groups = self.get_groups()
        groups.columns.name = 'cols_groups'
        groups.index.name = 'techs'
        self.solution = self.solution.merge(xr.DataArray(groups).to_dataset(name='groups'))
        # Add shares
        shares = self.get_shares(groups)
        shares.columns.name = 'cols_shares'
        shares.index.name = 'techs'
        self.solution = self.solution.merge(xr.DataArray(shares).to_dataset(name='shares'))
        # Add time resolution
        self.solution = self.solution.merge(self.data['_time_res'].copy(deep=True).to_dataset(name='time_res'))
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

    def get_ec(self, what='prod'):
        es = self.get_var('es_' + what)
        try:
            ec = self.get_var('ec_' + what)
        except exceptions.ModelError:  # ec has no data
            # Skip all the rest and return es straight away
            return es
        # For those techs that have an `ec`, replace their `es` with `ec`,
        # for the others, `ec` is `es`, so no change needed
        for carrier in ec.coords['c'].values:
            for tech in ec.coords['y'].values:
                es.loc[dict(c=carrier, y=tech)] = ec.loc[dict(c=carrier, y=tech)]
        return es  # the variable is called es, but the thing is now ec

    def get_ec_sum(self):
        ec = self.get_ec('prod') + self.get_ec('con')
        return ec.fillna(0)

    def get_node_variables(self):
        detail = ['s', 'rs']
        p = xr.Dataset({v: self.get_var(v) for v in detail})
        try:
            p['rbs'] = self.get_var('rbs')
        except exceptions.ModelError:
            # `rbs` doesn't exist in the model or exists without data
            p['rbs'] = p['rs'].copy()  # get same dimensions
            p['rbs'].loc[:] = 0
        p['e'] = self.get_ec_sum()
        return p

    def get_node_parameters(self):
        detail = ['s_cap', 'r_cap', 'r_area', 'e_cap', 'e_cap_net']
        result = xr.Dataset({v: self.get_var(v) for v in detail})
        try:
            result['rb_cap'] = self.get_var('rb_cap')
        except exceptions.ModelError:
            result['rb_cap'] = result['r_cap'].copy()  # get same dimensions
            result['rb_cap'].loc[:] = 0
        return result

    def get_costs(self, t_subset=None):
        """Get costs."""
        if t_subset is None:
            cost_fixed = self.get_var('cost_con') + self.get_var('cost_op_fixed')
            cost_variable = self.get_var('cost_op_variable')
        else:
            # len_adjust is the fraction of construction and fixed costs
            # that is accrued to the chosen t_subset. NB: construction and fixed
            # operation costs are calculated for a whole year
            len_adjust = (sum(self.data['_time_res'].to_series().iloc[t_subset])
                          / sum(self.data['_time_res'].to_series()))

            # Adjust for the fact that fixed costs accrue over a smaller length
            # of time as per len_adjust
            cost_fixed = self.get_var('cost_con') + self.get_var('cost_op_fixed')
            cost_fixed = cost_fixed * len_adjust

            # Adjust for the fact that variable costs are only accrued over
            # the t_subset period
            cost_op_var = self.get_var('cost_op_var')[{'t': t_subset}].sum(dim='t')
            cost_op_fuel = self.get_var('cost_op_fuel')[{'t': t_subset}].sum(dim='t')
            cost_op_rb = self.get_var('cost_op_rb')[{'t': t_subset}].sum(dim='t')

            cost_variable = cost_op_var + cost_op_fuel + cost_op_rb

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

        p = xr.Dataset({'ec_' + i: (self.get_ec(i)[dict(t=t_subset)]
                        * weights).sum(dim='t')
                        for i in ['prod', 'con']})
        for i in ['es_prod', 'es_con']:
            p[i] = (self.get_var(i)[dict(t=t_subset)] * weights).sum(dim='t')
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
                    lc = sol['costs'].loc[dict(k=cost)] / sol['ec_prod'].loc[dict(c=carrier)]
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
                cf = sol['ec_prod'].loc[dict(c=carrier)] / (sol['e_cap_net'] * time_res_sum)
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
        df.loc[:, 'carrier'] = df.index.map(lambda y: self.get_carrier(y))
        get_src_c = lambda y: self.get_source_carrier(y)
        df.loc[:, 'source_carrier'] = df.index.map(get_src_c)
        df.loc[:, 'stack_weight'] = df.index.map(lambda y: self.get_weight(y))
        df.loc[:, 'color'] = df.index.map(lambda y: self.get_color(y))
        return df

    def get_summary(self, sort_by='e_cap', carrier='power'):
        sol = self.solution

        # Total (over locations) capacity factors per carrier
        time_res_sum = self._get_time_res_sum()
        with np.errstate(divide='ignore', invalid='ignore'):  # don't warn about division by zero
            cf = (sol['ec_prod'].loc[dict(c=carrier)].sum(dim='x')
                  / (sol['e_cap_net'].sum(dim='x') * time_res_sum)).to_pandas()
        df = pd.DataFrame({'cf': cf})

        # Total (over locations) levelized costs per carrier
        for k in sorted(sol['levelized_cost'].coords['k'].values):
            with np.errstate(divide='ignore', invalid='ignore'):  # don't warn about division by zero
                df['levelized_cost_' + k] = (sol['costs'].loc[dict(k=k)].sum(dim='x')
                                   / sol['ec_prod'].loc[dict(c=carrier)].sum(dim='x'))

        # Add totals per carrier
        df['e_prod'] = sol['ec_prod'].loc[dict(c=carrier)].sum(dim='x')
        df['e_con'] = sol['ec_con'].loc[dict(c=carrier)].sum(dim='x')

        # Add other carrier-independent stuff
        df['e_cap'] = sol['e_cap'].sum(dim='x')
        df['r_area'] = sol['r_area'].sum(dim='x')
        df['s_cap'] = sol['s_cap'].sum(dim='x')

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
            costs = self.get_costs(t_subset=slice(0, stepsize)).to_dataset(name='costs')
            cost_vars.append(costs)

            timesteps = [time_res.at[t] for t in self.m.t][0:stepsize]
            d.attrs['time_res_sum'] += sum(timesteps)

            # Save state of storage for carry over to next iteration
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
        for param in _TIMESERIES_PARAMS:
            subset_name = 'y_def_' + param
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
        for param in _TIMESERIES_PARAMS:
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
