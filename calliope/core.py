"""
Copyright (C) 2013 Stefan Pfenninger.
Licensed under the Apache 2.0 License (see LICENSE file).

core.py
~~~~~~~

Core model functionality via the Model class.

"""

from __future__ import print_function
from __future__ import division

import datetime
import inspect
import itertools
import json
import os
import random
import shutil
import time

import coopr.opt as co
import coopr.pyomo as cp
import numpy as np
import pandas as pd
from pyutilib.services import TempfileManager

from . import analysis
from . import constraints
from . import locations
from . import transmission
from . import time_tools
from . import utils


def _expand_module_placeholder(s):
    return utils.replace(s, placeholder='module',
                         replacement=os.path.dirname(__file__))


class Model(object):
    """
    Calliope: a multi-scale energy systems (MUSES) modeling framework

    """
    def __init__(self, config_run=None, override=None):
        """
        Args:
            config_run : path to YAML file with run settings OR and AttrDict
                         containing settings. If not given, the path
                         ``{{ module }}/config/run.yaml`` is used as the
                         default.
            override : provide any additional options or override options
                       from ``config_run`` by passing an AttrDict
                       of the form ``{'model_settings': 'foo.yaml'}``.
                       Any option possible in ``run.yaml`` can be specified
                       in the dict, inluding ``override.`` options.

        """
        super(Model, self).__init__()
        self.debug = utils.AttrDict()
        self.initialize_configuration(config_run, override)
        # Other initialization tasks
        self.data = utils.AttrDict()
        self.initialize_parents()
        self.initialize_sets()
        self.read_data()
        self.mode = self.config_run.mode
        self.initialize_availability()
        self.initialize_time()

    def initialize_configuration(self, config_run, override):
        self.flush_option_cache()
        # Load run configuration
        config_path = os.path.join(os.path.dirname(__file__), 'config')
        if not config_run:
            config_run = os.path.join(config_path, 'run.yaml')
        self.config_run_path = config_run
        if isinstance(config_run, str):
            # 1) config_run is a string, assume it's a path
            cr = utils.AttrDict.from_yaml(config_run)
            # self.run_id is used to set an output folder for logs, if
            # debug.keepfiles is set to True
            self.run_id = os.path.splitext(os.path.basename(config_run))[0]
        else:
            # 2) config_run is not a string, assume it's an AttrDict
            cr = config_run
            assert isinstance(cr, utils.AttrDict)
            # we have no filename so we just use current date/time
            self.run_id = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.config_run = cr
        if override:
            assert isinstance(override, utils.AttrDict)
            for k in override.keys_nested():
                cr.set_key(k, override.get_key(k))
        # If manually specify a run_id in debug, overwrite the generated one
        if 'debug.run_id' in cr.keys_nested():
            self.run_id = cr.debug.run_id
        # Ensure 'input.model' is a list
        if not isinstance(cr.input.model, list):
            cr.input.model = [cr.input.model]
        # Expand {{ module }} placeholder
        cr.input.model = [_expand_module_placeholder(i)
                          for i in cr.input.model]
        cr.input.data_path = _expand_module_placeholder(cr.input.data_path)
        # Interpret relative config paths as relative to run.yaml
        cr.input.model = [utils.ensure_absolute(i, self.config_run_path)
                          for i in cr.input.model]
        cr.input.data_path = utils.ensure_absolute(cr.input.data_path,
                                              self.config_run_path)
        # Load all model config files and combine them into one AttrDict
        o = utils.AttrDict.from_yaml(os.path.join(config_path,
                                                  'defaults.yaml'))
        for path in cr.input.model:
            # The input files are allowed to override defaults
            o.union(utils.AttrDict.from_yaml(path), allow_override=True)
        # Override config_model settings if specified in config_run
        if ('override' in cr
                and isinstance(cr.override, utils.AttrDict)):
            for k in cr.override.keys_nested():
                o.set_key(k, cr.override.get_key(k))
        # Initialize locations
        o.locations = locations.process_locations(o.locations)
        # Store initialized configuration on model object
        self.config_model = o
        self.flush_option_cache()

    def get_timeres(self, verify=False):
        """Returns resolution of data in hours. Needs a properly
        formatted ``set_t.csv`` file to work.

        If ``verify=True``, verifies that the entire file is at the same
        resolution. ``model.get_timeres(verify=True)`` can be called
        after Model initialization to verify this.

        """
        path = self.config_run.input.data_path
        df = pd.read_csv(os.path.join(path, 'set_t.csv'), index_col=0,
                         header=None, parse_dates=[1])
        seconds = (df.iat[0, 0] - df.iat[1, 0]).total_seconds()
        if verify:
            for i in range(len(df) - 1):
                assert ((df.iat[i, 0] - df.iat[i+1, 0]).total_seconds()
                        == seconds)
        hours = abs(seconds) / 3600
        return hours

    def initialize_availability(self):
        # Availability in time/space is only used if dynamic timestep
        # adjustment is invoked
        d = self.data
        d['a'] = utils.AttrDict()
        # Append 'a' to d.params, so that it is included in parameter updates!
        d.params.append('a')
        # Fill in default values for a, so that something is there even in
        # case no dynamic timestepper is called
        for y in d._y_def_r:
            avail = self.get_option(y + '.constraints.availability')
            d.a[y] = pd.DataFrame(avail, index=d._t, columns=d._x)

    def initialize_time(self):
        """
        Performs time resolution reduction, if set up in configuration
        """
        cr = self.config_run
        t = cr.get_key('time.summarize', default=False)
        s = time_tools.TimeSummarizer()
        if t == 'mask':
            if cr.get_key('time.mask_function', default=False):
                eval('mask_src = time_masks.'
                     + cr.time_tools.mask_function + '(m.data)')
                mask = time.masks_to_resolution_series([mask_src])
            elif cr.get_key('time.mask_file', default=False):
                mask = pd.read_csv(utils.ensure_absolute(cr.time.mask_file,
                                                         self.config_run_path),
                                   index_col=0, header=None)[1]
                mask = mask.astype(int)
            else:
                raise KeyError('Define either mask_function or mask_file.')
            s.dynamic_timestepper(self.data, mask)
        elif t == 'uniform':
            s.reduce_resolution(self.data, cr.time.resolution)

    def prev(self, t):
        """Using the timesteps set of this model instance, return `t-1`,
        even if the set is not continuous.

        E.g., if t is [0, 1, 2, 6, 7, 8], model.prev(6) will return 2.

        """
        # Create an index to look up t, and save it for later use
        try:
            # Check if _t_index exists
            self._t_index.name
        except AttributeError:
            self._t_index = pd.Index(self.data._t)
        # Get the location of t in the index and use it to retrieve
        # the desired value, raising an error if it's <0
        loc = self._t_index.get_loc(t) - 1
        if loc >= 0:
            return self.data._t.iat[loc]
        else:
            raise KeyError('<0')

    def get_option(self, option, x=None, default=None):
        key = (option, x, default)
        try:
            result = self.option_cache[key]
        except KeyError:
            result = self.option_cache[key] = self._get_option(*key)
        return result

    def _get_option(self, option, x=None, default=None):
        """Retrieves options from model settings for the given tech,
        falling back to the default if the option is not defined for the
        tech.

        If ``x`` is given, will attempt to use location-specific override
        from the location matrix first before falling back to model-wide
        settings.

        If ``default`` is given, it is used as a fallback if no default value
        can be found in the regular inheritance chain. If ``default`` is None
        and the regular inheritance chain defines no default, an error
        is raised.

        If the first segment of the option containts ':', it will be
        interpreted as implicit tech subsetting: e.g. asking for
        'hvac:r1' implicitly uses 'hvac:r1' with the parent 'hvac', even
        if that has not been defined, to search the option inheritance
        chain.

        Examples: model.get_option('ccgt.costs.om_var') or
                  model.get_option('csp.weight') or
                  model.get_option('csp.r', x='33') or

        """
        o = self.config_model
        d = self.data

        def _get_option(option):
            try:
                result = o.get_key('techs.' + option)
            except KeyError:
                # Example: 'ccgt.costs.om_var'
                tech = option.split('.')[0]  # 'ccgt'
                remainder = '.'.join(option.split('.')[1:])  # 'costs.om_var'
                if ':' in tech:
                    parent = tech.split(':')[0]
                else:
                    parent = o.get_key('techs.' + tech + '.parent')  # 'defaults'
                try:
                    result = _get_option(parent + '.' + remainder)
                except KeyError:
                    if default:
                        result = _get_option(default)
                    elif tech == 'default':
                        raise KeyError('Reached top of inheritance chain '
                                       'and no default defined.')
                    else:
                        raise KeyError('Can not get parent for {} '
                                       'and no default defined.'.format(tech))
            return result

        def _get_location_option(key, location):
            # Raises KeyError if the specific _override column does not exist
            result = d.locations.ix[location, '_override.' + key]
            # Also raise KeyError if the result is NaN, i.e. if no
            # location-specific override has been defined
            try:
                if np.isnan(result):
                    raise KeyError
            # Have to catch this because np.isnan not implemented for strings
            except TypeError:
                pass
            return result

        if x:
            try:
                result = _get_location_option(option, x)
            # If can't find a location-specific option, fall back to model-wide
            except KeyError:
                result = _get_option(option)
        else:
            result = _get_option(option)
        # Deal with 'inf' settings
        if result == 'inf':
            result = float('inf')
        return result

    def set_option(self, option, value):
        """Set ``option`` to ``value``. Returns None on success."""
        # TODO add support for setting option at a specific x
        # TODO add support for changing defaults?
        o = self.config_model
        o.set_key('techs.' + option, value)
        self.flush_option_cache()

    def flush_option_cache(self):
        self.option_cache = {}

    def get_name(self, y):
        try:
            return self.get_option(y + '.name')
        except KeyError:
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
        if y in self.data._y_transmission:
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
        Return the member technologies of a group. If ``in_model`` is True,
        only members defined in the current model are returned.

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
                                  groups.

            ``expand_transmission`` : if True, return in-model
                                      transmission technologies in the
                                      form ``tech:location``.

        """
        def _get(self, group, memberset):
            members = [i for i in self.parents if self.parents[i] == group]
            if members:
                for i, member in enumerate(members):
                    if not head_nodes_only:
                        # FIXME this doesn't actually work, need to fix
                        # on the other hand, is it needed for anything?
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
                             if (y in self.data._y
                                 or y in self.data.transmission_y)])
            # Expand transmission techs
            if expand_transmission:
                for y in list(memberset):
                    if y in self.data.transmission_y:
                        memberset.remove(y)
                        memberset.update([yt
                                          for yt in self.data._y_transmission
                                          if yt.startswith(y + ':')])
        return list(memberset)

    @utils.memoize_instancemethod
    def get_eff_ref(self, var, y, x=None):
        """Get reference efficiency, falling back to efficiency if no
        reference efficiency has been set."""
        base = y + '.constraints.' + var
        try:
            eff_ref = self.get_option(base + '_eff_ref', x=x)
        except KeyError:
            eff_ref = False
        if eff_ref is False:
            eff_ref = self.get_option(base + '_eff', x=x)
        # NOTE: Will cause errors in the case where (1) eff_ref is not defined
        # and (2) eff is set to "file". That is ok however because in this edge
        # case eff_ref should be manually set as there is no straightforward
        # way to derive it from the time series file.
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
        if peak < float(df.min()):
            scale = float(df.min())
        else:
            scale = float(df.max())
        return (df / scale) * peak * adjustment

    def initialize_parents(self):
        o = self.config_model
        try:
            self.parents = {i: o.techs[i].parent for i in o.techs.keys()
                            if i != 'defaults'}
        except KeyError:
            tech = inspect.trace()[-1][0].f_locals['i']
            print(o.techs[tech].keys())
            if 'parent' in o.techs[tech].keys():
                raise KeyError('Technology `' + tech + '` defines no parent!')
        # Verify that all parents are themselves actually defined
        for k, v in self.parents.iteritems():
            if v not in o.techs.keys():
                raise KeyError('Parent `' + v + '` of technology `' +
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

    def initialize_sets(self):
        o = self.config_model
        d = self.data
        path = self.config_run.input.data_path
        #
        # t: Timesteps set
        #
        table_t = pd.read_csv(os.path.join(path, 'set_t.csv'), header=None)
        table_t.index = [datetime.datetime.strptime(dt, '%Y-%m-%d %H:%M:%S')
                         for dt in table_t[1]]
        if self.config_run.get_key('subset_t', default=False):
            table_t = table_t.loc[self.config_run.subset_t[0]:
                                  self.config_run.subset_t[1]]
            self.slice = slice(table_t[0][0], table_t[0][-1] + 1)
        else:
            self.slice = slice(None)
        d._t = pd.Series([int(t) for t in table_t[0].tolist()])
        d._dt = pd.Series(table_t.index, index=d._t.tolist())
        # First set time_res_data and time_res_static across all data
        d.time_res_data = self.get_timeres()
        d.time_res_static = d.time_res_data
        # From time_res_data, initialize time_res_series
        d.time_res_series = pd.Series(d.time_res_data, index=d._t.tolist())
        # Last index t for which model may still use startup exceptions
        d.startup_time_bounds = d._t[int(o.startup_time / d.time_res_data)]
        #
        # y: Technologies set
        #
        d._y = set()
        try:
            for k, v in o.locations.iteritems():
                for y in v.techs:
                    d._y.add(y)
        except KeyError:
            raise UserWarning('The region `' + k + '` does not allow '
                              'any technologies via `techs`. Must give '
                              'at least one technology per region.')
        d._y = list(d._y)
        if self.config_run.get_key('subset_y', default=False):
            d._y = [y for y in d._y if y in self.config_run.subset_y]
        # Subset of transmission technologies, if any defined (used below)
        # (not yet added to d._y here)
        if ('links' in o) and (o.links is not None):
            d._y_transmission = transmission.get_transmission_techs(o.links)
            d.transmission_y = list(set([v.keys()[0]
                                    for k, v in o.links.iteritems()]))
        else:
            d._y_transmission = []
            d.transmission_y = []
        # Subset of conversion technologies
        d._y_conversion = [y for y in d._y if self.ischild(y, of='conversion')]
        # Subset of supply, demand, storage technologies
        d._y_pc = [y for y in d._y
                   if not self.ischild(y, of='conversion')
                   or self.ischild(y, of='transmission')]
        # Subset of technologies that define es_prod/es_con
        d._y_prod = ([y for y in d._y if not self.ischild(y, of='demand')]
                     + d._y_transmission)
        d._y_con = ([y for y in d._y if not self.ischild(y, of='supply')]
                    + d._y_transmission)
        #
        # c: Carriers set
        #
        d._c = set()
        for y in d._y:  # Only add carriers for allowed technologies
            d._c.update([self.get_option(y + '.carrier')])
            if self.get_option(y + '.source_carrier'):
                d._c.update([self.get_option(y + '.source_carrier')])
        d._c = list(d._c)
        #
        # x: Locations set
        #
        d._x = o.locations.keys()
        if self.config_run.get_key('subset_x', default=False):
            d._x = [x for x in d._x if x in self.config_run.subset_x]
        #
        # Locations settings matrix and transmission technologies
        #
        d.locations = locations.generate_location_matrix(o.locations,
                                                         techs=d._y)
        # For simplicity, only keep the locations that are actually in set `x`
        d.locations = d.locations.ix[d._x, :]
        # Add transmission technologies to y, if any defined
        if d._y_transmission:
            d._y.extend(d._y_transmission)
            # Add transmission tech columns to locations matrix
            for y in d._y_transmission:
                d.locations[y] = 0
            # Create representation of location-tech links
            tree = transmission.explode_transmission_tree(o.links, d._x)
            # Populate locations matrix with allowed techs and overrides
            if tree:
                for x in tree:
                    for y in tree[x]:
                        # Allow the tech
                        d.locations.at[x, y] = 1
                        # Add constraints if needed
                        for c in tree[x][y].keys_nested():
                            colname = '_override.' + y + '.' + c
                            if not colname in d.locations.columns:
                                d.locations[colname] = np.nan
                            d.locations.at[x, colname] = tree[x][y].get_key(c)
        #
        # k: Cost classes set
        #
        classes = [o.techs[k].costs.keys() for k in o.techs
                   if k != 'defaults'  # Prevent 'default' from entering set
                   if 'costs' in o.techs[k]]
        # Flatten list and make sure 'monetary' is in it
        classes = ([i for i in itertools.chain.from_iterable(classes)]
                   + ['monetary'])
        d._k = list(set(classes))  # Remove any duplicates with a set roundtrip

    def read_data(self):
        """
        Read parameter data from CSV files, if needed. Data that may be
        defined in CSV files is read before generate_model() so that it
        only has to be read from disk once, even if generate_model() is
        repeatedly called.

        Note on indexing: if subset_t is set, a subset of the data is
        selected, and all data including their indices are subsetted.
        d._t maps a simple [0, data length] index to the actual t index
        used.

        Data is stored in the `self.data`  for each
        `param` and technology `y`: ``self.data[param][y]``

        """
        @utils.memoize
        def _get_option_from_csv(filename):
            """Read CSV time series"""
            d_path = os.path.join(self.config_run.input.data_path, filename)
            # [self.slice] to do time subset if needed
            # Results in e.g. d.r_eff['csp'] being a dataframe
            # of efficiencies for each time step t at location x
            df = pd.read_csv(d_path, index_col=0)[self.slice]
            # Fill columns that weren't defined with NaN
            # missing_cols = list(set(self.data._x) - set(df.columns))
            # for c in missing_cols:
            #     df[c] = np.nan
            return df

        d = self.data
        self.debug.data_sources = utils.AttrDict()
        # Data for storage initialization parameter
        d.s_init = pd.DataFrame(index=d._x, columns=d._y_pc)
        for y in d.s_init.columns:
            for x in d.s_init.index:
                d.s_init.at[x, y] = self.get_option(y + '.constraints.s_init',
                                                    x=x)
        # Parameters that may be defined over (x, t) for a given technology y
        d.params = ['r', 'e_eff']
        d._y_def_r = set()
        d._y_def_e_eff = set()
        # TODO allow params in d.params to be defined only over
        # x instead of either static or over (x, t) via CSV!
        for param in d.params:
            d[param] = utils.AttrDict()
            for y in d._y:
                d[param][y] = pd.DataFrame(np.nan, index=d._t,
                                           columns=d._x)
                # TODO this whole process could be refactored for efficiency
                # to read files only once,
                # create a dict of files: {'f1.csv': ['x1', 'x2'],
                #                          'f2.csv': ['x3'],
                #                          'model_config': ['x4, x5']}
                for x in d._x:
                    option = self.get_option(y + '.constraints.' + param, x=x)
                    k = param + '.' + y + '.' + x
                    if isinstance(option, str) and option.startswith('file'):
                        if param == 'r':
                            d._y_def_r.add(y)
                        elif param == 'e_eff':
                            d._y_def_e_eff.add(y)
                        try:
                            # Parse 'file=filename' option
                            f = option.split('=')[1]
                        except IndexError:
                            # If set to just 'file', set filename with y and
                            # param, e.g. 'csp_r_eff.csv'
                            f = y + '_' + param + '.csv'
                        df = _get_option_from_csv(f)
                        # Set x_map if that option has been set
                        try:
                            x_map = self.get_option(y + '.x_map', x=x)
                        except KeyError:
                            x_map = None
                        # Now, if x_map is available, remap cols accordingly
                        if x_map:
                            # Unpack dict from string
                            x_map = {x.split(':')[0].strip():
                                     x.split(':')[1].strip()
                                     for x in x_map.split(',')}
                            df = df[x_map.keys()]  # Keep only keys in x_map
                            # Then remap column names
                            df.columns = [x_map[c] for c in df.columns]
                        try:
                            d[param][y].loc[:, x] = df[x]
                            self.debug.data_sources.set_key(k, 'file:' + f)
                        except KeyError:
                            # If could not be read from file, set it to zero
                            d[param][y].loc[:, x] = 0
                            # Depending on whether or not the tech is allowed
                            # at this location, set _FILE_NOT_FOUND_ or _NA_
                            # for the data source
                            if self.data.locations.at[x, y] == 0:
                                self.debug.data_sources.set_key(k, '_NA_')
                            else:
                                v = 'file:_NOT_FOUND_'
                                self.debug.data_sources.set_key(k, v)
                    else:
                        d[param][y].loc[:, x] = option
                        self.debug.data_sources.set_key(k, 'model_config')
                        if (param == 'r' and option != float('inf')):
                            d._y_def_r.add(y)
                    # Convert power to energy for r, if necessary
                    if param == 'r':
                        r_unit = self.get_option(y + '.constraints.r_unit')
                        if r_unit == 'power':
                            r_scale = d.time_res_data
                            d[param][y].loc[:, x] = (d[param][y].loc[:, x]
                                                     * r_scale)
                    # Scale r to a given maximum if necessary
                    scale = self.get_option(y + '.constraints.r_scale_to_peak',
                                            x=x)
                    if param == 'r' and scale:
                        scaled = self.scale_to_peak(d[param][y][x], scale)
                        d[param][y].loc[:, x] = scaled
        ds = self.debug.data_sources
        missing_data = [k for k in ds.keys_nested()
                        if ds.get_key(k) == 'file:_NOT_FOUND_']
        if len(missing_data) > 0:
            message = ('The following parameter values could not be read '
                       'from file. They were automatically set to `0`: '
                       + ', '.join(missing_data))
            raise UserWarning(message)

    def _get_t_max_demand(self):
        t_max_demands = utils.AttrDict()
        for c in self.data._c:
            ys = [y for y in self.data._y
                  if self.get_option(y + '.carrier') == c]
            r_carrier = pd.Panel(self.data.r).loc[ys].sum(axis=2)
            t_max_demand = r_carrier[r_carrier < 0].sum(axis=1).idxmin()
            # Adjust for reduced resolution, only if t_max_demand not 0 anyway
            if t_max_demand != 0:
                t_max_demand = max([t for t in self.data._t
                                    if t < t_max_demand])
            t_max_demands[c] = t_max_demand
        return t_max_demands

    def add_constraint(self, constraint, *args, **kwargs):
        constraint(self, *args, **kwargs)

    def _param_populator(self, src, t_start=None):
        """Returns a `getter` function that returns either
        (x, t)-specific values for parameters that define such, or
        always the same static value if only a static value is given.

        """
        def getter(m, y, x, t):
            if isinstance(src[y], pd.core.frame.DataFrame):
                if t_start:
                    return float(src[y].loc[t_start + t, x])
                else:
                    return float(src[y].loc[t, x])
            else:
                return float(src[y])
        return getter

    def update_parameters(self):
        mi = self.instance
        d = self.data
        t_start = self.t_start

        for param in d.params:
            initializer = self._param_populator(d[param], t_start)
            y_set = self.param_sets[param]
            param_object = getattr(mi, param)
            for y in y_set:
                for x in mi.x:
                    for t in mi.t:
                        param_object[y, x, t] = initializer(mi, y, x, t)

        s_init_initializer = lambda m, y, x: float(d.s_init.at[x, y])
        for y in mi.y_pc:
            for x in mi.x:
                mi.s_init[y, x] = s_init_initializer(mi, y, x)

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
        m = cp.ConcreteModel()
        o = self.config_model
        d = self.data
        self.m = m
        self.t_start = t_start
        self.t_max_demand = self._get_t_max_demand()

        #
        # Sets
        #
        # Time steps
        if self.mode == 'plan':
            m.t = cp.Set(initialize=d._t, ordered=True)
        elif self.mode == 'operate':
            t_end = (t_start + o.opmode.horizon / d.time_res_data) - 1
            self.t_end = int(t_end)
            # If t_end is beyond last timestep, cap it to last one
            # TODO this is a hack and shouldn't be necessary?
            if t_end > d._t.iat[-1]:
                t_end = d._t.iat[-1]
            m.t = cp.Set(initialize=d._t.loc[self.t_start:self.t_end],
                         ordered=True)
        # The rest
        m.c = cp.Set(initialize=d._c, ordered=True)  # Carriers
        m.y = cp.Set(initialize=d._y, ordered=True)  # Technologies
        m.y_prod = cp.Set(initialize=d._y_prod, within=m.y,
                          ordered=True)  # Production technologies
        m.y_con = cp.Set(initialize=d._y_con, within=m.y,
                         ordered=True)  # Production technologies
        m.y_pc = cp.Set(initialize=d._y_pc, within=m.y,
                        ordered=True)  # Production/consumption technologies
        m.y_trans = cp.Set(initialize=d._y_transmission, within=m.y,
                           ordered=True)  # Transmission technologies
        m.y_conv = cp.Set(initialize=d._y_conversion, within=m.y,
                          ordered=True)  # Conversion technologies
        m.y_def_r = cp.Set(initialize=d._y_def_r, within=m.y)
        m.y_def_e_eff = cp.Set(initialize=d._y_def_e_eff, within=m.y)
        m.x = cp.Set(initialize=d._x, ordered=True)  # Nodes
        m.k = cp.Set(initialize=d._k, ordered=True)  # Cost classes

        #
        # Parameters
        #

        self.param_sets = {'r': m.y_def_r,
                           'a': m.y_def_r,
                           'e_eff': m.y_def_e_eff}
        for param in d.params:
            initializer = self._param_populator(d[param], t_start)
            y = self.param_sets[param]
            setattr(m, param, cp.Param(y, m.x, m.t, initialize=initializer,
                                       mutable=True))

        s_init_initializer = lambda m, y, x: float(d.s_init.at[x, y])
        m.s_init = cp.Param(m.y_pc, m.x, initialize=s_init_initializer,
                            mutable=True)

        #
        # Variables and constraints
        #
        # 1. Required
        required = [constraints.base.node_resource,
                    constraints.base.node_energy_balance,
                    constraints.base.node_constraints_build,
                    constraints.base.node_constraints_operational,
                    constraints.base.node_costs,
                    constraints.base.transmission_constraints,
                    constraints.base.model_constraints]
        for c in required:
            self.add_constraint(c)

        if self.mode == 'plan':
            self.add_constraint(constraints.planning.system_margin)

        # 2. Optional
        if o.get_key('constraints_pre_load', default=False):
            eval(o.constraints_pre_load)  # TODO potentially disastrous!
        if o.get_key('constraints', default=False):
            for c in o.constraints:
                self.add_constraint(eval(c))  # TODO this works but is unsafe!

        # 3. Objective function
        self.add_constraint(constraints.base.model_objective)

    def run(self):
        """Instantiate and solve the model"""
        start_time = time.time()
        if self.mode == 'plan':
            self.generate_model()  # Generated model goes to self.m
            self.solve()
        elif self.mode == 'operate':
            self.solve_iterative()
        else:
            raise UserWarning('Invalid mode')
        self.runtime = int(time.time() - start_time)
        if self.config_run.output.save is True:
            self.save_solution()

    def solve(self, warmstart=False, save_json=False):
        """
        Args:
            warmstart : (default False) re-solve an updated model
                        instance
            save_json : (default False) Save optimization results to
                        the given file as JSON.

        Returns: None

        """
        m = self.m
        cr = self.config_run
        if not warmstart:
            self.instance = m.create()
            solver_io = cr.get_key('solver_io', default=False)
            if solver_io:
                self.opt = co.SolverFactory(cr.solver, solver_io=solver_io)
            else:
                self.opt = co.SolverFactory(cr.solver)
            # Set solver options from run_settings file, if it exists
            try:
                for k in cr.solver_options.keys_nested():
                    self.opt.options[k] = cr.solver_options.get_key(k)
            except KeyError:
                pass
            if cr.get_key('debug.symbolic_solver_labels', default=False):
                self.opt.symbolic_solver_labels = True
        if cr.get_key('debug.keepfiles', default=False):
            self.opt.keepfiles = True
            if self.mode == 'plan':
                logdir = os.path.join('Logs', self.run_id)
            elif self.mode == 'operate':
                logdir = os.path.join('Logs', self.run_id
                                      + '_' + str(self.t_start))
            if (cr.get_key('debug.delete_old_logs', default=False)and os.path.exists(logdir)):
                shutil.rmtree(logdir)
            os.makedirs(logdir)
            TempfileManager.tempdir = logdir
        # Always preprocess instance, for both cold start and warm start
        self.instance.preprocess()
        # Silencing output by redirecting stdout and stderr
        with utils.capture_output() as out:
            if warmstart:
                results = self.opt.solve(self.instance, warmstart=True,
                                         tee=True)
            else:
                results = self.opt.solve(self.instance, tee=True)
        if save_json:
            with open(save_json, 'w') as f:
                json.dump(results.json_repn(), f, indent=4)
        self.results = results
        self.pyomo_output = out
        self.load_results()
        self.load_solution()

    def process_solution(self):
        # Add levelized cost
        self.solution['levelized_cost'] = self.get_levelized_cost()
        # Add capacity factor
        self.calculate_capacity_factor()
        # Add metadata
        self.solution['metadata'] = self.get_metadata()
        # Add summary
        self.solution['summary'] = self.get_summary()
        # Add shares
        self.solution['shares'] = self.get_shares()
        # Add time resolution, and give it a nicer index
        time_res = self.data.time_res_series
        time_res.index = self.solution.node.major_axis
        self.solution['time_res'] = time_res

    def load_solution(self):
        sol = {'node': self.get_node_variables(),
               'totals': self.get_totals(),
               'costs': self.get_costs(),
               'parameters': self.get_node_parameters()}
        self.solution = utils.AttrDict(sol)
        self.process_solution()

    def get_var(self, var, dims=None):
        """Return output for variable `var` as a series, dataframe or panel

        Args:
            var : variable name as string, e.g. 'es_prod'
            dims : list of indices as strings, e.g. ('y', 'x', 't');
                   if not given, they are auto-detected
        """
        # TODO currently assumes fixed position of 't' (always last)
        m = self.m
        var = getattr(m, var)
        # Get set
        s = var._pprint()[0][1][1].set_tuple
        # Get dims
        dims = [i.name for i in s]
        result = pd.DataFrame.from_dict(var.extract_values(), orient='index')
        result.index = pd.MultiIndex.from_tuples(result.index, names=dims)
        result = result[0]
        # Unstack and sort by time axis
        if len(dims) == 1:
            result = result.sort_index()
        else:
            # if len(dims) is 2, we already have a well-formed panel
            result = result.unstack(level=0)
            if len(dims) == 2:
                result = result.sort_index()
            if len(dims) == 3:
                result = result.to_panel()
                result = result.transpose(0, 2, 1)  # Flip time into major_axis
                result = result.sort_index(1)
            if len(dims) == 4:
                # Panel4D needs manual treatment until pandas supports to_panel
                # for anything else but the regular 3d panel
                p = {}
                for i in result.columns:
                    p[i] = result[i].unstack(0).to_panel().transpose(0, 2, 1)
                result = pd.Panel4D(p)
                result = result.sort_index(1)
        # Nicify time axis
        if 't' in dims:
            t = getattr(m, 't')
            if self.t_start == 0 or self.t_start == None:
                new_index = self.data._dt.loc[t.first():t.last()].tolist()
            else:
                new_index = self.data._dt.loc[self.t_start:self.t_end].tolist()
            if len(dims) <= 2:
                result.index = new_index
            else:
                result.major_axis = new_index
        return result

    def get_e(self):
        es_prod = self.get_var('es_prod')
        es_con = self.get_var('es_con')
        e = {}
        for i in es_prod.labels:
            e[i] = (es_prod[i].to_frame().add(es_con[i].to_frame(),
                                              fill_value=0)).to_panel()
        e = pd.Panel4D(e)
        return e

    # def get_system_variables(self):
    #     e = self.get_e()
    #     e = e.sum(axis=3).transpose(0, 2, 1)
    #     return e

    def get_node_variables(self):
        try:
            detail = ['s', 'rs', 'rs_']
            p = pd.Panel4D({v: self.get_var(v) for v in detail})
        except AttributeError:
            detail = ['s', 'rs']
            p = pd.Panel4D({v: self.get_var(v) for v in detail})
        # Add 'e:c' items for each c in carrier
        temp = self.get_e()
        for c in self.data._c:
            p['e' + ':' + c] = temp.loc[c, :, :, :]
        return p

    def get_node_parameters(self, built_only=False):
        """If built_only is True, disregard locations where e_cap==0"""
        detail = ['s_cap', 'r_cap', 'r_area', 'e_cap']
        result = pd.Panel({v: self.get_var(v) for v in detail})
        if built_only:
            result = result.to_frame()
            result = result[result.e_cap > 0].dropna(axis=1)
        return result

    def get_costs(self, t_subset=None):
        """Get costs."""
        # FIXME need to implement t_subset here, e.g. by calculating
        # costs ex-post based on all the relevant tech parameters
        cost = self.get_var('cost').transpose('major_axis', 'minor_axis',
                                              'items')
        cost.loc[:, 'total', :] = cost.sum(1)
        return cost

    def get_totals(self, t_subset=None):
        """Get total produced and consumed per technology and location."""
        if t_subset is None:
            t_subset = slice(None)
        p = pd.Panel4D({e: self.get_var(e)
                        .iloc[:, :, t_subset, :]
                        .sum(axis='major')
                        for e in ['es_prod', 'es_con']})
        p = p.transpose('items', 'labels', 'minor_axis', 'major_axis')
        for l in p.labels:
            p.loc[l, :, 'total', :] = p[l].sum(1)
        return p

    def get_levelized_cost(self):
        """
        Get levelized costs.

        NB: Only production, not consumption, is used in calculations.

        """
        # TODO a better way to include consumption as well
        sol = self.solution
        p4d = {}
        for cost in self.data._k:
            p = {}
            for carrier in self.data._c:
                # Levelized cost of electricity (LCOE)
                lc = sol.costs[cost] / sol.totals[carrier].es_prod
                lc[np.isinf(lc)] = 0
                #lcoe.loc['total', :] = sol.costs.cost.sum(0) / sol.costs.e_prod.sum(0)
                p[carrier] = lc
            p4d[cost] = pd.Panel(p)
        return pd.Panel4D(p4d)

    def calculate_capacity_factor(self):
        """
        Calculates capacity factor and adds it to ``solultion.totals``

        NB: Only production, not consumption, is used in calculations.

        """
        m = self.m
        sol = self.solution
        time_res = self.data.time_res_series
        e_cap = sol.parameters['e_cap']
        for carrier in sol.totals.labels:
            try:  # Try loading time_res_sum from operational mode
                time_res_sum = self.data.time_res_sum
            except KeyError:
                time_res_sum = sum(time_res.at[t] for t in m.t)
            cf = sol.totals[carrier].es_prod / (e_cap * time_res_sum)
            cf.loc['total', :] = (sol.totals[carrier].es_prod.loc['total', :]
                                  / (e_cap.sum(0) * time_res_sum))
            cf = cf.fillna(0)
            sol.totals.loc[carrier, 'cf', :, :] = cf.T  # .T needed, but why?

    def get_metadata(self):
        df = pd.DataFrame(index=self.data._y)
        df.loc[:, 'type'] = df.index.map(lambda y: self.get_parent(y))
        df.loc[:, 'name'] = df.index.map(lambda y: self.get_name(y))
        df.loc[:, 'carrier'] = df.index.map(lambda y: self.get_carrier(y))
        get_src_c = lambda y: self.get_source_carrier(y)
        df.loc[:, 'source_carrier'] = df.index.map(get_src_c)
        df.loc[:, 'weight'] = df.index.map(lambda y: self.get_weight(y))
        df.loc[:, 'color'] = df.index.map(lambda y: self.get_color(y))
        return df

    def get_summary(self, sort_by='capacity', carrier='power'):
        sol = self.solution
        # Capacity factor summed across carriers, this only works
        # if a tech doesn't produce more than one carrier
        df = pd.DataFrame({'cf': sol.totals.loc[carrier, 'cf', 'total', :]})
        # Add different costs, summing up over all carriers
        for k in sorted(sol.levelized_cost.labels):
            # .loc[cost_class, carrier, location, tech]
            df['cost_' + k] = sol.levelized_cost.loc[k, carrier, 'total', :]
        # Add totals per carrier
        df['production'] = sol.totals.loc[carrier, 'es_prod', 'total', :]
        df['consumption'] = sol.totals.loc[carrier, 'es_con', 'total', :]
        # Add other carrier-independent stuff
        df['capacity'] = sol.parameters['e_cap'].sum()
        df['area'] = sol.parameters['r_area'].sum()
        return df.sort(columns=sort_by, ascending=False)

    def get_shares(self):
        ggm = self.get_group_members
        s = pd.Series({k: '|'.join(ggm(k, head_nodes_only=True))
                      for k in self.config_model.techs
                      if ggm(k, head_nodes_only=True) != []
                      and ggm(k, head_nodes_only=True) is not None})

        df = pd.DataFrame(s, columns=['members'])
        df['group'] = df.index.map(lambda y: self.get_option(y + '.group'))
        df['type'] = df.index.map(self.get_parent)

        for var in ['production', 'consumption', 'capacity']:
            for index, row in df.iterrows():
                group_members = row['members'].split('|')
                group_type = row['type']
                share = analysis.get_group_share(self.solution, group_members,
                                                 group_type, var=var)
                df.at[index, var] = share
        return df

    def load_solution_iterative(self, node_vars, total_vars, cost_vars):
        def _totaler(var):
            return sum([var[i].fillna(0).to_frame()
                       for i in range(len(var))]).to_panel()
        totals = _totaler(total_vars)
        costs = _totaler(cost_vars)
        sol = {'node': pd.concat(node_vars, axis=2),
               'totals': totals,
               'costs': costs,
               'parameters': self.get_node_parameters()}
        self.solution = utils.AttrDict(sol)
        self.process_solution()

    def solve_iterative(self):
        """
        Solve iterative by updating model parameters.

        Returns None on success, storing results under self.solution

        """
        o = self.config_model
        d = self.data
        time_res = d.time_res_series
        window_adj = int(o.opmode.window / d.time_res_data)
        steps = [t for t in d._t if (t % window_adj) == 0]
        # Remove the last step - since we look forward at each step,
        # it would take us beyond actually existing data
        steps = steps[:-1]
        node_vars = []
        total_vars = []
        cost_vars = []
        d.time_res_sum = 0
        self.generate_model(t_start=steps[0])
        for index, step in enumerate(steps):
            if index == 0:
                self.solve(warmstart=False)
            else:
                self.t_start = step
                t_end = (step + o.opmode.horizon / d.time_res_data) - 1
                self.t_end = int(t_end)
                self.update_parameters()
                self.solve(warmstart=True)
            self.load_results()
            # Gather relevant model results over decision interval, so
            # we only grab [0:window/time_res] steps!
            # panel = self.get_system_variables()
            # system_vars.append(panel.iloc[:, 0:target_index, :])
            if index == (len(steps) - 1):
                # Final iteration saves data from entire horizon
                target_index = int(o.opmode.horizon / d.time_res_static)
            else:
                # Non-final iterations only save data from window
                target_index = int(o.opmode.window / d.time_res_static)
            # Get node variables
            panel4d = self.get_node_variables()
            node_vars.append(panel4d.iloc[:, :, 0:target_index, :])
            # Get totals
            totals = self.get_totals(t_subset=slice(0, target_index))
            total_vars.append(totals)
            # Get costs
            cost = self.get_costs(t_subset=slice(0, target_index))
            cost_vars.append(cost)
            timesteps = [time_res.at[t] for t in self.m.t][0:target_index]
            d.time_res_sum += sum(timesteps)
            # Save state of storage for carry over to next iteration
            s = self.get_var('s')
            # NB: -1 to convert from length --> index
            storage_state_index = target_index - 1
            d.s_init = s.iloc[:, storage_state_index, :]
        self.load_solution_iterative(node_vars, total_vars, cost_vars)

    def load_results(self):
        """Load results into model instance for access via model variables."""
        r = self.instance.load(self.results)
        if r is False:
            print(self.results.Problem)
            print(self.results.Solver)
            raise UserWarning('Could not load results into model instance.')

    def save_solution(self, how='hdf'):
        """Save model solution. ``how`` can be ``'hdf'`` or ``'csv'``"""
        # Create output dir, but ignore if it already exists
        try:
            os.makedirs(self.config_run.output.path)
        except OSError:  # Hoping this isn't raised for more serious stuff
            pass
        if how == 'hdf':
            self._save_hdf()
        elif how == 'csv':
            self._save_csv()
        else:
            raise ValueError('Unsupported value for how: {}'.format(how))

    def _save_hdf(self):
        """
        Save solution as HDF5 to the file ``solution.hdf`` in
        ``self.config_run.output.path``

        """
        sol = self.solution
        store_file = os.path.join(self.config_run.output.path, 'solution.hdf')
        # Raise error if file exists already, to make sure we don't destroy
        # existing data
        if os.path.exists(store_file):
            raise IOError('File `{}` exists, aborting.'.format(store_file))
        # Set compression to highest level (9), using blosc, which is fast
        # Also set mode to 'w' so existing file will be overwritten
        store = pd.HDFStore(store_file, mode='w',
                            complevel=9, complib='blosc')
        store.put('locations', self.data.locations)
        for key in sol:
            # Use .append instead of .add for Panel4D compatibility
            store.append(key, sol[key])
        store.close()

    def _save_csv(self):
        """Save solution as CSV files to ``self.config_run.output.path``"""
        d = self.data
        sol = self.solution
        output_files = {'locations.csv': d.locations,
                        'node_parameters.csv': sol.parameters.to_frame(),
                        'costs.csv': sol.costs.to_frame(),
                        'metadata.csv': sol.metadata,
                        'summary.csv': sol.summary,
                        'shares.csv': sol.shares,
                        'time_res.csv': sol.time_res}
        for var in sol.node.labels:
            k = 'node_variables_{}.csv'.format(var.replace(':', '_'))
            v = sol.node[var].to_frame()
            output_files[k] = v
        for c in self.data._c:
            k = 'totals_{}.csv'.format(c)
            output_files[k] = sol.totals[c].to_frame()
        for cost_class in self.data._k:
            k = 'levelized_cost_{}.csv'.format(cost_class)
            output_files[k] = sol.levelized_cost[cost_class].to_frame()
        # Write all files to output dir
        for k, v in output_files.iteritems():
            v.to_csv(os.path.join(self.config_run.output.path, k))
