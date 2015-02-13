"""
Copyright (C) 2013-2015 Stefan Pfenninger.
Licensed under the Apache 2.0 License (see LICENSE file).

core.py
~~~~~~~

Core model functionality via the Model class.

"""

import datetime
import importlib
import inspect
import itertools
import logging
import os
import random
import shutil
import sys
import time
import warnings

import pyomo.opt as popt
import pyomo.core as po
import pyomo.environ  # Necessary for solver plugins etc.
import numpy as np
import pandas as pd
from pyutilib.services import TempfileManager

from . import exceptions
from . import constraints
from . import locations
from . import output
from . import transmission
from . import time_functions
from . import time_tools
from . import utils

# Enable simple format when printing ModelWarnings
formatwarning_orig = warnings.formatwarning


def _formatwarning(message, category, filename, lineno, line=None):
    """Formats ModelWarnings as "Warning: message" without extra crud"""
    if category == exceptions.ModelWarning:
        return 'Warning: ' + str(message) + '\n'
    else:
        return formatwarning_orig(message, category, filename, lineno, line)

warnings.formatwarning = _formatwarning

# Get list of techs pre-defined in defaults.yaml
module_config = os.path.join(os.path.dirname(__file__), 'config')
o = utils.AttrDict.from_yaml(os.path.join(module_config, 'defaults.yaml'))
DEFAULT_TECHS = list(o.techs.keys())


def _load_function(source):
    """
    Returns a function from a module, given a source string of the form:

        'module.submodule.subsubmodule.function_name'

    """
    module_string, function_string = source.rsplit('.', 1)
    modules = [i for i in sys.modules.keys() if 'calliope' in i]
    # Check if module already loaded, if so, don't re-import it
    if (module_string in modules):
        module = sys.modules[module_string]
    elif ('calliope.' + module_string) in modules:
        module = sys.modules['calliope.' + module_string]
    # Else load the module
    else:
        try:
            module = importlib.import_module(module_string)
        except ImportError:
            module = importlib.import_module('calliope.' + module_string)
    return getattr(module, function_string)


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
            overlap = set(DEFAULT_TECHS) & set(new_o.techs.keys())
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
    Calliope: a multi-scale energy systems (MUSES) modeling framework

    Parameters
    ----------
    config_run : str or AttrDict, default None
        Path to YAML file with run settings, or AttrDict containing run
        settings. If not given, the included default run and model
        settings are used.
    override : AttrDict, default None
        Provide any additional options or override options from
        ``config_run`` by passing an AttrDict of the form
        ``{'model_settings': 'foo.yaml'}``. Any option possible in
        ``run.yaml`` can be specified in the dict, inluding ``override.``
        options.

    """
    def __init__(self, config_run=None, override=None):
        super(Model, self).__init__()
        self.debug = utils.AttrDict()
        self.initialize_configuration(config_run, override)
        # Other initialization tasks
        self.data = utils.AttrDict()
        # Initialize the option getter
        self._get_option = utils.option_getter(self.config_model, self.data)
        self.initialize_parents()
        self.initialize_sets()
        self.read_data()
        self.mode = self.config_run.mode
        self.initialize_availability()
        self.initialize_time()

    def override_model_config(self, override_dict):
        o = self.config_model
        od = override_dict
        if 'data_path' in od.keys_nested():
            # If run_config overrides data_path, interpret it as
            # relative to the run_config file's path
            od['data_path'] = utils.relative_path(od['data_path'],
                                                  self.config_run_path)
        o.union(od, allow_override=True, allow_replacement=True)

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
            # debug.keepfiles is set to True
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

    def get_timeres(self, verify=False):
        """Returns resolution of data in hours. Needs a properly
        formatted ``set_t.csv`` file to work.

        If ``verify=True``, verifies that the entire file is at the same
        resolution. ``model.get_timeres(verify=True)`` can be called
        after Model initialization to verify this.

        """
        path = self.config_model.data_path
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
            d.a[y] = pd.DataFrame(avail, index=d._dt.index, columns=d._x)

    def initialize_time(self):
        """
        Performs time resolution reduction, if set up in configuration
        """
        cr = self.config_run
        s = time_tools.TimeSummarizer()
        time = cr.get_key('time', default=False)
        if time:
            if 'function' in time and 'file' in time:
                e = exceptions.ModelError
                raise e('`time.function` and `time.file` cannot be'
                        'given at the same time.')
            if cr.get_key('time.function', default=False):
                options = cr.get_key('time.function_options',
                                     default=False)
                func_string = 'time_functions.' + cr.time.function
                mask_func = _load_function(func_string)
                if options:
                    mask_src = mask_func(self.data, **options)
                else:
                    mask_src = mask_func(self.data)
                if mask_src.name == 'mask':
                    getter = time_functions.masks_to_resolution_series
                    res_series = getter([mask_src])
                else:
                    # mask_src.name is 'resolution_series', no further
                    # processing needed
                    res_series = mask_src
            elif cr.get_key('time.file', default=False):
                res_file = utils.relative_path(cr.time.file,
                                               self.config_run_path)
                res_series = pd.read_csv(res_file, index_col=0, header=None)[1]
                res_series = res_series.astype(int)[self.slice]
            s.dynamic_timestepper(self.data, res_series)

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
            self._t_index = pd.Index(self.data._dt.index)
        # Get the location of t in the index and use it to retrieve
        # the desired value, raising an error if it's <0
        loc = self._t_index.get_loc(t) - 1
        if loc >= 0:
            return self.data._dt.index[loc]
        else:
            e = exceptions.ModelError
            raise e('Attempted to get a timestep earlier than the first one.')

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
        o = self.config_model
        d = self.data
        if x is None:
            o.set_key('techs.' + option, value)
        else:  # Setting a specific x
            d.locations.at[x, '_override.' + option] = value
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
        if y in self.data._y_transmission:
            return 'transmission'
        else:
            while True:
                parent = self.get_option(y + '.parent')
                if parent == 'defaults':
                    break
                y = parent
            return y

    def get_date_index(self, date):
        return self.data._dt[self.data._dt == date].index[0]

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
        except exceptions.OptionNotSetError:
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
        if peak < 0:
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
            if 'parent' not in list(o.techs[tech].keys()):
                e = exceptions.ModelError
                raise e('Technology `' + tech + '` defines no parent!')
        # Verify that no technologies apart from the default technologies
        # inherit from 'defaults'
        for k, v in self.parents.items():
            if k not in DEFAULT_TECHS and v == 'defaults':
                e = exceptions.ModelError
                raise e('Tech `' + k + '` inherits from `defaults` but ' +
                        'should inherit from a built-in default technology.')
        # Verify that all parents are themselves actually defined
        for k, v in self.parents.items():
            if v not in list(o.techs.keys()):
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

    def _initialize_transmission(self):
        o = self.config_model
        d = self.data
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

    def initialize_sets(self):
        o = self.config_model
        d = self.data
        path = o.data_path
        #
        # t: Timesteps set
        #
        table_t = pd.read_csv(os.path.join(path, 'set_t.csv'), header=None,
                              index_col=1, parse_dates=[1])
        table_t.columns = ['t_int']
        if self.config_run.get_key('subset_t', default=False):
            table_t = table_t.loc[self.config_run.subset_t[0]:
                                  self.config_run.subset_t[1]]
            self.slice = slice(table_t.iat[0, 0], table_t.iat[-1, 0] + 1)
        else:
            self.slice = slice(None)
        _t = pd.Series([int(t) for t in table_t['t_int'].tolist()])
        d._dt = pd.Series(table_t.index, index=_t.tolist())
        # First set time_res_data and time_res_static across all data
        # `time_res_data` never changes, so always reflects the spacing
        # of time step indices
        d.time_res_data = self.get_timeres()
        # `time_res_static` is updated after time resolution adjustments,
        # so does not reflect the spacing of time step indicex
        d.time_res_static = d.time_res_data
        d.time_res_native = 1  # In the beginning, time_res is native
        # From time_res_data, initialize time_res_series
        d.time_res_series = pd.Series(d.time_res_data, index=d._dt.index)
        # Last index t for which model may still use startup exceptions
        d.startup_time_bounds = d._dt.index[int(o.startup_time
                                                / d.time_res_data)]
        #
        # y: Technologies set
        #
        d._y = set()
        try:
            for k, v in o.locations.items():
                for y in v.techs:
                    d._y.add(y)
        except KeyError:
            e = exceptions.ModelError
            raise e('The region `' + k + '` does not allow '
                    'any technologies via `techs`. Must give '
                    'at least one technology per region.')
        d._y = list(d._y)
        if self.config_run.get_key('subset_y', default=False):
            d._y = [y for y in d._y if y in self.config_run.subset_y]
        # Subset of transmission technologies, if any defined
        # Used to initialized transmission techs further below
        # (not yet added to d._y here)
        if ('links' in o) and (o.links is not None):
            d._y_transmission = transmission.get_transmission_techs(o.links)
            d.transmission_y = list(set([list(v.keys())[0]
                                    for k, v in o.links.items()]))
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
        # Subset of technologies that allow rb
        d._y_rb = [y for y in d._y
                   if self.get_option(y + '.constraints.allow_rb') is True]
        # Subset of technologies with parasitics (carrier efficiency != 1.0)
        d._y_p = [y for y in d._y
                  if self.get_option(y + '.constraints.c_eff') != 1.0]
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
        d._x = list(o.locations.keys())
        if self.config_run.get_key('subset_x', default=False):
            d._x = [x for x in d._x if x in self.config_run.subset_x]
        #
        # Locations settings matrix and transmission technologies
        #
        d.locations = locations.generate_location_matrix(o.locations,
                                                         techs=d._y)
        # For simplicity, only keep the locations that are actually in set `x`
        d.locations = d.locations.loc[d._x, :]
        #
        # Initialize transmission technologies
        #
        self._initialize_transmission()
        #
        # self.data._y is now complete, ensure that all techs conform to the
        # rule that only "head" techs can be used in the model
        #
        for y in self.data._y:
            if self.get_option(y + '.parent') in self.data._y:
                e = exceptions.ModelError
                raise e('Only technologies without children can be used '
                        'in the model definition '
                        '({}, {}).'.format(y, self.get_option(y + '.parent')))
        #
        # k: Cost classes set
        #
        classes = [list(o.techs[k].costs.keys()) for k in o.techs
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
        d._dt.index maps a simple [0, data length] index to the actual
        t index used.

        Data is stored in the `self.data`  for each
        `param` and technology `y`: ``self.data[param][y]``

        """
        @utils.memoize
        def _get_option_from_csv(filename):
            """Read CSV time series"""
            d_path = os.path.join(self.config_model.data_path, filename)
            # [self.slice] to do time subset if needed
            # Results in e.g. d.r_eff['csp'] being a dataframe
            # of efficiencies for each time step t at location x
            df = pd.read_csv(d_path, index_col=0)[self.slice]
            # Fill columns that weren't defined with NaN
            # missing_cols = list(set(self.data._x) - set(df.columns))
            # for c in missing_cols:
            #     df[c] = np.nan

            # Ensure that the read file's index matches the data's timesteps
            mismatch = df.index.difference(d._dt.index)
            if len(mismatch) > 0:
                e = exceptions.ModelError
                entries = mismatch.tolist()
                raise e('File has invalid index. Ensure that it has the same '
                        'date range and resolution as set_t.csv: {}.\n\n'
                        'Invalid entries: {}'.format(filename, entries))
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
        # TODO could allow params in d.params to be defined only over
        # x instead of either static or over (x, t) via CSV!
        for param in d.params:
            d[param] = utils.AttrDict()
            for y in d._y:
                d[param][y] = pd.DataFrame(np.nan, index=d._dt.index,
                                           columns=d._x)
                # TODO this whole process could be refactored for efficiency
                # to read files only once,
                # create a dict of files: {'f1.csv': ['x1', 'x2'],
                #                          'f2.csv': ['x3'],
                #                          'model_config': ['x4, x5']}
                for x in d._x:
                    # If this y is actually not defined at this x,
                    # and is also not a transmission tech,
                    # continue (but set the param to 0 first)
                    # TODO this is a bit of a hack -- e.g. the extra check
                    # for transmission tech is necessary because we set
                    # e_eff to 0 for all transmission (as transmission techs
                    # don't show up in the config_model.locations[x].techs)
                    # Keep an eye out in case this causes other problems
                    if (y not in self.config_model.locations[x].techs
                            and y not in d._y_transmission):
                        d[param][y].loc[:, x] = 0
                        continue
                    option = self.get_option(y + '.constraints.' + param, x=x)
                    k = param + '.' + y + '.' + x
                    if (isinstance(option, str)
                            and option.startswith('file')):
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
                        # Set up x_map if that option has been set
                        try:
                            x_map = self.get_option(y + '.x_map', x=x)
                        except exceptions.OptionNotSetError:
                            x_map = None
                        # If x_map is available, remap the current col
                        if x_map:
                            # TODO this is a hack and will take up a lot
                            # of memory due to data duplication in case
                            # of a lot of mappings pointing to the same
                            # column in the data
                            # Format is <name in model config>:<name in data>
                            x_map_dict = {i.split(':')[0].strip():
                                          i.split(':')[1].strip()
                                          for i in x_map.split(',')}
                            x_map_str = 'x_map: \'{}\''.format(x_map)
                            # Get the mapping for this x from x_map
                            # NB not removing old columns in case
                            # those are also used somewhere!
                            try:
                                x_m = x_map_dict[x]
                            except KeyError:
                                e = exceptions.ModelError
                                raise e('x_map defined but does not map '
                                        'location defined in model config: '
                                        '{}, with {}'.format(x, x_map_str))
                            if x_m not in df.columns:
                                e = exceptions.ModelError
                                raise e('Trying to map to to a column not '
                                        'contained in data: {}, for region '
                                        '{}, with {}'
                                        .format(x_m, x, x_map_str))
                            df[x] = df[x_m]
                        try:
                            d[param][y].loc[:, x] = df[x]
                            self.debug.data_sources.set_key(k, 'file:' + f)
                        except KeyError:
                            # If could not be read from file, set it to zero
                            d[param][y].loc[:, x] = 0
                            # Depending on whether or not the tech is allowed
                            # at this location, set _NA_ for the data source,
                            # or raise an error
                            if self.data.locations.at[x, y] == 0:
                                self.debug.data_sources.set_key(k, '_NA_')
                            else:
                                w = exceptions.ModelWarning
                                message = ('Could not load data for {}, '
                                           'with given option: '
                                           '{}'.format(k, option))
                                warnings.warn(message, w)
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
        missing_data = [kk for kk in ds.keys_nested()
                        if ds.get_key(kk) == 'file:_NOT_FOUND_']
        if len(missing_data) > 0:
            message = ('The following parameter values could not be read '
                       'from file. They were automatically set to `0`: '
                       + ', '.join(missing_data))
            warnings.warn(message, exceptions.ModelWarning)

    def _get_t_max_demand(self):
        t_max_demands = utils.AttrDict()
        for c in self.data._c:
            ys = [y for y in self.data._y
                  if self.get_option(y + '.carrier') == c]
            r_carrier = pd.Panel(self.data.r.as_dict()).loc[ys].sum(axis=2)
            t_max_demand = r_carrier[r_carrier < 0].sum(axis=1).idxmin()
            # Adjust for reduced resolution, only if t_max_demand not 0 anyway
            if t_max_demand != 0:
                t_max_demand = max([t for t in self.data._dt.index
                                    if t <= t_max_demand])
            t_max_demands[c] = t_max_demand
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

    def _param_populator(self, src, t_offset=0):
        """Returns a `getter` function that returns either
        (x, t)-specific values for parameters that define such, or
        always the same static value if only a static value is given.

        """
        def getter(m, y, x, t):
            if isinstance(src[y], pd.core.frame.DataFrame):
                return float(src[y].loc[t + t_offset, x])
            else:
                return float(src[y])
        return getter

    def update_parameters(self, t_offset):
        mi = self.instance
        d = self.data

        for param in d.params:
            initializer = self._param_populator(d[param], t_offset)
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

    def _set_t_end(self):
        # t_end is the timestep previous to t_start + horizon,
        # because the .loc[start:end] slice includes the end
        try:
            self.t_end = self.prev(int(self.t_start
                                   + self.config_model.opmode.horizon
                                   / self.data.time_res_data))
        except KeyError:
            # If t_end is beyond last timestep, cap it to last one, and
            # log the occurance
            t_bound = self.data._dt.index[-1]
            msg = 'Capping t_end to {}'.format(t_bound)
            logging.debug(msg)
            self.t_end = t_bound
        print('\n\n***\n{}-{}\n***\n\n'.format(self.t_start, self.t_end))

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
        m = po.ConcreteModel()
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
            m.t = po.Set(initialize=d._dt.index, ordered=True)
        elif self.mode == 'operate':
            self._set_t_end()
            m.t = po.Set(initialize=d._dt.loc[self.t_start:self.t_end].index,
                         ordered=True)
        # Carriers
        m.c = po.Set(initialize=d._c, ordered=True)
        # Locations
        m.x = po.Set(initialize=d._x, ordered=True)
        # Cost classes
        m.k = po.Set(initialize=d._k, ordered=True)
        #
        # Technologies and various subsets of technologies
        #
        m.y = po.Set(initialize=d._y, ordered=True)
        # Production technologies
        m.y_prod = po.Set(initialize=d._y_prod, within=m.y, ordered=True)
        # Production technologies
        m.y_con = po.Set(initialize=d._y_con, within=m.y, ordered=True)
        # Production/consumption technologies
        m.y_pc = po.Set(initialize=d._y_pc, within=m.y, ordered=True)
        # Transmission technologies
        m.y_trans = po.Set(initialize=d._y_transmission, within=m.y,
                           ordered=True)
        # Conversion technologies
        m.y_conv = po.Set(initialize=d._y_conversion, within=m.y,
                          ordered=True)
        # Technologies with specified `r`
        m.y_def_r = po.Set(initialize=d._y_def_r, within=m.y)
        # Technologies with specified `e_eff`
        m.y_def_e_eff = po.Set(initialize=d._y_def_e_eff, within=m.y)
        # Technologies that allow `rb`
        m.y_rb = po.Set(initialize=d._y_rb, within=m.y)
        # Technologies with parasitics
        m.y_p = po.Set(initialize=d._y_p, within=m.y)
        # Technologies without parasitics
        m.y_np = po.Set(initialize=set(d._y) - set(d._y_p), within=m.y)

        #
        # Parameters
        #

        self.param_sets = {'r': m.y_def_r,
                           'a': m.y_def_r,
                           'e_eff': m.y_def_e_eff}
        for param in d.params:
            initializer = self._param_populator(d[param])
            y = self.param_sets[param]
            setattr(m, param, po.Param(y, m.x, m.t, initialize=initializer,
                                       mutable=True))

        s_init_initializer = lambda m, y, x: float(d.s_init.at[x, y])
        m.s_init = po.Param(m.y_pc, m.x, initialize=s_init_initializer,
                            mutable=True)

        #
        # Variables and constraints
        #
        # 1. Required
        required = [constraints.base.node_resource,
                    constraints.base.node_energy_balance,
                    constraints.base.node_constraints_build,
                    constraints.base.node_constraints_operational,
                    constraints.base.node_constraints_transmission,
                    constraints.base.node_parasitics,
                    constraints.base.node_costs,
                    constraints.base.model_constraints]
        for c in required:
            self.add_constraint(c)

        if self.mode == 'plan':
            self.add_constraint(constraints.planning.system_margin)

        # 2. Optional
        if o.get_key('constraints', default=False):
            for c in o.constraints:
                self.add_constraint(_load_function(c))

        # 3. Objective function
        default_obj = 'constraints.objective.objective_cost_minimization'
        objective = o.get_key('objective', default=default_obj)
        self.add_constraint(_load_function(objective))

    def _log_time(self):
        self.runtime = int(time.time() - self.start_time)
        logging.info('Runtime: ' + str(self.runtime) + ' secs')

    def run(self, iterative_warmstart=True):
        """
        Instantiate and solve the model

        """
        o = self.config_model
        d = self.data
        cr = self.config_run
        self.start_time = time.time()
        if self.mode == 'plan':
            self.generate_model()  # Generated model goes to self.m
            self.solve()
            self.load_solution()
        elif self.mode == 'operate':
            assert len(self.data.time_res_series.unique()) == 1, \
                'Operational mode only works with uniform time step lengths.'
            assert (d.time_res_static <= o.opmode.horizon and
                    d.time_res_static <= o.opmode.window), \
                'Timestep length must be smaller than horizon and window.'
            # solve_iterative() generates, solves, and loads the solution
            self.solve_iterative(iterative_warmstart)
        else:
            e = exceptions.ModelError
            raise e('Invalid model mode: `{}`'.format(self.mode))
        self._log_time()
        if cr.get_key('output.save', default=False) is True:
            output_format = cr.get_key('output.format', default=['hdf'])
            if not isinstance(output_format, list):
                output_format = [output_format]
            for fmt in output_format:
                self.save_solution(fmt)
            save_constr = cr.get_key('output.save_constraints', default=False)
            if save_constr:
                options = cr.get_key('output.save_constraints_options',
                                     default={})
                output.generate_constraints(self.solution,
                                            output_path=save_constr,
                                            **options)

    def solve(self, warmstart=False):
        """
        Args:
            warmstart : (default False) re-solve an updated model
                        instance

        Returns: None

        """
        m = self.m
        cr = self.config_run
        if not warmstart:
            self.instance = m.create()
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
                self.opt.symbolic_solver_labels = True
        if cr.get_key('debug.keepfiles', default=False):
            self.opt.keepfiles = True
            if self.mode == 'plan':
                logdir = os.path.join('Logs', self.run_id)
            elif self.mode == 'operate':
                logdir = os.path.join('Logs', self.run_id
                                      + '_' + str(self.t_start))
            if (cr.get_key('debug.delete_old_logs', default=False)
                    and os.path.exists(logdir)):
                shutil.rmtree(logdir)
            os.makedirs(logdir)
            TempfileManager.tempdir = logdir
        # Always preprocess instance, for both cold start and warm start
        self.instance.preprocess()

        def _solve(warmstart):
            if warmstart:
                results = self.opt.solve(self.instance, warmstart=True,
                                         tee=True)
            else:
                results = self.opt.solve(self.instance, tee=True)
            return results

        if cr.get_key('debug.echo_solver_log', default=False):
            self.results = _solve(warmstart)
        else:
            # Silencing output by redirecting stdout and stderr
            with utils.capture_output() as self.pyomo_output:
                self.results = _solve(warmstart)
        self.load_results()

    def process_solution(self):
        # Add levelized cost
        self.solution['levelized_cost'] = self.get_levelized_cost()
        # Add capacity factor
        self.solution['capacity_factor'] = self.get_capacity_factor()
        # Add metadata
        self.solution['metadata'] = self.get_metadata()
        # Add summary
        self.solution['summary'] = self.get_summary()
        # Add shares
        self.solution['shares'] = self.get_shares()
        # Add time resolution, and give it a nicer index
        time_res = self.data.time_res_series.copy()
        time_res.index = self.solution.node.major_axis
        self.solution['time_res'] = time_res
        # Add model and run config
        self.solution['config_run'] = self.config_run
        self.solution['config_model'] = self.config_model

    def load_solution(self):
        sol = {'node': self.get_node_variables(),
               'totals': self.get_totals(),
               'costs': self.get_costs(),
               'parameters': self.get_node_parameters()}
        self.solution = utils.AttrDict(sol)
        self.process_solution()

    def get_var(self, var, dims=None):
        """
        Return output for variable `var` as a series, dataframe or panel

        Args:
            var : variable name as string, e.g. 'es_prod'

            dims : list of indices as strings, e.g. ('y', 'x', 't');
            if not given, they are auto-detected

        """
        # FIXME: even if `dims` were given where `t` is not in the last
        # position, the code below assumes that `t` is in the last position,
        # and won't work otherwise. All model variables are formulated
        # so that `t` is always last, so this should not be a problem.
        m = self.m
        try:
            var = getattr(m, var)
        except AttributeError:
            raise exceptions.ModelError('Variable {} inexistent.'.format(var))
        # Get dims
        if not dims:
            dims = [i.name for i in var.index_set().set_tuple]
        result = pd.DataFrame.from_dict(var.get_values(), orient='index')
        if result.empty:
            raise exceptions.ModelError('Variable {} has no data.'.format(var))
        result.index = pd.MultiIndex.from_tuples(result.index, names=dims)
        result = result[0]  # Get the only column in the dataframe
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
            if self.t_start is None:
                new_index = self.data._dt.loc[t.first():t.last()].tolist()
            else:
                new_index = self.data._dt.loc[self.t_start:self.t_end].tolist()
            if len(dims) <= 2:
                result.index = new_index
            else:
                result.major_axis = new_index
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
        for carrier in ec.labels:
            for tech in ec.items:
                es[carrier][tech] = ec[carrier][tech]
        return es  # the variable is called es, but the thing is now ec

    def get_ec_sum(self):
        prod = self.get_ec('prod')
        con = self.get_ec('con')
        ec = {}
        for i in prod.labels:
            ec[i] = (prod[i].to_frame().add(con[i].to_frame(),
                                            fill_value=0)).to_panel()
        return pd.Panel4D(ec)

    def get_node_variables(self):
        detail = ['s', 'rs']
        p = pd.Panel4D({v: self.get_var(v) for v in detail})
        try:
            p['rbs'] = self.get_var('rbs')
        except exceptions.ModelError:
            # `rbs` doesn't exist in the model or exists without data
            p['rbs'] = 0
        # Add 'e:c' items for each c in carrier
        temp = self.get_ec_sum()
        for c in self.data._c:
            p['e' + ':' + c] = temp.loc[c, :, :, :]
        return p

    def get_node_parameters(self, built_only=False):
        """If built_only is True, disregard locations where e_cap==0"""
        detail = ['s_cap', 'r_cap', 'r_area', 'e_cap', 'e_cap_net']
        result = pd.Panel({v: self.get_var(v) for v in detail})
        try:
            result['rb_cap'] = self.get_var('rb_cap')
        except exceptions.ModelError:
            result['rb_cap'] = 0
        if built_only:
            result = result.to_frame()
            result = result[result.e_cap > 0].dropna(axis=1).to_panel()
        return result

    def get_costs(self, t_subset=None):
        """Get costs."""
        def _factor(var, t_subset):
            """Return the fraction of var within t_subset, used to
               calculate the correct fraction of operational costs."""
            if var.ndim == 4:  # Make Panel4D into Panel by summing carriers
                var = var.sum(0)
            return (var.iloc[:, t_subset, :].sum(1)
                    / var.sum(1)).fillna(0)

        if t_subset is None:
            t_subset = slice(None)

        T = lambda x: pd.Panel.transpose(x, 'major_axis',
                                         'minor_axis', 'items')

        len_adjust = (len(self.data.time_res_series.iloc[t_subset])
                      / len(self.m.t))

        cost = T(self.get_var('cost_con').add(self.get_var('cost_op_fixed')))
        # Adjust for the fact that fixed costs accrue over a smaller length
        # of time as per len_adjust
        cost = cost * len_adjust
        cost_var = T(self.get_var('cost_op_var'))
        cost_fuel = T(self.get_var('cost_op_fuel'))
        cost_rb = T(self.get_var('cost_op_rb'))
        # Adjust for the fact that var and fuel costs are only accrued over
        # the t_subset period
        if t_subset:
            es_prod = self.get_var('es_prod')
            # Broadcast multiplication along items (i.e. cost classes)
            cost_var = cost_var.multiply(_factor(es_prod, t_subset),
                                         axis='items')
            rs = self.get_var('rs')
            cost_fuel = cost_fuel.multiply(_factor(rs, t_subset),
                                           axis='items')
            try:
                rbs = self.get_var('rbs')
                cost_rb = cost_rb.multiply(_factor(rbs, t_subset),
                                           axis='items')
            except exceptions.ModelError:
                pass  # If rbs doesn't exist in the data, ModelError is raised,
                # and we simply move on...
        cost = cost.add(cost_var).add(cost_fuel).add(cost_rb)
        return cost

    def get_totals(self, t_subset=None):
        """Get total produced and consumed per technology and location."""
        if t_subset is None:
            t_subset = slice(None)
        p = pd.Panel4D({'ec_' + i: self.get_ec(i)
                        .iloc[:, :, t_subset, :]
                        .sum(axis='major')
                        for i in ['prod', 'con']})
        for i in ['es_prod', 'es_con']:
            p[i] = self.get_var(i).iloc[:, :, t_subset, :].sum(axis='major')
        p = p.transpose('items', 'labels', 'minor_axis', 'major_axis')
        return p

    def get_levelized_cost(self):
        """
        Get levelized costs.

        NB: Only production, not consumption, is used in calculations.

        """
        sol = self.solution
        p4d = {}
        for cost in self.data._k:
            p = {}
            for carrier in self.data._c:
                # Levelized cost of electricity (LCOE)
                lc = sol.costs[cost] / sol.totals[carrier].ec_prod
                lc[np.isinf(lc)] = 0
                lc.loc['total', :] = (sol.costs[cost].sum(0)
                                      / sol.totals[carrier].ec_prod.sum(0))
                p[carrier] = lc
            p4d[cost] = pd.Panel(p)
        return pd.Panel4D(p4d)

    def get_capacity_factor(self):
        """
        Get capacity factor.

        NB: Only production, not consumption, is used in calculations.

        """
        m = self.m
        sol = self.solution
        time_res = self.data.time_res_series
        e_cap = sol.parameters['e_cap_net']
        cfs = {}
        for carrier in sol.totals.labels:
            try:  # Try loading time_res_sum from operational mode
                time_res_sum = self.data.time_res_sum
            except KeyError:
                time_res_sum = sum(time_res.at[t] for t in m.t)
            cf = sol.totals[carrier].ec_prod / (e_cap * time_res_sum)
            cf.loc['total', :] = (sol.totals[carrier].ec_prod.sum(0)
                                  / (e_cap.sum(0) * time_res_sum))
            cf = cf.fillna(0)
            cfs[carrier] = cf
        return pd.Panel(cfs)

    def get_metadata(self):
        df = pd.DataFrame(index=self.data._y)
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
        # Capacity factor per carrier
        df = pd.DataFrame({'cf': sol.capacity_factor.loc[carrier, 'total', :]})
        # Costs per carrier
        for k in sorted(sol.levelized_cost.labels):
            # .loc[cost_class, carrier, location, tech]
            df['cost_' + k] = sol.levelized_cost.loc[k, carrier, 'total', :]
        # Add totals per carrier
        df['e_prod'] = sol.totals.loc[carrier, 'ec_prod', :, :].sum(0)
        df['e_con'] = sol.totals.loc[carrier, 'ec_con', :, :].sum(0)
        # Add other carrier-independent stuff
        df['e_cap'] = sol.parameters['e_cap'].sum()
        df['r_area'] = sol.parameters['r_area'].sum()
        return df.sort(columns=sort_by, ascending=False)

    def get_shares(self):
        from . import analysis
        ggm = self.get_group_members
        s = pd.Series({k: '|'.join(ggm(k, head_nodes_only=True))
                      for k in self.config_model.techs
                      if ggm(k, head_nodes_only=True) != []
                      and ggm(k, head_nodes_only=True) is not None})

        df = pd.DataFrame(s, columns=['members'])
        gg = lambda y: self.get_option(y + '.group',
                                       default='defaults.group',
                                       ignore_inheritance=True)
        df['group'] = df.index.map(gg)
        df['type'] = df.index.map(self.get_parent)

        for var in ['e_prod', 'e_con', 'e_cap']:
            for index, row in df.iterrows():
                group_members = row['members'].split('|')
                group_type = row['type']
                share = analysis.get_group_share(self.solution, group_members,
                                                 group_type, var=var)
                df.at[index, var] = share
        return df

    def load_solution_iterative(self, node_vars, total_vars, cost_vars):
        def _panel_sum(panels):
            # Sums a list of 3d or 4d panels
            result = panels[0]  # Take first item
            for i in range(1, len(panels)):  # Start at 1 as we dealt with 0
                result = result.add(panels[i])
            return result
        totals = _panel_sum(total_vars)
        costs = _panel_sum(cost_vars)
        sol = {'node': pd.concat(node_vars, axis=2),
               'totals': totals,
               'costs': costs,
               'parameters': self.get_node_parameters()}
        self.solution = utils.AttrDict(sol)
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
        time_res = d.time_res_series
        window_adj = int(o.opmode.window / d.time_res_data)
        steps = [t for t in d._dt.index if (t % window_adj) == 0]
        print(d.time_res_data)
        print(d.time_res_static)
        print(steps)
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
                stepsize = int(o.opmode.horizon / d.time_res_static)
            else:
                # Non-final iterations only save data from window
                stepsize = int(o.opmode.window / d.time_res_static)
            print(index)
            print(stepsize)
            print('\n')
            # Get node variables
            panel4d = self.get_node_variables()
            node_vars.append(panel4d.iloc[:, :, 0:stepsize, :])
            # Get totals
            totals = self.get_totals(t_subset=slice(0, stepsize))
            total_vars.append(totals)
            # Get costs
            cost = self.get_costs(t_subset=slice(0, stepsize))
            cost_vars.append(cost)
            timesteps = [time_res.at[t] for t in self.m.t][0:stepsize]
            print(self.m.t.value)
            d.time_res_sum += sum(timesteps)
            # Save state of storage for carry over to next iteration
            s = self.get_var('s')
            # Convert from timestep length to absolute index
            storage_state_index = stepsize - 1
            assert (isinstance(storage_state_index, int) or
                    storage_state_index.is_integer())
            storage_state_index = int(storage_state_index)
            print('storage_state_index: {}'.format(storage_state_index))
            d.s_init = s.iloc[:, storage_state_index, :]
        self.load_solution_iterative(node_vars, total_vars, cost_vars)

    def load_results(self):
        """Load results into model instance for access via model variables."""
        r = self.instance.load(self.results)
        if r is False:
            logging.critical(self.results.Problem)
            logging.critical(self.results.Solver)
            w = exceptions.ModelWarning
            warnings.warn('Could not load results into model instance.', w)

    def save_solution(self, how):
        """Save model solution. ``how`` can be 'hdf' or 'csv'

        CSV is supported for legacy purposes but usually, the
        HDF option should be used.

        """
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
            raise ValueError('Unsupported value for `how`: {}'.format(how))

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
            i = 0
            alt_file = os.path.join(self.config_run.output.path,
                                    'solution_{}.hdf')
            while os.path.exists(alt_file.format(i)):
                i += 1
            alt_file = alt_file.format(i)  # Now "pick" the first free filename
            message = ('File `{}` exists, '
                       'using `{}` instead.'.format(store_file, alt_file))
            logging.warning(message)
            store_file = alt_file
        # Use zlib compression (HDF standard)
        # Using mode 'w' means existing file will be overwritten
        store = pd.HDFStore(store_file, mode='w',
                            complevel=3, complib='zlib')
        # Now save solution -- except for config_model and config_run, since
        # they are AttrDicts so are saved separately below
        for key in [k for k in sol if k not in ['config_model', 'config_run']]:
            # Use .append instead of .add for Panel4D compatibility
            store.append(key, sol[key])
        # Now, save config_model and config_run as YAML strings
        config = pd.Series({key: sol[key].to_yaml() for key in ['config_model', 'config_run']})
        store.append('config', config)
        store.close()
        # Return the path we used
        return store_file

    def _save_csv(self):
        """Save solution as CSV files to ``self.config_run.output.path``"""
        d = self.data
        sol = self.solution
        output_files = {'node_parameters.csv': sol.parameters.to_frame(),
                        'costs.csv': sol.costs.to_frame(),
                        'capacity_factor.csv': sol.capacity_factor.to_frame(),
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
        for k, v in output_files.items():
            v.to_csv(os.path.join(self.config_run.output.path, k))
        # Also save model and run configuration
        self.config_run.to_yaml(os.path.join(self.config_run.output.path,
                                             'config_run.yaml'))
        self.config_model.to_yaml(os.path.join(self.config_run.output.path,
                                               'config_model.yaml'))
