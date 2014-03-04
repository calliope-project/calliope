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
import itertools
import json
import os
import shutil

import coopr.opt as co
import coopr.pyomo as cp
import numpy as np
import pandas as pd
from pyutilib.services import TempfileManager

from . import constraints
from . import locations
from . import transmission
from . import time
from . import time_masks
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
                       of the form ``{'input.path': 'foo', ...}``.
                       Any option possible in ``run.yaml`` can be specified
                       in the dict, inluding ``override.`` options.

        """
        super(Model, self).__init__()
        self.initialize_configuration(config_run, override)
        # Other initialization tasks
        self.data = utils.AttrDict()
        self.initialize_sets()
        self.initialize_parents()
        self.read_data()
        self.mode = self.config_run.mode
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
        # Ensure 'input.model' is a list
        if not isinstance(cr.input.model, list):
            cr.input.model = [cr.input.model]
        # Expand {{ module }} placeholder
        cr.input.model = [_expand_module_placeholder(i)
                          for i in cr.input.model]
        cr.input.path = _expand_module_placeholder(cr.input.path)
        # Interpret relative config paths as relative to run.yaml
        cr.input.model = [utils.ensure_absolute(i, self.config_run_path)
                          for i in cr.input.model]
        cr.input.path = utils.ensure_absolute(cr.input.path,
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
        path = self.config_run.input.path
        df = pd.read_csv(os.path.join(path, 'set_t.csv'), index_col=0,
                         header=None, parse_dates=[1])
        seconds = (df.iat[0, 0] - df.iat[1, 0]).total_seconds()
        if verify:
            for i in range(len(df) - 1):
                assert ((df.iat[i, 0] - df.iat[i+1, 0]).total_seconds()
                        == seconds)
        hours = abs(seconds) / 3600
        return hours

    def initialize_time(self):
        """
        Performs time resolution reduction, if set up in configuration
        """
        cr = self.config_run
        t = cr.get_key('time.summarize', default=False)
        s = time.TimeSummarizer()
        if t == 'mask':
            if (cr.get_key('time.mask_function', default=False) and
                    cr.get_key('time.mask_file', default=False)):
                raise KeyError('Define either mask_function or mask_file.')
            elif cr.get_key('time.mask_function', default=False):
                eval('mask_src = time_masks.'
                     + cr.time.mask_function + '(m.data)')
                mask = time_masks.masks_to_resolution_series([mask_src])
            elif cr.get_key('time.mask_file', default=False):
                mask = pd.read_csv(utils.ensure_absolute(cr.time.mask_file,
                                                         self.config_run_path),
                                   index_col=0, header=None)[1]
                mask = mask.astype(int)
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

    def get_group_members(self, group, in_model=True):
        """
        Return the member technologies of a group. If in_model is True,
        only members defined in the current model are returned.

        """
        def _get(self, group, memberset):
            members = [i for i in self.parents if self.parents[i] == group]
            if members:
                for i, member in enumerate(members):
                    members[i] = _get(self, member, memberset)
                return members
            else:
                memberset.add(group)
        if not group in self.parents:
            return None
        memberset = set()
        _get(self, group, memberset)
        member_techs = list(memberset)
        if in_model:
            member_techs = [y for y in member_techs
                            if (y in self.data._y
                                or y in self.data._y_transmission)]
        return member_techs

    @utils.memoize_instancemethod
    def get_eff_ref(self, var, y, x=None):
        """Get reference efficiency, falling back to efficiency if no
        reference efficiency has been set."""
        base = y + '.constraints.' + var
        eff_ref = self.get_option(base + '_eff_ref', x=x)
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
        self.parents = {i: o.techs[i].parent for i in o.techs.keys()
                        if i != 'defaults'}

    def initialize_sets(self):
        o = self.config_model
        d = self.data
        path = self.config_run.input.path
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
        for i in o.locations.itervalues():
            assert isinstance(i.techs, list)
            for y in i.techs:
                d._y.add(y)
        d._y = list(d._y)
        if self.config_run.get_key('subset_y', default=False):
            d._y = [y for y in d._y if y in self.config_run.subset_y]
        # Subset of transmission technologies, if any defined (used below)
        # (not yet added to d._y here)
        if ('links' in o) and (o.links is not None):
            d.transmission_y = transmission.get_transmission_techs(o.links)
            d._y_transmission = list(set([v.keys()[0]
                                     for k, v in o.links.iteritems()]))
        else:
            d.transmission_y = []
            d._y_transmission = []
        # Subset of conversion technologies
        # (already contained in d._y here but separated out as well)
        d.conversion_y = [y for y in d._y
                          if o.techs[y].parent == 'conversion']
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
        if d.transmission_y:
            d._y.extend(d.transmission_y)
            # Add transmission tech columns to locations matrix
            for y in d.transmission_y:
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
            d_path = os.path.join(self.config_run.input.path, filename)
            # [self.slice] to do time subset if needed
            # Results in e.g. d.r_eff['csp'] being a dataframe
            # of efficiencies for each time step t at location x
            df = pd.read_csv(d_path, index_col=0)[self.slice]
            # Fill columns that weren't defined with zeros
            missing_cols = list(set(self.data._x) - set(df.columns))
            for c in missing_cols:
                df[c] = 0
            return df

        d = self.data
        # Parameters that may defined over (x, t) for a given technology y
        d.params = ['r', 'e_eff']

        # TODO allow params in d.params to be defined only over
        # x instead of either static or over (x, t) via CSV!
        for param in d.params:
            d[param] = utils.AttrDict()
            for y in d._y:
                d[param][y] = pd.DataFrame(0, index=d._t[self.slice],
                                           columns=d._x)
                for x in d._x:
                    option = self.get_option(y + '.constraints.' + param, x=x)

                    if isinstance(option, str) and option.startswith('file'):
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
                        d[param][y][x] = df[x]
                    else:
                        d[param][y][x] = option
                    # Scale r to a given maximum if necessary
                    scale = self.get_option(y + '.constraints.r_scale_to_peak',
                                            x=x)
                    if param == 'r' and scale:
                        scaled = self.scale_to_peak(d[param][y][x], scale)
                        d[param][y][x] = scaled

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
            t_end = t_start + o.opmode.horizon / d.time_res_data
            # If t_end is beyond last timestep, cap it to last one
            # TODO this is a hack and shouldn't be necessary?
            if t_end > d._t.iat[-1]:
                t_end = d._t.iat[-1]
            m.t = cp.Set(initialize=d._t.loc[t_start:t_end],
                         ordered=True)
        # The rest
        m.c = cp.Set(initialize=d._c, ordered=True)  # Carriers
        m.y = cp.Set(initialize=d._y, ordered=True)  # Technologies
        m.x = cp.Set(initialize=d._x, ordered=True)  # Nodes
        m.k = cp.Set(initialize=d._k, ordered=True)  # Cost classes

        #
        # Parameters
        #
        def param_populator(src):
            """Returns a `getter` function that returns either
            (x, t)-specific values for parameters that define such, or
            always the same static value if only a static value is given.

            """
            def getter(m, y, x, t):
                if isinstance(src[y], pd.core.frame.DataFrame):
                    return float(src[y].loc[t, x])
                else:
                    return float(src[y])
            return getter

        for param in d.params:
            initializer = param_populator(d[param])
            setattr(m, param, cp.Param(m.y, m.x, m.t, initialize=initializer))

        m.time_res = (cp.Param(m.t,
                      initialize=lambda m, t: float(d.time_res_series.loc[t])))

        #
        # Constraints
        #
        # 1. Required
        required = [constraints.base.node_energy_balance,
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
        if self.mode == 'plan':
            self.generate_model()  # Generated model goes to self.m
            self.solve()
            self.load_results()
            if self.config_run.output.save is True:
                self.save_outputs()
        elif self.mode == 'operate':
            self.solve_iterative()
        else:
            raise UserWarning('Invalid mode')

    def solve(self, save_json=False):
        """
        Args:
            save_json : (default False) Save optimization results to
                        the given file as JSON.

        Returns: None

        """
        m = self.m
        instance = m.create()
        instance.preprocess()
        opt = co.SolverFactory(self.config_run.solver)  # could set solver_io='python'
        # Set solver options from run_settings file, if it exists
        try:
            for k, v in self.config_run.solver_options.iteritems():
                opt.options[k] = v
        except KeyError:
            pass
        if self.config_run.get_key('debug.symbolic_solver_labels',
                                   default=False):
            opt.symbolic_solver_labels = True
        if self.config_run.get_key('debug.keepfiles', default=False):
            opt.keepfiles = True
            if self.mode == 'plan':
                logdir = os.path.join('Logs', self.run_id)
            elif self.mode == 'operate':
                logdir = os.path.join('Logs', self.run_id
                                      + '_' + str(self.t_start))
            if self.config_run.get_key('debug.delete_old_logs',
                                       default=False):
                shutil.rmtree(logdir)
            os.makedirs(logdir)
            TempfileManager.tempdir = logdir
        # Silencing output by redirecting stdout and stderr
        with utils.capture_output() as out:
            results = opt.solve(instance)
        if save_json:
            with open(save_json, 'w') as f:
                json.dump(results.json_repn(), f, indent=4)
        self.instance = instance
        self.opt = opt
        self.results = results
        self.pyomo_output = out

    def get_var(self, var, dims=None):
        """Return output for variable `var` as a series, dataframe or panel

        Args:
            var : variable name as string, e.g. 'e'
            dims : list of indices as strings, e.g. ('y', 'x', 't');
                   if not given, they are auto-detected
        """
        # TODO currently assumes fixed position of 't' (always last)
        var = getattr(self.m, var)
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
            t = getattr(self.m, 't')
            new_index = self.data._dt.loc[t.first():t.last()].tolist()
            if len(dims) <= 2:
                result.index = new_index
            else:
                result.major_axis = new_index
        return result

    def get_system_variables(self):
        e = self.get_var('e')
        e = e.sum(axis=3).transpose(0, 2, 1)
        return e

    def get_node_variables(self):
        detail = ['s', 'rs']  # TODO removed 'rsecs', 'os'
        p = pd.Panel4D({v: self.get_var(v) for v in detail})
        detail_carrier = ['es_prod', 'es_con', 'e']
        for d in detail_carrier:
            temp = self.get_var(d)
            for c in self.data._c:
                p[d + ':' + c] = temp.loc[c, :, :, :]
        return p

    def get_node_parameters(self, built_only=False):
        """If built_only is True, disregard locations where e_cap==0"""
        detail = ['s_cap', 'r_cap', 'r_area', 'e_cap']
        result = pd.Panel({v: self.get_var(v) for v in detail})
        if built_only:
            result = result.to_frame()
            result = result[result.e_cap > 0].dropna(axis=1)
        return result

    def get_costs(self):
        """

        Get costs

        NB: Currently only counts e_prod towards costs, and only reports
        costs for the carrier ``power``!

        """
        # Levelized cost of electricity (LCOE)
        # TODO currently counting only e_prod for costs, makes sense?
        cost = self.get_var('cost').loc[:, 'monetary', :]
        e_prod = self.get_var('e_prod')
        # TODO ugly hack to limit to power
        e_prod = e_prod.loc['power', :, :, :].sum(axis='major')  # sum over t
        lcoe = cost / e_prod
        lcoe[np.isinf(lcoe)] = 0
        lcoe_total_zones = cost.sum(0) / e_prod.sum(0)
        lcoe = lcoe.append(pd.DataFrame(lcoe_total_zones.to_dict(), index=['total']))
        lcoe_total_techs = cost.sum(1) / e_prod.sum(1)
        lcoe['total'] = lcoe_total_techs
        lcoe.at['total', 'total'] = cost.sum().sum() / e_prod.sum().sum()
        # Capacity factor (CF)
        # NB: using .abs() to sum absolute values of e so cf always positive
        e = self.get_var('e')
        # TODO ugly hack to limit to power
        e = e.loc['power', :, :, :].abs().sum(axis='major')
        e_cap = self.get_var('e_cap')
        cf = e / (e_cap * sum(self.m.time_res[t] for t in self.m.t))
        cf = cf.fillna(0)
        time = sum(self.m.time_res[t] for t in self.m.t)
        cf_total_zones = e.sum(0) / (e_cap.sum(0) * time)
        cf_total_techs = e.sum(1) / (e_cap.sum(1) * time)
        cf = cf.append(pd.DataFrame(cf_total_zones.to_dict(), index=['total']))
        cf['total'] = cf_total_techs
        cf.at['total', 'total'] = e.sum().sum() / (e_cap.sum().sum() * time)
        # Combine everything
        df = pd.Panel({'lcoe': lcoe, 'cf': cf})
        return df

    def _costs_iterative(self, cost_vars, cost_weights, node_vars):
        def get_cf_i(i):
            return cost_vars[i]['cf'] * cost_weights[i]

        def get_lcoe_i(i):
            return cost_vars[i]['lcoe'] * node_vars[i]['e:power'].sum(axis=1)

        # Capacity factor (cf)
        for i in range(len(cost_weights)):
            if i == 0:
                cf = get_cf_i(i)
            else:
                cf = cf + get_cf_i(i)
        cf = cf / sum(cost_weights)
        # LCOE
        power = {}
        for i in range(len(cost_weights)):
            power[i] = node_vars[i]['e:power'].sum(axis=1)
            if i == 0:
                lcoe = get_lcoe_i(i)
            else:
                lcoe = lcoe + get_lcoe_i(i)
        power = pd.Panel(power).sum(axis=0)
        lcoe = lcoe / power
        return pd.Panel({'lcoe': lcoe, 'cf': cf})

    def solve_iterative(self):
        """
        Returns None on success, storing results under
        self.iterative_solution

        """
        o = self.config_model
        d = self.data
        window_adj = int(o.opmode.window / d.time_res_data)
        steps = [t for t in d._t if (t % window_adj) == 0]
        # Remove the last step - since we look forward at each step,
        # it would take us beyond actually existing data
        steps = steps[:-1]
        system_vars = []
        node_vars = []
        cost_vars = []
        cost_weights = []
        for index, step in enumerate(steps):
            self.generate_model(t_start=step)
            self.solve()
            self.load_results()
            # Gather relevant model results over decision interval, so
            # we only grab [0:window/time_res] steps!
            panel = self.get_system_variables()
            if index == (len(steps) - 1):
                # Final iteration saves data from entire horizon
                target_index = int(o.opmode.horizon / d.time_res_static)
            else:
                # Non-final iterations only save data from window
                target_index = int(o.opmode.window / d.time_res_static)
            system_vars.append(panel.iloc[:, 0:target_index, :])
            # Get node variables
            panel4d = self.get_node_variables()
            node_vars.append(panel4d.iloc[:, :, 0:target_index, :])
            # Get cost
            cost = self.get_costs()
            cost_vars.append(cost)
            cost_weights.append(target_index)
            # Save state of storage for carry over to next iteration
            s = self.get_var('s')
            # NB: -1 to convert from length --> index
            storage_state_index = target_index - 1
            d.s_init = s.iloc[:, storage_state_index, :]
        # self.operate_results = {'system': system_vars,
        #                         'node': node_vars,
        #                         'cost': cost,
        #                         'cost_vars': cost_vars,
        #                         'cost_weights': cost_weights}
        costs = self._costs_iterative(cost_vars, cost_weights, node_vars)
        self.operate_results = {'system': pd.concat(system_vars, axis=1),
                                'node': pd.concat(node_vars, axis=2),
                                'costs': costs}

    def load_results(self):
        """Load results into model instance for access via model variables."""
        r = self.instance.load(self.results)
        if r is False:
            raise UserWarning('Could not load results into model instance.')

    def save_outputs(self, carrier='power'):
        """Save model outputs as CSV to ``self.config_run.output.path``"""
        locations = self.data.locations
        if self.mode == 'plan':
            system_variables = self.get_system_variables()
            node_variables = self.get_node_variables()
            costs = self.get_costs()
        elif self.mode == 'operate':
            system_variables = self.iterative_solution['system']
            node_variables = self.iterative_solution['node']
            costs = self.iterative_solution['costs']
        node_parameters = self.get_node_parameters(built_only=False)
        output_files = {'locations.csv': locations,
                        'system_variables.csv': system_variables[carrier],
                        'node_parameters.csv': node_parameters.to_frame(),
                        'costs_lcoe.csv': costs.lcoe,
                        'costs_cf.csv': costs.cf}
        for var in node_variables.labels:
            k = 'node_variables_{}.csv'.format(var.replace(':', '_'))
            v = node_variables[var].to_frame()
            output_files[k] = v
        # Create output dir, but ignore if it already exists
        try:
            os.makedirs(self.config_run.output.path)
        except OSError:  # Hoping this isn't raised for more serious stuff
            pass
        # Write all files to output dir
        for k, v in output_files.iteritems():
            v.to_csv(os.path.join(self.config_run.output.path, k))
