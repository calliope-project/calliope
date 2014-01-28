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

import coopr.opt as co
import coopr.pyomo as cp
import numpy as np
import pandas as pd
from pyutilib.services import TempfileManager

from . import constraints
from . import locations
from . import techs
from . import transmission
from . import utils


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
        self.initialize_techs()
        self.read_data()
        self.mode = self.config_run.mode

    def _load_config(self, path):
        # Deal with possible import statement in loaded config file
        # Additional files specified by import will be merged into
        # the loaded config file in order of appearance
        loaded_config = utils.AttrDict.from_yaml(path)
        if 'import' in loaded_config:
            for k in loaded_config['import']:
                k = utils.replace(k, placeholder='module',
                                  replacement=os.path.dirname(__file__))
                if not os.path.isabs(k):
                    k = os.path.join(os.path.dirname(path), k)
                sub_config = self._load_config(k)
                loaded_config.union(sub_config)
            # Remove 'import' key from loaded_config, no longer need it
            loaded_config.pop('import', None)
        return loaded_config

    def initialize_configuration(self, config_run, override):
        # Load run configuration
        config_path = os.path.join(os.path.dirname(__file__), 'config')
        if not config_run:
            config_run = os.path.join(config_path, 'run.yaml')
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
        # Get all 'input.' keys
        input_keys = cr.input.keys()
        # Expand {{ module }} placeholder
        for p in ['input.' + k for k in input_keys]:
            cr.set_key(p, utils.replace(cr.get_key(p),
                       placeholder='module',
                       replacement=os.path.dirname(__file__)))
        # Load all model config files and combine them into one AttrDict
        o = utils.AttrDict.from_yaml(os.path.join(config_path,
                                                  'defaults.yaml'))
        for path in [cr.get_key('input.' + k) for k in input_keys
                     if 'path' not in k]:
            o.union(self._load_config(path))
        self.config_model = o
        # Override config_model settings if specified in config_run
        if ('override' in cr
                and isinstance(cr.override, utils.AttrDict)):
            for k in cr.override.keys_nested():
                o.set_key(k, cr.override.get_key(k))

    def initialize_techs(self):
        """Perform any tech-specific setup by instantiating tech classes"""
        self.technologies = utils.AttrDict()
        for t in self.data._y:
            try:
                techname = t.capitalize() + 'Technology'
                self.technologies[t] = getattr(techs, techname)(model=self)
            except AttributeError:
                self.technologies[t] = techs.Technology(model=self, name=t)

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

    # @utils.memoize_instancemethod
    def get_option(self, option, x=None, default=None):
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
                    else:
                        raise KeyError
            return result

        def _get_location_option(key, location):
            # Raises KeyError if the specific _override column does not exist
            result = d.locations.ix[location, '_override.' + key]
            # Also raise KeyError if the result is NaN, i.e. if no
            # location-specific override has been defined
            if not isinstance(result, str) and np.isnan(result):
                raise KeyError
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
        # First set time_res_static across all data
        d.time_res_static = self.get_timeres()
        # From time_res_static, initialize time_res_series
        d.time_res_series = pd.Series(d.time_res_static, index=d._t.tolist())
        # Last index t for which model may still use startup exceptions
        d.startup_time_bounds = d._t[int(o.startup_time / d.time_res_static)]
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
        # Subset of transmission technologies (used below)
        # (not yet added to d._y here)
        d.transmission_y = transmission.get_transmission_techs(o.links)
        # Subset of conversion technologies
        # (already contained in d._y here but separated out as well)
        d.conversion_y = [y for y in d._y
                          if o.techs[y].parent == 'conversion']
        #
        # c: Carriers set
        #
        d._c = set()
        for y in d._y:  # Only add carriers for allowed technologies
            d._c.update([o.techs[y].carrier])
            if 'secondary_carrier' in o.techs[y]:
                d._c.update([o.techs[y].secondary_carrier])
        d._c = list(d._c)
        #
        # x: Locations set
        #
        d._x = locations.get_locations(o.locations)
        if self.config_run.get_key('subset_x', default=False):
            d._x = [x for x in d._x if x in self.config_run.subset_x]
        #
        # Locations settings matrix and transmission technologies
        #
        d.locations = locations.generate_location_matrix(o.locations,
                                                         techs=d._y)
        # For simplicity, only keep the locations that are actually in set `x`
        d.locations = d.locations.ix[d._x, :]
        # Add transmission technologies to y
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

        #
        # Sets
        #
        # Time steps
        if self.mode == 'plan':
            m.t = cp.Set(initialize=d._t, ordered=True)
        elif self.mode == 'operate':
            horizon_adj = int(o.opmode.horizon / d.time_res_static)
            m.t = cp.Set(initialize=d._t[t_start:t_start+horizon_adj],
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
            self.process_outputs()
        elif self.mode == 'operate':
            # TODO this needs to be improved and get same level of
            # functionality as mode == 'plan', at least to save results
            # to disk!
            self.operate_results = self.solve_iterative()
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
            logdir = os.path.join('Logs', self.run_id)
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

    def get_var(self, var, dims):
        """Return output for variable `var` as a series, dataframe or panel

        Args:
            var : variable name as string, e.g. 'e'
            dims : list of indices as strings, e.g. ('y', 'x', 't')
        """
        def _get_value(value):
            try:
                return cp.value(value)
            except ValueError:
                # Catch this for uninitialized values
                return np.nan

        var = getattr(self.m, var)
        dims_strings = dims
        dims = [getattr(self.m, dim) for dim in dims]
        d = self.data
        if len(dims) == 1:
            result = pd.Series([_get_value(var[i])
                                for i in sorted(dims[0].value)])
            idx = dims[0]
            result.index = sorted(idx.value)
        elif dims_strings == ['y', 'x']:
            result = pd.DataFrame(0.0, index=sorted(dims[1].value),
                                  columns=sorted(dims[0].value))
            for i, v in var.iteritems():
                result.loc[i[1], i[0]] = _get_value(v)
        elif len(dims_strings) == 3:
            result = pd.Panel(data=None, items=sorted(dims[0].value),
                              major_axis=sorted(dims[2].value),
                              minor_axis=sorted(dims[1].value))
            for i, v in var.iteritems():
                result.loc[i[0], i[2], i[1]] = _get_value(v)
        elif len(dims_strings) == 4:
            result = pd.Panel4D(data=None, labels=sorted(dims[0].value),
                                items=sorted(dims[1].value),
                                major_axis=sorted(dims[3].value),
                                minor_axis=sorted(dims[2].value))
            for i, v in var.iteritems():
                result.loc[i[0], i[1], i[3], i[2]] = _get_value(v)
        # Nicify time column
        if 't' in dims_strings:
            if len(dims) == 1:
                    result.index = d._dt.loc[idx.first():idx.last()]
            # TODO currently no case where dims=2 and 't' in dims, but
            # that may change in the future!
            else:
                if len(dims) == 3:
                    axis_atrributes = ['item', 'minor_axis', 'major_axis']
                elif len(dims) == 4:
                    axis_atrributes = ['labels', 'item', 'minor_axis',
                                       'major_axis']
                t_loc = [i for i, x in enumerate(dims_strings) if x == 't'][0]
                new_index = d._dt.loc[dims[t_loc].first():dims[t_loc].last()]
                setattr(result, axis_atrributes[t_loc], new_index.tolist())
        return result

    def get_system_variables(self):
        e = self.get_var('e', ['c', 'y', 'x', 't'])
        e = e.sum(axis=3).transpose(0, 2, 1)
        return e

    def get_node_variables(self):
        detail = ['s', 'rs', 'rsecs', 'os']
        p = pd.Panel4D({v: self.get_var(v, ['y', 'x', 't']) for v in detail})
        detail_carrier = ['es_prod', 'es_con', 'e']
        for d in detail_carrier:
            temp = self.get_var(d, ['c', 'y', 'x', 't'])
            for c in self.data._c:
                p[d + ':' + c] = temp.loc[c, :, :, :]
        return p

    def get_node_parameters(self, built_only=False):
        """If built_only is True, disregard locations where e_cap==0"""
        detail = ['s_cap', 'r_cap', 'r_area', 'e_cap']
        result = pd.Panel({v: self.get_var(v, ['y', 'x']) for v in detail})
        if built_only:
            result = result.to_frame()
            result = result[result.e_cap > 0].dropna(axis=1)
        return result

    def get_costs(self):
        """Get costs

        NB: Currently only counts e_prod towards costs, and only reports
        costs for the carrier ``power``!

        """
        # Levelized cost of electricity (LCOE)
        # TODO currently counting only e_prod for costs, makes sense?
        cost = self.get_var('cost', ['y', 'x', 'k']).loc[:, 'monetary', :]
        e_prod = self.get_var('e_prod', ['c', 'y', 'x', 't'])
        # TODO ugly hack to limit to power
        e_prod = e_prod.loc['power', :, :, :].sum(axis='major')  # sum over t
        lcoe = cost / e_prod
        lcoe[np.isinf(lcoe)] = 0
        lcoe_total = cost.sum() / e_prod.sum()
        lcoe = lcoe.append(pd.DataFrame(lcoe_total.to_dict(), index=['total']))
        # Capacity factor (CF)
        # NB: using .abs() to sum absolute values of e so cf always positive
        e = self.get_var('e', ['c', 'y', 'x', 't'])
        # TODO ugly hack to limit to power
        e = e.loc['power', :, :, :].abs().sum(axis='major')
        e_cap = self.get_var('e_cap', ['y', 'x'])
        cf = e / (e_cap * sum(self.m.time_res[t] for t in self.m.t))
        cf = cf.fillna(0)
        cf_total = e.sum() / (e_cap.sum() * sum(self.m.time_res[t]
                                                for t in self.m.t))
        cf = cf.append(pd.DataFrame(cf_total.to_dict(), index=['total']))
        # Combine everything
        df = pd.Panel({'lcoe': lcoe, 'cf': cf})
        return df

    def solve_iterative(self):
        o = self.config_model
        d = self.data
        window_adj = int(o.opmode.window / d.time_res_static)
        steps = [t for t in d._t if (t % window_adj) == 0]
        system_vars = []
        node_vars = []
        for step in steps:
            self.generate_model(t_start=step)
            self.solve()
            self.instance.load(self.results)
            # Gather relevant model results over decision interval, so
            # we only grab [0:window/time_res] steps!
            df = self.get_system_variables()
            system_vars.append(df.iloc[0:window_adj])
            # Get node variables
            panel4d = self.get_node_variables()
            node_vars.append(panel4d)
            # Save stage of storage for carry over to next iteration
            s = self.get_var('s', ['y', 'x', 't'])
            # NB: -1 to convert from length --> index
            storage_state_index = window_adj - 1
            d.s_init = s.iloc[:, storage_state_index, :]
        return (pd.concat(system_vars),
                pd.concat(node_vars, axis=2))

    def process_outputs(self):
        """Load results into model instance for access via model
        variables, and if ``self.config_run.output.save`` is ``True``,
        save outputs to CSV.

        """
        r = self.instance.load(self.results)
        if r is False:
            raise UserWarning('Could not load results into model instance.')
        # Save results to disk if specified in configuration
        if self.config_run.output.save is True:
            self.save_outputs()

    def save_outputs(self):
        """Save model outputs as CSV to ``self.config_run.output.path``"""
        locations = self.data.locations
        system_variables = self.get_system_variables()
        node_parameters = self.get_node_parameters(built_only=False)
        node_variables = self.get_node_variables()
        costs = self.get_costs()
        output_files = {'locations.csv': locations,
                        'system_variables.csv': system_variables,
                        'node_parameters.csv': node_parameters.to_frame(),
                        'costs_lcoe.csv': costs.lcoe,
                        'costs_cf.csv': costs.cf}
        for var in node_variables.labels:
            k = 'node_variables_{}.csv'.format(var)
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
