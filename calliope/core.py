from __future__ import print_function
from __future__ import division

import datetime
import json
import os

import coopr.opt as co
import coopr.pyomo as cp
import numpy as np
import pandas as pd
from pyutilib.services import TempfileManager
import yaml

from . import constraints
from . import techs
from . import utils


class Model(object):
    """
    Calliope -- a multi-scale energy systems (MUSES) model

    Canonical use in an IPython notebook cell:

        import calliope
        model = calliope.Model()
        model.run()

    """
    def __init__(self, config_model=None, config_run=None):
        """
        Args:
            options : override default YAML file containing model settings.
        """
        super(Model, self).__init__()
        # Load settings
        if not config_model:
            config_model = os.path.join(os.path.dirname(__file__),
                                        'model_settings.yaml')
        if not config_run:
            config_run = os.path.join(os.path.dirname(__file__),
                                      'run_settings.yaml')
        self.config_model_file = config_model
        self.config_run_file = config_run
        self.config_model = utils.AttrDict(yaml.load(open(config_model, 'r')))
        self.config_run = utils.AttrDict(yaml.load(open(config_run, 'r')))
        # Override config_model settings if specified in config_run
        if ('override' in self.config_run
                and isinstance(self.config_run.override, utils.AttrDict)):
            for k, v in self.config_run.override.iteritems():
                self.config_model[k] = v
        o = self.config_model  # For easier access
        # Perform any tech-specific setup by instantiating tech classes
        self.technologies = utils.AttrDict()
        for t in o.techs:
            try:
                techname = t.capitalize() + 'Technology'
                self.technologies[t] = techs.getattr(techname)(o=o)
            except AttributeError:
                self.technologies[t] = techs.Technology(o=o, name=t)
        # Calculate depreciation coefficients for LEC
        self.data = utils.AttrDict()
        d = self.data
        d.depreciation = utils.AttrDict()
        for y in o.techs:
            interest = self.get_option('depreciation', y, 'interest')
            plant_life = self.get_option('depreciation', y, 'plant_life')
            dep = ((interest * (1 + interest) ** plant_life)
                   / (((1 + interest) ** plant_life) - 1))
            d.depreciation[y] = dep
        self.read_data()

    # def get_timeres(self):
    #     """Backwards (GAMS) compatible method to get time resolution."""
    #     # TODO: needs updating!
    #     path = self.config_run.input.path
    #     with open(os.path.join(path, 'ModelSettings.gms'), 'r') as f:
    #         gms = f.read()
    #     return float(gms.split('time_resolution')[-1].strip())

    def get_timeres(self):
        """Temporary hack: assume data always has 1.0 time resolution"""
        return 1.0

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

    #@utils.memoize
    def get_option(self, group, tech, option=None, allow_default=True):
        """Retrieves options from model settings for the given tech,
        falling back to the default if the option is not defined for the
        tech.

        If allow_default is False, then the lookup will raise a KeyError
        if no setting exists for the given tech/option combination, if
        it is True, the default will be looked up instead.

        Examples: model.get_option('costs', 'ccgt', 'om_var') or
                  model.get_option('tech_weights', 'csp')

        """
        o = self.config_model
        key = group + '.{}'
        if option:
            key += '.' + option
        try:
            result = o.get_key(key.format(tech))
        except KeyError:
            if allow_default:
                result = o.get_key(key.format('default'))
            else:
                raise KeyError
        # Deal with 'inf' settings
        if result == 'inf':
            # NB: somewhat of a hack because actually returning float('inf')
            # causes strange problems so it is probably not compatible with
            # either Pyomo or CPLEX or both!
            result = 1e15
        return result

    def read_data(self):
        """
        Read input data.

        Note on indexing: if subset_t is set, a subset of the data is
        selected, and all data including their indices are subsetted.
        d._t maps a simple [0, data length] index to the actual t index
        used.

        Data is stored in the `self.data` AttrDict as follows:

            self.data.<parameter> = z

        where z is

        * an AttrDict containing the below items if <parameter> differs
          across different technologies `y`.
        * or directly one of the below if <parameter> is uniform across
          all technologies `y`.

        The data container is one of:

        * a pandas DataFrame if <parameter> is defined over `t` and `x`
        * a pandas Series if <parameter> is defined over `t`
        * a Dict or AttrDict if <parameter> is defined over `x`
        * a single value if <parameter> is uniform across `t` and `x`

        Each <parameter> starting with an _ (underscore) will be made into
        a Pyomo object in the generate_model() method, else it will be accessed
        directly.

        """
        o = self.config_model
        d = self.data
        path = self.config_run.input.path
        #
        # t: Time steps set (read from CSV file)
        #
        table_t = pd.read_csv(os.path.join(path, 'set_t.csv'), header=None)
        table_t.index = [datetime.datetime.strptime(dt, '%Y-%m-%d %H:%M:%S')
                         for dt in table_t[1]]
        if self.config_run.subset_t:
            table_t = table_t.loc[self.config_run.subset_t[0]:
                                  self.config_run.subset_t[1]]
            s = slice(table_t[0][0], table_t[0][-1] + 1)
        else:
            s = slice(None)
        d._t = pd.Series([int(t) for t in table_t[0].tolist()])
        d._dt = pd.Series(table_t.index, index=d._t.tolist())
        # First set time_res_static across all data
        d.time_res_static = self.get_timeres()
        # From time_res_static, initialize time_res_series
        d.time_res_series = pd.Series(d.time_res_static, index=d._t.tolist())
        #
        # x: Locations set (read from CSV file)
        #
        table_x = pd.read_csv(os.path.join(path, 'set_x.csv'))
        d._x = [int(i) for i in table_x.columns.tolist()]
        if self.config_run.get_key('subset_x', default=False):
            d._x = sorted(self.config_run.subset_x)
        #
        # y: Technologies set (read from YAML settings)
        #
        d._y = o.techs
        #
        # Energy resource and efficiencies that may be defined over (x, t)
        # for a given technology y
        params_in_time_and_space = ['r', 'r_eff', 'e_eff']

        # Important -- the order of precendence is as follows:
        #   * custom setting in YAML model_settings (applied over all x, t for
        #     the given technology)
        #   * CSV file with explicit values for all x, t
        #   * default setting in YAML model_settings
        #   * KeyError if not defined anywhere at all
        for i in params_in_time_and_space:
            setattr(d, i, utils.AttrDict())
        for y in d._y:
            for i in params_in_time_and_space:
                # Check if y-i combination doesn't exist in model_settings YAML
                if i in o.constraints[y]:
                    # First try: Custom YAML setting
                    getattr(d, i)[y] = self.get_option('constraints', y, i,
                                                       allow_default=False)
                else:
                    try:
                        # Second try: CSV file
                        # File format e.g. 'csp_r_eff.csv'
                        d_path = os.path.join(path, y + '_' + i + '.csv')
                        # [s] to do time subset if needed
                        # Results in e.g. d.r_eff['csp'] being a dataframe
                        # of efficiencies for each time step t at location x
                        df = pd.read_csv(d_path, index_col=0)[s]
                        # Columns (x) from str to int
                        df.columns = [int(c) for c in df.columns]
                        getattr(d, i)[y] = df
                    except IOError:
                        # Final try: Default YAML setting
                        getattr(d, i)[y] = self.get_option('constraints',
                                                           'default', i)
                        # TODO raise helpful error message if this fails
        #
        # Demand (TODO: replace this with a general node)
        #
        # Aggregate power demand [kWh]
        d.D = pd.read_csv(os.path.join(path, self.config_run.input.demand),
                          index_col=0, header=None)[s]
        # Normalize demand to (0, 1), then multiply by the desired demand peak,
        # scaling this peak according to time_res
        d.D_max = self.config_run.input.D_max
        d.D = (d.D / float(d.D.max())) * d.D_max * d.time_res_static
        # Columns (x): str -> int
        d.D.columns = [int(c) for c in d.D.columns]
        # Last index t for which model may still use startup exceptions
        d.startup_time_bounds = d._t[int(o.startup_time / d.time_res_static)]

    def generate_model(self, mode='plan', t_start=None):
        """
        Generate the model and store it under the property `m`.

        Args:
            mode : 'plan' or 'operate'
            t_start : must be specified for mode=='operate'

        """
        #
        # Setup
        #
        m = cp.ConcreteModel()
        o = self.config_model
        d = self.data
        self.mode = mode

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
        m.x = cp.Set(initialize=d._x, ordered=True)  # Locations
        m.y = cp.Set(initialize=d._y, ordered=True)  # Technologies

        #
        # Parameters
        #
        def param_populator(src):
            """Returns a `getter` function that returns either
            (x, t)-specific values for parameters that define such, or
            always the same static value if only a static value is given.

            """
            # TODO also allow for setting over t or x only?
            def getter(m, y, x, t):
                if isinstance(src[y], pd.core.frame.DataFrame):
                    return float(src[y].loc[t, x])
                else:
                    return float(src[y])
            return getter

        # was dni
        m.r = cp.Param(m.y, m.x, m.t, initialize=param_populator(d.r))
        # was n_sf
        m.r_eff = cp.Param(m.y, m.x, m.t, initialize=param_populator(d.r_eff))
        # was n_el
        m.e_eff = cp.Param(m.y, m.x, m.t, initialize=param_populator(d.e_eff))

        # table_D is a pandas dataframe with one column, named 1
        m.D = cp.Param(m.t, initialize=lambda m, t: float(d.D.loc[t, 1]))
        m.time_res = (cp.Param(m.t,
                      initialize=lambda m, t: float(d.time_res_series.loc[t])))

        #
        # Constraints
        #
        # 1. Required
        constraints.node_energy_balance(m, o, d, self)
        constraints.node_constraints_build(m, o, d, self)
        constraints.node_constraints_operational(m, o, d)
        constraints.node_costs(m, o, d, self)
        constraints.model_slack(m, o, d)
        constraints.model_constraints(m, o, d)

        # 2. Optional
        # (none yet)

        # 3. Objective function
        constraints.model_objective(m, o, d, self)

        # Add Pyomo model object as a property
        self.m = m

    def run(self, save_json=False):
        """Instantiate and solve the model"""
        self.generate_model()  # Generated model is attached to `m` property
        self.solve(save_json)
        self.process_outputs()

    def solve(self, save_json=False):
        """
        Args:
            save_json : (default False) Save optimization results to
                        the given file as JSON.

        Returns:
            (instance, opt, results)

        """
        m = self.m
        instance = m.create()
        instance.preprocess()
        opt = co.SolverFactory('cplex')  # could set solver_io='python'
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
            logid = os.path.splitext(os.path.basename(self.config_run_file))[0]
            logdir = os.path.join('Logs', logid)
            # TODO this should be done during __init__ so it can fail sooner
            os.makedirs(logdir)
            TempfileManager.tempdir = logdir
        # Silencing output by redirecting stdout and stderr
        with utils.capture_output() as out:
            results = opt.solve(instance)
        # Copy results file to script folder
        # TODO
        # opt.log_file
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
        var = getattr(self.m, var)
        dims = [getattr(self.m, dim) for dim in dims]
        d = self.data
        if len(dims) == 1:
            result = pd.Series([cp.value(var[i])
                                for i in sorted(dims[0].value)])
            idx = dims[0]
            if idx.name == 't':
                result.index = d._dt.loc[idx.first():idx.last()]
            elif idx.name == 'x':
                result.index = sorted(idx.value)
        elif [i.name for i in dims] == ['y', 'x']:
            result = pd.DataFrame(0, index=sorted(dims[1].value),
                                  columns=sorted(dims[0].value))
            for i, v in var.iteritems():
                result.loc[i[1], i[0]] = cp.value(v)
        elif [i.name for i in dims] == ['y', 'x', 't']:
            result = pd.Panel(data=None, items=sorted(dims[0].value),
                              major_axis=sorted(dims[2].value),
                              minor_axis=sorted(dims[1].value))
            for i, v in var.iteritems():
                result.loc[i[0], i[2], i[1]] = cp.value(v)
            new_index = d._dt.loc[dims[2].first():dims[2].last()]
            result.major_axis = new_index.tolist()
        return result

    def get_system_variables(self):
        e = self.get_var('e', ['y', 'x', 't'])
        e = e.sum(axis=2)
        e['slack'] = self.get_var('slack', ['t'])
        e['D'] = self.get_var('D', ['t'])  # TODO temp while D still exists
        return e

    def get_node_variables(self):
        detail = ['s', 'rs', 'bs', 'es', 'os', 'e']
        return pd.Panel4D({v: self.get_var(v, ['y', 'x', 't'])
                          for v in detail})

    def get_node_parameters(self, built_only=False):
        """If built_only is True, disregard locations where e_cap==0"""
        detail = ['s_cap', 'r_cap', 'r_area', 'e_cap']
        result = pd.Panel({v: self.get_var(v, ['y', 'x']) for v in detail})
        if built_only:
            result = result.to_frame()
            result = result[result.e_cap > 0].dropna(axis=1)
        return result

    def get_costs(self):
        # Levelized cost of electricity (LCOE)
        cost = self.get_var('cost', ['y', 'x'])
        e = self.get_var('e', ['y', 'x', 't']).sum(axis='major')  # sum over t
        lcoe = cost / e
        lcoe[np.isinf(lcoe)] = 0
        lcoe_total = cost.sum() / e.sum()
        lcoe = lcoe.append(pd.DataFrame(lcoe_total.to_dict(), index=['total']))
        # Capacity factor (CF)
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
        # TODO needs updating for Calliope compatibility
        m = self.m
        d = self.data
        o = self.config_model
        window_adj = int(o.opmode.window / d.time_res_static)
        steps = [t for t in d._t if (t % window_adj) == 0]
        aggregates = []
        plantlevels = []
        for step in steps:
            m = self.generate_model(mode='operate', t_start=step)
            instance, opt, results = self.solve()
            instance.load(results)
            # Gather relevant model results over decision interval, so
            # we only grab [0:window/time_res] steps!
            df = self.get_aggregate_variables()
            aggregates.append(df.iloc[0:window_adj])
            # Get plant variables
            panel = self.get_plantlevel_variables()
            plantlevels.append(panel.iloc[:, 0:window_adj, :])
            # Get E_stor state at the end of the interval to pass on
            # to the next iteration
            _E_stor = self.get_var(m.E_stor, [m.t, m.i])
            # NB: -1 to convert from length --> index
            storage_state_index = window_adj - 1
            self.d.E_init = _E_stor.iloc[storage_state_index, :]
        return (pd.concat(aggregates), pd.concat(plantlevels, axis=1))

    def process_outputs(self):
        # Load results into model instance for access via model variables
        r = self.instance.load(self.results)
        if r is False:
            raise UserWarning('Could not load results into model instance.')
        # Save results to disk if specified in configuration
        if self.config_run.output.save is True:
            # TODO probably doesn't work in Calliope
            system_variables = self.get_system_variables()
            node_parameters = self.get_node_parameters(built_only=False)
            node_variables = self.get_node_variables().to_frame()
            costs = self.get_costs().to_frame()
            output_files = {'system_variables.csv': system_variables,
                            'node_parameters.csv': node_parameters,
                            'node_variables.csv': node_variables,
                            'costs.csv': costs}
            # Create output dir, but ignore if it already exists
            try:
                os.makedirs(self.config_run.output.path)
            except OSError:  # Hoping this isn't raised for more serious stuff
                pass
            # Write all files to output dir
            for k, v in output_files.iteritems():
                v.to_csv(os.path.join(self.config_run.output.path, k))
