from __future__ import print_function
from __future__ import division

import datetime
import json
import math
import os

import coopr.opt as co
import coopr.pyomo as cp
import numpy as np
import pandas as pd
from pyutilib.services import TempfileManager
import yaml

from . import utils


class Lisa(object):
    """
    Large-scale international solar power arrangement (Lisa) model

    Canonical use in an IPython notebook cell:

        model = lisa.Lisa()
        model.run()
        lisa.utils.notify()

    """
    def __init__(self, config_model=None, config_run=None):
        """
        Args:
            options : override default YAML file containing model settings.
        """
        super(Lisa, self).__init__()
        # Load settings
        if not config_model:
            config_model = os.path.join(os.path.dirname(__file__),
                                        'model_settings.yaml')
        if not config_run:
            config_run = os.path.join(os.path.dirname(__file__),
                                      'run_settings.yaml')
        self.config_model = utils.AttrDict(yaml.load(open(config_model, 'r')))
        self.config_run = utils.AttrDict(yaml.load(open(config_run, 'r')))
        # Override config_model settings if specificed in config_run
        for k, v in self.config_run.override.iteritems():
            self.config_model[k] = v
        o = self.config_model  # For easier access
        # Redefine some parameters based on given options
        if o.use_scale_sf:
            o.sf_max = o.scale_sf * o.P_max
        if o.use_E_time:
            r_temp_amb = 25
            r_temp_op = 590
            tmax = r_temp_op - (r_temp_op - r_temp_amb) * 0.05
            carnot_mod = 1 - math.sqrt((r_temp_amb + 273) / (tmax + 273))
            o.E_max = o.E_time * o.P_max / carnot_mod
        # Calculate depreciation coefficient for LEC
        self.data = utils.AttrDict()
        d = self.data
        d.dep_csp = ((o.interest.csp * (1 + o.interest.csp) ** o.plant_life)
                     / (((1 + o.interest.csp) ** o.plant_life) - 1))
        d.dep_noncsp = ((o.interest.noncsp * (1 + o.interest.noncsp)
                        ** o.plant_life)
                        / (((1 + o.interest.noncsp) ** o.plant_life) - 1))
        self.read_data()

    def read_data(self):
        """
        Read input data.

        """
        o = self.config_model
        d = self.data
        path = self.config_run.input.path
        # Read files
        table_t = pd.read_csv(os.path.join(path, 'datetimes.csv'), header=None)
        d._t = [int(t) for t in table_t[0].tolist()]
        table_i = pd.read_csv(os.path.join(path, 'PlantSet.csv'))
        d._i = [int(i) for i in table_i.columns.tolist()]
        if self.config_run.subset_i:
            d._i = sorted(self.config_run.subset_i)
        d._dt = [datetime.datetime.strptime(dt, '%Y-%m-%d %H:%M:%S')
                 for dt in table_t[1]]
        # Combined efficiency factor in time step t for plant i [1]
        d.n_el = pd.read_csv(os.path.join(path, 'EfficiencyTable.csv'),
                             index_col=0)
        # DNI [W/m2]
        d.dni = pd.read_csv(os.path.join(path, 'DNITable.csv'),
                            index_col=0)
        # Solar field efficiency [1]
        d.n_sf = pd.read_csv(os.path.join(path,
                             'SolarfieldEfficiencyTable.csv'), index_col=0)
        # Aggregate power demand [kWh]
        d.D = (self.config_run.input.D_scale *
               pd.read_csv(os.path.join(path, self.config_run.input.demand),
               index_col=0, header=None))
        # Columns: str -> int
        for table in [d.n_el, d.dni, d.n_sf, d.D]:
            table.columns = [int(c) for c in table.columns]
        # Maximum demand over all timesteps
        d.D_max = float(d.D.max())
        # Last index t for which model may still use startup exceptions
        d.startup_time_bounds = d._t[(o.startup_time // o.time_res)]

    def generate_model(self, mode='plan', t_start=None):
        """
        Generate the model and store it under the property `m`.

        Args:
            mode : 'plan' or 'operate'
            t_start : must be specified for mode=='operate'

        """
        m = cp.ConcreteModel()
        o = self.config_model
        d = self.data

        if not np.iterable(o.E_init):
            # Set it to default for all i
            d.E_init = pd.Series(o.E_init, index=d._i)
        else:
            d.E_init = o.E_init

        #
        # Sets
        #

        # Time steps
        if mode == 'plan':
            m.t = cp.Set(initialize=d._t, ordered=True)
        elif mode == 'operate':
            horizon_adj = o.opmode.horizon // o.time_res
            m.t = cp.Set(initialize=d._t[t_start:t_start+horizon_adj],
                         ordered=True)
        # Sites
        m.i = cp.Set(initialize=d._i, ordered=True)

        #
        # Parameters
        #

        m.n_el = cp.Param(m.t, m.i,
                          initialize=lambda m, t, i: float(d.n_el.loc[t, i]))
        m.dni = cp.Param(m.t, m.i,
                         initialize=lambda m, t, i: float(d.dni.loc[t, i]))
        m.n_sf = cp.Param(m.t, m.i,
                          initialize=lambda m, t, i: float(d.n_sf.loc[t, i]))
        # table_D is a pandas dataframe with one column, named 1
        m.D = cp.Param(m.t, initialize=lambda m, t: float(d.D.loc[t, 1]))

        #
        # Variables
        #

        m.cost_csp = cp.Var(m.i, within=cp.NonNegativeReals)
        m.cost_csp_con = cp.Var(m.i, within=cp.NonNegativeReals)
        m.cost_csp_op = cp.Var(m.i, within=cp.NonNegativeReals)
        m.cost_noncsp = cp.Var(within=cp.NonNegativeReals)
        m.cost_noncsp_con = cp.Var(within=cp.NonNegativeReals)
        m.cost_noncsp_op = cp.Var(within=cp.NonNegativeReals)
        m.cost_slack = cp.Var(within=cp.NonNegativeReals)
        m.Q_sf = cp.Var(m.t, m.i, within=cp.NonNegativeReals)  # Solar field heat in
        m.Q_gen = cp.Var(m.t, m.i, within=cp.NonNegativeReals)  # Generator heat out
        m.Q_diss = cp.Var(m.t, m.i, within=cp.NonNegativeReals)  # Dissipated heat out
        m.Q_bak = cp.Var(m.t, m.i, within=cp.NonNegativeReals)  # Backup burner heat in
        m.P = cp.Var(m.t, m.i, within=cp.NonNegativeReals)
        m.P_noncsp = cp.Var(m.t, within=cp.NonNegativeReals)
        m.P_slack = cp.Var(m.t, within=cp.NonNegativeReals)
        m.E_stor = cp.Var(m.t, m.i, within=cp.NonNegativeReals)
        m.E_built = cp.Var(m.i, within=cp.NonNegativeReals)
        m.sf_built = cp.Var(m.i, within=cp.NonNegativeReals)
        m.P_built = cp.Var(m.i, within=cp.NonNegativeReals)
        m.P_noncsp_built = cp.Var(within=cp.NonNegativeReals)

        #
        # Objective
        #

        def obj_rule(m):
            return (o.lambda_csp * sum(m.cost_csp[i] for i in m.i)
                    + o.lambda_noncsp * m.cost_noncsp
                    + o.lambda_slack * m.cost_slack)

        m.obj = cp.Objective(sense=cp.minimize)
        #m.obj.domain = cp.NonNegativeReals

        #
        # Constraints
        #

        def c_cost_csp_rule(m, i):
            return m.cost_csp[i] == m.cost_csp_con[i] + m.cost_csp_op[i]

        def c_cost_csp_con_rule(m, i):
            return (m.cost_csp_con[i] == d.dep_csp
                    * ((len(m.t) * o.time_res) / 8760)
                    * (o.cost.csp_stor * m.E_built[i]
                    + (o.cost.csp_sf + o.csp_rec_per_sf * o.cost.csp_rec)
                    * m.sf_built[i] + o.cost.csp_pb * m.P_built[i]))

        def c_cost_csp_op_rule(m, i):
            return (m.cost_csp_op[i] == m.cost_csp_con[i] * o.cost.csp_omfrac
                    + (o.cost.csp_omvar * sum(m.P[t, i] for t in m.t)))
            # If hybridization allowed:
            # Also add + (cost_burner * sum(m.Q_bak[t, i] for t in m.t))

        def c_cost_noncsp_rule(m):
            return m.cost_noncsp == (m.cost_noncsp_con + m.cost_noncsp_op)

        def c_cost_noncsp_con_rule(m):
            return (m.cost_noncsp_con == d.dep_noncsp
                    * ((len(m.t) * o.time_res) / 8760)
                    * m.P_noncsp_built * o.cost.noncsp_build)

        def c_cost_noncsp_op_rule(m):
            # return (m.cost_noncsp_op == m.cost_noncsp_con * cost_noncsp_omfrac
            #         + sum(m.P_noncsp[t] for t in m.t
            #               if t > startup_time_bounds)
            #         * cost_noncsp_fuel)
            return (m.cost_noncsp_op == m.cost_noncsp_con * o.cost.noncsp_omfrac
                    + sum(m.P_noncsp[t] for t in m.t) * o.cost.noncsp_fuel)

        def c_cost_slack_rule(m):
            return m.cost_slack == sum(m.P_slack[t] for t in m.t)

        def c_P_rule(m, t, i):
            return m.P[t, i] == m.Q_gen[t, i] * m.n_el[t, i]

        def c_Q_sf_rule(m, t, i):
            return (m.Q_sf[t, i] == m.dni[t, i]
                    * 0.001 * m.sf_built[i] * m.n_sf[t, i])

        def c_Q_balance_rule(m, t, i):
            if m.t.order_dict[t] > 1:
                E_stor_minus_one = ((1 - o.mu_stor)
                                    ** o.time_res) * m.E_stor[t - 1, i]
            else:
                E_stor_minus_one = d.E_init[i]
            return (m.E_stor[t, i] == E_stor_minus_one
                    + o.eff_stor * m.Q_sf[t, i] + m.Q_bak[t, i]
                    - m.Q_gen[t, i] - m.Q_diss[t, i])

        def c_Q_bak_rule(m, t, i):
            # Q_bak is allowed only during the hours within startup_time
            if t < d.startup_time_bounds:
                return m.Q_bak[t, i] <= (o.time_res
                                         * m.P_built[i]) / m.n_el[t, i]
            else:
                return m.Q_bak[t, i] == 0

        def c_pwr_max_rule(m, t, i):
            return m.P[t, i] <= o.time_res * m.P_built[i]

        def c_storage_max_rule(m, t, i):
            return m.E_stor[t, i] <= m.E_built[i]

        def c_noncsp_rule(m, t):
            return m.P_noncsp[t] <= o.time_res * m.P_noncsp_built

        def c_fleet_rule(m, t):
            return (sum(m.P[t, i] for i in m.i) + m.P_noncsp[t]
                    + m.P_slack[t] >= m.D[t])

        def c_csp_hourly_rule(m, t):
            return (sum(m.P[t, i] for i in m.i) + m.P_slack[t]
                    >= m.D[t] - (o.noncsp_avail * d.D_max))

        def c_storage_built_rule(m, i):
            if mode == 'plan':
                return m.E_built[i] <= o.E_max
            elif mode == 'operate':
                return m.E_built[i] == o.E_max

        def c_solarfield_built_rule(m, i):
            if mode == 'plan':
                return m.sf_built[i] <= o.sf_max
            elif mode == 'operate':
                return m.sf_built[i] == o.sf_max

        def c_powerblock_built_rule(m, i):
            if mode == 'plan':
                return m.P_built[i] <= o.P_max
            elif mode == 'operate':
                return m.P_built[i] == o.P_max

        def c_noncsp_built_rule(m):
            P_noncsp_max = o.noncsp_avail * (d.D_max / o.time_res)
            return m.P_noncsp_built == P_noncsp_max  # [TEMP] was <=

        # Build the constraints
        m.c_cost_csp = cp.Constraint(m.i)
        m.c_cost_csp_con = cp.Constraint(m.i)
        m.c_cost_csp_op = cp.Constraint(m.i)
        m.c_cost_noncsp = cp.Constraint()
        m.c_cost_noncsp_con = cp.Constraint()
        m.c_cost_noncsp_op = cp.Constraint()
        m.c_cost_slack = cp.Constraint()
        m.c_P = cp.Constraint(m.t, m.i)
        m.c_Q_sf = cp.Constraint(m.t, m.i)
        m.c_Q_balance = cp.Constraint(m.t, m.i)
        m.c_Q_bak = cp.Constraint(m.t, m.i)
        m.c_pwr_max = cp.Constraint(m.t, m.i)
        m.c_storage_max = cp.Constraint(m.t, m.i)
        m.c_noncsp = cp.Constraint(m.t)
        m.c_fleet = cp.Constraint(m.t)
        m.c_csp_hourly = cp.Constraint(m.t)
        m.c_storage_built = cp.Constraint(m.i)
        m.c_solarfield_built = cp.Constraint(m.i)
        m.c_powerblock_built = cp.Constraint(m.i)
        m.c_noncsp_built = cp.Constraint()

        self.m = m

    #
    # --- INSTANTIATE & SOLVE THE MODEL ---
    #

    def run(self):
        self.generate_model()  # model gets attached to model.m property
        self.solve()
        self.process_outputs()

    def solve(self, debug=False, save_json=False):
        """
        Args:
            debug : (default False)
            save_json : (default False) Save optimization results to
                        the given file as JSON.

        Returns:
            (instance, opt, results)

        """
        m = self.m
        instance = m.create()
        instance.preprocess()
        opt = co.SolverFactory('cplex')  # could set solver_io='python'
        # opt.options["threads"] = 4
        if debug:
            opt.keepfiles = True
            opt.symbolic_solver_labels = True
            TempfileManager.tempdir = 'Logs'
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
        """Return output for variable `var` as a series or dataframe

        Args:
            dims : list of indices, e.g. (m.t, m.i)
        """
        d = self.data
        if len(dims) == 1:
            df = pd.Series([cp.value(var[i]) for i in sorted(dims[0].value)])
            idx = dims[0]
            if idx.name == 't':
                df.index = d._dt[idx.first():idx.last() + 1]
            elif idx.name == 'i':
                df.index = sorted(idx.value)
        elif len(dims) == 2:
            df = pd.DataFrame(0, index=sorted(dims[0].value),
                              columns=sorted(dims[1].value))
            for i, v in var.iteritems():
                df.loc[i[0], i[1]] = cp.value(v)
            df.index = d._dt[dims[0].first():dims[0].last() + 1]
        return df

    def get_aggregate_variables(self):
        m = self.m
        D = self.get_var(m.D, [m.t])
        P_slack = self.get_var(m.P_slack, [m.t])
        P_noncsp = self.get_var(m.P_noncsp, [m.t])
        P = self.get_var(m.P, [m.t, m.i]).sum(1)
        return pd.DataFrame({'D': D, 'P_slack': P_slack,
                            'P_noncsp': P_noncsp, 'P': P})

    def get_plantlevel_variables(self):
        m = self.m
        detail = {'Q_sf': m.Q_sf,
                  'Q_gen': m.Q_gen,
                  'Q_bak': m.Q_bak,
                  'Q_diss': m.Q_diss,
                  'E_stor': m.E_stor,
                  'P': m.P}
        return pd.Panel({k: self.get_var(v, [m.t, m.i])
                        for k, v in detail.iteritems()})

    def get_plant_parameters(self, built_only=False):
        """If built_only==True, disregard locations where P_built==0"""
        m = self.m
        detail = {'P_built': m.P_built,
                  'E_built': m.E_built,
                  'sf_built': m.sf_built}
        df = pd.DataFrame({k: self.get_var(v, [m.i])
                          for k, v in detail.iteritems()})
        if built_only:
            df = df[df.P_built > 0]
        return df

    def get_costs(self):
        m = self.m
        d = self.data
        o = self.config_model
        # lcoe per plant and total
        cost_csp = self.get_var(m.cost_csp, [m.i])
        P = self.get_var(m.P, [m.t, m.i]).sum()  # sum over t
        lcoe = cost_csp / P
        lcoe[np.isinf(lcoe)] = 0
        lcoe_total = cost_csp.sum() / P.sum()
        # cf per plant and total
        P_built = self.get_var(m.P_built, [m.i])
        cf = P / (P_built * len(d._dt) * o.time_res)
        cf = cf.fillna(0)
        cf_total = P.sum() / (P_built.sum() * len(d._dt) * o.time_res)
        # combine
        df = pd.DataFrame({'lcoe': lcoe, 'cf': cf})
        df = df.append(pd.DataFrame({'lcoe': lcoe_total, 'cf': cf_total},
                                    index=['total']))
        # add non-CSP
        try:
            lcoe_noncsp = m.cost_noncsp.value / m.P_noncsp_built.value
        except ZeroDivisionError:
            lcoe_noncsp = 0
        cf_noncsp = (self.get_var(m.P_noncsp, [m.t]).sum()
                     / (m.P_noncsp_built.value * len(d._dt) * o.time_res))
        if np.isnan(cf_noncsp):
            cf_noncsp = 0
        df = df.append(pd.DataFrame({'lcoe': lcoe_noncsp, 'cf': cf_noncsp},
                                    index=['noncsp']))
        return df

    def solve_iterative(self):
        m = self.m
        d = self.data
        o = self.config_model
        window_adj = o.opmode.window // o.time_res
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
            overall = self.get_aggregate_variables()
            plant_parameters = self.get_plant_parameters(built_only=False)
            plants_output = self.get_plantlevel_variables().to_frame()
            costs = self.get_costs()
            output_files = {'overall.csv': overall,
                            'plant_parameters.csv': plant_parameters,
                            'plants_output.csv': plants_output,
                            'costs.csv': costs}
            # Create output dir, but ignore if it already exists
            try:
                os.makedirs(self.config_run.output.path)
            except OSError:  # Hoping this isn't raised for more serious stuff
                pass
            # Write all files to output dir
            for k, v in output_files.iteritems():
                v.to_csv(os.path.join(self.config_run.output.path, k))
