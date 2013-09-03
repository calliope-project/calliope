from __future__ import print_function
from __future__ import division
import os
import math
import json
import datetime

import numpy as np
import pandas as pd
import coopr.opt as co
import coopr.pyomo as cp

from csp_model_settings import *

#
# --- INITIAL SETUP ---
#

# Redefine some parameters based on given options
if use_scale_sf:
    sf_max = scale_sf * P_max
if use_E_time:
    r_temp_amb = 25
    r_temp_op = 590
    tmax = r_temp_op - (r_temp_op - r_temp_amb) * 0.05
    carnot_mod = 1 - math.sqrt((r_temp_amb + 273) / (tmax + 273))
    E_max = E_time * P_max / carnot_mod

# Calculate depreciation coefficient for LEC
dep_csp = ((interest_csp * (1 + interest_csp) ** plant_life)
           / (((1 + interest_csp) ** plant_life) - 1))
dep_noncsp = ((interest_noncsp * (1 + interest_noncsp) ** plant_life)
              / (((1 + interest_noncsp) ** plant_life) - 1))

#
# --- READ INPUT DATA ---
#

table_t = pd.read_csv(os.path.join(path_input, 'datetimes.csv'), header=None)
list_t = [int(t) for t in table_t[0].tolist()]
table_i = pd.read_csv(os.path.join(path_input, 'PlantSet.csv'))
list_i = [int(i) for i in table_i.columns.tolist()]
if i_subset:
    list_i = sorted(i_subset)

DTINDEX = [datetime.datetime.strptime(dt, '%Y-%m-%d %H:%M:%S')
           for dt in table_t[1]]

# Combined efficiency factor in time step t for plant i [1]
table_n_el = pd.read_csv(os.path.join(path_input, 'EfficiencyTable.csv'),
                         index_col=0)
# DNI [W/m2]
table_dni = pd.read_csv(os.path.join(path_input, 'DNITable.csv'), index_col=0)
# Solar field efficiency [1]
table_n_sf = pd.read_csv(os.path.join(path_input,
                         'SolarfieldEfficiencyTable.csv'), index_col=0)
# Aggregate power demand [kWh]
table_D = D_scale * pd.read_csv(os.path.join(path_input, demand_filename),
                                index_col=0, header=None)

# Columns: str -> int
for table in [table_n_el, table_dni, table_n_sf, table_D]:
    table.columns = [int(c) for c in table.columns]

D_max = float(table_D.max())  # Maximum demand over all timesteps

# last value of index t for which model may still use startup exceptions
startup_time_bounds = list_t[(startup_time // time_res)]


#
# --- FORMULATE MODEL ---
#

def generate_model(mode='plan', t_start=None,
                   horizon=None, E_init=E_init):
    """
    Args:
        mode : 'plan' or 'operate'
        t_start : must be specified for mode=='operate'
        horizon : must be specified for mode=='operate'
        E_init : E_init vector, useful for mode=='operate'

    """
    m = cp.ConcreteModel()

    if not np.iterable(E_init):
        # Set it to default for all i
        E_init = pd.Series(E_init, index=list_i)

    #
    # Sets
    #

    # Time steps
    if mode == 'plan':
        m.t = cp.Set(initialize=list_t, ordered=True)
    elif mode == 'operate':
        m.t = cp.Set(initialize=list_t[t_start:t_start+horizon],
                     ordered=True)
    # Sites
    m.i = cp.Set(initialize=list_i, ordered=True)

    #
    # Parameters
    #

    m.n_el = cp.Param(m.t, m.i,
                      initialize=lambda m, t, i: float(table_n_el.loc[t, i]))
    m.dni = cp.Param(m.t, m.i,
                     initialize=lambda m, t, i: float(table_dni.loc[t, i]))
    m.n_sf = cp.Param(m.t, m.i,
                      initialize=lambda m, t, i: float(table_n_sf.loc[t, i]))
    # table_D is a pandas dataframe with one column, named 1
    m.D = cp.Param(m.t, initialize=lambda m, t: float(table_D.loc[t, 1]))

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
        return (lambda_csp * sum(m.cost_csp[i] for i in m.i)
                + lambda_noncsp * m.cost_noncsp + lambda_slack * m.cost_slack)

    m.obj = cp.Objective(sense=cp.minimize)
    #m.obj.domain = cp.NonNegativeReals

    #
    # Constraints
    #

    def c_cost_csp_rule(m, i):
        return m.cost_csp[i] == m.cost_csp_con[i] + m.cost_csp_op[i]

    def c_cost_csp_con_rule(m, i):
        return (m.cost_csp_con[i] == dep_csp * ((len(m.t) * time_res) / 8760)
                * (cost_csp_stor * m.E_built[i]
                + (cost_csp_sf + csp_rec_per_sf * cost_csp_rec) * m.sf_built[i]
                + cost_csp_pb * m.P_built[i]))

    def c_cost_csp_op_rule(m, i):
        return (m.cost_csp_op[i] == m.cost_csp_con[i] * cost_csp_omfrac
                + (cost_csp_omvar * sum(m.P[t, i] for t in m.t)))
        # If hybridization allowed:
        # Also add + (cost_burner * sum(m.Q_bak[t, i] for t in m.t))

    def c_cost_noncsp_rule(m):
        return m.cost_noncsp == (m.cost_noncsp_con + m.cost_noncsp_op)

    def c_cost_noncsp_con_rule(m):
        return (m.cost_noncsp_con == dep_noncsp * ((len(m.t) * time_res) / 8760)
                * m.P_noncsp_built * cost_noncsp_build)

    def c_cost_noncsp_op_rule(m):
        # return (m.cost_noncsp_op == m.cost_noncsp_con * cost_noncsp_omfrac
        #         + sum(m.P_noncsp[t] for t in m.t
        #               if t > startup_time_bounds)
        #         * cost_noncsp_fuel)
        return (m.cost_noncsp_op == m.cost_noncsp_con * cost_noncsp_omfrac
                + sum(m.P_noncsp[t] for t in m.t) * cost_noncsp_fuel)

    def c_cost_slack_rule(m):
        return m.cost_slack == sum(m.P_slack[t] for t in m.t)

    def c_P_rule(m, t, i):
        return m.P[t, i] == m.Q_gen[t, i] * m.n_el[t, i]

    def c_Q_sf_rule(m, t, i):
        return m.Q_sf[t, i] == m.dni[t, i] * 0.001 * m.sf_built[i] * m.n_sf[t, i]

    def c_Q_balance_rule(m, t, i):
        if m.t.order_dict[t] > 1:
            E_stor_minus_one = ((1 - mu_stor) ** time_res) * m.E_stor[t - 1, i]
        else:
            E_stor_minus_one = E_init[i]
        return (m.E_stor[t, i] == E_stor_minus_one + eff_stor * m.Q_sf[t, i]
                + m.Q_bak[t, i] - m.Q_gen[t, i] - m.Q_diss[t, i])

    def c_Q_bak_rule(m, t, i):
        # Q_bak is allowed only during the hours within startup_time
        if t < startup_time_bounds:
            return m.Q_bak[t, i] <= (time_res * m.P_built[i]) / m.n_el[t, i]
        else:
            return m.Q_bak[t, i] == 0

    def c_pwr_max_rule(m, t, i):
        return m.P[t, i] <= time_res * m.P_built[i]

    def c_storage_max_rule(m, t, i):
        return m.E_stor[t, i] <= m.E_built[i]

    def c_noncsp_rule(m, t):
        return m.P_noncsp[t] <= time_res * m.P_noncsp_built

    def c_fleet_rule(m, t):
        return (sum(m.P[t, i] for i in m.i) + m.P_noncsp[t]
                + m.P_slack[t] >= m.D[t])

    def c_csp_hourly_rule(m, t):
        return (sum(m.P[t, i] for i in m.i) + m.P_slack[t]
                >= m.D[t] - (noncsp_avail * D_max))

    def c_storage_built_rule(m, i):
        if mode == 'plan':
            return m.E_built[i] <= E_max
        elif mode == 'operate':
            return m.E_built[i] == E_max

    def c_solarfield_built_rule(m, i):
        if mode == 'plan':
            return m.sf_built[i] <= sf_max
        elif mode == 'operate':
            return m.sf_built[i] == sf_max

    def c_powerblock_built_rule(m, i):
        if mode == 'plan':
            return m.P_built[i] <= P_max
        elif mode == 'operate':
            return m.P_built[i] == P_max

    def c_noncsp_built_rule(m):
        P_noncsp_max = noncsp_avail * (D_max / time_res)
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

    return m


#
# --- INSTANTIATE & SOLVE THE MODEL ---
#

def solve(m, debug=False, save_json=False):
    """
    Args:
        debug : (default False)
        save_json : (default False) Save optimization results to
                    disk as results.json

    Returns:
        (instance, opt, results)

    """
    instance = m.create()
    instance.preprocess()
    opt = co.SolverFactory('cplex')  # could set solver_io='python'
    # opt.options["threads"] = 4
    if debug:
        opt.keepfiles = True
        opt.symbolic_solver_labels = True
        opt.tempdir = 'Logs'
    results = opt.solve(instance)
    # Copy results file to script folder
    # TODO
    # opt.log_file
    if save_json:
        with open(results_file_json, 'w') as f:
            json.dump(results.json_repn(), f, indent=4)
    return instance, opt, results


def get_var(var, dims):
    """Return output for variable `var` as a series or dataframe

    Args:
        dims : list of indices, e.g. (m.t, m.i)
    """
    if len(dims) == 1:
        df = pd.Series([cp.value(var[i]) for i in sorted(dims[0].value)])
        idx = dims[0]
        if idx.name == 't':
            df.index = DTINDEX[idx.first():idx.last() + 1]
        elif idx.name == 'i':
            df.index = sorted(idx.value)
    elif len(dims) == 2:
        df = pd.DataFrame(0, index=sorted(dims[0].value),
                          columns=sorted(dims[1].value))
        for i, v in var.iteritems():
            df.loc[i[0], i[1]] = cp.value(v)
        df.index = DTINDEX[dims[0].first():dims[0].last() + 1]
    return df


def get_aggregate_variables(m):
    D = get_var(m.D, [m.t])
    P_slack = get_var(m.P_slack, [m.t])
    P_noncsp = get_var(m.P_noncsp, [m.t])
    P = get_var(m.P, [m.t, m.i]).sum(1)
    return pd.DataFrame({'D': D, 'P_slack': P_slack,
                        'P_noncsp': P_noncsp, 'P': P})


def get_plantlevel_variables(m):
    detail = {'Q_sf': m.Q_sf,
              'Q_gen': m.Q_gen,
              'Q_bak': m.Q_bak,
              'Q_diss': m.Q_diss,
              'E_stor': m.E_stor,
              'P': m.P}
    return pd.Panel({k: get_var(v, [m.t, m.i])
                    for k, v in detail.iteritems()})


def get_plant_parameters(m, built_only=False):
    """If built_only==True, disregard locations where P_built==0"""
    detail = {'P_built': m.P_built,
              'E_built': m.E_built,
              'sf_built': m.sf_built}
    df = pd.DataFrame({k: get_var(v, [m.i]) for k, v in detail.iteritems()})
    if built_only:
        df = df[df.P_built > 0]
    return df


def get_costs(m):
    # lcoe per plant and total
    cost_csp = get_var(m.cost_csp, [m.i])
    P = get_var(m.P, [m.t, m.i]).sum()  # sum over t
    lcoe = cost_csp / P
    lcoe[np.isinf(lcoe)] = 0
    lcoe_total = cost_csp.sum() / P.sum()
    # cf per plant and total
    P_built = get_var(m.P_built, [m.i])
    cf = P / (P_built * len(DTINDEX) * time_res)
    cf = cf.fillna(0)
    cf_total = P.sum() / (P_built.sum() * len(DTINDEX) * time_res)
    # combine
    df = pd.DataFrame({'lcoe': lcoe, 'cf': cf})
    df = df.append(pd.DataFrame({'lcoe': lcoe_total, 'cf': cf_total},
                                index=['total']))
    # add non-CSP
    try:
        lcoe_noncsp = m.cost_noncsp.value / m.P_noncsp_built.value
    except ZeroDivisionError:
        lcoe_noncsp = 0
    cf_noncsp = (get_var(m.P_noncsp, [m.t]).sum()
                 / (m.P_noncsp_built.value * len(DTINDEX) * time_res))
    if np.isnan(cf_noncsp):
        cf_noncsp = 0
    df = df.append(pd.DataFrame({'lcoe': lcoe_noncsp, 'cf': cf_noncsp},
                                index=['noncsp']))
    return df


def solve_iterative(horizon=48, window=24, E_init=E_init):
    steps = [t for t in list_t if t % (window // time_res) == 0]
    # TODO currently E_init is passed as global, not good!
    aggregates = []
    plantlevels = []
    for step in steps:
        m = generate_model(mode='operate', t_start=step,
                           horizon=horizon // time_res,
                           E_init=E_init)
        instance, opt, results = solve(m)
        instance.load(results)
        # Gather relevant model results over decision interval, so
        # we only grab [0:window/time_res] steps!
        df = get_aggregate_variables(m)
        aggregates.append(df.iloc[0:window // time_res])
        # Get plant variables
        panel = get_plantlevel_variables(m)
        plantlevels.append(panel.iloc[:, 0:window // time_res, :])
        # Get E_stor state at the end of the interval to pass on
        # to the next iteration
        _E_stor = get_var(m.E_stor, [m.t, m.i])
        storage_state_index = (window // time_res) - 1  # -1 for len --> idx
        E_init = _E_stor.iloc[storage_state_index, :]
    return (pd.concat(aggregates), pd.concat(plantlevels, axis=1))


if __name__ == '__main__':
    m = generate_model()
    instance, opt, results = solve(m)
