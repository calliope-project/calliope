from __future__ import print_function
from __future__ import division
import os
import math
import json

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

#
# --- FORMULATE MODEL ---
#

model = cp.ConcreteModel()
m = model

#
# Sets
#

m.t = cp.Set(initialize=list_t, ordered=True)  # Time steps
m.i = cp.Set(initialize=list_i, ordered=True)  # Plant sites

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
    # t_to_start_counting = startup_time / time_res
    # return (m.cost_noncsp_op == m.cost_noncsp_con * cost_noncsp_omfrac
    #         + sum(m.P_noncsp[t] for t in m.t
    #               if m.t.order_dict[t] > t_to_start_counting)
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
        E_stor_minus_one = E_init
    return (m.E_stor[t, i] == E_stor_minus_one + eff_stor * m.Q_sf[t, i]
            + m.Q_bak[t, i] - m.Q_gen[t, i] - m.Q_diss[t, i])


def c_Q_bak_rule(m, t, i):
    # Q_bak is allowed only during the hours within startup_time
    if m.t.order_dict[t] < (startup_time / time_res):
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
    return m.E_built[i] <= E_max


def c_solarfield_built_rule(m, i):
    return m.sf_built[i] <= sf_max


def c_powerblock_built_rule(m, i):
    return m.P_built[i] <= P_max


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


#
# --- INSTANTIATE & SOLVE THE MODEL ---
#

def solve(debug=False, save_json=False):
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
        opt.tempdir = 'logs'
    results = opt.solve(instance)
    # Copy results file to script folder
    # TODO
    # opt.log_file
    if save_json:
        with open(results_file_json, 'w') as f:
            json.dump(results.json_repn(), f, indent=4)
    return instance, opt, results


if __name__ == '__main__':
    instance, opt, results = solve()
