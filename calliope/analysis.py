"""
Copyright (C) 2013-2015 Stefan Pfenninger.
Licensed under the Apache 2.0 License (see LICENSE file).

analysis.py
~~~~~~~~~~~

Functionality to analyze model results.

"""

import itertools
import logging

try:
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
except ImportError:
    logging.debug('Matplotlib could not be imported, '
                  'no plotting will be available.')
import numpy as np
import pandas as pd

from . import utils
from . import analysis_utils as au


def plot_carrier_production(solution, carrier='power', subset_t=None,
                            **kwargs):
    """
    Generate a stackplot of the production by the given ``carrier``.

    Parameters
    ----------
    solution : solution container
    carrier : str, default 'power'
    subset_t : 2-tuple of str or None, default None
        Specify a date subset for which to plot, for example,
        ('2005-02-01', '2005-02-10') , or ('2005-02-01', None)
    **kwargs : passed to ``plot_timeseries`` (see its documentation)

    """
    data = solution.node['e:{}'.format(carrier)].sum(axis=2)
    if subset_t:
        data = data.loc[slice(*subset_t), :]
    plot_timeseries(solution, data, carrier=carrier, **kwargs)


def plot_timeseries(solution, data, carrier='power', demand='demand_power',
                    types=['supply', 'conversion', 'storage', 'unmet_demand'],
                    colormap=None, ticks=None):
    """
    Generate a stackplot of the ``data`` for the given ``carrier``,
    plotting the ``demand`` on top.

    Use ``plot_carrier_production`` for a simpler way to plot production by
    a given carrier.

    Parameters
    ----------
    solution : solution container
    data : pandas DataFrame
        subset of solution to plot
    carrier : str, default 'power'
        name of the carrier to plot
    demand : str, default 'demand_power'
        name of a demand tech whose time series to plot on top
    types : list, default ['supply', 'conversion', 'storage', 'unmet_demand']
        Technology types to include in the plot.
    colormap : matplotlib colormap
        Colormap to use, if not given, the colors specified for each
        technology in the solution's metadata are used.
    ticks : str, default None
        Where to draw x-axis (time axis) ticks. By default (None),
        auto-detects, but can manually set to either 'hourly', 'daily',
        or 'monthly'.

    """
    # Determine ticks
    if not ticks:
        timespan = (data.index[-1] - data.index[0]).days
        if timespan <= 2:
            ticks = 'hourly'
        elif timespan < 14:
            ticks = 'daily'
        else:
            ticks = 'monthly'
    # Set up time series to plot, dividing it by time_res_series
    time_res = solution.time_res
    plot_df = data.divide(time_res, axis='index')
    # Get tech stack and names
    df = solution.metadata[solution.metadata.carrier == carrier]
    query_string = au._get_query_string(types)
    stacked_techs = df.query(query_string).index.tolist()
    # Put stack in order according to stack_weights
    weighted = df.stack_weight.order(ascending=False).index.tolist()
    stacked_techs = [y for y in weighted if y in stacked_techs]
    names = [df.at[y, 'name'] for y in stacked_techs]
    # If no colormap given, derive one from colors given in metadata
    if not colormap:
        colors = [solution.metadata.at[i, 'color'] for i in stacked_techs]
        colormap = ListedColormap(colors)
    # Plot!
    ax = au.stack_plot(plot_df, stacked_techs, colormap=colormap,
                       alpha=0.9, ticks=ticks, legend='right', names=names)
    ax.plot(plot_df[demand].index,
            plot_df[demand] * -1,
            color='black', lw=1, ls='-')
    return ax


def plot_installed_capacities(solution,
                              types=['supply', 'conversion', 'storage'],
                              unit_multiplier=1.0,
                              unit_label='kW',
                              **kwargs):
    """
    Plot installed capacities (``e_cap``) with a bar plot.

    Parameters
    ----------
    types : list, default ['supply', 'conversion', 'storage']
        Technology types to include in the plot.
    unit_multiplier : float or int, default 1.0
        Multiply installed capacities by this value for plotting.
    unit_label : str, default 'kW'
        Label for capacity values, adjust this when
        changing ``unit_multiplier``.
    **kwargs : are passed to ``pandas.DataFrame.plot()``

    """
    query_string = au._get_query_string(types)
    supply_cap = solution.metadata.query(query_string).index.tolist()

    df = solution.parameters.e_cap.loc[:, supply_cap]

    weighted = solution.metadata.stack_weight.order(ascending=False).index.tolist()
    stacked_techs = [y for y in weighted if y in df.columns]

    df = df.loc[:, stacked_techs] * unit_multiplier

    names = [solution.metadata.at[y, 'name'] for y in df.columns]
    colors = [solution.metadata.at[i, 'color'] for i in df.columns]
    colormap = ListedColormap(colors)
    proxies = [plt.Rectangle((0, 0), 1, 1, fc=i)
               for i in colors]

    # Order the locations nicely, but only take those locations that actually
    # exists in the current solution
    if ('metadata' in solution.config_model and
            'location_ordering' in solution.config_model.metadata):
        meta_config = solution.config_model.metadata
        for index, item in enumerate(meta_config.location_ordering):
            if item in df.index:
                df.at[item, 'ordering'] = index
        df = df.sort('ordering', ascending=False)
        df = df.drop('ordering', axis=1)

    ax = df.plot(kind='barh', stacked=True, legend=False, colormap=colormap,
                 **kwargs)
    leg = au.legend_on_right(ax, style='custom', artists=proxies, labels=names)

    ylab = ax.set_ylabel('')
    xlab = ax.set_xlabel('Installed capacity ({})'.format(unit_label))

    return ax


def plot_transmission(solution, tech='hvac', carrier='power',
                      labels='utilization',
                      figsize=(15, 15), fontsize=9):
    """
    Plot transmission links on a map. Requires that model metadata have
    been defined with a lat/lon for each model location and a boundary for
    the map display.

    Requires Basemap and NetworkX to be installed.

    Parameters
    ----------
    solution : solution container
    tech : str, default 'hvac'
        Which transmission technology to plot.
    carrier : str, default 'power'
        Which carrier to plot transmission for.
    labels : str, default 'utilization'
        Determines how transmission links are labeled, either
        `transmission` or `utilization`.
    figsize : (int, int), default (15, 15)
        Size of resulting figure.
    fontsize : int, default 9
        Font size of figure labels.

    """
    import networkx as nx

    # Determine maximum that could have been transmitted across a link
    def get_edge_capacity(solution, a, b):
        hrs = solution.time_res.sum()
        cap = solution.parameters.at['e_cap_net', a, '{}:'.format(tech) + b] * hrs
        return cap

    # Get annual power transmission between zones
    zones = sorted(solution.node.minor_axis.tolist())
    trans_tech = lambda x: '{}:{}'.format(tech, x)
    df = pd.DataFrame({zone: solution.node.loc['e:{}'.format(carrier),
                                               trans_tech(zone), :, :].sum()
                      for zone in zones})

    # Set smaller than zero to zero --
    # we only want to know about 'production' from
    # transmission, not their consumptions
    df[df < 0] = 0

    # Create directed graph
    G = nx.from_numpy_matrix(df.as_matrix().T, create_using=nx.DiGraph())
    G = nx.relabel_nodes(G, dict(list(zip(list(range(len(zones))), zones))))

    # Transmission
    edge_transmission = {edge: int(round(df.at[edge[1], edge[0]] / 1e6))
                         for edge in G.edges()}

    # Utilization ratio
    edge_use = {(a, b): (df.at[a, b] + df.at[b, a])
                / get_edge_capacity(solution, a, b)
                for (a, b) in G.edges()}

    # Set edge labels
    if labels == 'utilization':
        edge_labels = {k: '{:.2f}'.format(v) for k, v in edge_use.items()}
    elif labels == 'transmission':
        edge_labels = edge_transmission

    # Set edge colors
    edge_colors = [edge_use[i] for i in G.edges()]

    ax, m = au.plot_graph_on_map(solution.config_model,
                                 edge_colors, edge_labels,
                                 figsize, fontsize)

    return ax


def get_delivered_cost(solution, cost_class='monetary', carrier='power',
                       count_unmet_demand=False, unit_multiplier=1.0):
    """
    Get the levelized cost per unit of energy delivered for the given
    ``cost_class`` and ``carrier``.

    Parameters
    ----------
    solution : solution container
    cost_class : str, default 'monetary'
    carrier : str, default 'power'
    count_unmet_demand : bool, default False
        Whether to count the cost of unmet demand in the final
        delivered cost
    unit_multiplier : float or int, default 1.0
        Adjust unit of the returned cost value. For example, if model units
        are kW and kWh, ``unit_multiplier=1.0`` will return cost per kWh, and
        ``unit_multiplier=0.001`` will return cost per MWh.

    """
    summary = solution.summary
    meta = solution.metadata
    carrier_subset = meta[meta.carrier == carrier].index.tolist()
    if count_unmet_demand is False:
        carrier_subset.remove('unmet_demand_' + carrier)
    cost = solution.costs.loc[cost_class, :, carrier_subset].sum().sum()
    # Actually, met_demand also includes demand "met" by unmet_demand
    met_demand = summary.at['demand_' + carrier, 'e_con']
    try:
        unmet_demand = summary.at['unmet_demand_' + carrier, 'e_con']
    except KeyError:
        unmet_demand = 0
    if count_unmet_demand is False:
        demand = met_demand + unmet_demand  # unmet_demand is positive, add it
    else:
        demand = met_demand

    return cost / demand * -1


def get_group_share(solution, techs, group_type='supply',
                    var='e_prod'):
    """
    From ``solution.summary``, get the share of the given list of ``techs``
    from the total for the given ``group_type``, for the given ``var``.

    """
    summary = solution.summary
    meta = solution.metadata
    group = meta.query('type == "' + group_type + '"').index.tolist()
    supply_total = summary.loc[group, var].sum()
    supply_group = summary.loc[techs, var].sum()
    return supply_group / supply_total


def get_supply_groups(solution):
    """
    Get individual supply technologies and those groups that define
    group == True, for purposes of calculating diversity of supply

    """
    # idx_1: group is True and '|' in members
    grp_1 = solution.shares.query('group == True & type == "supply"')
    idx_1 = grp_1[(grp_1.members != grp_1.index)
                  & (grp_1.members.str.contains('\|'))].index.tolist()
    # idx_2: group is False and no '|' in members
    grp_2 = solution.shares.query('group == False & type == "supply"')
    idx_2 = grp_2[grp_2.members == grp_2.index].index.tolist()
    # Also drop entries from idx_2 that are already covered by
    # groups in idx_1
    covered = [i.split('|')
               for i in solution.shares.loc[idx_1, 'members'].tolist()]
    covered_flat = [i for i in itertools.chain.from_iterable(covered)]
    idx_2 = [i for i in idx_2 if i not in covered_flat]
    return idx_1 + idx_2


def get_unmet_demand_hours(solution, carrier='power', details=False):
    """
    Get information about unmet demand from ``solution``.

    Parameters
    ----------
    solution : solution container
    carrier : str, default 'power'
    details : bool, default False
        By default, only the number of hours with unmet are returned. If
        details is True, a dict with 'hours', 'timesteps', and 'dates' keys
        is returned instead.

    """
    unmet = solution.node['e:' + carrier]['unmet_demand_' + carrier].sum(1)
    timesteps = len(unmet[unmet > 0])
    hours = solution.time_res[unmet > 0].sum()
    if details:
        return {'hours': hours, 'timesteps': timesteps,
                'dates': unmet[unmet > 0].index}
    else:
        return hours


def areas_below_resolution(solution, resolution):
    """
    Returns a list of (start, end) timestamp tuples delimiting those
    areas in the solution below the given timestep resolution (in hours).

    """
    selected = solution.time_res[solution.time_res < resolution]
    return list(au._get_ranges(selected.index.tolist()))


def get_swi(solution, shares_var='e_cap', exclude_patterns=['unmet_demand']):
    """
    Returns the Shannon-Wiener diversity index.

    :math:`SWI = -1 \\times \sum_{i=1}^{I} p_{i} \\times \ln(p_{i})`

    where where I is the number of categories and :math:`p_{i}`
    is each category's share of the total (between 0 and 1).

    :math:`SWI` is zero when there is perfect concentration.

    """
    techs = get_supply_groups(solution)
    for pattern in exclude_patterns:
        techs = [t for t in techs if pattern not in t]
    swi = -1 * sum((p * np.log(p))
                   for p in [solution.shares.at[y, shares_var] for y in techs]
                   if p > 0)
    return swi


def get_hhi(solution, shares_var='e_cap', exclude_patterns=['unmet_demand']):
    """
    Returns the Herfindahl-Hirschmann diversity index.

    :math:`HHI = \sum_{i=1}^{I} p_{i}^2`

    where :math:`p_{i}` is the percentage share of each technology i (0-100).

    :math:`HHI` ranges between 0 and 10,000. A value above 1800 is
    considered a sign of a concentrated market.

    """
    techs = get_supply_groups(solution)
    for pattern in exclude_patterns:
        techs = [t for t in techs if pattern not in t]
    hhi = sum((solution.shares.at[y, shares_var] * 100.) ** 2 for y in techs)
    return hhi


def get_domestic_supply_index(solution):
    """
    Assuming that ``solution`` specifies a ``domestic`` cost class to
    give each technology a domesticity score, return the total domestic
    supply index for the given solution.

    """
    idx = solution.metadata.query('type == "supply"').index.tolist()
    dom = (solution.costs.domestic.loc[:, idx].sum().sum() /
           solution.totals.loc['power', 'ec_prod', :, :].sum().sum())
    return dom


class DummyModel(object):
    def __init__(self, solution):
        """
        Create a dummy model object from a model solution,
        which gives direct access to the ``config_model`` AttrDict and
        the ``get_option``, ``get_cost``, and ``get_depreciation``
        methods.

        Furthermore, it provides several ``recompute_`` methods
        to recalculate costs for a given technology after modifying the
        underlying data.

        """
        self.config_model = solution.config_model
        self.solution = solution
        data_keys = [k for k in solution.keys() if 'data/' in k]
        self.data = utils.AttrDict()
        for k in data_keys:
            # Turn 'data/r/pv' into 'r/pv', then turn 'r/pv' into 'r.pv'
            data_k = k.split('/', 1)[1] \
                      .replace('/', '.')
            self.data.set_key(data_k, solution[k])

        # Add getters
        self._get_option = utils.option_getter(self.config_model, solution)
        self._get_cost = utils.cost_getter(self._get_option)
        self._get_dep = utils.depreciation_getter(self._get_option)

    def get_option(self, *args, **kwargs):
        return self._get_option(*args, **kwargs)

    def get_cost(self, *args, **kwargs):
        return self._get_cost(*args, **kwargs)

    def get_depreciation(self, *args, **kwargs):
        return self._get_dep(*args, **kwargs)

    def recompute_investment_costs(self, y, k='monetary',
                                   cost_adjustment=1.0):
        """
        Recompute investment costs for the given technology ``y``.

        ``cost_adjustment`` allows linear scaling of the costs by the
        given amount.

        """
        get_cost = self.get_cost
        solution = self.solution
        cost_con = pd.DataFrame()
        cost_con['e_cap'] = (get_cost('e_cap', y, k)
                             * solution.parameters['e_cap'][y])
        cost_con['r_cap'] = (get_cost('r_cap', y, k)
                             * solution.parameters['r_cap'][y])
        cost_con['r_area'] = (get_cost('r_area', y, k)
                              * solution.parameters['r_area'][y])
        cost_con['rb_cap'] = (get_cost('rb_cap', y, k)
                              * solution.parameters['rb_cap'][y])
        cost_con['s_cap'] = (get_cost('s_cap', y, k)
                             * solution.parameters['s_cap'][y])
        cost_con['total'] = cost_con.sum(axis=1)
        cost_con['total_per_e_cap'] = (cost_con['total']
                                       / solution.parameters['e_cap'][y])
        cost_con.at['total', 'total'] = cost_con['total'].sum()
        total_per_e_cap = (cost_con.at['total', 'total']
                           / solution.parameters['e_cap'][y].sum())
        cost_con.at['total', 'total_per_e_cap'] = total_per_e_cap
        return cost_con * cost_adjustment

    def recompute_operational_costs(self, y, k='monetary', carrier='power',
                                    cost_adjustment=1.0):
        """
        Recompute operational costs for the given technology ``y``.

        ``cost_adjustment`` lineary scales the construction costs used
        internally, but is not applied to the rest of the operational
        costs.

        """
        get_cost = self.get_cost
        get_depreciation = self.get_depreciation
        solution = self.solution
        # Here we want gross production
        production = solution.totals[carrier].es_prod[y]
        fuel = (solution.node.loc['rs', y, :, :].sum()
                / self.get_option(y + '.constraints.r_eff'))
        fuel_rb = (solution.node.loc['rbs', y, :, :].sum()
                   / self.get_option(y + '.constraints.rb_eff'))
        icf = self.recompute_investment_costs
        cost_con = icf(y, k=k, cost_adjustment=cost_adjustment)['total'].sum()
        cost_op = pd.DataFrame()
        cost_op['om_frac'] = (get_cost('om_frac', y, k) * cost_con
                              * get_depreciation(y, k)
                              * (solution.time_res.sum() / 8760))
        cost_op['om_fixed'] = (get_cost('om_fixed', y, k)
                               * solution.parameters['e_cap'][y]
                               * (sum(solution.time_res) / 8760))
        cost_op['om_var'] = get_cost('om_var', y, k) * production
        cost_op['om_fuel'] = get_cost('om_fuel', y, k) * fuel
        cost_op['om_rb'] = get_cost('om_rb', y, k) * fuel_rb
        cost_op['total'] = cost_op.sum(axis=1)
        return cost_op

    def recompute_levelized_costs(self, y, k='monetary', carrier='power',
                                  cost_adjustment=1.0):
        """
        Recompute levelized costs for the given technology ``y``.

        ``cost_adjustment`` lineary scales the construction costs used
        internally to compute levelized costs.

        """
        get_depreciation = self.get_depreciation
        solution = self.solution
        icf = self.recompute_investment_costs
        cost_con = icf(y, k=k, cost_adjustment=cost_adjustment)['total']
        ocf = self.recompute_operational_costs
        cost_op = ocf(y, k=k, carrier=carrier,
                      cost_adjustment=cost_adjustment)['total']
        # Here we want net production
        production = solution.totals[carrier].ec_prod[y]
        lc = (get_depreciation(y, k) * (solution.time_res.sum() / 8760)
              * cost_con + cost_op) / production
        lc.at['total'] = (get_depreciation(y, k)
                          * (solution.time_res.sum() / 8760)
                          * cost_con.total
                          # * cost_con.drop(['total'], axis=0).sum()
                          + cost_op.sum()) / production.sum()
        return lc
