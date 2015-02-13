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


def legend_on_right(ax, style='default', artists=None, labels=None, **kwargs):
    """Draw a legend on outside on the right of the figure given by 'ax'"""
    box = ax.get_position()
    # originally box.width * 0.8 but 1.0 solves some problems
    # it just means that the box becomes wider, which is ok though!
    ax.set_position([box.x0, box.y0, box.width * 1.0, box.height])
    if style == 'square':
        artists, labels = get_square_legend(ax.legend())
        l = ax.legend(artists, labels, loc='center left',
                      bbox_to_anchor=(1, 0.5), **kwargs)
    elif style == 'custom':
        l = ax.legend(artists, labels, loc='center left',
                      bbox_to_anchor=(1, 0.5), **kwargs)
    else:
        l = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), **kwargs)
    return l


def legend_below(ax, style='default', columns=5, artists=None, labels=None,
                 **kwargs):
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])
    if style == 'square':
        artists, labels = get_square_legend(ax.legend())
        l = ax.legend(artists, labels, loc='upper center',
                      bbox_to_anchor=(0.5, -0.05), ncol=columns, **kwargs)
    elif style == 'custom':
        l = ax.legend(artists, labels, loc='upper center',
                      bbox_to_anchor=(0.5, -0.05), ncol=columns, **kwargs)
    else:
        l = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
                      ncol=columns, **kwargs)
    return l


def get_square_legend(lgd):
    rects = [plt.Rectangle((0, 0), 1, 1,
             fc=l.get_color()) for l in lgd.get_lines()]
    labels = [l.get_label() for l in lgd.get_lines()]
    return (rects, labels)


def stack_plot(df, stack=None, figsize=None, colormap='jet', legend='default',
               ticks='daily', names=None, ax=None, leg_title=None,
               leg_fontsize=None, **kwargs):
    """
    if stack is None, the columns of the passed df are used

    legend can be 'default' or 'right', can set legend title with `leg_title`
    ticks can be 'hourly', 'daily', 'monthly'

    kwargs get passed to ax.stackplot()

    """
    if not stack:
        stack = df.columns
    if not ax:
        if not figsize:
            figsize = (16, 4)
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
    colors = plt.get_cmap(colormap)(np.linspace(0, 1.0, len(stack)))
    fills = ax.stackplot(df.index, df[stack].T, label=stack, colors=colors,
                         **kwargs)
    # Rename the tech stack with friendly names, if given, for legend plotting
    if names:
        stack = names
    # Legend via proxy artists
    # Based on https://github.com/matplotlib/matplotlib/issues/1943
    proxies = [plt.Rectangle((0, 0), 1, 1, fc=i.get_facecolor()[0])
               for i in fills]
    if legend == 'default':
        l = ax.legend(reversed(proxies), reversed(stack), title=leg_title)
    elif legend == 'right':
        l = legend_on_right(ax, artists=reversed(proxies),
                            labels=reversed(stack),
                            style='custom', title=leg_title)
    if leg_title and leg_fontsize:
        plt.setp(l.get_title(), fontsize=leg_fontsize)
    # Format x datetime axis
    # Based on http://stackoverflow.com/a/9627970/397746
    import matplotlib.dates as mdates
    if ticks == 'monthly':
        formatter = mdates.DateFormatter('%b %Y')
        locator = mdates.MonthLocator()
    if ticks == 'daily':
        formatter = mdates.DateFormatter('%d-%m-%Y')
        locator = mdates.DayLocator()
    if ticks == 'hourly':
        formatter = mdates.DateFormatter('%H:%M\n%d-%m-%Y')
        locator = mdates.HourLocator(byhour=[0])
        minor_formatter = mdates.DateFormatter('%H:%M')
        minor_locator = mdates.HourLocator(byhour=list(range(1, 24)))
        plt.gca().xaxis.set_minor_formatter(minor_formatter)
        plt.gca().xaxis.set_minor_locator(minor_locator)
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.gca().xaxis.set_major_locator(locator)
    return ax


def _get_query_string(types, additional_types=None):
    query_string = ''
    if additional_types:
        types = types + additional_types
    formatted_types = ['type == "{}"'.format(t) for t in types]
    query_string = ' | '.join(formatted_types)
    return query_string


def plot_solution(solution, data, carrier='power', demand='demand_power',
                  additional_types=None, colormap=None, ticks=None):
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
    query_string = _get_query_string(['supply', 'conversion', 'storage',
                                      'unmet_demand'], additional_types)
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
    ax = stack_plot(plot_df, stacked_techs, colormap=colormap,
                    alpha=0.9, ticks=ticks, legend='right', names=names)
    ax.plot(plot_df[demand].index,
            plot_df[demand] * -1,
            color='black', lw=1, ls='-')
    return ax


def plot_installed_capacities(solution, additional_types=None, **kwargs):
    """
    Arguments:

    additional_types: list of additional technology types to include,
    default is 'supply', 'conversion', 'storage'

    Additional kwargs are passed to pandas.DataFrame.plot()

    """
    query_string = _get_query_string(['supply', 'conversion', 'storage'],
                                     additional_types)
    supply_cap = solution.metadata.query(query_string).index.tolist()

    df = solution.parameters.e_cap.loc[:, supply_cap]

    weighted = solution.metadata.stack_weight.order(ascending=False).index.tolist()
    stacked_techs = [y for y in weighted if y in df.columns]

    df = df.loc[:, stacked_techs] / 1e6

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
    leg = legend_on_right(ax, style='custom', artists=proxies, labels=names)

    ylab = ax.set_ylabel('')
    xlab = ax.set_xlabel('Installed capacity [GW]')

    return ax


def plot_transmission(solution, tech='hvac', carrier='power',
                      labels='utilization',
                      figsize=(15, 15), fontsize=9):
    """
    Plots transmission links on a map.

    `labels` determines how transmission links are labeled,
    can be `transmission` or `utilization`.

    NB: Requires Basemap and NetworkX to be installed.

    """
    from mpl_toolkits.basemap import Basemap
    import networkx as nx
    from calliope.lib import nx_pylab

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

    # Set up basemap
    bounds = solution.config_model.metadata.map_boundary
    bounds_width = bounds[2] - bounds[0]  # lon --> width
    bounds_height = bounds[3] - bounds[1]  # lat --> height
    m = Basemap(projection='merc', ellps='WGS84',
                llcrnrlon=bounds[0], llcrnrlat=bounds[1],
                urcrnrlon=bounds[2], urcrnrlat=bounds[3],
                lat_ts=bounds[1] + bounds_width / 2,
                resolution='i',
                suppress_ticks=True)

    # Node positions
    pos = solution.config_model.metadata.location_coordinates
    pos = {i: m(pos[i][1], pos[i][0]) for i in pos}  # Flip lat, lon to x, y!

    # Create plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, axisbg='w', frame_on=False)
    m.drawmapboundary(fill_color=None, linewidth=0)
    m.drawcoastlines(linewidth=0.2, color='#626262')

    # Draw the graph

    # Using nx_pylab to be able to set zorder below the edges
    nx_pylab.draw_networkx_nodes(G, pos, node_color='#CCCCCC',
                                 node_size=300, zorder=0)

    # Using nx_pylab from lib to get arrow_style option
    nx_pylab.draw_networkx_edges(G, pos, width=3,
                                 edge_color=edge_colors,
                                 # This works for edge_use
                                 edge_vmin=0.0, edge_vmax=1.0,
                                 edge_cmap=plt.get_cmap('seismic'),
                                 arrows=True, arrow_style='->')

    labels = nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels,
                                          rotate=False, font_size=fontsize)

    # Add a map scale
    scale = m.drawmapscale(
        bounds[0] + bounds_width * 0.05, bounds[1] + bounds_height * 0.05,
        bounds[0], bounds[1],
        100,
        barstyle='simple', labelstyle='simple',
        fillcolor1='w', fillcolor2='#555555',
        fontcolor='#555555', fontsize=fontsize
    )

    return ax


def get_delivered_cost(solution, cost_class='monetary', carrier='power',
                       count_unmet_demand=False):
    summary = solution.summary
    meta = solution.metadata
    carrier_subset = meta[meta.carrier == carrier].index.tolist()
    if count_unmet_demand is False:
        carrier_subset.remove('unmet_demand_' + carrier)
    cost = solution.costs.loc[cost_class, :, carrier_subset].sum().sum()
    # Actually, met_demand also includes demand "met" by unmet_demand
    met_demand = summary.at['demand_' + carrier, 'consumption']
    try:
        unmet_demand = summary.at['unmet_demand_' + carrier, 'consumption']
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


def get_unmet_load_hours(solution, carrier='power', details=False):
    unmet = solution.node['e:' + carrier]['unmet_demand_' + carrier].sum(1)
    timesteps = len(unmet[unmet > 0])
    hours = solution.time_res[unmet > 0].sum()
    if details:
        return {'hours': hours, 'timesteps': timesteps,
                'dates': unmet[unmet > 0].index}
    else:
        return hours


def _get_ranges(dates):
    # Modified from http://stackoverflow.com/a/6934267/397746
    while dates:
        end = 1
        timedelta = dates[end] - dates[end - 1]
        try:
            while dates[end] - dates[end - 1] == timedelta:
                end += 1
        except IndexError:
            pass

        yield (dates[0], dates[end - 1])
        dates = dates[end:]


def areas_below_resolution(solution, resolution):
    """
    Returns a list of (start, end) tuples for areas in the solution
    below the given timestep resolution.

    """
    selected = solution.time_res[solution.time_res < resolution]
    return list(_get_ranges(selected.index.tolist()))


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
    idx = solution.metadata.query('type == "supply"').index.tolist()
    dom = (solution.costs.domestic.loc[:, idx].sum().sum() /
           solution.totals.loc['power', 'ec_prod', :, :].sum().sum())
    return dom


def aggregate_parameters(solution, iterations=None, how='max'):
    """
    Get aggregated plant sizes across all planning runs.
    Used for operational runs.

    """
    if not iterations:
        iterations = solution.iterations.index.tolist()
    solution_panel = pd.Panel4D({i: solution.solutions[i].parameters
                                 for i in iterations})
    aggregator = getattr(solution_panel, how)
    return aggregator(axis=0)


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
        cost_con['total_per_kW'] = (cost_con['total']
                                    / solution.parameters['e_cap'][y])
        cost_con.at['total', 'total'] = cost_con['total'].sum()
        total_per_kW = (cost_con.at['total', 'total']
                        / solution.parameters['e_cap'][y].sum())
        cost_con.at['total', 'total_per_kW'] = total_per_kW
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
