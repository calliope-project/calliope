"""
Copyright (C) 2013-2017 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

analysis.py
~~~~~~~~~~~

Functionality to analyze model results.

"""

import logging

try:
    from matplotlib.patches import Rectangle
    from matplotlib.colors import ListedColormap
except ImportError:
    logging.debug('Matplotlib could not be imported, '
                  'no plotting will be available.')
import numpy as np
import pandas as pd

from . import analysis_utils as au


def plot_carrier_production(solution, carrier='power', subset=dict(),
                            **kwargs):
    """
    Generate a stackplot of the production by the given ``carrier``.

    Parameters
    ----------
    solution : model solution xarray.Dataset
    carrier : str, optional
        Name of the carrier to plot, default 'power'.
    subset : dict, optional
        Specify an additional subset of Dataset coordinates, for example,
        dict(t=slice('2005-02-01', '2005-02-10').
    **kwargs : optional
        Passed to ``plot_timeseries``.

    """
    data = solution['e'].loc[dict(c=carrier, **subset)].sum(dim='x')
    return plot_timeseries(solution, data, carrier=carrier, demand='demand_{}'.format(carrier), **kwargs)


def plot_timeseries(
        solution, data, carrier='power', demand='demand_power',
        tech_types=[
            'supply', 'supply_plus', 'conversion',
            'conversion_plus', 'storage', 'unmet_demand'],
        colormap=None, ticks=None,
        resample_options=None, resample_func=None,
        add_legend=True, ax=None):
    """
    Generate a stackplot of ``data`` for the given ``carrier``,
    plotting ``demand`` on top.

    Use ``plot_carrier_production`` for a simpler way to plot production by
    a given carrier.

    Parameters
    ----------
    solution : model solution xarray.Dataset
    data : xarray.Dataset
        Subset of solution to plot.
    carrier : str, optional
        Name of the carrier to plot, default 'power'.
    demand : str, optional
        Name of a demand tech whose time series to plot on top,
        default 'demand_power'.
    tech_types : list, optional
        Technology types to include in the plot. Default list is
        ['supply', 'supply_plus' , 'conversion', 'storage', 'unmet_demand'].
    colormap : matplotlib colormap, optional
        Colormap to use. If not given, the colors specified for each
        technology in the solution's metadata are used.
    ticks : str, optional
        Where to draw x-axis (time axis) ticks. By default (None),
        auto-detects, but can manually set to either 'hourly', 'daily',
        or 'monthly'.
    resample_options : dict, optional
        Give options for pandas.DataFrame.resample in a dict, to resample
        the entire time series prior to plotting. Both resample_options
        and resample_func must be given for resampling to happen.
        Default None.
    resample_func : string, optional
        Give the name of the aggregating function to use when resampling,
        e.g. "mean" or "sum". Default None.

    """
    # Determine ticks
    if not ticks:
        days = (data.coords['t'].values[-1] - data.coords['t'].values[0])
        timespan = days / np.timedelta64(1, 'D')
        # timespan = days.astype('timedelta64[D]').astype(int)
        if timespan <= 2:
            ticks = 'hourly'
        elif timespan < 14:
            ticks = 'daily'
        else:
            ticks = 'monthly'
    # Set up time series to plot, dividing it by time_res_series
    time_res = solution['time_res'].to_pandas()
    data = data.to_pandas().T
    plot_df = data.divide(time_res, axis='index')
    if resample_options and resample_func:
        plot_df = getattr(plot_df.resample(**resample_options), resample_func)()
    df_metadata = solution['metadata'].to_pandas()
    # Get pandas DataFrame based on carrier being plotted
    df = df_metadata.query('carrier_in == "{}" or '
                           'carrier_out == "{}"'.format(carrier, carrier))
    # Specifically add pertinent conversion_plus technologies to df
    if 'conversion_plus' in tech_types:
        df_cp = df_metadata.query('type == "conversion_plus"')
        for y, columns in df_cp.iterrows():
            if carrier in columns.carrier_in or carrier in columns.carrier_out:
                df = df.append(df_metadata.loc[y])
        df.drop_duplicates(inplace=True)
    # Get tech stack and names
    stacked_techs = df_metadata.query('type in {}'.format(tech_types)).index.tolist()
    # Put stack in order according to stack_weights
    weighted = df.stack_weight.sort_values(ascending=False).index.tolist()
    stacked_techs = [y for y in weighted if y in stacked_techs]
    names = [df.at[y, 'name'] for y in stacked_techs]
    # If no colormap given, derive one from colors given in metadata
    if not colormap:
        colors = [df.at[i, 'color'] for i in stacked_techs]
        colormap = ListedColormap(colors)
    # Plot!
    if ax is None:
        ax = au.stack_plot(plot_df, stacked_techs, colormap=colormap,
                           alpha=0.9, ticks=ticks, legend=None, names=names)
    else:
        au.stack_plot(plot_df, stacked_techs, colormap=colormap,
                      alpha=0.9, ticks=ticks, legend=None, names=names,
                      ax=ax)
    ax.plot(plot_df[demand].index,
            plot_df[demand] * -1,
            color='red', lw=1.5, ls='--', label=df.at[demand, 'name'])
    # Add legend here rather than in stack_plot so we get demand too
    if add_legend:
        au.legend_outside_ax(ax, where='right')
    return ax


def plot_installed_capacities(
        solution,
        tech_types=[
            'supply', 'supply_plus', 'conversion',
            'conversion_plus', 'storage'
        ],
        unit_multiplier=1.0,
        unit_label='kW',
        **kwargs):
    """
    Plot installed capacities (``e_cap``) with a bar plot.

    Parameters
    ----------
    solution : model solution xarray.Dataset
    tech_types : list, optional
        Technology types to include in the plot. Default is
        ['supply', 'supply_plus', 'conversion', 'conversion_plus', 'storage']
    unit_multiplier : float or int, optional
        Multiply installed capacities by this value for plotting.
        Defaults to 1.0
    unit_label : str, optional
        Label for capacity values. Default is 'kW', adjust this
        when changing ``unit_multiplier``.
    **kwargs : optional
        are passed to ``pandas.DataFrame.plot()``

    """
    query_string = au._get_query_string(tech_types)
    md = solution.metadata.to_pandas()
    supply_cap = md.query(query_string).index.tolist()

    df = solution['e_cap'].loc[dict(y=supply_cap)].to_pandas()

    weighted = md.stack_weight.sort_values(ascending=False).index.tolist()
    stacked_techs = [y for y in weighted if y in df.columns]

    df = df.loc[:, stacked_techs] * unit_multiplier

    names = [md.at[y, 'name'] for y in df.columns]
    colors = [md.at[i, 'color'] for i in df.columns]
    colormap = ListedColormap(colors)
    proxies = [Rectangle((0, 0), 1, 1, fc=i)
               for i in colors]

    # Order the locations nicely, but only take those locations that actually
    # exists in the current solution
    if ('metadata' in solution.config_model and
            'location_ordering' in solution.config_model.metadata):
        meta_config = solution.config_model.metadata
        for index, item in enumerate(meta_config.location_ordering):
            if item in df.index:
                df.at[item, 'ordering'] = index
        df = df.sort_values(by='ordering', ascending=False)
        df = df.drop('ordering', axis=1)

    ax = df.plot(kind='barh', stacked=True, legend=False, colormap=colormap,
                 **kwargs)
    leg = au.legend_outside_ax(ax, where='right', artists=proxies, labels=names)

    ylab = ax.set_ylabel('')
    xlab = ax.set_xlabel('Installed capacity ({})'.format(unit_label))

    return ax


def plot_transmission(solution, tech='ac_transmission', carrier='power',
                      labels='utilization',
                      figsize=(15, 15), fontsize=9,
                      show_scale=True,
                      ax=None,
                      **kwargs):
    """
    Plot transmission links on a map. Requires that model metadata have
    been defined with a lat/lon for each model location and a boundary for
    the map display.

    Requires Basemap and NetworkX to be installed.

    Parameters
    ----------
    solution : solution container
    tech : str, default 'ac_transmission'
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
    show_scale : bool, default True
        Plot a distance scale on the map.
    ax : matplotlib axes, default None
    **kwargs : are passed to ``analysis_utils.plot_graph_on_map()``

    """
    import networkx as nx

    # Determine maximum that could have been transmitted across a link
    def get_edge_capacity(solution, a, b):
        hrs = solution['time_res'].to_pandas().sum()
        cap = (solution['e_cap_net'].loc[dict(x=a, y='{}:'.format(tech) + b)]
                                    .to_pandas() * hrs)
        return cap

    # Get annual power transmission between zones
    zones = sorted(list(solution.coords['x'].values))
    trans_tech = lambda x: '{}:{}'.format(tech, x)
    def get_annual_power_transmission(zone):
        try:
            return (solution['e'].loc[dict(c=carrier, y=trans_tech(zone))]
                                .sum(dim='t').to_pandas())
        except KeyError:
            return 0
    df = pd.DataFrame({zone: get_annual_power_transmission(zone)
                      for zone in zones}, index=zones)

    # Set negative and very small (erroneous) values to zero --
    # Positive values = energy received at a node by transmission,
    # negative values = energy sent by transmission.
    # The absolute value sent and recieved between two nodes will be
    # different when there is line loss (i.e. negative values will have a
    # larger absolute value than the positive value for the same transmission
    # line), which is why we take the recieving (positive) value.
    df[df < df.max().max()*1e-10] = 0

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

    ax, m = au.plot_graph_on_map(solution.config_model, G=G,
                                 edge_colors=edge_colors,
                                 edge_labels=edge_labels,
                                 figsize=figsize, fontsize=fontsize,
                                 ax=ax, show_scale=show_scale, **kwargs)

    return ax


def get_delivered_cost(solution, cost_class='monetary', carrier='power',
                       count_unmet_demand=False):
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
        delivered cost.

    """
    summary = solution.summary.to_pandas()
    meta = solution.metadata.to_pandas()
    carrier_subset = meta[meta.carrier_out == carrier].index.tolist()
    if count_unmet_demand is False:
        try:
            carrier_subset.remove('unmet_demand_' + carrier)
        except ValueError:  # no unmet demand technology
            pass
    cost = (solution['costs'].loc[dict(k=cost_class, y=carrier_subset)]
                             .to_pandas().sum().sum())
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


def get_levelized_cost(solution, cost_class='monetary', carrier='power',
                       groups=None, locations=None,
                       unit_multiplier=1.0):
    """
    Get the levelized cost per unit of energy produced for the given
    ``cost_class`` and ``carrier``, optionally for a subset of technologies
    given by ``groups`` and a subset of ``locations``.

    Parameters
    ----------
    solution : solution container
    cost_class : str, default 'monetary'
    carrier : str, default 'power'
    groups : list, default None
        Limit the computation to members of the given groups (see the
        groups table in the solution for valid groups). Defaults to
        ['supply', 'supply_plus'] if not given.
    locations : str or iterable, default None
        Limit the computation to the given location or locations.
    unit_multiplier : float or int, default 1.0
        Adjust unit of the returned cost value. For example, if model units
        are kW and kWh, ``unit_multiplier=1.0`` will return cost per kWh, and
        ``unit_multiplier=0.001`` will return cost per MWh.

    """
    if groups is None:
        groups = ['supply', 'supply_plus']

    members = []
    for g in groups:
        try:
            members.extend(solution.groups.to_pandas().at[g, 'members'].split('|'))
        except KeyError:
            pass

    if locations is None:
        locations_slice = slice(None)
    elif isinstance(locations, (str, float, int)):
        # Make sure that locations is a list if it's a single value
        locations_slice = [locations]
    else:
        locations_slice = locations

    cost = solution['costs'].loc[dict(k=cost_class, x=locations_slice, y=members)]
    c_prod = solution['c_prod'].loc[dict(c=carrier, x=locations_slice, y=members)]

    if locations is None:
        cost = cost.sum(dim='x').to_pandas()
        c_prod = c_prod.sum(dim='x').to_pandas()
    else:
        cost = cost.to_pandas()
        c_prod = c_prod.to_pandas()

    return (cost / c_prod) * unit_multiplier


def get_group_share(solution, techs, group, var='e_prod'):
    """
    From ``solution.summary``, get the share of the given list of ``techs``
    from the total for the given ``group``, for the given ``var``.

    """
    summary = solution.summary
    meta = solution.metadata.to_pandas()
    members = meta.query('type == "' + group + '"').index.tolist()
    if group == 'transmission':
        # Special case for transmission techs, we only want
        # the base tech names, since we're looking this
        # up in the summary table
        def transmission_basenames(group):
            return list(set([i.split(':')[0] for i in group]))
        techs = transmission_basenames(techs)
        members = transmission_basenames(members)
    supply_total = summary.loc[members, var].sum()
    supply_group = summary.loc[techs, var].sum()
    try:
        with np.errstate(divide='ignore', invalid='ignore'):
            return supply_group / supply_total
    except ZeroDivisionError:
        # FIXME it seems that on some systems, supply_ are not numpy.floats
        # but regular Python floats, leading to a ZeroDivisionError
        # in cases such as demand groups
        return np.nan


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
    unmet = (solution['e']
             .loc[dict(c=carrier, y='unmet_demand_' + carrier)]
             .sum(dim='x')
             .to_pandas())
    timesteps = len(unmet[unmet > 0])
    hours = solution.time_res.to_pandas()[unmet > 0].sum()
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
    # TODO: add unit tests
    time_res = solution.time_res.to_pandas()
    selected = time_res[time_res < resolution]
    return list(au._get_ranges(selected.index.tolist()))


def get_swi(solution, shares_var='e_cap', exclude_patterns=['unmet_demand']):
    """
    Returns the Shannon-Wiener diversity index.

    :math:`SWI = -1 \\times \sum_{i=1}^{I} p_{i} \\times \ln(p_{i})`

    where where I is the number of categories and :math:`p_{i}`
    is each category's share of the total (between 0 and 1).

    :math:`SWI` is zero when there is perfect concentration.

    """
    # TODO: add unit tests
    techs = au.get_supply_groups(solution)
    for pattern in exclude_patterns:
        techs = [t for t in techs if pattern not in t]
    swi = -1 * sum((p * np.log(p))
                   for p in [solution.shares.to_pandas().at[y, shares_var]
                   for y in techs]
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
    # TODO: add unit tests
    techs = au.get_supply_groups(solution)
    for pattern in exclude_patterns:
        techs = [t for t in techs if pattern not in t]
    hhi = sum((solution.shares.to_pandas().at[y, shares_var] * 100.) ** 2
              for y in techs)
    return hhi


def get_domestic_supply_index(solution):
    """
    Assuming that ``solution`` specifies a ``domestic`` cost class to
    give each technology a domesticity score, return the total domestic
    supply index for the given solution.

    """
    # TODO: add unit tests
    idx = solution.metadata.query('type == "supply"').index.tolist()
    dom = (solution.costs.loc[dict(k='domestic', y=idx)].sum().sum() /
           solution['c_prod'].loc[dict(c='power')].sum().sum())
    return dom


def map_results(results, func, as_frame=False):
    """
    Applies ``func`` to each model solution in ``results``, returning
    a pandas DataFrame (if as_frame is True) or Series,
    indexed by the run names (if available).

    """
    # TODO: add unit tests
    def map_func(x):
        try:
            return func(x)
        except Exception:
            return np.nan

    iterations = list(results.solutions.keys())
    items = [map_func(results.solutions[i]) for i in iterations]
    idx = list(results.solutions.keys())
    try:
        idx = results.iterations.loc[idx, 'name']
    except Exception:
        pass
    if as_frame:
        return pd.DataFrame(items, index=idx)
    else:
        return pd.Series(items, index=idx)
