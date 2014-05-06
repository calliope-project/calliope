"""
Copyright (C) 2013 Stefan Pfenninger.
Licensed under the Apache 2.0 License (see LICENSE file).

analysis.py
~~~~~~~~~~~

Functionality to analyze model results.

"""

from __future__ import print_function
from __future__ import division

import matplotlib.pyplot as plt
import numpy as np


def legend_on_right(ax, style='default', artists=None, labels=None):
    """Draw a legend on outside on the right of the figure given by 'ax'"""
    box = ax.get_position()
    # originally box.width * 0.8 but 1.0 solves some problems
    # it just means that the box becomes wider, which is ok though!
    ax.set_position([box.x0, box.y0, box.width * 1.0, box.height])
    if style == 'square':
        artists, labels = get_square_legend(ax.legend())
        l = ax.legend(artists, labels, loc='center left',
                      bbox_to_anchor=(1, 0.5))
    elif style == 'custom':
        l = ax.legend(artists, labels, loc='center left',
                      bbox_to_anchor=(1, 0.5))
    else:
        l = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    return l


def legend_below(ax, style='default', columns=5, artists=None, labels=None):
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])
    if style == 'square':
        artists, labels = get_square_legend(ax.legend())
        l = ax.legend(artists, labels, loc='upper center',
                      bbox_to_anchor=(0.5, -0.05), ncol=columns)
    elif style == 'custom':
        l = ax.legend(artists, labels, loc='upper center',
                      bbox_to_anchor=(0.5, -0.05), ncol=columns)
    else:
        l = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
                      ncol=columns)
    return l


def get_square_legend(lgd):
    rects = [plt.Rectangle((0, 0), 1, 1,
             fc=l.get_color()) for l in lgd.get_lines()]
    labels = [l.get_label() for l in lgd.get_lines()]
    return (rects, labels)


def stack_plot(df, stack, figsize=None, colormap='jet', legend='default',
               ticks='daily', names=None, **kwargs):
    """
    legend can be 'default' or 'right'
    ticks can be 'hourly', 'daily', 'monthly'

    """
    if not figsize:
        figsize = (16, 4)
    colors = plt.get_cmap(colormap)(np.linspace(0, 1.0, len(stack)))
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
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
        ax.legend(reversed(proxies), reversed(stack))
    elif legend == 'right':
        legend_on_right(ax, artists=reversed(proxies), labels=reversed(stack),
                        style='custom')
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
        minor_locator = mdates.HourLocator(byhour=range(1, 24))
        plt.gca().xaxis.set_minor_formatter(minor_formatter)
        plt.gca().xaxis.set_minor_locator(minor_locator)
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.gca().xaxis.set_major_locator(locator)
    return ax


def plot_solution(solution, data, demand='demand_power',
                  colormap='jet', ticks=None):
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
    df = solution.metadata
    stacked_techs = df[(df['type'] == 'supply')
                       | (df['type'] == 'storage')
                       | (df['type'] == 'unmet_demand')].index.tolist()
    # Put stack in order according to stack_weights
    weighted = df.weight.order(ascending=False).index.tolist()
    stacked_techs = [y for y in weighted if y in stacked_techs]
    names = [df.at[y, 'name'] for y in stacked_techs]
    # Plot!
    ax = stack_plot(plot_df, stacked_techs, colormap=colormap,
                    alpha=0.9, ticks=ticks, legend='right', names=names)
    ax.plot(plot_df[demand].index,
            plot_df[demand] * -1,
            color='black', lw=1, ls='-')
    return ax


def get_delivered_cost(solution, cost_class='monetary', carrier='power'):
    summary = solution.summary
    meta = solution.metadata
    carrier_subset = meta[meta.carrier == carrier].index.tolist()
    cost = solution.costs.loc[cost_class, 'total', carrier_subset].sum()
    delivered = summary.at['demand_' + carrier, 'consumption'] * 1e6
    try:
        unmet = summary.at['unmet_demand_' + carrier, 'consumption'] * 1e6
    except KeyError:
        unmet = 0
    return cost / (delivered - unmet) * -1


def get_group_share(solution, techs, group_type='supply',
                    var='production'):
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
