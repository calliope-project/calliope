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
    # TODO check how pandas does its very nice formatting for df.plot()
    import matplotlib.dates as mdates
    if ticks == 'monthly':
        formatter = mdates.DateFormatter('%b %Y')
        locator = mdates.MonthLocator()
    if ticks == 'daily':
        formatter = mdates.DateFormatter('%d-%m-%Y')
        locator = mdates.DayLocator()
    if ticks == 'hourly':
        formatter = mdates.DateFormatter('%h:%m %d-%m-%Y')
        locator = mdates.HourLocator()
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.gca().xaxis.set_major_locator(locator)
    return ax


def get_delivered_cost(solution, carrier='power'):
    summary = solution.summary
    carrier_subset = summary[summary.carrier == carrier].index.tolist()
    cost = solution.costs.cost.loc['total', carrier_subset].sum()
    delivered = summary.at['demand_' + carrier, 'consumption (GWh)'] * 1e6
    try:
        unmet = summary.at['unmet_demand_' + carrier,
                           'consumption (GWh)'] * 1e6
    except KeyError:
        unmet = 0
    return cost / (delivered - unmet) * -1
