"""
Copyright (C) 2013-2015 Stefan Pfenninger.
Licensed under the Apache 2.0 License (see LICENSE file).

analysis_utils.py
~~~~~~~~~~~~~~~~~

Helper functions for analysis.py.

"""

try:
    import matplotlib.pyplot as plt
except ImportError:
    pass  # This is logged in analysis.py
import numpy as np
import pandas as pd


def legend_on_right(ax, style='default', artists=None, labels=None, **kwargs):
    """Draw a legend on outside on the right of the figure given by ``ax``"""
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
    """Draw a legend on outside below the figure given by ``ax``"""
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


def _get_query_string(types):
    formatted_types = ['type == "{}"'.format(t) for t in types]
    query_string = ' | '.join(formatted_types)
    return query_string


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


def plot_graph_on_map(config_model, G=None,
                      edge_colors=None, edge_labels=None,
                      figsize=(15, 15), fontsize=9,
                      arrow_style='->',
                      rotate_labels=False,
                      bounds=None,
                      scale_left_distance=0.05,
                      scale_bottom_distance=0.05):
    from mpl_toolkits.basemap import Basemap
    import networkx as nx
    from calliope.lib import nx_pylab

    # Set up basemap
    if not bounds:
        bounds = config_model.metadata.map_boundary
    bounds_width = bounds[2] - bounds[0]  # lon --> width
    bounds_height = bounds[3] - bounds[1]  # lat --> height
    m = Basemap(projection='merc', ellps='WGS84',
                llcrnrlon=bounds[0], llcrnrlat=bounds[1],
                urcrnrlon=bounds[2], urcrnrlat=bounds[3],
                lat_ts=bounds[1] + bounds_width / 2,
                resolution='i',
                suppress_ticks=True)

    # Node positions
    pos = config_model.metadata.location_coordinates
    pos = {i: m(pos[i][1], pos[i][0]) for i in pos}  # Flip lat, lon to x, y!

    # Create plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, axisbg='w', frame_on=False)
    m.drawmapboundary(fill_color=None, linewidth=0)
    m.drawcoastlines(linewidth=0.2, color='#626262')

    # Draw the graph
    if G:
        # Using nx_pylab to be able to set zorder below the edges
        nx_pylab.draw_networkx_nodes(G, pos, node_color='#CCCCCC',
                                     node_size=300, zorder=0)

        # Using nx_pylab from lib to get arrow_style option
        nx_pylab.draw_networkx_edges(G, pos, width=3,
                                     edge_color=edge_colors,
                                     # This works for edge_use
                                     edge_vmin=0.0, edge_vmax=1.0,
                                     edge_cmap=plt.get_cmap('seismic'),
                                     arrows=True, arrow_style=arrow_style)

        # bbox = dict(color='white', alpha=0.5, edgecolor=None)
        labels = nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels,
                                              rotate=rotate_labels,
                                              font_size=fontsize)

    # Add a map scale
    scale = m.drawmapscale(
        bounds[0] + bounds_width * scale_left_distance,
        bounds[1] + bounds_height * scale_bottom_distance,
        bounds[0], bounds[1],
        100,
        barstyle='simple', labelstyle='simple',
        fillcolor1='w', fillcolor2='#555555',
        fontcolor='#555555', fontsize=fontsize
    )

    return ax, m
