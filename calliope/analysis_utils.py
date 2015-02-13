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
