"""
Copyright (C) 2013-2017 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

analysis_utils.py
~~~~~~~~~~~~~~~~~

Helper functions for analysis.py.

"""

import itertools

try:
    import matplotlib.pyplot as plt
except ImportError:
    pass  # This is logged in analysis.py
import numpy as np
import pandas as pd

from . import utils


def legend_outside_ax(ax, where='right', artists=None, labels=None, **kwargs):
    """
    Draw a legend outside the figure given by ``ax``.

    """
    box = ax.get_position()

    # Positions depending on `where`
    # 3-tuples: (loc, bbox_to_anchor, positions for ax.set_position())
    POS = {
        'right': ('center left', (1, 0.5),
                  [box.x0, box.y0, box.width * 1.0, box.height]),
        'left': ('upper center', (05., -0.05),
                 [box.x0, box.y0, box.width * 1.0, box.height]),
        'bottom': ('center right', (-0.35, 0.5),
                   [box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
    }
    try:
        leg_loc, leg_bbox, new_ax_pos = POS[where]
    except KeyError:
        raise ValueError('Invalid for `where`: {}'.format(where))

    ax.set_position(new_ax_pos)

    # Use custom artists and labels?
    if artists and labels:
        args = [artists, labels]
    else:
        args = []

    legend = ax.legend(*args, loc=leg_loc, bbox_to_anchor=leg_bbox, **kwargs)

    return legend


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

    colors = plt.get_cmap(colormap)(np.linspace(0, 1.0, len(stack))).tolist()
    labels = names if names else stack  # Labels are friendly names, if given
    fills = ax.stackplot(df.index, df[stack].T, colors=colors, labels=labels,
                         **kwargs)

    # A little hack to reverse the ordering of the just-added PolyCollections
    ax.collections = [i for i in ax.collections if i not in fills]
    ax.collections += reversed(fills)

    # Add legend
    if legend:
        if legend == 'default':
            l = ax.legend(title=leg_title)
        elif legend in ['right', 'left', 'bottom']:
            l = legend_outside_ax(ax, where=legend, title=leg_title)
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
                      scale_bottom_distance=0.05,
                      ax=None, show_scale=True,
                      map_resolution='i'):
    from mpl_toolkits.basemap import Basemap
    import networkx as nx
    from calliope.lib import nx_pylab

    if all(['lat' in i or 'lon' in i for i in
            config_model.metadata.location_coordinates.as_dict_flat().keys()]):
        coord_system = 'geographic'
    elif all(['x' in x or 'y' in x for x in
            config_model.metadata.location_coordinates.as_dict_flat().keys()]):
        coord_system = 'cartesian'
    else:
        raise KeyError('unidentified coordinate system. Expecting data in '
                       'the format {lat: N, lon: M} or {x: N, y: M} for user '
                       'coordinate values of N, M.')
    # Set up basemap
    if not bounds:
        bounds = config_model.metadata.map_boundary

    # Create plot
    if not ax:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, facecolor='w', frame_on=False)

    # Node positions
    pos = config_model.metadata.location_coordinates
    if coord_system == 'geographic':
        bounds_width = bounds.upper_right.lon - bounds.lower_left.lon # lon --> width
        bounds_height = bounds.upper_right.lat - bounds.lower_left.lat  # lat --> height
        m = Basemap(projection='merc', ellps='WGS84',
                llcrnrlon=bounds.lower_left.lon,
                llcrnrlat=bounds.lower_left.lat,
                urcrnrlon=bounds.upper_right.lon,
                urcrnrlat=bounds.upper_right.lat,
                lat_ts=bounds.lower_left.lat + bounds_width / 2,
                resolution=map_resolution,
                suppress_ticks=True)
        m.drawmapboundary(fill_color=None, linewidth=0)
        m.drawcoastlines(linewidth=0.2, color='#626262')
        pos = {i: m(pos[i].lon, pos[i].lat) for i in pos} # translate lat, lon to basemap positions
        # Adding node names just above node points
        pos_offset = {i: (pos[i][0], pos[i][1] + 20) for i in pos}
    elif coord_system == 'cartesian':
        pos = {i: (pos[i].x, pos[i].y) for i in pos}
        # Adding node names just above node points
        pos_offset = {i: (pos[i][0], pos[i][1] + 0.2) for i in pos}
        # m has to be defined as it is returned
        m = None

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

        # Adding node names just above node points
        nx.draw_networkx_labels(G, pos_offset, font_size=fontsize)

    # Add a map scale
    if show_scale and coord_system=='geographic':
        scale = m.drawmapscale(
            bounds.lower_left.lon + bounds_width * scale_left_distance,
            bounds.lower_left.lat + bounds_height * scale_bottom_distance,
            bounds.lower_left.lon, bounds.lower_left.lat,
            100,
            barstyle='simple', labelstyle='simple',
            fillcolor1='w', fillcolor2='#555555',
            fontcolor='#555555', fontsize=fontsize
        )
    return ax, m


def plot_shapefile_polys(shapefile, ax, m, cmap):
    """
    Plot polygons from the given shapefile on a basemap.

    Parameters
    ----------
    shapefile : shapefile object
        Shapefile opened with fiona (``fiona.open(path_to_shapefile)``)
    ax : matplotlib ax object
        Axes to plot on
    m : basemap object
    cmap : matplotlib colormap
        Colormap to use to color the individual polygons in the shapefile

    """
    from descartes import PolygonPatch
    from matplotlib.collections import PatchCollection
    import shapely
    import shapely.ops

    # Break up MultiPolygons into Polygons, keeping track of
    # which shape they belong to
    polygons = []
    names = []
    for i in shapefile:
        # name = i['properties'][name_property]
        name = i['id']
        geom = shapely.geometry.shape(i['geometry'])
        if geom.geom_type == 'MultiPolygon':
            for poly in geom:
                polygons.append(poly)
                names.append(name)
        else:
            polygons.append(geom)
            names.append(name)

    # Create patches with basemap-transformed coordinates
    def TransformedPolygonPatch(poly):
        return PolygonPatch(shapely.ops.transform(m, poly),
                            ec='#555555', lw=.2, alpha=1., zorder=4)

    polygon_patches = [TransformedPolygonPatch(p) for p in polygons]

    # Create name->color_id mapping dict
    colors = {name: idx for idx, name in enumerate(list(set(names)))}
    n = plt.Normalize(0, len(colors))

    pc = PatchCollection(polygon_patches, match_original=True)
    # Now go through all names and set the color appropriately
    pc.set_facecolor(cmap(n([colors[i] for i in names])))
    pc.set_alpha(0.2)
    ax.add_collection(pc)

    return ax


def add_labels(ax, m, label_coordinates, label_positioning, fontsize=13):
    """
    Add labels to a basemap ``m`` on the given ``ax``.

    Parameters
    ----------
    label_coordinates : dict
        Dict of coordinates for each label in the form of
        {label: (lat, lon)}
    label_positioning : dict
        Dict of positions for each label in the form {label: position},
        where position can be 'c' (center), 'r' (right), 'l' (left), or
        'b' (bottom)

    """
    align_x = 25000
    align_y = 35000
    for zone, pos in label_positioning.items():
        lat, lon = label_coordinates[zone]
        x, y = m(lon, lat)
        if pos == 'c':
            # no x/y adjustment for central labels..
            ha = 'center'
        if pos == 'l':
            x -= align_x
            ha = 'right'
        if pos == 'r':
            x += align_x
            ha = 'left'
        if pos == 'b':
            y -= align_y
            ha = 'center'
        ax.add_artist(plt.Text(x, y, zone,
                               fontsize=fontsize,
                               color='black', backgroundcolor='white',
                               horizontalalignment=ha,
                               verticalalignment='bottom',
                               zorder=10))
    return ax


def _get_supply_groups(solution):
    """
    Get individual supply technologies and those groups that define
    group == True, for purposes of calculating diversity of supply

    """
    # idx_1: group is 'True' and '|' in members
    groups = solution.groups.to_pandas()
    grp_1 = groups.query('group == "True" & (type == "supply" | type == "supply_plus")')
    idx_1 = grp_1[(grp_1.members != grp_1.index)
                  & (grp_1.members.str.contains('\|'))].index.tolist()
    # idx_2: group is 'False' and no '|' in members
    grp_2 = groups.query('group == "False" & (type == "supply" | type == "supply_plus")')
    idx_2 = grp_2[grp_2.members == grp_2.index].index.tolist()
    # Also drop entries from idx_2 that are already covered by
    # groups in idx_1
    covered = [i.split('|')
               for i in groups.loc[idx_1, 'members'].tolist()]
    covered_flat = [i for i in itertools.chain.from_iterable(covered)]
    idx_2 = [i for i in idx_2 if i not in covered_flat]
    return idx_1 + idx_2


def get_float_formatter_func(precision=None, thousands_sep=False,
                             zero_string='0'):
    """
    Returns a function that gives ``zero_string`` if the float is 0, else
    a string with the given digits of precision.

    """

    def float_formatter(f):
        if isinstance(f, str):
            return f

        if thousands_sep:
            ts = ','
        else:
            ts = ''
        if precision is not None:
            prec = '.' + str(precision) + 'f'
        elif float(f).is_integer():
            prec = '.0f'
        else:
            prec = ''

        fmt = '{:' + ts + prec + '}'

        try:
            if f == 0:
                return zero_string
            elif (precision is not None) or (thousands_sep is not None):
                return fmt.format(f)
            else:
                return str(f)
        except ValueError:
            return f
    return float_formatter


def write_latex_table(df, path, formatters=None, no_textbackslash=False,
                      **kwargs):
    """
    Write a LaTeX table from the given ``df`` (a pandas DataFrame)
    to the given ``path``, optionally using the given ``formatters``
    (see pandas.DataFrame.to_latex documentation).

    Additional **kwargs are passed to pandas.DataFrame.to_latex.

    """
    first_line = '\\begin{tabularx}{0.95\\textwidth}{l' + 'X' * len(df.columns) + '}'
    last_line = '\\end{tabularx}'

    latex_lines = df.to_latex(formatters=formatters, **kwargs).split('\n')
    latex_lines[0] = first_line
    latex_lines[-2] = last_line  # latex_lines[-1] is trailing newline

    # Fix \ chars
    if no_textbackslash:
        latex_lines = [l.replace('\\textbackslash', '\\') for l in latex_lines]

    with open(path, 'w') as f:
        f.write('\n'.join(latex_lines))


def df_transmission_matrix(config_model, tech, constraint='e_cap.max'):
    """
    Returns a transmission matrix in the form of a pandas DataFrame,
    for the given constraint.

    """
    zones = sorted(list(config_model.locations.keys()))

    df = pd.DataFrame(0.0, index=zones, columns=zones)

    for z1 in zones:
        for z2 in zones:
            key = z1 + ',' + z2
            if (key in config_model.links
                    and tech in config_model.links[key]):
                constr_attrdict = config_model.links[key][tech]['constraints']
                val = constr_attrdict.get_key(constraint)
                if val == 'inf':
                    val = pd.np.inf
                df.at[z1, z2] = val
                df.at[z2, z1] = val

    return df


def df_tech_table(model, columns, parents=['supply', 'supply_plus']):
    """
    Returns a pandas DataFrame of technologies from the given model with
    the given parent tech, with  a column for each of the tech
    constraints/options given in ``columns``.

    """
    get_any_option = utils.any_option_getter(model)
    cm = model.config_model
    techs = []
    for p in parents:
        techs.extend([
            k for k in cm.techs
            if model.ischild(k, p) and 'name' in cm.techs[k]
        ])
    data = []
    for t in techs:
        item = {'name': cm.techs[t].name}
        for c in columns:
            item[c] = get_any_option(t + '.' + c)
        data.append(item)
    return pd.DataFrame.from_records(data)


def flatten(d):
    return {k: d.get_key(k) for k in d.keys_nested()}


def df_locations_table(model, string_in_cols=None, settings_to_get=None):
    """
    Returns a pandas DataFrame of locations from the given model.

    If string_in_cols specified, extracts data from location table.

    If settings_to_get specified, extracts data from model configuration
    directly.

    """
    target = model.config_model.locations
    l = []
    for k in target:
        if string_in_cols:
            item = flatten(target[k])
        if settings_to_get:
            item = {s: model.get_option(s, x=k) for s in settings_to_get}
        item['name'] = k
        l.append(item)
    df = pd.DataFrame.from_records(l)
    if string_in_cols:
        # Always keep the 'name' column
        df = df[[c for c in df.columns if string_in_cols in c] + ['name']]
    return df
