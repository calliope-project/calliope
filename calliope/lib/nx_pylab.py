"""
Improved networkx plotting functionality

Source: https://networkx.lanl.gov/trac/ticket/434
Author: Aric Hagberg, https://github.com/hagberg
License: BSD license, see LICENSE_NetworkX

"""


import networkx as nx
import matplotlib.patches as patches
import matplotlib.collections as collections
import numpy as np
import math
import matplotlib.cbook as cb
import matplotlib.colors as mcolors
import matplotlib as mpl
import matplotlib.cm as cm

def get_color_dict(color,item_list,vmin=None,vmax=None,cmap=None):
    """ Determine color rgb color values given a data list.

    This function is used to take a sequence of data and convert it
    into a dictionary which of possible rgb values. It can take a
    number of different types of data explained below. If no colormap
    is given it will return the grayscale for scalar values.

    Parameters:
    -----------
    color: A string, scalar, or iterable of either.
           This can be a color in a variety of formats. A matplotlib
           a single color, which can be a matplotlib color specification
           e.g. 'rgbcmykw', a hex color string '#00FFFF', a standard
           html name like 'aqua', or a numeric value to be scaled with
           vmax or vmin. It can also be a list of any of these types
           as well as a dictionary of any of these types.
    item_list: a list
               A list of keys which correspond to the values given in
               color.
    vmin: A scalar
          The minimum value if scalar values are given for color
    vmax: A scalar
          The maximum value if scalar values are given for color
    cmap: A matplotlib colormap
          A colormap to be used if scalar values are given.

    Returns:
    -------
    color_dict: dict
                A dictionary of rgb colors keyed by values in item_list
    """

    CC = mcolors.ColorConverter()

    if cb.is_string_like(color):
        return {}.fromkeys(item_list,CC.to_rgb(color))
    elif cb.is_scalar(color):
        CN = mcolors.Normalize(0.0,1.0)
        if cmap is not None:
            return {}.fromkeys(item_list,cmap(CN(color)))
        else:
            return {}.fromkeys(item_list,CC.to_rgb(str(CN(color))))
    elif cb.iterable(color) and not cb.is_string_like(color):
        try:
            vals = [color[i] for i in item_list]
        except:
            vals = color
        keys = item_list
        if len(item_list)>len(vals):
            raise nx.NetworkXError("Must provide a value for each item")
        if cb.alltrue([cb.is_string_like(c) for c in vals]):
            return dict(zip(keys,[CC.to_rgb(v) for v in vals]))
        elif cb.alltrue([cb.is_scalar(c) for c in vals]):
            if vmin is None:
                vmin = float(min(vals))
            if vmax is None:
                vmax = float(max(vals))
            CN = mcolors.Normalize(vmin,vmax)
            if cmap is not None:
                return dict(zip(keys,[cmap(CN(v)) for v in vals]))
            else:
                return dict(zip(keys,[CC.to_rgb(str(CN(v))) for v in vals]))
        elif cb.alltrue([cb.iterable(c) and not cb.is_string(c) for c in vals]):
            return dict(zip(keys,vals))
    else:
        raise nx.NetworkXError("Could not convert colors")

def is_weighted(G):
    """ Determine if a graph G is weighted

    Checks each edge to see if it has attribute 'weight' if it does
    return True, otherwise false. This checks if the entire graph is
    weighted not partial.

    Parameters:
    ----------
    G: A networkx Graph

    Returns:
    --------
    weighted :  A bool
       Determines whether the graph is weighted.
    """
    weighted = True
    for (u,v) in G.edges():
        weighted = weighted and ('weight' in G.edge[u][v])
        if not weighted:
            return weighted
    return weighted

def is_weighted(self):
    """ Determine if a graph G is weighted

    Checks each edge to see if it has attribute 'weight' if it does
    return True, otherwise false. This checks if the entire graph is
    weighted not partial.

    Parameters:
    ----------
    G: A networkx Graph

    Returns:
    --------
    weighted :  A bool
       Determines whether the graph is weighted.
    """
    weighted = True
    for (u,v) in self.edges():
        weighted = weighted and ('weight' in self.edge[u][v])
        if not weighted:
            return weighted
    return weighted

def edge_width_weight(G,edgelist=None):
    """Automatically calculate a normalized reasonable line width for
    a weighted graph

    Parameters:
    -----------
    G: A networkx Graph
    edgelist: A list
      Edges to calculate the weights for if None, usesall edges
    Returns:
    --------
    weight_dict: A dictionary
       Line weights that displays nicely in matplotlib.
    """
    if edgelist is None:
        edgelist = G.edges()
    lw = {}
    for (u,v) in edgelist:
        lw[(u,v)] = G.edge[u][v]['weight']
    maxw = max(lw.values())
    minw = float(min(lw.values())) #to ensure floats later
    return dict(zip(lw.keys(), \
                    map((lambda x: 19.5*((x-minw)/(maxw-minw)) + 0.5), \
                        lw.values())))

def edge_color_weight(G,edgelist=None):
    """Automatically calculate a normalized reasonable color for
    a weighted graph

    Parameters:
    -----------
    G: A networkx Graph
    edgelist: A list
      Edges to calculate the weights for if None, uses all edges
    Returns:
    --------
    weight_dict: A dictionary
       Values between 0-1 that displays nicely in matplotlib.
    """
    cl = {}
    if edgelist is None:
        edgelist=G.edges()
    for (u,v) in edgelist:
        cl[(u,v)] = G[u][v]['weight']
    maxw = max(cl.values())
    minw = float(min(cl.values()))
    return dict(zip(cl.keys(), \
                    map((lambda x: (x-minw)/(maxw - minw)),cl.values())))

def draw(G, pos=None, ax=None, hold=None, **kwds):
    """Draw the graph G with Matplotlib (pylab).

    Draw the graph as a simple representation with no node
    labels or edge labels and using the full Matplotlib figure area
    and no axis labels by default.  See draw_networkx() for more
    full-featured drawing that allows title, axis labels etc.

    Parameters
    ----------
    G : graph
       A networkx graph

    pos : dictionary, optional
       A dictionary with nodes as keys and positions as values.
       If not specified a spring layout positioning will be computed.
       See networkx.layout for functions that compute node positions.

    ax : Matplotlib Axes object, optional
       Draw the graph in specified Matplotlib axes.

    hold: bool, optional
       Set the Matplotlib hold state.  If True subsequent draw
       commands will be added to the current axes.

    **kwds: optional keywords
       See networkx.draw_networkx() for a description of optional keywords.

    Examples
    --------
    >>> G=nx.dodecahedral_graph()
    >>> nx.draw(G)
    >>> nx.draw(G,pos=nx.spring_layout(G)) # use spring layout

    See Also
    --------
    draw_networkx()
    draw_networkx_nodes()
    draw_networkx_edges()
    draw_networkx_labels()
    draw_networkx_edge_labels()

    Notes
    -----
    This function has the same name as pylab.draw and pyplot.draw
    so beware when using

    >>> from networkx import *

    since you might overwrite the pylab.draw function.

    Good alternatives are:

    With pylab:

    >>> import pylab as P #
    >>> import networkx as nx
    >>> G=nx.dodecahedral_graph()
    >>> nx.draw(G)  # networkx draw()
    >>> P.draw()    # pylab draw()

    With pyplot

    >>> import matplotlib.pyplot as plt
    >>> import networkx as nx
    >>> G=nx.dodecahedral_graph()
    >>> nx.draw(G)  # networkx draw()
    >>> plt.draw()  # pyplot draw()

    Also see the NetworkX drawing examples at
    http://networkx.lanl.gov/gallery.html


    """
    try:
        import matplotlib.pylab as pylab
    except ImportError:
        raise ImportError, "Matplotlib required for draw()"
    except RuntimeError:
        print "Matplotlib unable to open display"
        raise

    cf=pylab.gcf()
    cf.set_facecolor('w')
    if ax is None:
        if cf._axstack() is None:
            ax=cf.add_axes((0,0,1,1))
        else:
            ax=cf.gca()

 # allow callers to override the hold state by passing hold=True|False
    b = pylab.ishold()
    h = kwds.pop('hold', None)
    if h is not None:
        pylab.hold(h)
    try:
        draw_networkx(G,pos=pos,ax=ax,**kwds)
        ax.set_axis_off()
        pylab.draw_if_interactive()
    except:
        pylab.hold(b)
        raise
    pylab.hold(b)
    return

def draw_networkx(G, pos=None, with_labels=True, **kwds):
    """Draw the graph G using Matplotlib.

    Draw the graph with Matplotlib with options for node positions,
    labeling, titles, and many other drawing features.
    See draw() for simple drawing without labels or axes.

    Parameters
    ----------
    G : graph
       A networkx graph

    pos : dictionary, optional
       A dictionary with nodes as keys and positions as values.
       If not specified a spring layout positioning will be computed.
       See networkx.layout for functions that compute node positions.

    ax : Matplotlib Axes object, optional
       Draw the graph in the specified Matplotlib axes.

    with_labels:  bool, optional
       Set to True (default) to draw labels on the nodes.

    nodelist: list, optional
       Draw only specified nodes (default G.nodes())

    edgelist: list
       Draw only specified edges(default=G.edges())

    node_size: scalar or array
       Size of nodes (default=300).  If an array is specified it must be the
       same length as nodelist.

    node_color: color string, or array of floats
       Node color. Can be a single color format string (default='r'),
       or a  sequence of colors with the same length as nodelist.
       If numeric values are specified they will be mapped to
       colors using the cmap and vmin,vmax parameters. Can also be a
       dictionary keyed by node, and can be in any matplotlib acceptable
       color value.

    node_shape:  string
       The shape of the node.  Specification is as matplotlib.scatter
       marker, one of 'so^>v<dph8' (default='o').

    alpha: float
       The node transparency (default=1.0)

    cmap: Matplotlib colormap
       Colormap for mapping intensities of nodes (default=None)

    vmin,vmax: floats
       Minimum and maximum for node colormap scaling (default=None)

    width: float
       Line width of edges (default =1.0)

    edge_color: color string, or array of floats
       Edge color. Can be a single color format string (default='r'),
       or a sequence of colors with the same length as edgelist.
       If numeric values are specified they will be mapped to
       colors using the edge_cmap and edge_vmin,edge_vmax parameters.

    edge_ cmap: Matplotlib colormap
       Colormap for mapping intensities of edges (default=None)

    edge_vmin,edge_vmax: floats
       Minimum and maximum for edge colormap scaling (default=None)

    style: string
       Edge line style (default='solid') (solid|dashed|dotted,dashdot)

    labels: dictionary
       Node labels in a dictionary keyed by node of text labels (default=None)

    font_size: int
       Font size for text labels (default=12)

    font_color: string
       Font color string (default='k' black)

    font_weight: string
       Font weight (default='normal')

    font_family: string
       Font family (default='sans-serif')

    Notes
    -----
    Any keywords not listed above are passed through to draw_networkx_nodes(),
    draw_networkx_edges(), and draw_networkx_labels().  For finer control
    of drawing you can call those functions directly.

    Examples
    --------
    >>> G=nx.dodecahedral_graph()
    >>> nx.draw(G)
    >>> nx.draw(G,pos=nx.spring_layout(G)) # use spring layout

    >>> import pylab
    >>> limits=pylab.axis('off') # turn of axis

    Also see the NetworkX drawing examples at
    http://networkx.lanl.gov/gallery.html

    See Also
    --------
    draw()
    draw_networkx_nodes()
    draw_networkx_edges()
    draw_networkx_labels()
    draw_networkx_edge_labels()

    """
    try:
        import matplotlib.pylab as pylab
    except ImportError:
        raise ImportError, "Matplotlib required for draw()"
    except RuntimeError:
        print "Matplotlib unable to open display"
        raise

    if pos is None:
        pos=nx.drawing.spring_layout(G) # default to spring layout

    node_patches=draw_networkx_nodes(G, pos, **kwds)
    edge_patches=draw_networkx_edges(G, pos, node_patches, **kwds)
    if with_labels:
        draw_networkx_labels(G, pos, **kwds)
    pylab.draw_if_interactive()

def draw_networkx_nodes(G, pos,
                        nodelist=None,
                        node_size=300,
                        node_color='r',
                        node_shape='o',
                        alpha=1.0,
                        cmap=None,
                        vmin=None,
                        vmax=None,
                        ax=None,
                        linewidth=None,
                        zorder=None,
                        **kwds):
    """Draw the nodes of the graph G.

    This draws only the nodes of the graph G.

    Parameters
    ----------
    G : graph
       A networkx graph

    pos : dictionary
       A dictionary with nodes as keys and positions as values.
       If not specified a spring layout positioning will be computed.
       See networkx.layout for functions that compute node positions.

    ax : Matplotlib Axes object, optional
       Draw the graph in the specified Matplotlib axes.

    nodelist: list, optional
       Draw only specified nodes (default G.nodes())

    edgelist: list
       Draw only specified edges(default=G.edges())

    node_size: scalar or array
       Size of nodes (default=300).  If an array is specified it must be the
       same length as nodelist.

    node_color: color string, or array of floats
       Node color. Can be a single color format string (default='r'),
       or a  sequence of colors with the same length as nodelist.
       If numeric values are specified they will be mapped to
       colors using the cmap and vmin,vmax parameters.  See
       matplotlib.scatter for more details.

    node_shape:  string
       The shape of the node.  Specification is as matplotlib.scatter
       marker, one of 'so^>v<dph8' (default='o').

    alpha: float
       The node transparency (default=1.0)

    cmap: Matplotlib colormap
       Colormap for mapping intensities of nodes (default=None)

    vmin,vmax: floats
       Minimum and maximum for node colormap scaling (default=None)

    width`: float
       Line width of edges (default =1.0)


    Notes
    -----
    Any keywords not listed above are passed through to Matplotlib's
    scatter function.

    Examples
    --------
    >>> G=nx.dodecahedral_graph()
    >>> nodes=nx.draw_networkx_nodes(G,pos=nx.spring_layout(G))

    Also see the NetworkX drawing examples at
    http://networkx.lanl.gov/gallery.html

    See Also
    --------
    draw()
    draw_networkx()
    draw_networkx_edges()
    draw_networkx_labels()
    draw_networkx_edge_labels()



    """
    try:
        import matplotlib.pylab as pylab
        import numpy
    except ImportError:
        raise ImportError, "Matplotlib required for draw()"
    except RuntimeError:
        print "Matplotlib unable to open display"
        raise

    if zorder is None:
        zorder = 2

    if ax is None:
        ax=pylab.gca()

    if nodelist is None:
        nodelist=G.nodes()

    if not nodelist or len(nodelist)==0:  # empty nodelist, no drawing
        return None

    try:
        xy=numpy.asarray([pos[v] for v in nodelist])
    except KeyError,e:
        raise nx.NetworkXError('Node %s has no position.'%e)
    except ValueError:
        raise nx.NetworkXError('Bad value in node positions.')

    syms =  { # a dict from symbol to (numsides, angle)
            's' : (4,math.pi/4.0,0),   # square
            'o' : (0,0,3),            # circle
            '^' : (3,0,0),             # triangle up
            '>' : (3,math.pi/2.0,0),   # triangle right
            'v' : (3,math.pi,0),       # triangle down
            '<' : (3,3*math.pi/2.0,0), # triangle left
            'd' : (4,0,0),             # diamond
            'p' : (5,0,0),             # pentagram
            'h' : (6,0,0),             # hexagon
            '8' : (8,0,0),             # octagon
            '+' : (4,0,0),             # plus
            'x' : (4,math.pi/4.0,0)    # cross
            }

    temp_x = map(lambda p: p[0],pos.values())
    temp_y = map(lambda p: p[1],pos.values())
    minx = np.amin(temp_x)
    maxx = np.amax(temp_x)
    miny = np.amin(temp_y)
    maxy = np.amax(temp_y)

    w = max(maxx-minx,1.0)
    h = max(maxy-miny,1.0)
    #for scaling

    area2radius = lambda a: math.sqrt((a*w*h)/(ax.figure.get_figheight()*ax.figure.get_figwidth()*ax.figure.dpi*ax.figure.dpi*math.pi*.75*.75))

    if cb.iterable(node_size):
        try:
            vals = node_size.values()
        except:
            vals = node_size
        node_size = dict(zip(nodelist,map(area2radius,vals)))
    else:
        node_size = {}.fromkeys(nodelist,area2radius(node_size))
    for n in node_size:
        if node_size[n] == 0.0:
            node_size[n] = .00001

    if cmap is None:
        cmap = cm.get_cmap(mpl.rcParams['image.cmap'])

    n_colors = get_color_dict(node_color,nodelist,vmin,vmax,cmap)

    sym = syms[node_shape]
    numsides,rotation,symstyle=syms[node_shape]

    node_patches = {}
    for n in nodelist:
        if symstyle==0:
            node_patches[n] = patches.RegularPolygon(pos[n],
                                                     numsides,
                                                     orientation=rotation,
                                                     radius=node_size[n],
                                                     facecolor=n_colors[n],
                                                     edgecolor='k',
                                                     alpha=alpha,
                                                     linewidth=linewidth,
                                                     transform=ax.transData,
                                                     zorder=zorder)


        elif symstyle==3:
            node_patches[n] = patches.Circle(pos[n],
                                             radius=node_size[n],
                                             facecolor=n_colors[n],
                                             edgecolor='k',
                                             alpha=alpha,
                                             linewidth=linewidth,
                                             transform=ax.transData,
                                             zorder=zorder)
        ax.add_patch(node_patches[n])




    # the pad is a little hack to deal with the fact that we don't
    # want to transform all the symbols whose scales are in points
    # to data coords to get the exact bounding box for efficiency
    # reasons.  It can be done right if this is deemed important
    temp_x = xy[:,0]
    temp_y = xy[:,1]
    minx = np.amin(temp_x)
    maxx = np.amax(temp_x)
    miny = np.amin(temp_y)
    maxy = np.amax(temp_y)

    w = maxx-minx
    h = maxy-miny
    padx, pady = 0.05*w, 0.05*h
    corners = (minx-padx, miny-pady), (maxx+padx, maxy+pady)
    ax.update_datalim(corners)
#    ax.autoscale()
    ax.autoscale_view()
    ax.set_aspect('equal')
#   pylab.axes(ax)
    #pylab.sci(node_collection)
    #node_collection.set_zorder(2)
    return node_patches

def draw_networkx_edges(G, pos, node_patches=None,
                        edgelist=None,
                        width=None,
                        edge_color=None,
                        style='solid',
                        alpha=None,
                        edge_cmap=None,
                        edge_vmin=None,
                        edge_vmax=None,
                        ax=None,
                        arrows=True,
                        arrow_style=None,
                        connection_style='arc3',
                        color_weights=False,
                        width_weights=False,
                        **kwds):
    """Draw the edges of the graph G

    This draws only the edges of the graph G.

    Parameters
    ----------
    G : graph
       A networkx graph

    pos : dictionary
       A dictionary with nodes as keys and positions as values.
       If not specified a spring layout positioning will be computed.
       See networkx.layout for functions that compute node positions.

    ax : Matplotlib Axes object, optional
       Draw the graph in the specified Matplotlib axes.

    alpha: float
       The edge transparency (default=1.0)

    width`: float
       Line width of edges (default =1.0)

    edge_color: color string, or array of floats
       Edge color. Can be a single color format string (default='r'),
       or a sequence of colors with the same length as edgelist.
       If numeric values are specified they will be mapped to
       colors using the edge_cmap and edge_vmin,edge_vmax parameters.

    edge_ cmap: Matplotlib colormap
       Colormap for mapping intensities of edges (default=None)

    edge_vmin,edge_vmax: floats
       Minimum and maximum for edge colormap scaling (default=None)

    style: string
       Edge line style (default='solid') (solid|dashed|dotted,dashdot)
    arrow: Bool
       Whether to draw arrows or not for directed graphs
    arrow_style: string
       Arrow style used by matplotlib see FancyArrowPatch
    connection_style: string
       Connection style used by matplotlib, see FancyArrowPatch
    color_weights: Bool
       Whether to color the edges of a graph by their weight if the
       graph has any.
    width_weights: Bool
       Whether to vary the thicknes of an edge by their weight, if the
       graph has any.

    Examples
    --------
    >>> G=nx.dodecahedral_graph()
    >>> edges=nx.draw_networkx_edges(G,pos=nx.spring_layout(G))

    Also see the NetworkX drawing examples at
    http://networkx.lanl.gov/gallery.html

    See Also
    --------
    draw()
    draw_networkx()
    draw_networkx_nodes()
    draw_networkx_labels()
    draw_networkx_edge_labels()

    """
    try:
        import matplotlib
        import matplotlib.pylab as pylab
        import matplotlib.cbook as cb
        from matplotlib.colors import colorConverter,Colormap
        from matplotlib.collections import LineCollection
        import numpy
    except ImportError:
        raise ImportError, "Matplotlib required for draw()"
    except RuntimeError:
        print "Matplotlib unable to open display"
        raise

    if ax is None:
        ax=pylab.gca()

    if edgelist is None:
        edgelist=G.edges()

    if not edgelist or len(edgelist)==0: # no edges!
        return None

    # set edge positions
    edge_pos=numpy.asarray([(pos[e[0]],pos[e[1]]) for e in edgelist])


    if width is None and width_weights and is_weighted(G):
        lw = edge_width_weight(G,edgelist)
        if alpha is None:
            alpha = 0.75
    elif width is None:
        lw = {}.fromkeys(edgelist,1.0)
    elif cb.iterable(width):
        try:
            lwvals = width.values()
        except:
            lwvals = width
        lw = dict(zip(edgelist,lwvals))
    elif cb.is_scalar(width):
        lw = {}.fromkeys(edgelist,width)
    else:
        raise nx.NetworkXError("Must provide a single scalar value or a list \
                                of values for line width or None")


    if edge_cmap is None:
        edge_cmap = cm.get_cmap(mpl.rcParams['image.cmap'])

    if edge_color is None and color_weights and is_weighted(G):
        edge_color = edge_color_weight(G,edgelist)
        if alpha is None:
            alpha = 0.75
    elif edge_color is None:
        edge_color = 'k'

    e_colors = get_color_dict(edge_color,edgelist,edge_vmin,edge_vmax,edge_cmap)

    edge_patches = {}

    if arrow_style is None:
        if G.is_directed():
            arrow_style = '-|>'
        else:
            arrow_style = '-'

    if node_patches is None:
        node_patches = {}.fromkeys(G.nodes(),None)
    for (u,v) in edgelist:
        edge_patches[(u,v)] = patches.FancyArrowPatch(posA=pos[u],
                                                      posB=pos[v],
                                                      arrowstyle=arrow_style,
                                                      connectionstyle=connection_style,
                                                      patchA=node_patches[u],
                                                      patchB=node_patches[v],
                                                      shrinkA=0.0,
                                                      shrinkB=0.0,
                                                      mutation_scale=20.0,
                                                      alpha=alpha,
                                                      color=e_colors[(u,v)],
                                                      lw = lw[(u,v)],
                                                      linestyle=style,
                                                      zorder=1)
        ax.add_patch(edge_patches[(u,v)])

    # update view
    minx = numpy.amin(numpy.ravel(edge_pos[:,:,0]))
    maxx = numpy.amax(numpy.ravel(edge_pos[:,:,0]))
    miny = numpy.amin(numpy.ravel(edge_pos[:,:,1]))
    maxy = numpy.amax(numpy.ravel(edge_pos[:,:,1]))

    w = maxx-minx
    h = maxy-miny
    padx, pady = 0.05*w, 0.05*h
    corners = (minx-padx, miny-pady), (maxx+padx, maxy+pady)
    ax.update_datalim( corners)
    ax.autoscale_view()

    return edge_patches


def draw_networkx_labels(G, pos,
                         labels=None,
                         font_size=12,
                         font_color='k',
                         font_family='sans-serif',
                         font_weight='normal',
                         alpha=1.0,
                         ax=None,
                         **kwds):
    """Draw node labels on the graph G

    Parameters
    ----------
    G : graph
       A networkx graph

    pos : dictionary, optional
       A dictionary with nodes as keys and positions as values.
       If not specified a spring layout positioning will be computed.
       See networkx.layout for functions that compute node positions.

    ax : Matplotlib Axes object, optional
       Draw the graph in the specified Matplotlib axes.

    alpha: float
       The text transparency (default=1.0)

    labels: dictionary
       Node labels in a dictionary keyed by node of text labels (default=None)

    font_size: int
       Font size for text labels (default=12)

    font_color: string
       Font color string (default='k' black)

    font_weight: string
       Font weight (default='normal')

    font_family: string
       Font family (default='sans-serif')


    Examples
    --------
    >>> G=nx.dodecahedral_graph()
    >>> labels=nx.draw_networkx_labels(G,pos=nx.spring_layout(G))

    Also see the NetworkX drawing examples at
    http://networkx.lanl.gov/gallery.html


    See Also
    --------
    draw()
    draw_networkx()
    draw_networkx_nodes()
    draw_networkx_edges()
    draw_networkx_edge_labels()
    """
    try:
        import matplotlib.pylab as pylab
        import matplotlib.cbook as cb
    except ImportError:
        raise ImportError, "Matplotlib required for draw()"
    except RuntimeError:
        print "Matplotlib unable to open display"
        raise

    if ax is None:
        ax=pylab.gca()

    if labels is None:
        labels=dict(zip(G.nodes(),G.nodes()))

    text_items={}  # there is no text collection so we'll fake one
    for (n,label) in labels.items():
        (x,y)=pos[n]
        if not cb.is_string_like(label):
            label=str(label) # this will cause "1" and 1 to be labeled the same
        t=ax.text(x, y,
                  label,
                  size=font_size,
                  color=font_color,
                  family=font_family,
                  weight=font_weight,
                  horizontalalignment='center',
                  verticalalignment='center',
                  transform = ax.transData,
                  clip_on=True,
                  )
        text_items[n]=t

    return text_items

def draw_networkx_edge_labels(G, pos,
                              edge_labels=None,
                              font_size=10,
                              font_color='k',
                              font_family='sans-serif',
                              font_weight='normal',
                              alpha=1.0,
                              bbox=None,
                              ax=None,
                              **kwds):
    """Draw edge labels.

    Parameters
    ----------
    G : graph
       A networkx graph

    pos : dictionary, optional
       A dictionary with nodes as keys and positions as values.
       If not specified a spring layout positioning will be computed.
       See networkx.layout for functions that compute node positions.

    ax : Matplotlib Axes object, optional
       Draw the graph in the specified Matplotlib axes.

    alpha: float
       The text transparency (default=1.0)

    labels: dictionary
       Node labels in a dictionary keyed by edge two-tuple of text
       labels (default=None), Only labels for the keys in the dictionary
       are drawn.

    font_size: int
       Font size for text labels (default=12)

    font_color: string
       Font color string (default='k' black)

    font_weight: string
       Font weight (default='normal')

    font_family: string
       Font family (default='sans-serif')

    bbox: Matplotlib bbox
       Specify text box shape and colors.

    clip_on: bool
       Turn on clipping at axis boundaries (default=True)

    Examples
    --------
    >>> G=nx.dodecahedral_graph()
    >>> edge_labels=nx.draw_networkx_edge_labels(G,pos=nx.spring_layout(G))

    Also see the NetworkX drawing examples at
    http://networkx.lanl.gov/gallery.html

    See Also
    --------
    draw()
    draw_networkx()
    draw_networkx_nodes()
    draw_networkx_edges()
    draw_networkx_labels()

    """
    try:
        import matplotlib.pylab as pylab
        import matplotlib.cbook as cb
        import numpy
    except ImportError:
        raise ImportError, "Matplotlib required for draw()"
    except RuntimeError:
        print "Matplotlib unable to open display"
        raise

    if ax is None:
        ax=pylab.gca()
    if edge_labels is None:
        labels=dict(zip(G.edges(),[d for u,v,d in G.edges(data=True)]))
    else:
        labels = edge_labels
    text_items={}
    for ((n1,n2),label) in labels.items():
        (x1,y1)=pos[n1]
        (x2,y2)=pos[n2]
        (x,y) = ((x1+x2)/2, (y1+y2)/2)
        angle=numpy.arctan2(y2-y1,x2-x1)/(2.0*numpy.pi)*360 # degrees
        # make label orientation "right-side-up"
        if angle > 90:
            angle-=180
        if angle < - 90:
            angle+=180
        # transform data coordinate angle to screen coordinate angle
        xy=numpy.array((x,y))
        trans_angle=ax.transData.transform_angles(numpy.array((angle,)),
                                                  xy.reshape((1,2)))[0]
        # use default box of white with white border
        if bbox is None:
            bbox = dict(boxstyle='round',
                        ec=(1.0, 1.0, 1.0),
                        fc=(1.0, 1.0, 1.0),
                        )
        if not cb.is_string_like(label):
            label=str(label) # this will cause "1" and 1 to be labeled the same
        t=ax.text(x, y,
                  label,
                  size=font_size,
                  color=font_color,
                  family=font_family,
                  weight=font_weight,
                  horizontalalignment='center',
                  verticalalignment='center',
                  rotation=trans_angle,
                  transform = ax.transData,
                  bbox = bbox,
                  zorder = 1,
                  clip_on=True,
                  )
        text_items[(n1,n2)]=t

    return text_items



def draw_circular(G, **kwargs):
    """Draw the graph G with a circular layout"""
    draw(G,circular_layout(G),**kwargs)

def draw_random(G, **kwargs):
    """Draw the graph G with a random layout."""
    draw(G,random_layout(G),**kwargs)

def draw_spectral(G, **kwargs):
    """Draw the graph G with a spectral layout."""
    draw(G,spectral_layout(G),**kwargs)

def draw_spring(G, **kwargs):
    """Draw the graph G with a spring layout"""
    draw(G,spring_layout(G),**kwargs)

def draw_shell(G, **kwargs):
    """Draw networkx graph with shell layout"""
    nlist = kwargs.get('nlist', None)
    if nlist != None:
        del(kwargs['nlist'])
    draw(G,shell_layout(G,nlist=nlist),**kwargs)

def draw_graphviz(G, prog="neato", **kwargs):
    """Draw networkx graph with graphviz layout"""
    pos=nx.drawing.graphviz_layout(G,prog)
    draw(G,pos,**kwargs)

def draw_nx(G,pos,**kwds):
    """For backward compatibility; use draw or draw_networkx"""
    draw(G,pos,**kwds)

# fixture for nose tests
def setup_module(module):
    from nose import SkipTest
    try:
        import pylab
    except:
        raise SkipTest("matplotlib not available")

def test():
    import matplotlib.pylab as pyb
    G=nx.path_graph(10,create_using=nx.DiGraph())
    draw(G)
    pyb.draw()
    pyb.show()

