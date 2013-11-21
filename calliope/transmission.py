"""
Functions to deal with transmission between nodes
"""

from __future__ import print_function
from __future__ import division

from . import nodes
from . import utils


def get_transmission_techs(model):
    transmission_y = []
    links = model.config_run.links
    for i in links:
        for y_base in links[i]:
            for x in nodes.explode_nodes(i):
                transmission_y.append(y_base + ':' + x)
    return transmission_y


def explode_transmission_tree(model):
    tree = utils.AttrDict()
    links = model.config_run.links
    for k in links:
        pairs = [nodes.explode_nodes(k)]
        pairs.append(list(reversed(pairs[0])))
        for x, remote_x in pairs:
            if x not in model.data._x:
                raise KeyError('Transmission line to inexistent node.')
            if x not in tree:
                tree[x] = utils.AttrDict()
            for y in links[k]:
                tree[x][y + ':' + remote_x] = links[k][y]
    return tree
