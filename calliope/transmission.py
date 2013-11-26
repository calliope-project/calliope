"""
Functions to deal with transmission between nodes
"""

from __future__ import print_function
from __future__ import division

from . import nodes
from . import utils


def get_remotes(y, x):
    y_remote = y.split(':')[0] + ':' + x
    x_remote = y.split(':')[1]
    return (y_remote, x_remote)


def get_transmission_techs(links):
    transmission_y = []
    for i in links:
        for y_base in links[i]:
            for x in nodes.explode_nodes(i):
                transmission_y.append(y_base + ':' + x)
    return transmission_y


def explode_transmission_tree(links, possible_x):
    tree = utils.AttrDict()
    for k in links:
        pairs = [nodes.explode_nodes(k)]
        pairs.append(list(reversed(pairs[0])))
        for x, remote_x in pairs:
            if x not in possible_x:
                raise KeyError('Transmission line to inexistent node.')
            if x not in tree:
                tree[x] = utils.AttrDict()
            for y in links[k]:
                tree[x][y + ':' + remote_x] = links[k][y]
    return tree
