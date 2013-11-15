"""
Functions to deal with nodes their configuration
"""

from __future__ import print_function
from __future__ import division

import pandas as pd

from . import utils


def _append_node(v, d):
    """Appends the settings for a node given by the AttrDict `v`
    to the AttrDict `d`

    """
    techs = [k for k in d.keys() if not k.startswith('_')]
    d._level.append(v.level)
    d._within.append(v.within)
    for y in techs:
        if y in v.techs:
            d[y].append(1)
        else:
            d[y].append(0)


def _explode_nodes(k):
    """Expands keys of the form '1--3' into the list form ['1', '2', '3'],
    and keys of the form '1,3,4' into the list form ['1', '3', '4'].
    Can deal with any combination, e.g. '1--3,6,9--12'.

    Always returns a list, even if `k` is just a single key, i.e.
    _explode_nodes('1') returns ['1'].

    """
    finalkeys = []
    subkeys = k.split(',')
    for sk in subkeys:
        if '--' in sk:
            begin, end = sk.split('--')
            finalkeys += [str(i).strip()
                          for i in range(int(begin), int(end)+1)]
        else:
            finalkeys += [sk.strip()]
    return finalkeys


def get_nodes(d):
    """ Return a list of all nodes in the given dictionary, expanding
    nodes in compact representation (such as '1--10') as needed.

    """
    l = []
    for k in d.keys():
        k = _explode_nodes(k)
        l.extend(k)
    return l


def generate_node_matrix(d, techs):
    """Generate a pandas DataFrame indexed by nodes, containing a column
    for each technology in `techs` and 1 if that node is allowed to
    use the technology, else 0.

    The DataFrame also contains _level and _within columns for grouping
    nodes into layers and grid zones, both currently unused.

    """
    # Beware: all keys of the AttrDict that don't start with '_' are
    # considered techs in append_node()!
    df = utils.AttrDict({'_node': [], '_level': [], '_within': []})
    for y in techs:
        df[y] = []
    for k, v in d.iteritems():
        if '--' in k or ',' in k:
            allnodes = _explode_nodes(k)
            for n in allnodes:
                df._node.append(n)
                _append_node(v, df)
        else:
            df._node.append(k)
            _append_node(v, df)
    df = pd.DataFrame(df)
    df.index = df._node
    df = df.drop(['_node'], axis=1)
    return df
