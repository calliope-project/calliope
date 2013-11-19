"""
Functions to deal with nodes their configuration
"""

from __future__ import print_function
from __future__ import division

import pandas as pd


def _generate_node(node, items, techs):
    """
    Returns a dict for a given node. Dict keys for permitted technologies
    are the only ones that don't start with '_'.

    Args:
        node : (str) name of the node
        items : (AttrDict) node settings
        techs : (list) list of available technologies
    """
    # Mandatory basics
    d = {'_node': node, '_level': items.level, '_within': items.within}
    # Override
    if 'override' in items:
        for k in items.override.keys_nested():
            d['_override.' + k] = items.override.get_key(k)
    # Permitted echnologies
    for y in techs:
        if y in items.techs:
            d[y] = 1
        else:
            d[y] = 0
    return d


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
    rows = []
    for k, v in d.iteritems():
        if '--' in k or ',' in k:
            allnodes = _explode_nodes(k)
            for n in allnodes:
                rows.append(_generate_node(n, v, techs))
        else:
            rows.append(_generate_node(k, v, techs))
    df = pd.DataFrame.from_records(rows)
    df.index = df._node
    df = df.drop(['_node'], axis=1)
    return df
