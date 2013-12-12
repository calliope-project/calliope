"""
Functions to deal with transmission between locations
"""

from __future__ import print_function
from __future__ import division

from . import locations
from . import utils


def get_remotes(y, x):
    """For a given pair of ``y`` (tech) and ``x`` (location), return
    ``(y_remote, x_remote)``, a tuple giving the corresponding indices
    of the remote location a transmission technology is connected to.
    """
    y_remote = y.split(':')[0] + ':' + x
    x_remote = y.split(':')[1]
    return (y_remote, x_remote)


def get_transmission_techs(links):
    """Extract a list of all transmission technologies needed for the
    given ``links`` (an AttrDict). Returns an empty list if ``links`` is
    empty.

    """
    if links is None:
        return []
    transmission_y = []
    for i in links:
        for y_base in links[i]:
            for x in locations.explode_locations(i):
                transmission_y.append(y_base + ':' + x)
    # Pass through set to remove duplicates
    return list(set(transmission_y))


def explode_transmission_tree(links, possible_x):
    """Return an AttrDict with configuration for all possible transmission
    technologies defined by ``links``, checking if they have been defined
    to a location within ``possible_x`` (which can be a list or othe iterable).

    Returns None if ``links`` empty.

    """
    if links is None:
        return None
    tree = utils.AttrDict()
    for k in links:
        pairs = [locations.explode_locations(k)]
        pairs.append(list(reversed(pairs[0])))
        for x, remote_x in pairs:
            if x not in possible_x:
                raise KeyError('Link to inexistent location.')
            if x == remote_x:
                raise KeyError('Link must be between different locations.')
            if x not in tree:
                tree[x] = utils.AttrDict()
            for y in links[k]:
                tree[x][y + ':' + remote_x] = links[k][y]
    return tree
