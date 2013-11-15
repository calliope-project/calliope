from __future__ import print_function
from __future__ import division

from contextlib import contextmanager
from cStringIO import StringIO


class AttrDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, arg=None):
        super(AttrDict, self).__init__()
        if isinstance(arg, dict):
            self.init_from_dict(arg)

    def init_from_dict(self, d):
        for k, v in d.iteritems():
            if isinstance(k, int):
                k = str(k)  # Keys must be strings, not ints
            if isinstance(v, dict):
                self[k] = AttrDict(v)
            else:
                self[k] = v

    def set_key(self, key, value):
        if '.' in key:
            key, remainder = key.split('.', 1)
            try:
                self[key].set_key(remainder, value)
            except KeyError:
                self[key] = AttrDict()
                self[key].set_key(remainder, value)
        else:
            self[key] = value

    def get_key(self, key, default=None):
        """If default is given and not None (it may be, for example, False),
        returns default if encounters a KeyError during lookup"""
        if '.' in key:
        # Nested key of form "foo.bar"
            key, remainder = key.split('.', 1)
            if default is not None:
                try:
                    value = self[key].get_key(remainder)
                except KeyError:
                    return default
            else:
                value = self[key].get_key(remainder)
        else:
        # Single, non-nested key of form "foo"
            if default is not None:
                return self.get(key, default)
            else:
                return self[key]
        return value

    def as_dict(self):
        d = {}
        for k, v in self.iteritems():
            if isinstance(v, AttrDict):
                d[k] = v.as_dict()
            else:
                d[k] = v
        return d

    def keys_nested(self, subkeys_as='list'):
        keys = []
        for k, v in self.iteritems():
            if isinstance(v, AttrDict):
                if subkeys_as == 'list':
                    keys.extend([k + '.' + kk for kk in v.keys_nested()])
                elif subkeys_as == 'dict':
                    keys.append({k: v.keys_nested(subkeys_as=subkeys_as)})
            else:
                keys.append(k)
        return keys


def invert_dict(d):
    """Inverts a dict of the form:

    {'a': ['b-1', 'b-2'], 'b': ['bbb'], 'c': 'foo'}

    to:

    {'b-1': 'a', 'b-2': 'a', 'bbb': 'b', 'foo': 'c'}

    """
    mapping = {}
    for key, v in [(v, k) for k, v in d.iteritems()]:
        if isinstance(key, list):
            for k in key:
                mapping[k] = v
        else:
            mapping[key] = v
    return mapping


@contextmanager
def capture_output():
    import sys
    old_out, old_err = sys.stdout, sys.stderr
    try:
        out = [StringIO(), StringIO()]
        sys.stdout, sys.stderr = out
        yield out
    finally:
        sys.stdout, sys.stderr = old_out, old_err
        out[0] = out[0].getvalue()
        out[1] = out[1].getvalue()


def memoize(f):
    """ Memoization decorator for a function taking one or more arguments."""
    class MemoDict(dict):
        def __getitem__(self, *key):
            return dict.__getitem__(self, key)

        def __missing__(self, key):
            ret = self[key] = f(*key)
            return ret

    return MemoDict().__getitem__
