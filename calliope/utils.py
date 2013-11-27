from __future__ import print_function
from __future__ import division

from contextlib import contextmanager
from cStringIO import StringIO
from functools import partial
import yaml


class AttrDict(dict):
    """A subclass of ``dict`` with key access by attributes::

        d = AttrDict({'a': 1, 'b': 2})
        d.a == 1  # True

    """
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, arg=None):
        super(AttrDict, self).__init__()
        if isinstance(arg, dict):
            self.init_from_dict(arg)

    def init_from_dict(self, d):
        """Initialize a new AttrDict from the given dict. Handles any
        nested dicts by turning them into AttrDicts too::

            d = AttrDict({'a': 1, 'b': {'x': 1, 'y': 2}})
            d.b.x == 1  # True

        """
        for k, v in d.iteritems():
            if isinstance(k, int):
                k = str(k)  # Keys must be strings, not ints
            if isinstance(v, dict):
                self[k] = AttrDict(v)
            else:
                self[k] = v

    @classmethod
    def from_yaml(self, f):
        """Returns an AttrDict initialized from the given path or
        file object ``f``, which must point to a YAML file.

        """
        if isinstance(f, str):
            with open(f, 'r') as src:
                return AttrDict(yaml.load(src))
        else:
            return AttrDict(yaml.load(f))

    def set_key(self, key, value):
        """Set the given ``key`` to the given ``value``. Handles nested
        keys, e.g.::

            d = AttrDict()
            d.set_key('foo.bar', 1)
            d.foo.bar == 1  # True

        """
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
        """Looks up the given ``key``. Like set_key(), deals with nested
        keys.

        If default is given and not None (it may be, for example, False),
        returns default if encounters a KeyError during lookup

        """
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
        """Return the AttrDict as a pure dict (with nested dicts if
        necessary).

        """
        d = {}
        for k, v in self.iteritems():
            if isinstance(v, AttrDict):
                d[k] = v.as_dict()
            else:
                d[k] = v
        return d

    def keys_nested(self, subkeys_as='list'):
        """Returns all keys in the AttrDict, including the keys of
        nested subdicts (which may be either regular dicts or AttrDicts).

        If ``subkeys_as='list'`` (default), then a (sorted) list of
        all keys is returned, in the form ``['a', 'b.b1']``.

        If ``subkeys_as='dict'``, a list containing keys and dicts of
        subkeys is returned, in the form ``['a', {'b': [b1]}]``. The list
        is sorted (subdicts first, then string keys).

        """
        keys = []
        for k, v in self.iteritems():
            if isinstance(v, AttrDict) or isinstance(v, dict):
                if subkeys_as == 'list':
                    keys.extend([k + '.' + kk for kk in v.keys_nested()])
                elif subkeys_as == 'dict':
                    keys.append({k: v.keys_nested(subkeys_as=subkeys_as)})
            else:
                keys.append(k)
        return sorted(keys)

    def union(self, other):
        """
        Merges the AttrDict in-place with the passed ``other`` dict or
        AttrDict. Keys in ``other`` take precedence, and nested keys
        are properly handled.

        """
        for k in other.keys_nested():
            self.set_key(k, other.get_key(k))


@contextmanager
def capture_output():
    """Capture stdout and stderr output of a wrapped function::

        with capture_output() as out:
            # do things that create stdout or stderr output

    Returns a list with the captured strings: ``[stderr, stdout]``

    """
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
    """ Memoization decorator for a function taking one or more
    arguments.

    """
    class MemoDict(dict):
        def __getitem__(self, *key):
            return dict.__getitem__(self, key)

        def __missing__(self, key):
            ret = self[key] = f(*key)
            return ret

    return MemoDict().__getitem__


class memoize_instancemethod(object):
    """Cache the return value of a method

    Source: http://code.activestate.com/recipes/577452/

    This class is meant to be used as a decorator of methods. The return
    value from a given method invocation will be cached on the instance
    whose method was invoked. All arguments passed to a method decorated
    with memoize must be hashable.

    If a memoized method is invoked directly on its class the result
    will not be cached. Instead the method will be invoked like a
    static method.

    """
    def __init__(self, func):
        self.func = func

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self.func
        return partial(self, obj)

    def __call__(self, *args, **kw):
        obj = args[0]
        try:
            cache = obj.__cache
        except AttributeError:
            cache = obj.__cache = {}
        key = (self.func, args[1:], frozenset(kw.items()))
        try:
            res = cache[key]
        except KeyError:
            res = cache[key] = self.func(*args, **kw)
        return res


def replace_all(paths, placeholder, replacement):
    """Replace all occurences of ``{{placeholder}}`` or
    ``{{ placeholder }}`` in ``paths`` (a list of strings) with
    ``replacement``. Returns a list of strings. Error if ``paths`` not
    a list of strings.

    """
    placeholders = ['{{ ' + placeholder + ' }}',
                    '{{' + placeholder + '}}']
    for i, path in enumerate(paths):
        for p in placeholders:
            new_path = paths[i].replace(p, replacement)
            paths[i] = new_path
    return paths
