"""
Copyright (C) 2013 Stefan Pfenninger.
Licensed under the Apache 2.0 License (see LICENSE file).

utils.py
~~~~~~~~

Various utility functions, particularly the AttrDict class (a subclass
of regular dict) used for managing model configuration.

"""

from __future__ import print_function
from __future__ import division

from contextlib import contextmanager
from cStringIO import StringIO
import functools
import os

import matplotlib.pyplot as plt
import yaml


class AttrDict(dict):
    """A subclass of ``dict`` with key access by attributes::

        d = AttrDict({'a': 1, 'b': 2})
        d.a == 1  # True

    Includes a range of additional methods to read and write to YAML,
    and to deal with nested keys.

    """
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, source_dict=None):
        super(AttrDict, self).__init__()
        if isinstance(source_dict, dict):
            self.init_from_dict(source_dict)

    def copy(self):
        """Override copy method so that it returns an AttrDict"""
        return AttrDict(dict(self).copy())

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
                self.set_key(k, v)

    @classmethod
    def from_yaml(cls, f, resolve_imports=True):
        """Returns an AttrDict initialized from the given path or
        file object ``f``, which must point to a YAML file.

        If ``resolve_imports`` is True, ``import:`` statements are
        resolved recursively, else they are treated like any other key.

        """
        if isinstance(f, str):
            with open(f, 'r') as src:
                loaded = cls(yaml.load(src))
        else:
            loaded = cls(yaml.load(f))
        if resolve_imports and 'import' in loaded:
            for k in loaded['import']:
                imported = cls.from_yaml(_resolve_path(f, k))
                loaded.union(imported)
            # 'import' key no longer needed, so we drop it
            loaded.pop('import', None)
        return loaded

    @classmethod
    def from_yaml_string(cls, string):
        """Returns an AttrDict initialized from the given string, which
        must be valid YAML.

        """
        return cls(yaml.load(string))

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
            except AttributeError:
                if self[key] is None:  # If the value is None, we replace it
                    self[key] = AttrDict()
                    self[key].set_key(remainder, value)
                # Else there is probably something there, and we don't just
                # want to overwrite so stop and warn the user
                else:
                    raise UserWarning('Cannot set nested key on non-dict key.')
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

    def to_yaml(self, path):
        """Saves the AttrDict to the given path as YAML file"""
        with open(path, 'w') as f:
            yaml.dump(self.as_dict(), f)

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

    def union(self, other, allow_override=False):
        """
        Merges the AttrDict in-place with the passed ``other`` dict or
        AttrDict. Keys in ``other`` take precedence, and nested keys
        are properly handled. If ``allow_override`` is False, a
        KeyError is raised if other tries to redefine an already
        defined key.

        """
        for k in other.keys_nested():
            if not allow_override and k in self.keys_nested():
                raise KeyError('Key defined twice: {}'.format(k))
            else:
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
        return functools.partial(self, obj)

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


def replace(string, placeholder, replacement):
    """Replace all occurences of ``{{placeholder}}`` or
    ``{{ placeholder }}`` in ``string`` with ``replacement``.

    """
    placeholders = ['{{ ' + placeholder + ' }}',
                    '{{' + placeholder + '}}']
    for p in placeholders:
        string = string.replace(p, replacement)
    return string


def _resolve_path(base_path, path):
    path = replace(path, placeholder='module',
                   replacement=os.path.dirname(__file__))
    if not os.path.isabs(path):
        path = os.path.join(os.path.dirname(base_path), path)
    return path


def stack_plot(df, stack, figsize=None, colormap='jet', **kwargs):
    if not figsize:
        figsize = (16, 4)
    colors = plt.get_cmap(colormap)(np.linspace(0, 1.0, len(stack)))
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    fills = ax.stackplot(df.index, df[stack].T, label=stack, colors=colors,
                         **kwargs)
    # Legend via proxy artists
    # Based on https://github.com/matplotlib/matplotlib/issues/1943
    proxies = [plt.Rectangle((0, 0), 1, 1, fc=i.get_facecolor()[0])
               for i in fills]
    ax.legend(reversed(proxies), reversed(stack))
    # Format x datetime axis
    # Based on http://stackoverflow.com/a/9627970/397746
    # TODO check how pandas does its very nice formatting for df.plot()
    import matplotlib.dates as mdates
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator())
    return ax
