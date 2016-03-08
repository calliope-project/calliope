"""
Copyright (C) 2013-2016 Stefan Pfenninger.
Licensed under the Apache 2.0 License (see LICENSE file).

utils.py
~~~~~~~~

Various utility functions, particularly the AttrDict class (a subclass
of regular dict) used for managing model configuration.

"""

from contextlib import contextmanager
from io import StringIO
import functools
import os

import numpy as np
import yaml

from . import exceptions


class __Missing(object):
    def __repr__(self):
        return ('MISSING')

    def __nonzero__(self):
        return False


_MISSING = __Missing()


class AttrDict(dict):
    """
    A subclass of ``dict`` with key access by attributes::

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
        """
        Initialize a new AttrDict from the given dict. Handles any
        nested dicts by turning them into AttrDicts too::

            d = AttrDict({'a': 1, 'b': {'x': 1, 'y': 2}})
            d.b.x == 1  # True

        """
        for k, v in d.items():
            # First, keys must be strings, not ints
            if isinstance(k, int):
                k = str(k)
            # Now, assign to the key, handling nested AttrDicts properly
            if isinstance(v, dict):
                self.set_key(k, AttrDict(v))
            elif isinstance(v, list):
                self.set_key(k, [i if not isinstance(i, dict) else AttrDict(i)
                                 for i in v])
            else:
                self.set_key(k, v)

    @classmethod
    def from_yaml(cls, f, resolve_imports=True):
        """
        Returns an AttrDict initialized from the given path or
        file object ``f``, which must point to a YAML file.

        If ``resolve_imports`` is True, ``import:`` statements are
        resolved recursively, else they are treated like any other key.

        When resolving import statements, anything defined locally
        overrides definitions in the imported file.

        """
        if isinstance(f, str):
            with open(f, 'r') as src:
                loaded = cls(yaml.load(src))
        else:
            loaded = cls(yaml.load(f))
        if resolve_imports and 'import' in loaded:
            for k in loaded['import']:
                imported = cls.from_yaml(relative_path(k, f))
                # loaded is added to imported (i.e. it takes precedence)
                imported.union(loaded)
                loaded = imported
            # 'import' key no longer needed, so we drop it
            loaded.pop('import', None)
        return loaded

    @classmethod
    def from_yaml_string(cls, string):
        """
        Returns an AttrDict initialized from the given string, which
        must be valid YAML.

        """
        return cls(yaml.load(string))

    def set_key(self, key, value):
        """
        Set the given ``key`` to the given ``value``. Handles nested
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
                    raise KeyError('Cannot set nested key on non-dict key.')
        else:
            self[key] = value

    def get_key(self, key, default=_MISSING):
        """
        Looks up the given ``key``. Like set_key(), deals with nested
        keys.

        If default is anything but ``_MISSING``, the given default is
        returned if the key does not exist.

        """
        if '.' in key:
            # Nested key of form "foo.bar"
            key, remainder = key.split('.', 1)
            if default != _MISSING:
                try:
                    value = self[key].get_key(remainder, default)
                except KeyError:
                    # subdict exists, but doesn't contain key
                    return default
                except AttributeError:
                    # key points to non-dict thing, so no get_key attribute
                    return default
            else:
                value = self[key].get_key(remainder)
        else:
            # Single, non-nested key of form "foo"
            if default != _MISSING:
                return self.get(key, default)
            else:
                return self[key]
        return value

    def del_key(self, key):
        """Delete the given key. Properly deals with nested keys."""
        if '.' in key:
            key, remainder = key.split('.', 1)
            try:
                del self[key][remainder]
            except KeyError:
                self[key].del_key(remainder)
        else:
            del self[key]

    def as_dict(self, flat=False):
        """
        Return the AttrDict as a pure dict (with nested dicts if
        necessary).

        """
        if not flat:
            return self.as_dict_nested()
        else:
            return self.as_dict_flat()

    def as_dict_nested(self):
        d = {}
        for k, v in self.items():
            if isinstance(v, AttrDict):
                d[k] = v.as_dict()
            elif isinstance(v, list):
                d[k] = [i if not isinstance(i, AttrDict) else i.as_dict()
                        for i in v]
            else:
                d[k] = v
        return d

    def as_dict_flat(self):
        d = {}
        keys = self.keys_nested()
        for k in keys:
            d[k] = self.get_key(k)
        return d

    def to_yaml(self, path=None, convert_objects=True, **kwargs):
        """
        Saves the AttrDict to the given path as a YAML file.

        If ``path`` is None, returns the YAML string instead.

        Any additional keyword arguments are passed to the YAML writer,
        so can use e.g. ``indent=4`` to override the default of 2.

        ``convert_objects`` (defaults to True) controls whether Numpy
        objects should be converted to regular Python objects, so that
        they are properly displayed in the resulting YAML output.

        """
        if convert_objects:
            result = self.copy()
            for k in result.keys_nested():
                # Convert numpy numbers to regular python ones
                v = result.get_key(k)
                if isinstance(v, np.floating):
                    result.set_key(k, float(v))
                elif isinstance(v, np.integer):
                    result.set_key(k, int(v))
            result = result.as_dict()
        else:
            result = self.as_dict()
        if path is not None:
            with open(path, 'w') as f:
                yaml.dump(result, f, **kwargs)
        else:
            return yaml.dump(result, **kwargs)

    def keys_nested(self, subkeys_as='list'):
        """
        Returns all keys in the AttrDict, sorted, including the keys of
        nested subdicts (which may be either regular dicts or AttrDicts).

        If ``subkeys_as='list'`` (default), then a list of
        all keys is returned, in the form ``['a', 'b.b1', 'b.b2']``.

        If ``subkeys_as='dict'``, a list containing keys and dicts of
        subkeys is returned, in the form ``['a', {'b': ['b1', 'b2']}]``.

        """
        keys = []
        for k, v in sorted(self.items()):
            if isinstance(v, AttrDict) or isinstance(v, dict):
                if subkeys_as == 'list':
                    keys.extend([k + '.' + kk for kk in v.keys_nested()])
                elif subkeys_as == 'dict':
                    keys.append({k: v.keys_nested(subkeys_as=subkeys_as)})
            else:
                keys.append(k)
        return keys

    def union(self, other, allow_override=False, allow_replacement=False):
        """
        Merges the AttrDict in-place with the passed ``other``
        AttrDict. Keys in ``other`` take precedence, and nested keys
        are properly handled.

        If ``allow_override`` is False, a KeyError is raised if
        other tries to redefine an already defined key.

        If ``allow_replacement``, allow "_REPLACE_" key to replace an
        entire sub-dict.

        """
        if allow_replacement:
            WIPE_KEY = '_REPLACE_'
            override_keys = [k for k in other.keys_nested()
                             if WIPE_KEY not in k]
            wipe_keys = [k.split('.' + WIPE_KEY)[0]
                         for k in other.keys_nested()
                         if WIPE_KEY in k]
        else:
            override_keys = other.keys_nested()
            wipe_keys = []
        for k in override_keys:
            if not allow_override and k in self.keys_nested():
                raise KeyError('Key defined twice: {}'.format(k))
            else:
                self.set_key(k, other.get_key(k))
        for k in wipe_keys:
            self.set_key(k, other.get_key(k + '.' + WIPE_KEY))


@contextmanager
def capture_output():
    """
    Capture stdout and stderr output of a wrapped function::

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


# This used to be a custom function, but as of Python 3.2 we can use
# the built-in lru_cache for simplicity
memoize = functools.lru_cache(maxsize=512)


class memoize_instancemethod(object):
    """
    Cache the return value of a method on a per-instance basis
    (as opposed to a per-class basis like functools.lru_cache does)

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
        key = (self.func, args[1:], frozenset(list(kw.items())))
        try:
            res = cache[key]
        except KeyError:
            res = cache[key] = self.func(*args, **kw)
        return res


def relative_path(path, base_path_file):
    """
    If ``path`` is not absolute, it is interpreted as relative to the
    path of the given ``base_path_file``.

    """
    # Check if base_path_file is a string because it might be an AttrDict
    if not os.path.isabs(path) and isinstance(base_path_file, str):
        path = os.path.join(os.path.dirname(base_path_file), path)
    return path


def option_getter(config_model, data):
    """Returns a get_option() function using the given config_model and data"""
    o = config_model
    d = data

    def get_option(option, x=None, default=None, ignore_inheritance=False):

        def _get_option(opt, fail=False):
            try:
                result = o.get_key('techs.' + opt)
            except KeyError:
                if ignore_inheritance:
                    return _get_option(default, fail)
                # 'ccgt.constraints.s_time' -> 'ccgt', 'constraints.s_time'
                tech, remainder = opt.split('.', 1)
                if ':' in tech:
                    parent = tech.split(':')[0]
                else:
                    # parent = e.g. 'defaults'
                    parent = o.get_key('techs.' + tech + '.parent')
                try:
                    result = _get_option(parent + '.' + remainder, fail)
                except KeyError:
                    e = exceptions.OptionNotSetError
                    if fail:
                        raise e('Failed to read option `{}` '
                                'with given default '
                                '`{}`'.format(option, default))
                    elif default:
                        result = _get_option(default, fail=True)
                    elif tech == 'defaults':
                        raise e('Reached top of inheritance chain '
                                'and no default defined for: '
                                '`{}`'.format(option))
                    else:
                        raise e('Can not get parent for `{}` '
                                'and no default defined '
                                '({}).'.format(tech, option))
            return result

        def _get_location_option(key, location):

            def getter(key, location):
                # NB: KeyErrors raised here are always caught within _get_option
                # so need no further information or handling
                # Raises KeyError if the specific _override column does not exist
                result = d.locations.at[location, '_override.' + key]
                # Also raise KeyError if the result is NaN, i.e. if no
                # location-specific override has been defined
                try:
                    if np.isnan(result):
                        raise KeyError
                # Have to catch this because np.isnan not implemented for strings
                except TypeError:
                    pass
                return result

            while True:
                try:
                    return getter(key, location)
                except KeyError:
                    parent_location = d.locations.at[location, '_within']
                    if parent_location:  # Will be None if no parent
                        return getter(key, parent_location)
                    else:
                        # Once top of "location inheritance" chain reached,
                        # raise KeyError, which will cause the calling function
                        # to fall back to non-location specific settings
                        raise

        if x:
            try:
                result = _get_location_option(option, x)
            # If can't find a location-specific option, fall back to model-wide
            except KeyError:
                result = _get_option(option)
        else:
            result = _get_option(option)
        # Deal with 'inf' settings
        if result == 'inf':
            result = float('inf')
        return result

    return get_option


def cost_getter(option_getter, costs_type='costs'):
    def get_cost(cost, y, k, x=None):
        return option_getter(y + '.' + costs_type + '.' + k + '.' + cost,
                             default=y + '.' + costs_type + '.default.' + cost,
                             x=x)
    return get_cost


def cost_per_distance_getter(option_getter):
    def get_cost_per_distance(cost, y, k, x):
        try:
            cost_option = y + '.costs_per_distance.' + k + '.' + cost
            cost = option_getter(cost_option)
            per_distance = option_getter(y + '.per_distance')
            distance = option_getter(y + '.distance', x=x)
            distance_cost = cost * (distance / per_distance)
        except exceptions.OptionNotSetError:
            distance_cost = 0
        return distance_cost
    return get_cost_per_distance


def depreciation_getter(option_getter):
    def get_depreciation_rate(y, k):
        interest = option_getter(y + '.depreciation.interest.' + k,
                                 default=y + '.depreciation.interest.default')
        plant_life = option_getter(y + '.depreciation.plant_life')
        if interest == 0:
            dep = 1 / plant_life
        else:
            dep = ((interest * (1 + interest) ** plant_life)
                   / (((1 + interest) ** plant_life) - 1))
        return dep
    return get_depreciation_rate


def any_option_getter(model):
    """
    Get any option from the given Model or SolutionModel, including
    ``costs.`` or ``costs_per_distance.`` options

    """
    get_cost = cost_getter(model.get_option)
    get_cost_pd = cost_getter(model.get_option, 'costs_per_distance')

    def get_any_option(option):
        if 'costs.' in option:
            y, rest, k, cost = option.split('.')
            return get_cost(cost, y, k)
        elif 'costs_per_distance.' in option:
            y, rest, k, cost = option.split('.')
            return get_cost_pd(cost, y, k)
        else:
            return model.get_option(option)
    return get_any_option
