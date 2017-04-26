"""
Copyright (C) 2013-2017 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

utils.py
~~~~~~~~

Various utility functions, particularly the AttrDict class (a subclass
of regular dict) used for managing model configuration.

"""

from contextlib import contextmanager
from io import StringIO
import functools
import logging
import os
import importlib
import sys

import numpy as np
import yaml

from . import exceptions


class __Missing(object):
    def __repr__(self):
        return ('MISSING')

    def __nonzero__(self):
        return False


_MISSING = __Missing()


def _yaml_load(src):
    """Load YAML from a file object or path with useful parser errors"""
    if not isinstance(src, str):
        try:
            src_name = src.name
        except AttributeError:
            src_name = '<yaml stringio>'
        # Force-load file streams as that allows the parser to print
        # much more context when it encounters an error
        src = src.read()
    else:
        src_name = '<yaml string>'
    try:
        return yaml.load(src)
    except yaml.YAMLError:
        logging.error('Parser error when reading YAML from {}.'.format(src_name))
        raise


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
                loaded = cls(_yaml_load(src))
        else:
            loaded = cls(_yaml_load(f))
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
        return cls(_yaml_load(string))

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


def _load_function(source):
    """
    Returns a function from a module, given a source string of the form:

        'module.submodule.subsubmodule.function_name'

    """
    module_string, function_string = source.rsplit('.', 1)
    modules = [i for i in sys.modules.keys() if 'calliope' in i]
    # Check if module already loaded, if so, don't re-import it
    if (module_string in modules):
        module = sys.modules[module_string]
    elif ('calliope.' + module_string) in modules:
        module = sys.modules['calliope.' + module_string]
    # Else load the module
    else:
        try:
            module = importlib.import_module(module_string)
        except ImportError:
            module = importlib.import_module('calliope.' + module_string)
    return getattr(module, function_string)


def plugin_load(name, builtin_module):
    try:  # First try importing as a third-party module
        func = _load_function(name)
    except ValueError:
        # ValueError raised if we got a string without '.',
        # which implies a builtin function,
        # so we attempt to load from the given module
        func_string = builtin_module + '.' + name
        func = _load_function(func_string)
    return func


def option_getter(config_model):
    """Returns a get_option() function using the given config_model and data"""

    def get_option(option, x=None, default=None, ignore_inheritance=False):

        def _get_option(opt, fail=False):
            try:
                result = config_model.get_key('techs.' + opt)
            except KeyError:
                if ignore_inheritance:
                    return _get_option(default, fail)
                # 'ccgt.constraints.s_time' -> 'ccgt', 'constraints.s_time'
                tech, remainder = opt.split('.', 1)
                if ':' in tech: # transmission
                    parent = tech.split(':')[0]
                else:
                    # parent = e.g. 'defaults'
                    parent = config_model.get_key('techs.' + tech + '.parent')
                try:
                    result = _get_option(parent + '.' + remainder, fail)
                except KeyError:
                    e = exceptions.OptionNotSetError
                    if fail:
                        raise e('Failed to read option `{}` '
                                'with given default '
                                '`{}`'.format(option, default))
                    elif default:
                        if not isinstance(default, str): #allow setting the default directly as anything that isn't a string - could do with being more robust
                            result = default
                        else:
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
            # NB1: KeyErrors raised here are always caught in get_option
            # so need no further information or handling
            return config_model.get_key(
                'locations.' + location + '.override.' + key
            )

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


def cost_getter(option_getter_func):
    def get_cost(cost, y, k, x=None, costs_type='costs'):
        return option_getter_func(y + '.' + costs_type + '.' + k + '.' + cost,
                             default=y + '.' + costs_type + '.default.' + cost,
                             x=x)
    return get_cost


def cost_per_distance_getter(config_model):
    option_getter_func = option_getter(config_model)
    def get_cost_per_distance(cost, y, k, x):
        _cost = cost_getter(option_getter_func)
        cost = _cost(cost, y, k, x, costs_type='costs_per_distance')
        tech, x2 = y.split(':')
        per_distance = option_getter_func(y + '.per_distance')
        link = config_model.get_key('links.'+ x + ',' + x2,
            default=config_model['links'].get(x2 + ',' + x))
        # link = None if no link exists
        if not link or tech not in link.keys():
            return 0
        try:
            distance = link.get_key(tech + '.distance')
        except KeyError:
            if cost > 0:
                e = exceptions.OptionNotSetError
                raise e('Distance must be defined for link: {} '
                        'and transmission tech: {}, as cost_per_distance '
                        'is defined'.format(x + ',' + x2, tech))
            else:
                return 0
        distance_cost = cost * (distance / per_distance)
        return distance_cost
    return get_cost_per_distance


def depreciation_getter(option_getter_func):
    def get_depreciation_rate(y, k):
        interest = option_getter_func(
            y + '.depreciation.interest.' + k,
            default=y + '.depreciation.interest.default')
        plant_life = option_getter_func(y + '.depreciation.plant_life')
        if interest == 0:
            dep = 1 / plant_life
        else:
            dep = ((interest * (1 + interest) ** plant_life)
                   / (((1 + interest) ** plant_life) - 1))
        return dep
    return get_depreciation_rate


def any_option_getter(model):
    """
    Get any option from the given Model, including
    ``costs.`` or ``costs_per_distance.`` options

    """
    get_cost = cost_getter(model.get_option)
    get_cost_pd = cost_per_distance_getter(model.get_option)

    def get_any_option(option, x=None, t=None):
        if 'costs.' in option:
            if t and x:
                y, cost_type, k, cost = option.split('.')
                try:
                    return model.data['_'.join([cost_type,k,cost])].sel(y=y,x=x,t=t)
                except:
                    return model.get_cost(cost, y, k, x=x, costs_type=cost_type)
            elif t and not x:
                e = exceptions.OptionNotSetError
                raise e('must define location for time dependant variable '
                    '`{}`'.format(option))
            else:
                y, cost_type, k, cost = option.split('.')
                return model.get_cost(cost, y, k, x=x, costs_type=cost_type)
        else:
            if t and x:
                y = option.split('.', 1)[0]
                field = option.rsplit('.', 1)[-1]
                try:
                    return model.data[field].sel(y=y,x=x,t=t)
                except:
                    return model.get_option(option, x=x)
            elif t and not x:
                e = exceptions.OptionNotSetError
                raise e('must define location for time dependant variable '
                        '`{}`'.format(option))
            elif x and not t:
                return model.get_option(option, x=x)
            else:
                if 'costs_per_distance.' in option:
                    y, rest, k, cost = option.split('.')
                    return get_cost_pd(cost, y, k)
                else:
                    return model.get_option(option)
    return get_any_option


def vincenty(coord1, coord2):
    """
    Vincenty's inverse method formula to calculate the distance in metres
    between two points on the surface of a spheroid (WGS84).
    modified from https://github.com/maurycyp/vincenty
    """

    a = 6378137  # equitorial radius in meters
    f = 1 / 298.257223563 # flattening from sphere to oblate spheroid
    b = a * (1 - f) # polar radius in meters

    max_iter = 200
    thresh = 1e-12

    # short-circuit coincident points
    if coord1[0] == coord2[0] and coord1[1] == coord2[1]:
        return 0

    U1 = np.arctan((1 - f) * np.tan(np.radians(coord1[0])))
    U2 = np.arctan((1 - f) * np.tan(np.radians(coord2[0])))
    L = np.radians(coord2[1] - coord1[1])
    Lambda = L

    sinU1 = np.sin(U1)
    cosU1 = np.cos(U1)
    sinU2 = np.sin(U2)
    cosU2 = np.cos(U2)

    for iteration in range(max_iter):
        sinLambda = np.sin(Lambda)
        cosLambda = np.cos(Lambda)
        sinSigma = np.sqrt((cosU2 * sinLambda) ** 2 +
                             (cosU1 * sinU2 - sinU1 * cosU2 * cosLambda) ** 2)
        if sinSigma == 0:
            return 0.0  # coincident points
        cosSigma = sinU1 * sinU2 + cosU1 * cosU2 * cosLambda
        sigma = np.arctan2(sinSigma, cosSigma)
        sinAlpha = cosU1 * cosU2 * sinLambda / sinSigma
        cosSqAlpha = 1 - sinAlpha ** 2
        try:
            cos2SigmaM = cosSigma - 2 * sinU1 * sinU2 / cosSqAlpha
        except ZeroDivisionError:
            cos2SigmaM = 0
        C = f / 16 * cosSqAlpha * (4 + f * (4 - 3 * cosSqAlpha))
        LambdaPrev = Lambda
        Lambda = L + (1 - C) * f * sinAlpha * (sigma + C * sinSigma *
                                               (cos2SigmaM + C * cosSigma *
                                                (-1 + 2 * cos2SigmaM ** 2)))
        if abs(Lambda - LambdaPrev) < thresh:
            break  # successful convergence
    else:
        return None  # failure to converge

    uSq = cosSqAlpha * (a ** 2 - b ** 2) / (b ** 2)
    A = 1 + uSq / 16384 * (4096 + uSq * (-768 + uSq * (320 - 175 * uSq)))
    B = uSq / 1024 * (256 + uSq * (-128 + uSq * (74 - 47 * uSq)))
    deltaSigma = B * sinSigma * (cos2SigmaM + B / 4 * (cosSigma *
                 (-1 + 2 * cos2SigmaM ** 2) - B / 6 * cos2SigmaM *
                 (-3 + 4 * sinSigma ** 2) * (-3 + 4 * cos2SigmaM ** 2)))
    D = b * A * (sigma - deltaSigma)

    return round(D)
