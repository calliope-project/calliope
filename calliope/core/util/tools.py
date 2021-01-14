"""
Copyright (C) since 2013 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

"""

from copy import deepcopy
import functools
import importlib
import operator
import os
import sys


def get_from_dict(data_dict, map_list):
    return functools.reduce(operator.getitem, map_list, data_dict)


def apply_to_dict(data_dict, map_list, func, args):
    getattr(get_from_dict(data_dict, map_list[:-1])[map_list[-1]], func)(*args)


memoize = functools.lru_cache(maxsize=2048)


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


def relative_path(base_path_file, path):
    """
    If ``path`` is not absolute, it is interpreted as relative to the
    path of the given ``base_path_file``.

    """
    # Check if base_path_file is a string because it might be an AttrDict
    if not os.path.isabs(path) and isinstance(base_path_file, str):
        path = os.path.join(os.path.dirname(base_path_file), path)
    return path


def load_function(source):
    """
    Returns a function from a module, given a source string of the form:

        'module.submodule.subsubmodule.function_name'

    """
    module_string, function_string = source.rsplit(".", 1)
    modules = sys.modules.keys()
    # Check if module already loaded, if so, don't re-import it
    if module_string in modules:
        module = sys.modules[module_string]
    # Else load the module
    else:
        module = importlib.import_module(module_string)
    return getattr(module, function_string)


def plugin_load(name, builtin_module):
    try:  # First try importing as a third-party module
        func = load_function(name)
    except ValueError:
        # ValueError raised if we got a string without '.',
        # which implies a builtin function,
        # so we attempt to load from the given module
        func_string = builtin_module + "." + name
        func = load_function(func_string)
    return func
