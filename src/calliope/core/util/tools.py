# Copyright (C) since 2013 Calliope contributors listed in AUTHORS.
# Licensed under the Apache 2.0 License (see LICENSE file).


import functools
import importlib
import operator
import os
import sys
from typing import Callable, TypeVar

import jsonschema
from typing_extensions import ParamSpec

from calliope.exceptions import print_warnings_and_raise_errors

P = ParamSpec("P")
T = TypeVar("T")


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


def copy_docstring(wrapper: Callable[P, T]):
    """
    Decorator to copy across a function docstring to the wrapped function.
    Any additional documentation in the wrapped function will be appended to the copied
    docstring.
    """

    def decorator(func: Callable) -> Callable[P, T]:
        func_doc = ""
        if wrapper.__doc__ is not None:
            func_doc += wrapper.__doc__
        if func.__doc__ is not None:
            func_doc += func.__doc__
        func.__doc__ = func_doc

        return func

    return decorator


def validate_dict(to_validate: dict, schema: dict, dict_descriptor: str) -> None:
    """
    Validate a dictionary under a given schema.

    Args:
        to_validate (dict): Dictionary to validate.
        schema (dict): Schema to validate with.
        dict_descriptor (str): Description of the dictionary to validate, to use if an error is raised.

    Raises:
        jsonschema.SchemaError: If the schema itself is malformed, a SchemaError will be raised at the first issue. Other issues than that raised may still exist.
        calliope.exceptions.ModelError: If the dictionary is not valid according to the schema, a list of the issues found will be collated and raised.
    """
    errors = []
    jsonschema.Draft7Validator.META_SCHEMA["additionalProperties"] = False
    validator = jsonschema.Draft7Validator(schema)
    try:
        validator.check_schema(schema)
    except jsonschema.SchemaError as err:
        path = ".".join(err.path)
        if path != "":
            path = f" at `{path}`"

        if err.context:
            message = err.context[0].args[0]
        else:
            message = err.args[0]
        raise jsonschema.SchemaError(
            message=f"The {dict_descriptor} schema is malformed{path}: {message}"
        )

    try:
        jsonschema.validate(to_validate, schema)
    except jsonschema.ValidationError:
        for error in sorted(validator.iter_errors(to_validate), key=str):
            path_ = error.json_path.lstrip("$.")
            message = error.args[0]
            if path_ == "":
                errors.append(message)
            else:
                errors.append(f"{path_}: {message}")
    if errors:
        print_warnings_and_raise_errors(
            errors=errors, during=f"validation of the {dict_descriptor} dictionary."
        )
