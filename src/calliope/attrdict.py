# Copyright (C) since 2013 Calliope contributors listed in AUTHORS.
# Licensed under the Apache 2.0 License (see LICENSE file).
"""AttrDict implementation (a subclass of regular dict) used for managing model configuration."""

import copy
import logging

from typing_extensions import Self

logger = logging.getLogger(__name__)


class __Missing:
    def __nonzero__(self):
        return False


_MISSING = __Missing()


class AttrDict(dict):
    """Extended `dict` class."""

    __name__ = "AttrDict"

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    def __init__(self, source_dict=None):
        """A subclass of ``dict`` with key access by attributes.

        Examples:
            d = AttrDict({'a': 1, 'b': 2})
            d.a == 1  # True

        Includes a range of additional methods to read and write to YAML,
        and to deal with nested keys.
        """
        super().__init__()

        if source_dict is not None:
            if isinstance(source_dict, dict):
                self.init_from_dict(source_dict)
            else:
                raise ValueError("Must pass a dict to AttrDict")

    def copy(self):
        """Override copy method so that it returns an AttrDict."""
        return AttrDict(self.as_dict().copy())

    def __deepcopy__(self, memo):
        """Override copy method so that it returns an AttrDict."""
        return AttrDict(copy.deepcopy(self.as_dict(), memo))

    def init_from_dict(self, d):
        """Initialize a new AttrDict from the given dict.

        Handles any nested dicts by turning them into AttrDicts too:
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
                # Modifying the list in-place so that if it is a modified
                # list subclass, e.g. CommentedSeq, it is not killed
                for i in range(len(v)):
                    if isinstance(v[i], dict):
                        v[i] = AttrDict(v[i])
                self.set_key(k, v)
            else:
                self.set_key(k, v)

    def set_key(self, key, value):
        """Set the given ``key`` to the given ``value``.

        Handles nested keys, e.g.:
            d = AttrDict()
            d.set_key('foo.bar', 1)
            d.foo.bar == 1  # True

        """
        if isinstance(value, dict) and not isinstance(value, AttrDict):
            value = AttrDict(value)
        if "." in key:
            key, remainder = key.split(".", 1)
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
                    raise KeyError("Cannot set nested key on non-dict key.")
        else:
            if key in self and isinstance(value, AttrDict):
                for k, v in value.items():
                    self[key].set_key(k, v)
            else:
                self[key] = value

    def get_key(self, key, default=_MISSING):
        """Looks up the given ``key``.

        Deals with nested keys.

        If default is anything but ``_MISSING``, the given default is
        returned if the key does not exist.
        """
        if "." in key:
            # Nested key of form "foo.bar"
            key, remainder = key.split(".", 1)
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
        if "." in key:
            key, remainder = key.split(".", 1)
            try:
                del self[key][remainder]
            except KeyError:
                self[key].del_key(remainder)

            # If we removed the last subkey, delete the parent key too
            if len(self[key].keys()) == 0:
                del self[key]

        else:
            del self[key]

    def as_dict(self, flat=False):
        """Return the AttrDict as a pure dict (with nested dicts if necessary)."""
        if flat:
            return self.as_dict_flat()
        else:
            return self.as_dict_nested()

    def as_dict_nested(self):
        """Return the AttrDict as a pure dict, converting nested AttrDicts."""
        d = {}
        for k, v in self.items():
            if isinstance(v, AttrDict):
                d[k] = v.as_dict()
            elif isinstance(v, list):
                d[k] = [i if not isinstance(i, AttrDict) else i.as_dict() for i in v]
            else:
                d[k] = v
        return d

    def as_dict_flat(self):
        """Return a flat dictionary."""
        d = {}
        keys = self.keys_nested()
        for k in keys:
            d[k] = self.get_key(k)
        return d

    def keys_nested(self, subkeys_as="list"):
        """Returns all keys in the AttrDict, including nested keys.

        Nested subdicts may be either regular dicts or AttrDicts.

        If ``subkeys_as='list'`` (default), then a list of
        all keys is returned, in the form ``['a', 'b.b1', 'b.b2']``.

        If ``subkeys_as='dict'``, a list containing keys and dicts of
        subkeys is returned, in the form ``['a', {'b': ['b1', 'b2']}]``.

        """
        keys = []
        for k, v in self.items():
            # Check if dict instance (which AttrDict is too),
            # and for non-emptyness of the dict
            if isinstance(v, dict) and v:
                if subkeys_as == "list":
                    keys.extend([k + "." + kk for kk in v.keys_nested()])
                elif subkeys_as == "dict":
                    keys.append({k: v.keys_nested(subkeys_as=subkeys_as)})
            else:
                keys.append(k)
        return keys

    def union(
        self,
        other: Self | dict,
        allow_override: bool = False,
        allow_replacement: bool = False,
    ):
        """In-place merge with another AttrDict.

        Args:
            other (Self | dict): other AttrDict to use in union. Takes precedence.
            allow_override (bool, optional): whether or not to allow overrides of
                already defined keys. Defaults to False.
            allow_replacement (bool, optional): allow "_REPLACE_" key to replace an
                entire sub-dict. Defaults to False.

        Raises:
            KeyError: `other` has an already defined key and `allow_override == False`
        """
        if not isinstance(other, AttrDict):
            other = AttrDict(other)
        self_keys = self.keys_nested()
        other_keys = other.keys_nested()
        if allow_replacement:
            WIPE_KEY = "_REPLACE_"
            override_keys = [k for k in other_keys if WIPE_KEY not in k]
            wipe_keys = [
                k.split("." + WIPE_KEY)[0] for k in other_keys if WIPE_KEY in k
            ]
        else:
            override_keys = other_keys
            wipe_keys = []
        for k in override_keys:
            if not allow_override and k in self_keys:
                raise KeyError(f"Key defined twice: {k}")
            else:
                other_value = other.get_key(k)
                # If other value is None, and would overwrite an entire subdict,
                # we skip it
                if not (
                    other_value is None and isinstance(self.get_key(k, None), AttrDict)
                ):
                    self.set_key(k, other_value)
        for k in wipe_keys:
            self.del_key(k)
            self.set_key(k, other.get_key(k + "." + WIPE_KEY))
