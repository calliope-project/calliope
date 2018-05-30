"""
Copyright (C) 2013-2018 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

attrdict.py
~~~~~~~~~~~

Implements the AttrDict class (a subclass of regular dict)
used for managing model configuration.

"""

import numpy as np
import ruamel.yaml as ruamel_yaml

from calliope.core.util.tools import relative_path
from calliope.core.util.logging import logger


class __Missing(object):
    def __nonzero__(self):
        return False


_MISSING = __Missing()


class CalliopeYAMLEmitter(ruamel_yaml.emitter.Emitter):

    def write_comment(self, comment):
        """
        Override write_comment such as to ensure that inline
        comments are two spaces from the end of a statement,
        while full-line comments are aligned with their line.

        """
        comment_too_close = comment.start_mark.column < self.column
        comment_too_far = comment.start_mark.column > self.column + 2

        if comment_too_close or comment_too_far:
            comment.start_mark.column = self.column + 2

        super().write_comment(comment)


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
        # Create a ruamel_yaml.YAML instance in order to get the
        # comment roundtripping that it provides by default
        yaml_ = ruamel_yaml.YAML()
        result = yaml_.load(src)
        if not isinstance(result, dict):
            raise ValueError('Could not parse {} as YAML'.format(src_name))
        return result
    except ruamel_yaml.YAMLError:
        logger.error(
            'Parser error when reading YAML '
            'from {}.'.format(src_name)
        )
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

    # We add a dict key that contains comment metadata, but override
    # the __iter__, keys, and items methods below in order to hide
    # this key from users -- it remains accessible with direct key
    # access but is not listed or iterated over
    __hidden_keys__ = ['__dict_comments__']

    def __init__(self, source_dict=None):
        super().__init__()

        setattr(self, '__dict_comments__', {})

        if source_dict is not None:
            if isinstance(source_dict, dict):
                self.init_from_dict(source_dict)
            else:
                raise ValueError('Must pass a dict to AttrDict')

    def __iter__(self):
        return iter(self.keys())

    def __delitem__(self, key):
        try:
            del self['__dict_comments__'][key]
        except KeyError:
            pass
        super().__delitem__(key)

    __delattr__ = __delitem__

    def keys(self, filtered=True):
        if filtered:
            return [
                i for i in super().keys()
                if i not in self.__hidden_keys__]
        else:
            return super().keys()

    def values(self, filtered=True):
        if filtered:
            return [
                i[1] for i in super().items()
                if i[0] not in self.__hidden_keys__]
        else:
            return super().values()

    def items(self, filtered=True):
        if filtered:
            return [
                i for i in super().items()
                if i[0] not in self.__hidden_keys__]
        else:
            return super().items()

    def copy(self):
        """Override copy method so that it returns an AttrDict"""
        return AttrDict(self.as_dict(filtered=False).copy())

    def init_from_dict(self, d):
        """
        Initialize a new AttrDict from the given dict. Handles any
        nested dicts by turning them into AttrDicts too::

            d = AttrDict({'a': 1, 'b': {'x': 1, 'y': 2}})
            d.b.x == 1  # True

        """
        if isinstance(d, ruamel_yaml.comments.CommentedMap):
            setattr(self, '__dict_comments__', dict(d.ca.items))

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

            # Deal with comments on collapsed dict notation, e.g. if a
            # YAML file says ``foo.bar.baz: 1  # A comment``
            if '.' in k and k in self['__dict_comments__']:
                base_key, comment_key = k.rsplit('.', 1)
                self.get_key(base_key)['__dict_comments__'][comment_key] = self['__dict_comments__'][k]
                del self['__dict_comments__'][k]

    @classmethod
    def _resolve_imports(cls, loaded, resolve_imports, base_path=None):
        if resolve_imports and 'import' in loaded:
            for k in loaded['import']:
                if base_path:
                    path = relative_path(base_path, k)
                else:
                    path = k
                imported = cls.from_yaml(path)
                # loaded is added to imported (i.e. it takes precedence)
                imported.union(loaded)
                loaded = imported
            # 'import' key no longer needed, so we drop it
            loaded.pop('import', None)
        return loaded

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
        loaded = cls._resolve_imports(loaded, resolve_imports, base_path=f)
        return loaded

    @classmethod
    def from_yaml_string(cls, string, resolve_imports=False):
        """
        Returns an AttrDict initialized from the given string, which
        must be valid YAML.

        """
        loaded = cls(_yaml_load(string))
        loaded = cls._resolve_imports(loaded, resolve_imports)
        return loaded

    def set_key(self, key, value):
        """
        Set the given ``key`` to the given ``value``. Handles nested
        keys, e.g.::

            d = AttrDict()
            d.set_key('foo.bar', 1)
            d.foo.bar == 1  # True

        """
        if isinstance(value, dict) and not isinstance(value, AttrDict):
            value = AttrDict(value)
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

            # If we removed the last subkey, delete the parent key too
            if len(self[key].keys()) == 0:
                del self[key]

        else:
            del self[key]

    def as_dict(self, flat=False, filtered=True):
        """
        Return the AttrDict as a pure dict (with nested dicts if
        necessary).

        """
        if flat:
            return self.as_dict_flat(filtered=filtered)
        else:
            return self.as_dict_nested(filtered=filtered)

    def as_dict_nested(self, filtered=True):
        d = {}
        for k, v in self.items(filtered=filtered):
            if isinstance(v, AttrDict):
                d[k] = v.as_dict(filtered=filtered)
            elif isinstance(v, list):
                d[k] = [
                    i if not isinstance(i, AttrDict)
                    else i.as_dict(filtered=filtered)
                    for i in v]
            else:
                d[k] = v
        return d

    def as_dict_flat(self, filtered=True):
        d = {}
        keys = self.keys_nested(filtered=filtered)
        for k in keys:
            d[k] = self.get_key(k)
        return d

    def _comment_key_dict(self, key):
        if '.' in key:
            base_key, comment_key = key.rsplit('.', 1)
            comment_dict = self.get_key(base_key).__dict_comments__
        else:
            comment_key = key
            comment_dict = self.__dict_comments__
        return comment_key, comment_dict

    def get_comments(self, key):
        """
        Get comments for the given key. Returns a dict of the form
        ``{'above': ..., 'inline': ..., 'below': ...}``.

        """
        comment_key, comment_dict = self._comment_key_dict(key)

        if comment_key not in comment_dict:
            raise KeyError(key)

        comments = comment_dict[comment_key]

        # comments is a list of four entries:
        # [1] is a list of comments above the key,
        # [2] is inline comment,
        # [3] is a list of comments below the key
        try:
            above = '\n'.join(i.value for i in comments[1])
        except (TypeError, AttributeError):
            above = None
        try:
            inline = comments[2].value
        except AttributeError:
            inline = None
        try:
            below = '\n'.join(i.value for i in comments[3])
        except (TypeError, AttributeError):
            below = None

        return {
            'above': above,
            'inline': inline,
            'below': below
        }

    def set_comment(self, key, comment, kind='inline', exist_ok=False):
        """
        Set a comment for the given key. Returns None if successful,
        KeyError if key does not exist, and ValueError if a comment
        already exists, unless exist_ok is set to False.

        Parameters
        ----------
        key : str
        comment : str
        kind : str, optional, default 'inline'
            Can be 'inline', 'above'.
        exist_ok : bool, optional, default False
            If True, existing comments are overwritten.

        """
        comment_key, comment_dict = self._comment_key_dict(key)

        assert isinstance(comment, str)
        if not comment.startswith('#'):
            comment = '# ' + comment

        token = ruamel_yaml.CommentToken(
            comment,
            start_mark=ruamel_yaml.CommentMark(0),
            end_mark=None
        )

        if kind == 'inline':
            comments = token
            pos = 2
        elif kind == 'above':
            comments = [token]
            pos = 1
        else:
            raise ValueError('Invalid kind')

        if comment_key not in comment_dict:
            comment_list = [None, None, None, None]
            comment_list[pos] = comments
            comment_dict[comment_key] = comment_list
        elif (  # Comment empty or allowed to overwrite
            comment_dict[comment_key][pos] is None
            or (kind == 'inline'
                and comment_dict[comment_key][pos].value.strip('#').strip() == '')
            or (kind == 'above'
                and ''.join([i.value for i in comment_dict[comment_key][pos]]).strip('#').strip() == '')
            or exist_ok
        ):
            comment_dict[comment_key][pos] = comments
        else:
            raise ValueError('Comment exists')

    def del_comments(self, key):
        """
        Remove any comments at the given key. Returns None if successful,
        KeyError if key has no comments.

        """
        comment_key, comment_dict = self._comment_key_dict(key)
        del comment_dict[comment_key]

    def as_commentedmap(self):
        commented_map = ruamel_yaml.comments.CommentedMap(
            self.as_dict_nested())
        commented_map.ca._items = self.get('__dict_comments__', {})

        for k, v in self.items():
            if isinstance(v, dict):
                commented_map[k] = v.as_commentedmap()

        return commented_map

    def to_yaml(self, path=None):
        """
        Saves the AttrDict to the ``path`` as a YAML file, or returns
        a YAML string if ``path`` is None.

        """
        result = self.copy()
        yaml_ = ruamel_yaml.YAML()
        yaml_.Emitter = CalliopeYAMLEmitter
        yaml_.indent = 2
        yaml_.block_seq_indent = 0

        # Numpy objects should be converted to regular Python objects,
        # so that they are properly displayed in the resulting YAML output
        for k in result.keys_nested():
            # Convert numpy numbers to regular python ones
            v = result.get_key(k)
            if isinstance(v, np.floating):
                result.set_key(k, float(v))
            elif isinstance(v, np.integer):
                result.set_key(k, int(v))
            # Lists are turned into seqs so that they are formatted nicely
            elif isinstance(v, list):
                result.set_key(k, yaml_.seq(v))

        result = result.as_commentedmap()

        if path is not None:
            with open(path, 'w') as f:
                yaml_.dump(result, f)
        else:
            return yaml_.dump(result)

    def keys_nested(self, subkeys_as='list', filtered=True):
        """
        Returns all keys in the AttrDict, sorted, including the keys of
        nested subdicts (which may be either regular dicts or AttrDicts).

        If ``subkeys_as='list'`` (default), then a list of
        all keys is returned, in the form ``['a', 'b.b1', 'b.b2']``.

        If ``subkeys_as='dict'``, a list containing keys and dicts of
        subkeys is returned, in the form ``['a', {'b': ['b1', 'b2']}]``.

        """
        keys = []
        for k, v in sorted(self.items(filtered=filtered)):
            # Check if dict instance (which AttrDict is too),
            # and for non-emptyness of the dict
            if isinstance(v, dict) and v:
                if subkeys_as == 'list':
                    if k == '__dict_comments__':
                        keys.append(k)
                    else:
                        keys.extend([
                            k + '.' + kk
                            for kk in v.keys_nested(filtered=filtered)
                        ])
                elif subkeys_as == 'dict':
                    keys.append({k: v.keys_nested(
                        subkeys_as=subkeys_as, filtered=filtered)
                    })
            else:
                keys.append(k)
        return keys

    def union(
            self, other,
            allow_override=False, allow_replacement=False,
            allow_subdict_override_with_none=False):
        """
        Merges the AttrDict in-place with the passed ``other``
        AttrDict. Keys in ``other`` take precedence, and nested keys
        are properly handled.

        If ``allow_override`` is False, a KeyError is raised if
        other tries to redefine an already defined key.

        If ``allow_replacement``, allow "_REPLACE_" key to replace an
        entire sub-dict.

        If ``allow_subdict_override_with_none`` is False (default),
        a key of the form ``this.that: None`` in other will be ignored
        if subdicts exist in self like ``this.that.foo: 1``, rather
        than wiping them.

        """
        self_keys = self.keys_nested(filtered=False)
        other_keys = other.keys_nested(filtered=False)
        if allow_replacement:
            WIPE_KEY = '_REPLACE_'
            override_keys = [k for k in other_keys
                             if WIPE_KEY not in k]
            wipe_keys = [k.split('.' + WIPE_KEY)[0]
                         for k in other_keys
                         if WIPE_KEY in k]
        else:
            override_keys = other_keys
            wipe_keys = []
        for k in override_keys:
            if k.endswith('__dict_comments__'):
                self.set_key(k, {**self.get_key(k, {}), **other.get_key(k, {})})
            elif not allow_override and k in self_keys:
                raise KeyError('Key defined twice: {}'.format(k))
            else:
                other_value = other.get_key(k)
                # If other value is None, and would overwrite an entire subdict,
                # we skip it
                if not (other_value is None and isinstance(self.get_key(k, None), AttrDict)):
                    self.set_key(k, other_value)
        for k in wipe_keys:
            self.set_key(k, other.get_key(k + '.' + WIPE_KEY))
