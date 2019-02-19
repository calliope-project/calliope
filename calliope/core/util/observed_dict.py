
"""
Copyright (C) 2013-2019 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

"""

from collections import Mapping
from calliope.core.attrdict import AttrDict


class ObservedDict(dict):
    """
    Dictionary subclass which updates an observer when there is a change in the
    values assigned to keys in the Dictionary.
    """
    def __init__(self, initial_dict, on_changed=None):
        if isinstance(initial_dict, str):
            initial_dict = AttrDict.from_yaml_string(initial_dict).as_dict()

        super().__init__(initial_dict)

        self.on_changed = on_changed

        for k, v in initial_dict.items():
            if isinstance(v, dict):
                super().__setitem__(
                    k, ObservedDict(v, on_changed=self.notify))
        self.notify()

    def __setitem__(self, key, value):
        if isinstance(value, dict):
            value = ObservedDict(value, on_changed=self.notify)
        super().__setitem__(key, value)
        self.notify()

    def update(self, other=None, **kwargs):
        if other is not None:
            for k, v in other.items() if isinstance(other, Mapping) else other:
                self[k] = v
        for k, v in kwargs.items():
            self[k] = v

    def notify(self, updated=None):
        if self.on_changed is not None:
            return self.on_changed(self)


class UpdateObserver(ObservedDict):
    """
    Dictionary subclass which observes a dictionary and updates an attribute of
    the model_data xarray dataset with a YAML string of that dictionary.
    This update takes place whenever a value is updated in the dictionary.

    Parameters
    ----------
    initial_dict : a dictionary to subclass
    name : the model_data attribute key to update
    observer : the model_data object
    """

    def __init__(self, *args, name, observer, **kwargs):
        self.observer = observer
        self.name = name
        super().__init__(*args, **kwargs)

    def notify(self, updated=None):
        temp_dict = {
            k: v for k, v in self.items()
            if (not isinstance(v, dict) and v is not None)
            or (isinstance(v, dict) and len(v.keys()) > 0)
        }
        self.observer.attrs[self.name] = AttrDict(temp_dict).to_yaml()