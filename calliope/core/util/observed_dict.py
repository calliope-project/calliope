
"""
Copyright (C) 2013-2019 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

"""

from collections import Mapping
from calliope.core.attrdict import AttrDict


class ObservedDict(dict):
    """
    Dictionary subclass which updates an attribute of the model_data xarray dataset
    with a yaml string of that dictionary. This update takes place whenever a
    value is updated in the dictionary.

    Parameters
    ----------
    initial_dict : a dictionary to subclass
    name : the model_data attribute key to update
    model_data : the model data object
    """

    def __init__(self, initial_dict, name=None, observer=None, parent=None):
        if isinstance(initial_dict, str):
            initial_dict = AttrDict.from_yaml_string(initial_dict).as_dict()
        for k, v in initial_dict.items():
            if isinstance(v, dict):
                _parent = self if parent is None else parent
                initial_dict[k] = ObservedDict(v, name, observer, parent=_parent)

        super().__init__(initial_dict)

        self.observer = observer
        self.name = name
        self.parent = parent
        if parent is None:  # initialise the key:value pair in observer
            observer.attrs[name] = AttrDict(initial_dict).to_yaml()

    def __setitem__(self, item, value):
        if isinstance(value, dict):
            _value = ObservedDict(value, self.name, self.observer, parent=self.parent)
        else:
            _value = value

        super().__setitem__(item, _value)

        # Update B
        if self.parent is not None:
            self.observer.attrs[self.name] = AttrDict(self.parent).to_yaml()
        else:
            self.observer.attrs[self.name] = AttrDict(self).to_yaml()

    def update(self, other=None, **kwargs):
        if other is not None:
            for k, v in other.items() if isinstance(other, Mapping) else other:
                self[k] = v
        for k, v in kwargs.items():
            self[k] = v
