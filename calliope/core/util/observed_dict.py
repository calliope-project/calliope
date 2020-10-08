"""
Copyright (C) since 2013 Calliope contributors listed in AUTHORS.
Licensed under the Apache 2.0 License (see LICENSE file).

"""

from collections.abc import Mapping
from calliope.core.attrdict import AttrDict


class ObservedDict(dict):
    """
    Dictionary subclass which updates UpdateObserverDict when there is a change
    in the values assigned to keys in the dictionary.
    """

    def __init__(self, initial_dict, initial_yaml_string, on_changed=None):
        if initial_yaml_string is not None:
            initial_dict = AttrDict.from_yaml_string(initial_yaml_string).as_dict()

        super().__init__(initial_dict)

        self.on_changed = on_changed

        for k, v in initial_dict.items():
            if isinstance(v, dict):
                super().__setitem__(k, ObservedDict(v, None, on_changed=self.notify))
        self.notify()

    def __setitem__(self, key, value):
        if isinstance(value, dict):
            value = ObservedDict(value, None, on_changed=self.notify)
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


class UpdateObserverDict(ObservedDict):
    """
    Dictionary subclass which observes a dictionary and updates an attribute of
    the model_data xarray dataset with a YAML string of that dictionary.
    This update takes place whenever a value is updated in the dictionary.

    Parameters
    ----------
    name : str
        The model_data attribute key to update.
    observer : xarray Dataset (e.g. calliope model_data)
        The Dataset whose attribute dictionary will contain the key `name`
        and value = a YAML string of the initial_dict.
    initial_dict : dict, optional, default = None
        An initial dictionary to copy for observing.
        One of initial_dict or initial_yaml_string must be defined.
    initial_yaml_string : str, optional, default = None
        A YAML string of an initial dictionary to copy for observing.
        One of initial_dict or initial_yaml_string must be defined.

    Returns
    -------
    Observed dictionary, which acts as a dictionary in every sense *except* that
    on changing or adding any key:value pairs in that dictionary, the YAML string
    stored at `observer.attrs[name]` will be updated to reflect the new Observed
    dicitonary.
    """

    def __init__(
        self,
        name,
        observer,
        initial_dict=None,
        initial_yaml_string=None,
        *args,
        **kwargs,
    ):
        self.observer = observer
        self.name = name

        check_input_args = [i is None for i in [initial_dict, initial_yaml_string]]
        if all(check_input_args) or not any(check_input_args):
            raise ValueError(
                "must supply one, and only one, of initial_dict or initial_yaml_string"
            )

        super().__init__(initial_dict, initial_yaml_string, *args, **kwargs)

    def notify(self, updated=None):
        temp_dict = {
            k: v
            for k, v in self.items()
            if (not isinstance(v, dict) and v is not None)
            or (isinstance(v, dict) and len(v.keys()) > 0)
        }
        self.observer.attrs[self.name] = AttrDict(temp_dict).to_yaml()
