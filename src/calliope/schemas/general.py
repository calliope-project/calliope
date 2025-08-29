# Copyright (C) since 2013 Calliope contributors listed in AUTHORS.
# Licensed under the Apache 2.0 License (see LICENSE file).
"""Defines generic calliope configuration types."""

import logging
from collections.abc import Hashable
from typing import Annotated, TypeVar

import jsonref
from annotated_types import Len
from pydantic import AfterValidator, BaseModel, Field, RootModel
from pydantic_core import PydanticCustomError
from typing_extensions import Self

from calliope.attrdict import AttrDict

LOGGER = logging.getLogger(__name__)
# ==
# Modified from https://github.com/pydantic/pydantic-core/pull/820#issuecomment-1670475909
T = TypeVar("T", bound=Hashable | list)


def _validate_unique_list(v: list) -> list:
    try:
        unique = set(v)
    except TypeError:
        unique = set([tuple(i) for i in v])
    if len(v) != len(unique):
        raise PydanticCustomError("unique_list", "List must be unique")
    return v


UniqueList = Annotated[
    list[T],
    AfterValidator(_validate_unique_list),
    Field(json_schema_extra={"uniqueItems": True}),
]
"""A list with no repeated values."""
# ==
NonEmptyList = Annotated[list[T], Len(min_length=1)]
"""A list with at least one value in it."""
NonEmptyUniqueList = Annotated[UniqueList[T], Len(min_length=1)]
"""A list with at least one value in it and no repeated values."""
AttrStr = Annotated[str, Field(pattern=r"^[^_^\d][\w]*$")]
"""Single word string in snake_case (e.g., wind_offshore)."""
NumericVal = int | Annotated[float, Field(allow_inf_nan=True)]
"""Numerical integer or float value. Can be `nan` or infinite (`float(inf)`)."""


class CalliopeDictModel(RootModel):
    """Pydantic Model that is used to store dictionaries with user-defined keys and Calliope pydantic model values."""

    def __setitem__(self, *args, **kwargs) -> None:
        """Do not allow direct item setting."""
        raise PydanticCustomError(
            "no_extra_dict",
            f"Cannot set a {self.__class__.__name__} directly. Use the `update` method instead, which will return a copy.",
        )

    def __getitem__(self, key):
        """Expose the root attribute when getting an item by key."""
        return self.root[key]

    def __repr__(self, *args, **kwargs):
        """Show the __repr__ of the root attribute when requesting the __repr__ of the class."""
        return self.root.__repr__(*args, **kwargs)

    def __rich_repr__(self):
        """Prettyprint the __repr__ of the root attribute when requesting the prettyprint of the class."""
        yield from self.root.items()

    def update(self, update_dict: dict, deep: bool = False) -> Self:
        """Return a new iteration of the model with updated fields.

        Args:
            update_dict (dict): Dictionary with which to update the base model.
            deep (bool, optional): Set to True to make a deep copy of the model. Defaults to False.

        Returns:
            BaseModel: New model instance.
        """
        new_dict: dict = {}
        # Iterate through dict to be updated and convert any sub-dicts into their respective pydantic model objects.
        # Wrapped in `AttrDict` to allow users to define dot notation nested configuration.
        for key, val in AttrDict(update_dict).items():
            key_class = self.root.get(key, None)
            if isinstance(key_class, CalliopeBaseModel):
                new_dict[key] = key_class.update(val, deep=deep)
            elif isinstance(key_class, CalliopeListModel):
                new_dict[key] = key_class.update(val)
            elif key_class == val:
                continue
            else:
                LOGGER.debug(f"Adding {self.__class__.__name__} entry: `{key}`")
                new_dict[key] = self.model_validate({key: val})[key]

        return self.model_validate(self.root | new_dict)


class CalliopeListModel(RootModel):
    """Pydantic Model that is used to store lists of Calliope pydantic models."""

    def __iter__(self):
        """Iterate over root attribute contents when iterating over class."""
        return iter(self.root)

    def __getitem__(self, item: int):
        """Expose the root attribute when getting an item by index value."""
        return self.root[item]

    def __repr__(self, *args, **kwargs):
        """Show the __repr__ of the root attribute when requesting the __repr__ of the class."""
        return self.root.__repr__(*args, **kwargs)

    def __rich_repr__(self):
        """Prettyprint the __repr__ of the root attribute when requesting the prettyprint of the class."""
        yield from self.root

    def update(self, update_list: list) -> Self:
        """Return a new iteration of the model fields entirely replaced.

        We do not allow updating individual items in the list as it's hard to guarantee the order of items in the list.

        Args:
            update_list (list): List with which to update the base model.

        Returns:
            BaseModel: New model instance.
        """
        return self.model_validate(update_list)


class CalliopeBaseModel(BaseModel):
    """A base class for creating pydantic models for Calliope models."""

    model_config = {
        "extra": "forbid",
        "frozen": True,
        "revalidate_instances": "always",
        "use_attribute_docstrings": True,
    }

    def __getitem__(self, item):
        """Allow attribute access via item lookup."""
        return getattr(self, item)

    def update(self, update_dict: dict, deep: bool = False) -> Self:
        """Return a new iteration of the model with updated fields.

        Args:
            update_dict (dict): Dictionary with which to update the base model.
            deep (bool, optional): Set to True to make a deep copy of the model. Defaults to False.

        Returns:
            BaseModel: New model instance.
        """
        new_dict: dict = {}
        # Iterate through dict to be updated and convert any sub-dicts into their respective pydantic model objects.
        # Wrapped in `AttrDict` to allow users to define dot notation nested configuration.
        for key, val in AttrDict(update_dict).items():
            key_class = getattr(self, key, None)
            if isinstance(key_class, CalliopeBaseModel | CalliopeDictModel):
                new_dict[key] = key_class.update(val, deep=deep)
            elif isinstance(key_class, CalliopeListModel):
                new_dict[key] = key_class.update(val)
            elif key_class == val:
                continue
            else:
                LOGGER.debug(
                    f"Updating {self.__class__.__name__} `{key}`: {key_class} -> {val}"
                )
                new_dict[key] = val
        updated = super().model_copy(update=new_dict, deep=deep)
        return updated.model_validate(updated.model_dump())

    @classmethod
    def model_no_ref_schema(cls) -> AttrDict:
        """Generate an AttrDict with the schema replacing $ref/$def for better readability.

        Returns:
            AttrDict: class schema.
        """
        schema_dict = AttrDict(cls.model_json_schema())
        schema_dict = AttrDict(jsonref.replace_refs(schema_dict))
        if "$defs" in schema_dict:
            schema_dict.del_key("$defs")
        return schema_dict
