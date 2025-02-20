# Copyright (C) since 2013 Calliope contributors listed in AUTHORS.
# Licensed under the Apache 2.0 License (see LICENSE file).
"""Defines generic calliope configuration types."""

import logging
from collections.abc import Hashable
from typing import Annotated, TypeVar

import jsonref
from annotated_types import Len
from pydantic import AfterValidator, BaseModel, Field
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
NumericVal = Annotated[float, int, Field(allow_inf_nan=True)]
"""Numerical integer or float value. Can be `nan` or infinite (`float(inf)`)."""


class CalliopeBaseModel(BaseModel):
    """A base class for creating pydantic models for Calliope models."""

    model_config = {
        "extra": "forbid",
        "frozen": True,
        "revalidate_instances": "always",
        "use_attribute_docstrings": True,
    }

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
            key_class = getattr(self, key)
            if isinstance(key_class, CalliopeBaseModel):
                new_dict[key] = key_class.update(val)
            else:
                LOGGER.info(
                    f"Updating {self.model_config['title']} `{key}`: {key_class} -> {val}"
                )
                new_dict[key] = val
        updated = super().model_copy(update=new_dict, deep=deep)
        updated.model_validate(updated)
        return updated

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
