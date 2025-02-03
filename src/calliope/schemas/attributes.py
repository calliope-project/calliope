"""Defines generic calliope configuration types."""

from collections.abc import Hashable
from typing import Annotated, TypeVar

from pydantic import AfterValidator, Field
from pydantic_core import PydanticCustomError

FIELD_REGEX = r"^[^_^\d][\w]*$"  # Regular string pattern for most calliope attributes
CnfStr = Annotated[str, Field(pattern=FIELD_REGEX)]

# ==
# Taken from https://github.com/pydantic/pydantic-core/pull/820#issuecomment-1670475909
T = TypeVar("T", bound=Hashable)


def _validate_unique_list(v: list[T]) -> list[T]:
    if len(v) != len(set(v)):
        raise PydanticCustomError("unique_list", "List must be unique")
    return v


UniqueList = Annotated[
    list[T],
    AfterValidator(_validate_unique_list),
    Field(json_schema_extra={"uniqueItems": True}),
]
# ==
