# Copyright (C) since 2013 Calliope contributors listed in AUTHORS.
# Licensed under the Apache 2.0 License (see LICENSE file).
"""Schema for dimensional data definition."""

from typing import Literal

from pydantic import Field, field_validator, model_validator
from typing_extensions import Self

from calliope.schemas.general import (
    AttrStr,
    CalliopeBaseModel,
    NonEmptyList,
    NonEmptyUniqueList,
    NumericVal,
)
from calliope.util.tools import listify

DataValue = str | bool | NumericVal | None
IndexValue = str | NumericVal
BaseTech = Literal["conversion", "demand", "storage", "supply", "transmission"]


class IndexedParam(CalliopeBaseModel):
    """Indexed parameter schema."""

    model_config = {"title": "Indexed parameter definition"}

    data: DataValue | NonEmptyList[DataValue]
    """
    Parameter value(s).
    If data is one value, will be applied to all dimension members.
    If a list, must be same length as the index array.
    """
    dims: AttrStr | NonEmptyUniqueList[AttrStr]
    """
    Model dimension(s) over which the parameter is indexed.
    Must be same length as the sub-arrays of `index`.
    I.e., if `index` does not have any sub-arrays or is simply a single value, `dims` must be of length 1.
    """
    index: IndexValue | NonEmptyUniqueList[IndexValue | NonEmptyUniqueList[IndexValue]]
    """
    Model dimension members to apply the parameter value(s) to.
    If an array of arrays, sub-arrays must have same length as number of `dims`.
    """


class IndexedTechNodeParam(IndexedParam):
    """Tech-specific parameter schema."""

    model_config = {"title": "Indexed `techs` parameter definition"}

    @field_validator("dims", mode="before")
    @classmethod
    def check_dims(
        cls, value: AttrStr | NonEmptyUniqueList[AttrStr]
    ) -> AttrStr | NonEmptyUniqueList[AttrStr]:
        """Ensure dimensions do not refer to techs or nodes."""
        forbidden = ["techs", "nodes"]
        if any(set(listify(value)) & set(forbidden)):
            raise ValueError(f"`dims` must not contain '{forbidden}', found '{value}'.")
        return value


class DimensionData(CalliopeBaseModel):
    """Calliope's generic dimension data schema."""

    model_config = {"title": "Generic dimension data", "extra": "allow"}

    active: bool = True
    __pydantic_extra__: dict[AttrStr, IndexedTechNodeParam | DataValue]


class CalliopeTech(DimensionData):
    """Calliope's technology dimension schema."""

    model_config = {"title": "Technology dimension data"}

    base_tech: BaseTech | None = None
    carrier_in: AttrStr | NonEmptyUniqueList[AttrStr] | None = None
    carrier_out: AttrStr | NonEmptyUniqueList[AttrStr] | None = None
    carrier_export: AttrStr | NonEmptyUniqueList[AttrStr] | None = None
    link_from: AttrStr | NonEmptyUniqueList[AttrStr] | None = None
    link_to: AttrStr | NonEmptyUniqueList[AttrStr] | None = None


class CalliopeNode(DimensionData):
    """Calliope's node dimension schema."""

    model_config = {"title": "Node dimension", "extra": "allow"}

    latitude: NumericVal | None = Field(default=None, ge=-90, le=90)
    longitude: NumericVal | None = Field(default=None, ge=-180, le=180)
    techs: None | dict[AttrStr, None | dict[AttrStr, DataValue | dict]]

    @model_validator(mode="after")
    def check_dependent_definitions(self) -> Self:
        """Ensure dependent settings are defined."""
        if type(self.latitude) is not type(self.longitude):
            raise ValueError(
                f"Invalid latitude/longitude definition. Types must match, found ({self.latitude}, {self.longitude})."
            )
        return self
