# Copyright (C) since 2013 Calliope contributors listed in AUTHORS.
# Licensed under the Apache 2.0 License (see LICENSE file).
"""Schema for dimensional data definition."""

from typing import ClassVar, Literal

from pydantic import Field, field_validator, model_validator
from typing_extensions import Self

from calliope.schemas.general import (
    AttrStr,
    CalliopeBaseModel,
    CalliopeDictModel,
    NonEmptyList,
    NumericVal,
    UniqueList,
)

DataValue = str | bool | NumericVal | None
IndexValue = str | NumericVal
BaseTech = Literal["conversion", "demand", "storage", "supply", "transmission"]
DimNames = Literal["techs", "nodes"]


class IndexedData(CalliopeBaseModel):
    """Indexed data schema."""

    model_config = {"title": "Indexed data definition"}

    data: DataValue | NonEmptyList[DataValue]
    """
    Parameter / lookup value(s).
    If data is one value, will be applied to all dimension members.
    If a list, must be same length as the index array.
    """
    dims: UniqueList[AttrStr] = Field(default_factory=list)
    """
    Model dimension(s) over which the data is indexed.
    Must be same length as the sub-arrays of `index`.
    I.e., if `index` does not have any sub-arrays or is simply a single value, `dims` must be of length 1.
    """
    index: list[UniqueList[IndexValue]] = Field(default_factory=lambda: [[]])
    """
    Model dimension members to apply the data value(s) to.
    If an array of arrays, sub-arrays must have same length as number of `dims`.
    """

    @field_validator("index", mode="before")
    @classmethod
    def listify_index(cls, value) -> list[list]:
        """Ensure index and dims are consistent."""
        if not isinstance(value, list):
            value = [[value]]
        if isinstance(value, list) and not any(isinstance(i, list) for i in value):
            value = [[v] for v in value]

        return value

    @field_validator("dims", mode="before")
    @classmethod
    def listify_dims(cls, value) -> list:
        """Ensure index and dims are consistent."""
        if not isinstance(value, list):
            value = [value]

        return value


class IndexedTechNodeParam(IndexedData):
    """Tech-specific data schema."""

    model_config = {"title": "Indexed `techs` data definition"}

    @field_validator("dims", mode="after")
    @classmethod
    def check_dims(cls, value: UniqueList[AttrStr]) -> UniqueList[AttrStr]:
        """Ensure dimensions do not refer to techs or nodes."""
        forbidden = ["techs", "nodes"]
        if any(set(value) & set(forbidden)):
            raise ValueError(f"`dims` must not contain '{forbidden}', found '{value}'.")
        return value


class DimensionData(CalliopeBaseModel):
    """Calliope's generic dimension data schema."""

    model_config = {"title": "Generic dimension data", "extra": "allow"}

    active: bool = True
    __pydantic_extra__: dict[
        AttrStr, IndexedTechNodeParam | DataValue | NonEmptyList[DataValue]
    ]


class CalliopeTech(DimensionData):
    """Calliope's technology dimension schema."""

    model_config = {"title": "Technology dimension data"}

    base_tech: BaseTech | None = None
    """
    One of the abstract base classes, used to derive specific defaults and
    to activate technology-specific constraints.
    """


class CalliopeTransmissionTech(CalliopeTech):
    """Calliope's transmission technology dimension schema."""

    model_config = {"title": "Transmission technology dimension data"}

    base_tech: Literal["transmission"] | None = None
    """
    One of the abstract base classes, used to derive specific defaults and
    to activate technology-specific constraints.
    """

    link_from: AttrStr
    """Node from which the transmission technology links."""
    link_to: AttrStr
    """Node to which the transmission technology links."""
    one_way: bool = False
    """Whether the transmission technology only allows flow in one direction (from `link_from` to `link_to`)."""


class CalliopeTechs(CalliopeDictModel):
    """Calliope Techs dictionary."""

    root: dict[AttrStr, CalliopeTransmissionTech | CalliopeTech] = Field(
        default_factory=dict
    )

    _dim: ClassVar[DimNames] = "techs"
    """The name of the dimension in YAML used for validation and error messages."""

    @field_validator("*", mode="before")
    @classmethod
    def no_none_entries(cls, value) -> dict:
        """All entries are of dict type."""
        if value is None:
            return {}

        for key, entry in value.items():
            if entry is None:
                value[key] = {}
        return value


class CalliopeNode(DimensionData):
    """Calliope's node dimension schema."""

    model_config = {"title": "Node dimension", "extra": "allow"}

    latitude: NumericVal | None = Field(default=None, ge=-90, le=90)
    """Latitude (WGS84 / EPSG4326)."""
    longitude: NumericVal | None = Field(default=None, ge=-180, le=180)
    """Longitude (WGS84 / EPSG4326)."""
    techs: CalliopeTechs = Field(default_factory=CalliopeTechs)
    """
    Technologies present at this node. Also allows to override technology data.
    """

    @model_validator(mode="after")
    def check_dependent_definitions(self) -> Self:
        """Ensure dependent settings are defined."""
        if type(self.latitude) is not type(self.longitude):
            raise ValueError(
                f"Invalid latitude/longitude definition. Types must match, found ({self.latitude}, {self.longitude})."
            )
        return self


class CalliopeNodes(CalliopeDictModel):
    """Calliope Nodes dictionary."""

    root: dict[AttrStr, CalliopeNode] = Field(default_factory=dict)

    _dim: ClassVar[DimNames] = "nodes"
    """The name of the dimension in YAML used for validation and error messages."""


class CalliopeDataDef(CalliopeDictModel):
    """Calliope data definition dictionary."""

    root: dict[AttrStr, DataValue | IndexedData] = Field(default_factory=dict)
