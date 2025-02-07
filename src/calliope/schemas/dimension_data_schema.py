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
IndexValue = AttrStr | NumericVal
BaseTech = Literal["conversion", "demand", "storage", "supply", "transmission"]


class EmptyDict(CalliopeBaseModel):
    """Empty dictionary schema.

    Allows validating cases where only empty dicts OR fully defined data is allowed.
    """

    model_config = {"title": "Empty dictionary"}


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


class CalliopeDimensionData(CalliopeBaseModel):
    """Calliope's generic dimension data schema."""

    model_config = {"title": "Generic dimension data", "extra": "allow"}

    name: str | None = None
    __pydantic_extra__: dict[AttrStr, IndexedParam | DataValue]


class IndexedTechParam(IndexedParam):
    """Tech-specific parameter schema."""

    model_config = {"title": "Indexed `techs` parameter definition"}

    @field_validator("dims", mode="before")
    @classmethod
    def _forbidden(
        cls, value: AttrStr | NonEmptyUniqueList[AttrStr]
    ) -> AttrStr | NonEmptyUniqueList[AttrStr]:
        forbidden = {"techs", "nodes"}
        if any(set(listify(value)) & forbidden):
            raise ValueError(f"Must not contain {forbidden}, found {value}.")
        return value


class CalliopeTech(CalliopeDimensionData):
    """Calliope's technology dimension schema."""

    model_config = {"title": "Technology dimension data"}

    active: bool = True
    base_tech: BaseTech
    carrier_in: AttrStr | NonEmptyUniqueList[AttrStr] | None = None
    carrier_out: AttrStr | NonEmptyUniqueList[AttrStr] | None = None
    carrier_export: AttrStr | NonEmptyUniqueList[AttrStr] | None = None
    link_from: AttrStr | NonEmptyUniqueList[AttrStr] | None = None
    link_to: AttrStr | NonEmptyUniqueList[AttrStr] | None = None

    @model_validator(mode="after")
    def check_base_tech_dependencies(self) -> Self:
        """Ensure technologies are defined correctly."""
        match self.base_tech:
            case "conversion":
                require = ["carrier_in", "carrier_out"]
                exclude = ["link_from", "link_to"]
            case "demand":
                require = ["carrier_in"]
                exclude = ["carrier_out", "link_from", "link_to"]
            case "storage":
                require = ["carrier_in", "carrier_out"]
                exclude = ["link_from", "link_to"]
            case "supply":
                require = ["carrier_out"]
                exclude = ["carrier_in", "link_from", "link_to"]
            case "transmission":
                require = ["carrier_in", "carrier_out", "link_from", "link_to"]
                exclude = []
            case _:
                raise ValueError(f"Invalid 'base_tech'. Must be one of {BaseTech}.")

        if not all([getattr(self, i) for i in require]) and any(
            [getattr(self, i) for i in exclude]
        ):
            raise ValueError(
                f"""Incorrect {self.base_tech} setup. Required: {require}. Invalid: {exclude}."""
            )

        return self


class CalliopeNode(CalliopeBaseModel):
    """Calliope's node dimension schema."""

    model_config = {"title": "Node dimension", "extra": "allow"}
    active: bool = True
    latitude: NumericVal | None = Field(default=None, ge=-90, le=90)
    longitude: NumericVal | None = Field(default=None, ge=-180, le=180)
    techs: EmptyDict | dict[AttrStr, None | dict[AttrStr, DataValue | IndexedTechParam]]
    available_area: NumericVal = Field(default=float("inf"))

    __pydantic_extra__: dict[AttrStr, IndexedTechParam | DataValue]

    @model_validator(mode="after")
    def check_dependent_definitions(self) -> Self:
        """Ensure dependent settings are defined."""
        if type(self.latitude) is not type(self.longitude):
            raise ValueError(
                f"Invalid latitude/longitude definition. Types must match, found ({self.latitude}, {self.longitude})."
            )
        return self
