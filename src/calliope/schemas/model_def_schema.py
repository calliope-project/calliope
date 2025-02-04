"""Calliope model definition schema."""

from typing import Literal

from pydantic import Field, model_validator
from typing_extensions import Self

from calliope.schemas.attributes import (
    CalliopeBaseModel,
    CnfStr,
    NonEmptyList,
    NonEmptyUniqueList,
)
from calliope.schemas.config_schema import CalliopeConfig
from calliope.schemas.data_table_schema import DataTable

DATA_T = CnfStr | bool | int | float | None
INDEX_T = CnfStr | int | float

BASE_TECH_T = Literal["conversion", "demand", "storage", "supply", "transmission"]


class IndexParam(CalliopeBaseModel):
    """Uniform dictionary for indexed parameters."""

    data: DATA_T | NonEmptyList[DATA_T]
    """
    Parameter value(s).
    If data is one value, will be applied to all dimension members.
    If a list, must be same length as the dims array.
    """
    dims: CnfStr | NonEmptyUniqueList[CnfStr]
    """
    Model dimension(s) over which the parameter is indexed.
    """
    index: INDEX_T | NonEmptyUniqueList[INDEX_T | NonEmptyUniqueList[INDEX_T]]
    """
    Model dimension members to apply the parameter value(s) to.
    Must have same length as number of `dims` (if list of lists, sub-lists must match that length).
    """


class Tech(CalliopeBaseModel):
    """Calliope's technology dimension schema."""

    model_config = {"title": "Technology dimension", "extra": "allow"}
    active: bool = True
    base_tech: BASE_TECH_T
    __pydantic_extra__: dict[CnfStr, IndexParam | DATA_T] = Field(
        union_mode="left_to_right"
    )

    @model_validator(mode="after")
    def check_base_tech_dependencies(self) -> Self:
        """Ensure technologies are defined correctly."""
        match self.base_tech:
            case "conversion":
                required = ["carrier_in", "carrier_out"]
                invalid = ["from", "to"]
            case "demand":
                required = ["carrier_in"]
                invalid = ["carrier_out", "from", "to"]
            case "storage":
                required = ["carrier_in", "carrier_out"]
                invalid = ["from", "to"]
            case "supply":
                required = ["carrier_out"]
                invalid = ["carrier_in", "from", "to"]
            case "transmission":
                required = ["carrier_in", "carrier_out", "from", "to"]
                invalid = []
            case _:
                raise ValueError(f"Invalid 'base_tech', must be one of {BASE_TECH_T}")

        if not all([hasattr(self, i) for i in required]) or any(
            [hasattr(self, i) for i in invalid]
        ):
            raise ValueError("Invalid 'base_tech' configuration.")

        return self


class CalliopeModelDef(CalliopeBaseModel):
    """Calliope model definition."""

    model_config = {"title": "Calliope Model Definition"}

    config: CalliopeConfig
    data_tables: DataTable
    # parameters: Parameters
    # techs: Techs
    # nodes: Nodes
